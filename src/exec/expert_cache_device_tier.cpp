// Host expert tier half of the device expert cache (#2621): pool sizing, the mailbox thread
// that serves tiered decode misses, and tiered prefill staging. Kernels: expert_cache_device.cu.
#include "exec/expert_cache_device.h"
#include "core/logging.h"
#include "memory/weight_snapshot.h"
#include "model/model.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace imp {

namespace {
constexpr int kMaxEntries = 3 * kDevExpertMaxTopK;
}  // namespace

bool DeviceExpertCache::init_tier_(const Model& model, const std::vector<char>& tiered, int top_k,
                                   int host_pool_mib) {
    const int n_layers = static_cast<int>(tiered.size());
    const int ne = n_experts_;
    size_t pb = 0, mb = 0;
    int n_tiered = 0;
    for (int l = 0; l < n_layers; ++l) {
        if (!tiered[l])
            continue;
        ++n_tiered;
        pb = std::max(pb, packed_bytes_[static_cast<size_t>(l) * 3 + 1]);
        mb = std::max(mb, ms_bytes_[static_cast<size_t>(l) * 3 + 1]);
    }
    const size_t unit = HostExpertTier::unit_bytes(pb, mb);
    const size_t need = static_cast<size_t>(n_tiered) * 3 * ne * unit;
    // Floor: one whole layer (touched-only prefill staging) plus one decode step.
    const size_t floor_b = (static_cast<size_t>(3) * ne + 3 * static_cast<size_t>(top_k)) * unit;
    constexpr size_t kHeadroom = 6ull << 30;
    size_t budget = 0;
    if (host_pool_mib > 0) {
        budget = static_cast<size_t>(host_pool_mib) << 20;
    } else {
        const size_t avail = host_mem_available_bytes();
        budget = avail > kHeadroom ? avail - kHeadroom : 0;
    }
    // LRU for every unit plus the prefill scratch (one layer) is the most the tier can use.
    size_t bytes = std::min(budget, need + static_cast<size_t>(3) * ne * unit) / unit * unit;
    const auto t0 = std::chrono::steady_clock::now();
    while (bytes >= floor_b) {
        tier_arena_ = PinnedBuffer::acquire_filled(bytes, HostPinnedKind::Mapped, [](void*) {});
        if (!tier_arena_.empty())
            break;
        bytes = bytes / 4 * 3 / unit * unit;
    }
    if (tier_arena_.empty()) {
        IMP_LOG_WARN(
            "Host expert tier: not built (budget %.2f GiB, floor %.2f GiB for one layer); "
            "the host LRU path serves decode",
            budget / (1024.0 * 1024.0 * 1024.0), floor_b / (1024.0 * 1024.0 * 1024.0));
        return false;
    }
    const size_t io_b = static_cast<size_t>(kMaxEntries) * sizeof(DevMissEntry) + 256 +
                        static_cast<size_t>(3) * ne * sizeof(DevExpertSrc) + 64 + sizeof(TierMailbox);
    tier_io_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(), io_b, HostPinnedKind::Mapped);
    if (tier_io_.empty() || !tier_io_.device()) {
        tier_arena_ = PinnedBuffer();
        return false;
    }
    char* h = static_cast<char*>(tier_io_.data());
    char* d = static_cast<char*>(tier_io_.device());
    const size_t miss_b = static_cast<size_t>(kMaxEntries) * sizeof(DevMissEntry);
    std::memset(h, 0, io_b);
    tier_misses_ = reinterpret_cast<DevMissEntry*>(h);
    tier_misses_d_ = reinterpret_cast<DevMissEntry*>(d);
    tier_n_miss_ = reinterpret_cast<int*>(h + miss_b);
    tier_n_miss_d_ = reinterpret_cast<int*>(d + miss_b);
    tier_stage_src_h_ = reinterpret_cast<DevExpertSrc*>(h + miss_b + 256);
    tier_stage_src_d_ = reinterpret_cast<DevExpertSrc*>(d + miss_b + 256);
    const size_t mb_off = (miss_b + 256 + static_cast<size_t>(3) * ne * sizeof(DevExpertSrc) + 63) / 64 * 64;
    tier_mb_ = reinterpret_cast<TierMailbox*>(h + mb_off);
    tier_mb_d_ = reinterpret_cast<TierMailbox*>(d + mb_off);

    tier_ = std::make_unique<HostExpertTier>(static_cast<char*>(tier_arena_.data()),
                                             static_cast<char*>(tier_arena_.device()), tier_arena_.bytes(),
                                             unit, n_layers * 3 * ne, 32, 3 * ne);
    for (int l = 0; l < n_layers; ++l) {
        if (!tiered[l])
            continue;
        const TransformerLayer& ly = model.layer(l);
        const std::vector<Tensor>* projs[3] = {&ly.expert_w_gate, &ly.expert_w_up, &ly.expert_w_down};
        for (int p = 0; p < 3; ++p) {
            const std::vector<Tensor>& ex = *projs[p];
            if (static_cast<int>(ex.size()) != ne)
                continue;  // absent gate projection
            const size_t i = static_cast<size_t>(l) * 3 + p;
            for (int e = 0; e < ne; ++e)
                tier_->set_source(static_cast<int>(i * ne + e),
                                  {ex[e].data, packed_bytes_[i], ex[e].scales, ms_bytes_[i]});
        }
    }
    tier_->bind_files();
    IMP_LOG_INFO(
        "Host expert tier: %d unit(s) x %.2f MiB = %.2f GiB pinned of %.2f GiB host-resident experts "
        "(%d layer(s), %.1f s to pin); misses read from the checkpoint",
        tier_->units(), unit / (1024.0 * 1024.0), tier_arena_.bytes() / (1024.0 * 1024.0 * 1024.0),
        need / (1024.0 * 1024.0 * 1024.0), n_tiered,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// Fills the sources of one tiered decode layer's misses (tier thread, no CUDA calls).
void DeviceExpertCache::tier_fill_(int layer) {
    const int n = *reinterpret_cast<volatile int*>(tier_n_miss_);
    if (n <= 0)
        return;
    int keys[kMaxEntries];
    HostExpertTier::UnitView views[kMaxEntries];
    const int base = layer * 3 * n_experts_;
    for (int i = 0; i < n; ++i)
        keys[i] = base + tier_misses_[i].key;
    if (!tier_->acquire(keys, n, views)) {
        IMP_LOG_ERROR("Host expert tier: %d misses exceed %d units on layer %d", n, tier_->units(), layer);
        for (int i = 0; i < n; ++i)
            tier_misses_[i].packed = tier_misses_[i].ms = nullptr;
        return;
    }
    for (int i = 0; i < n; ++i) {
        tier_misses_[i].packed = views[i].packed;
        tier_misses_[i].ms = views[i].ms;
    }
    std::atomic_thread_fence(std::memory_order_release);
}

// Requests arrive strictly one at a time: the stream waits for each ack before the next
// resolve. Spins while decode is active, sleeps 50 us between polls after 2 ms idle.
void DeviceExpertCache::tier_serve_() {
    volatile uint32_t* req = &tier_mb_->req_seq;
    volatile uint32_t* ack = &tier_mb_->ack_seq;
    uint32_t done = *ack;
    auto last = std::chrono::steady_clock::now();
    while (!tier_stop_.load(std::memory_order_acquire)) {
        const uint32_t s = *req;
        if (s == done) {
            if (std::chrono::steady_clock::now() - last > std::chrono::milliseconds(2))
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        const auto t_in = std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_acquire);
        tier_fill_(*reinterpret_cast<volatile int32_t*>(&tier_mb_->layer));
        std::atomic_thread_fence(std::memory_order_release);
        *ack = s;
        done = s;
        const auto t_out = std::chrono::steady_clock::now();
        if (tier_timing_.calls++ > 0)
            tier_timing_.gap_ms += std::chrono::duration<double, std::milli>(t_in - last).count();
        tier_timing_.busy_ms += std::chrono::duration<double, std::milli>(t_out - t_in).count();
        last = t_out;
    }
}

const DevExpertSrc* DeviceExpertCache::tier_stage_src_(int layer, const int32_t* expert_offsets, int e0,
                                                       int n, int p0, int p1, cudaStream_t stream) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (!tier_ || cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone)
        return nullptr;
    const int ne = n_experts_;
    // The previous staging kernel may still read units this call could evict.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return nullptr;
    std::vector<int32_t> offs(static_cast<size_t>(ne) + 1);
    if (cudaMemcpy(offs.data(), expert_offsets, offs.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        return nullptr;
    std::vector<int> projs;
    if (p0 < 0)
        projs = {0, 1, 2};
    else
        projs = {p0};
    if (p1 >= 0)
        projs.push_back(p1);
    std::vector<int> keys, idx;
    const int base = layer * 3 * ne;
    for (int p : projs) {
        if (packed_bytes_[static_cast<size_t>(layer) * 3 + p] == 0)
            continue;  // absent gate projection
        for (int e = std::max(0, e0); e < std::min(ne, e0 + n); ++e) {
            if (offs[e + 1] == offs[e])
                continue;
            keys.push_back(base + p * ne + e);
            idx.push_back(p * ne + e);
        }
    }
    std::vector<HostExpertTier::UnitView> views(keys.size());
    if (!keys.empty() && !tier_->acquire(keys.data(), static_cast<int>(keys.size()), views.data(), true))
        return nullptr;
    std::memset(tier_stage_src_h_, 0, static_cast<size_t>(3) * ne * sizeof(DevExpertSrc));
    for (size_t i = 0; i < keys.size(); ++i)
        tier_stage_src_h_[idx[i]] = DevExpertSrc{views[i].packed, views[i].ms, 0.0f};
    return tier_stage_src_d_;
}

}  // namespace imp
