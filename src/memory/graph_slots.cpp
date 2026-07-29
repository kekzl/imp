#include "memory/graph_slots.h"

#include "core/logging.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

namespace imp {

namespace {

// CUDA's natural allocation alignment. Every sub-buffer in a slot starts here
// so a slot-backed pointer is indistinguishable from a cudaMalloc'd one.
constexpr size_t kSubAlign = 256;

constexpr size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

constexpr size_t kSampleScratchBytes = kGraphSlotSampleScratchBytes;

// Scalar count: position, context_len, step_counter, step_limit, think_limit,
// think_count, in_think, think_exit_step, content_after_think, penalty_count.
constexpr int kNumScalars = 10;

}  // namespace

// ── layout ──────────────────────────────────────────────────────────────
//
// device slot:  [sample scratch][10 ints][stop ids][penalty ring]
// host slot:    [ring buffer][step counter][burst done][decode scratch]
//
// Each sub-buffer is kSubAlign-aligned, so the strides below are what one slot
// costs and slot i starts at i * stride.

static size_t device_stride_for(const GraphSlotCaps& caps) {
    size_t off = 0;
    off += align_up(kSampleScratchBytes, kSubAlign);
    off += align_up(kNumScalars * sizeof(int), kSubAlign);
    off += align_up(static_cast<size_t>(std::max(caps.stop_ids, 0)) * sizeof(int32_t), kSubAlign);
    off += align_up(static_cast<size_t>(std::max(caps.penalty_slots, 0)) * sizeof(int32_t), kSubAlign);
    return off;
}

static size_t host_stride_for(const GraphSlotCaps& caps) {
    size_t off = 0;
    off += align_up(static_cast<size_t>(std::max(caps.max_steps, 0)) * sizeof(int32_t), kSubAlign);
    off += align_up(sizeof(int), kSubAlign);   // step counter
    off += align_up(sizeof(int), kSubAlign);   // burst done
    off += align_up(sizeof(int32_t), kSubAlign);  // decode scratch
    return off;
}

GraphSlotPool::~GraphSlotPool() { close(); }

namespace {

class CudaHostPinnedAllocator final : public HostPinnedAllocator {
public:
    bool alloc(size_t bytes, void** out_host, void** out_device) override {
        void* hp = nullptr;
        const cudaError_t herr = cudaHostAlloc(&hp, bytes, cudaHostAllocMapped);
        if (herr != cudaSuccess) {
            IMP_LOG_WARN("graph slot pool: pinned host alloc failed (%zu B): %s", bytes,
                         cudaGetErrorString(herr));
            return false;
        }
        void* hdev = nullptr;
        const cudaError_t derr = cudaHostGetDevicePointer(&hdev, hp, 0);
        if (derr != cudaSuccess) {
            IMP_LOG_WARN("graph slot pool: host device pointer failed: %s", cudaGetErrorString(derr));
            cudaFreeHost(hp);
            return false;
        }
        *out_host = hp;
        *out_device = hdev;
        return true;
    }
    void free(void* host) override {
        if (host)
            IMP_CUDA_CHECK_LOG(cudaFreeHost(host));
    }
};

}  // namespace

HostPinnedAllocator& cuda_host_pinned_allocator() {
    static CudaHostPinnedAllocator a;
    return a;
}

MemError GraphSlotPool::open(Backend& backend, HostPinnedAllocator& host, const GraphSlotCaps& caps,
                             int num_slots) {
    std::lock_guard<std::mutex> lock(mu_);
    if (open_)
        return MemError::InvalidArgument;
    if (num_slots <= 0 || caps.max_steps <= 0 || caps.penalty_slots <= 0 || caps.stop_ids < 0)
        return MemError::InvalidArgument;

    slot_device_stride_ = device_stride_for(caps);
    slot_host_stride_ = host_stride_for(caps);
    const size_t dev_total = slot_device_stride_ * static_cast<size_t>(num_slots);
    const size_t host_total = slot_host_stride_ * static_cast<size_t>(num_slots);

    auto got = backend.acquire(dev_total, kSubAlign, RegionTag::EnginePersistent);
    if (!got)
        return got.error;

    void* hp = nullptr;
    void* hdev = nullptr;
    if (!host.alloc(host_total, &hp, &hdev))
        return MemError::OutOfMemory;

    region_ = std::move(got.region);
    host_alloc_ = &host;
    host_base_ = hp;
    host_device_base_ = hdev;
    caps_ = caps;
    num_slots_ = num_slots;
    device_bytes_ = dev_total;
    host_bytes_ = host_total;
    in_use_.assign(static_cast<size_t>(num_slots), false);
    declines_exhausted_ = 0;
    declines_too_small_ = 0;
    open_ = true;

    IMP_LOG_INFO("graph slot pool: %d slots (%.1f KiB device + %.1f KiB pinned host), caps: "
                 "max_steps=%d penalty_slots=%d stop_ids=%d",
                 num_slots, dev_total / 1024.0, host_total / 1024.0, caps.max_steps,
                 caps.penalty_slots, caps.stop_ids);
    return MemError::Ok;
}

void GraphSlotPool::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!open_)
        return;
    const int leaked = static_cast<int>(std::count(in_use_.begin(), in_use_.end(), true));
    if (leaked > 0) {
        // A lease outliving the pool means a runner is still holding addresses
        // we are about to free. Loud, because the graph would replay into it.
        IMP_LOG_ERROR("graph slot pool: closing with %d slot(s) still leased", leaked);
    }
    if (declines_exhausted_ || declines_too_small_) {
        IMP_LOG_INFO("graph slot pool: %llu declines (%llu exhausted, %llu too small)",
                     static_cast<unsigned long long>(declines_exhausted_ + declines_too_small_),
                     static_cast<unsigned long long>(declines_exhausted_),
                     static_cast<unsigned long long>(declines_too_small_));
    }
    region_.reset();
    if (host_base_ && host_alloc_)
        host_alloc_->free(host_base_);
    host_base_ = nullptr;
    host_alloc_ = nullptr;
    host_device_base_ = nullptr;
    in_use_.clear();
    num_slots_ = 0;
    device_bytes_ = 0;
    host_bytes_ = 0;
    open_ = false;
}

bool GraphSlotPool::is_open() const {
    std::lock_guard<std::mutex> lock(mu_);
    return open_;
}

GraphSlotView GraphSlotPool::carve_(int index) const {
    GraphSlotView v{};
    auto* d = static_cast<std::byte*>(region_.base()) + slot_device_stride_ * static_cast<size_t>(index);

    v.sample_scratch = d;
    d += align_up(kSampleScratchBytes, kSubAlign);

    int* scalars = reinterpret_cast<int*>(d);
    v.position = scalars + 0;
    v.context_len = scalars + 1;
    v.step_counter = scalars + 2;
    v.step_limit = scalars + 3;
    v.think_limit = scalars + 4;
    v.think_count = scalars + 5;
    v.in_think = scalars + 6;
    v.think_exit_step = scalars + 7;
    v.content_after_think = scalars + 8;
    v.penalty_count = scalars + 9;
    d += align_up(kNumScalars * sizeof(int), kSubAlign);

    v.stop_ids = reinterpret_cast<int32_t*>(d);
    d += align_up(static_cast<size_t>(caps_.stop_ids) * sizeof(int32_t), kSubAlign);

    v.penalty_ring = reinterpret_cast<int32_t*>(d);

    const size_t host_off = slot_host_stride_ * static_cast<size_t>(index);
    auto* h = static_cast<std::byte*>(host_base_) + host_off;
    auto* hd = static_cast<std::byte*>(host_device_base_) + host_off;
    const size_t ring_bytes = align_up(static_cast<size_t>(caps_.max_steps) * sizeof(int32_t), kSubAlign);
    const size_t scalar_bytes = align_up(sizeof(int), kSubAlign);

    v.h_ring = reinterpret_cast<int32_t*>(h);
    v.d_ring = reinterpret_cast<int32_t*>(hd);
    h += ring_bytes;
    hd += ring_bytes;

    v.h_step_counter = reinterpret_cast<int*>(h);
    v.d_step_counter_mapped = reinterpret_cast<int*>(hd);
    h += scalar_bytes;
    hd += scalar_bytes;

    v.h_burst_done = reinterpret_cast<int*>(h);
    v.d_burst_done_mapped = reinterpret_cast<int*>(hd);
    h += scalar_bytes;

    v.h_decode_scratch = reinterpret_cast<int32_t*>(h);
    return v;
}

GraphSlotLease GraphSlotPool::acquire(const GraphSlotCaps& need) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!open_)
        return {};
    if (need.max_steps > caps_.max_steps || need.penalty_slots > caps_.penalty_slots ||
        need.stop_ids > caps_.stop_ids) {
        ++declines_too_small_;
        return {};
    }
    for (int i = 0; i < num_slots_; ++i) {
        if (in_use_[static_cast<size_t>(i)])
            continue;
        in_use_[static_cast<size_t>(i)] = true;
        return GraphSlotLease(this, i, carve_(i));
    }
    ++declines_exhausted_;
    return {};
}

void GraphSlotPool::release_(int index) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!open_ || index < 0 || index >= num_slots_)
        return;
    in_use_[static_cast<size_t>(index)] = false;
}

int GraphSlotPool::free_slots() const {
    std::lock_guard<std::mutex> lock(mu_);
    return open_ ? static_cast<int>(std::count(in_use_.begin(), in_use_.end(), false)) : 0;
}

uint64_t GraphSlotPool::declines() const {
    std::lock_guard<std::mutex> lock(mu_);
    return declines_exhausted_ + declines_too_small_;
}

uint64_t GraphSlotPool::declines_exhausted() const {
    std::lock_guard<std::mutex> lock(mu_);
    return declines_exhausted_;
}

uint64_t GraphSlotPool::declines_too_small() const {
    std::lock_guard<std::mutex> lock(mu_);
    return declines_too_small_;
}

void GraphSlotLease::release() {
    if (pool_ && index_ >= 0)
        pool_->release_(index_);
    pool_ = nullptr;
    index_ = -1;
    view_ = {};
}

GraphSlotPool& graph_slot_pool() {
    static GraphSlotPool pool;
    return pool;
}

void graph_slot_pool_open_for(Backend& backend, int max_seq_len) {
    if (max_seq_len <= 0)
        return;
    GraphSlotCaps caps;
    caps.max_steps = max_seq_len;
    caps.penalty_slots = max_seq_len * 2;
    caps.stop_ids = 64;
    const MemError e = graph_slot_pool().open(backend, caps, 4);
    if (e != MemError::Ok)
        IMP_LOG_WARN("graph slot pool: open failed (%s) — per-burst allocation stays",
                     mem_error_name(e));
}

}  // namespace imp
