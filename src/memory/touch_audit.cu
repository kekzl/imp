#include "memory/touch_audit.h"
#include "memory/device_alloc_all.h"

#include "core/logging.h"
#include "memory/backend.h"
#include "memory/engine_arena.h"
#include "memory/mem_account.h"
#include "memory/vram_allocator.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <map>

namespace imp {

namespace {

constexpr int kT = 256;
constexpr size_t kPageWords = TouchAudit::kPage / sizeof(uint32_t);

// out[page] = position-weighted sum of the page's 32-bit words (a trailing 1..3 bytes are not read).
__global__ void touch_hash_k(const uint32_t* p, size_t words, unsigned long long* out) {
    __shared__ unsigned long long sh[kT];
    const size_t base = (size_t)blockIdx.x * kPageWords, end = min(base + kPageWords, words);
    unsigned long long h = 0;
    for (size_t i = base + threadIdx.x; i < end; i += kT)
        h += (unsigned long long)(p[i] ^ 0x9E3779B9u) * (0x100000001B3ull * (i - base + 1) | 1ull);
    sh[threadIdx.x] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long s = 0;
        for (int t = 0; t < kT; ++t)
            s += sh[t];
        out[blockIdx.x] = s;
    }
}

struct Range {
    const void* ptr = nullptr;
    size_t bytes = 0;
    std::string tag;
};

// What the three registries hold: VRAMAllocator allocations, Backend regions (the committed chunks of a
// growable one) and MemAccount::note_alloc rows (weights). A range inside an earlier one is dropped.
std::vector<Range> live_ranges(const VRAMAllocator& va) {
    std::vector<Range> all;
    for (const VRAMAllocator::Live& a : va.live())
        all.push_back({a.ptr, a.bytes, a.tag});
    std::vector<Backend::LiveRange> regions;
    cuda_malloc_backend().live_ranges(regions);
    if (Backend* vmm = vmm_backend())
        vmm->live_ranges(regions);
    // The engine arena by taking site instead of by committed chunk: arena:<file>, with the line for a
    // take of 8 MiB or more.
    const std::vector<ArenaAllocator::Take> takes = engine_arena().takes();
    for (const ArenaAllocator::Take& t : takes) {
        const std::string file = t.file ? t.file : "?";
        std::string tag = "arena:" + file.substr(file.find_last_of('/') + 1);
        if (t.bytes >= (size_t{8} << 20))
            tag += ":" + std::to_string(t.line);
        all.push_back({t.ptr, t.bytes, std::move(tag)});
    }
    for (const Backend::LiveRange& r : regions) {
        if (!takes.empty() && r.tag == RegionTag::EnginePersistent)
            continue;
        all.push_back({r.ptr, r.bytes, std::string("region:") + region_tag_name(r.tag)});
    }
    for (const MemAccount::Charged& c : MemAccount::instance().charged())
        all.push_back({c.ptr, c.bytes, c.pool ? c.pool : "unknown"});
    std::sort(all.begin(), all.end(), [](const Range& a, const Range& b) {
        return a.ptr != b.ptr ? a.ptr < b.ptr : a.bytes > b.bytes;
    });
    std::vector<Range> out;
    const char* end = nullptr;
    for (Range& r : all) {
        const char* p = static_cast<const char*>(r.ptr);
        if (end && p + r.bytes <= end)
            continue;
        end = p + r.bytes;
        out.push_back(std::move(r));
    }
    return out;
}

}  // namespace

TouchAudit::~TouchAudit() {
    if (d_out_)
        device_free_all(d_out_);
}

bool TouchAudit::hash_(const void* p, size_t bytes, std::vector<uint64_t>& out) {
    const size_t words = bytes / sizeof(uint32_t), pages = (words + kPageWords - 1) / kPageWords;
    out.assign(pages, 0);
    if (pages == 0)
        return true;
    if (pages > d_cap_) {
        if (d_out_)
            device_free_all(d_out_);
        d_out_ = nullptr, d_cap_ = 0;
        void* q = nullptr;
        if (device_alloc_all(dev_req(q, pages * sizeof(uint64_t))) != cudaSuccess)
            return false;
        d_out_ = static_cast<uint64_t*>(q), d_cap_ = pages;
    }
    touch_hash_k<<<static_cast<unsigned>(pages), kT>>>(static_cast<const uint32_t*>(p), words,
                                                       reinterpret_cast<unsigned long long*>(d_out_));
    IMP_CUDA_CHECK_LAUNCH();
    return cudaMemcpy(out.data(), d_out_, pages * sizeof(uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess;
}

std::unique_ptr<TouchAudit> TouchAudit::start(bool on, const VRAMAllocator& va) {
    auto audit = on ? std::make_unique<TouchAudit>() : nullptr;
    if (audit && !audit->snapshot(va))
        audit.reset();
    return audit;
}

bool TouchAudit::snapshot(const VRAMAllocator& va) {
    snap_.clear();
    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;
    size_t total = 0;
    for (const Range& a : live_ranges(va)) {
        Entry e;
        e.bytes = a.bytes;
        if (!hash_(a.ptr, a.bytes, e.pages)) {
            snap_.clear();
            IMP_LOG_WARN("vram touch audit: hashing %s (%.1f MiB) failed, audit off", a.tag.c_str(),
                         static_cast<double>(a.bytes) / 1048576.0);
            return false;
        }
        total += a.bytes;
        snap_.emplace(a.ptr, std::move(e));
    }
    IMP_LOG_INFO("vram touch audit: %zu ranges, %.0f MiB hashed in %zu KiB pages", snap_.size(),
                 static_cast<double>(total) / 1048576.0, kPage / 1024);
    return true;
}

std::vector<TouchAudit::Row> TouchAudit::compare(const VRAMAllocator& va) {
    std::vector<Row> rows;
    if (snap_.empty() || cudaDeviceSynchronize() != cudaSuccess)
        return rows;
    std::map<std::string, Row> by_tag;
    std::vector<uint64_t> now;
    for (const Range& a : live_ranges(va)) {
        Row& r = by_tag[a.tag];
        r.tag = a.tag;
        const auto it = snap_.find(a.ptr);
        if (it == snap_.end() || it->second.bytes != a.bytes || !hash_(a.ptr, a.bytes, now)) {
            r.fresh += a.bytes;
            continue;
        }
        r.bytes += a.bytes;
        ++r.allocs;
        size_t changed = 0;
        for (size_t i = 0; i < now.size(); ++i)
            changed += now[i] != it->second.pages[i];
        r.written += std::min(changed * kPage, a.bytes);
    }
    for (auto& [tag, r] : by_tag)
        rows.push_back(std::move(r));
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.bytes - a.written > b.bytes - b.written; });
    return rows;
}

void TouchAudit::report(const VRAMAllocator& va, const char* label) {
    const std::vector<Row> rows = compare(va);
    if (rows.empty())
        return;
    const double mib = 1048576.0;
    size_t bytes = 0, written = 0;
    IMP_LOG_INFO("===== VRAM TOUCH AUDIT [%s]: MiB held since warmup, MiB in pages written since =====",
                 label);
    for (const Row& r : rows) {
        bytes += r.bytes, written += r.written;
        IMP_LOG_INFO("touch %-26s %9.1f %9.1f %5.1f %%  (%d allocs, %.1f MiB new)", r.tag.c_str(),
                     static_cast<double>(r.bytes) / mib, static_cast<double>(r.written) / mib,
                     r.bytes ? 100.0 * static_cast<double>(r.written) / static_cast<double>(r.bytes) : 0.0,
                     r.allocs, static_cast<double>(r.fresh) / mib);
    }
    IMP_LOG_INFO("touch %-26s %9.1f %9.1f %5.1f %%", "TOTAL", static_cast<double>(bytes) / mib,
                 static_cast<double>(written) / mib,
                 bytes ? 100.0 * static_cast<double>(written) / static_cast<double>(bytes) : 0.0);
}

}  // namespace imp
