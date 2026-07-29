#include "memory/backend.h"
#include "memory/vram_query.h"
#include "core/logging.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace imp {

namespace {

constexpr int kNumTags = static_cast<int>(RegionTag::Other) + 1;

std::atomic<AllocPhase> g_phase{AllocPhase::Loading};
std::atomic<uint64_t> g_steady_allocs[kNumTags]{};
std::atomic<bool> g_steady_logged[kNumTags]{};

int tag_index(RegionTag t) {
    int i = static_cast<int>(t);
    return (i >= 0 && i < kNumTags) ? i : static_cast<int>(RegionTag::Other);
}

}  // namespace

const char* region_tag_name(RegionTag t) {
    switch (t) {
        case RegionTag::ModelResident:      return "model_resident";
        case RegionTag::EnginePersistent:   return "engine_persistent";
        case RegionTag::KvBlockPool:        return "kv_block_pool";
        case RegionTag::SwaBlockPool:       return "swa_block_pool";
        case RegionTag::ResidualRing:       return "residual_ring";
        case RegionTag::SsmState:           return "ssm_state";
        case RegionTag::RecurrentSnapshots: return "recurrent_snapshots";
        case RegionTag::ForwardScratch:     return "forward_scratch";
        case RegionTag::HostStaging:        return "host_staging";
        case RegionTag::Other:              return "other";
    }
    return "other";
}

const char* mem_error_name(MemError e) {
    switch (e) {
        case MemError::Ok:              return "ok";
        case MemError::OutOfMemory:     return "out_of_memory";
        case MemError::BudgetExceeded:  return "budget_exceeded";
        case MemError::NotGrowable:     return "not_growable";
        case MemError::InvalidArgument: return "invalid_argument";
    }
    return "unknown";
}

// --- phase guard ---

AllocPhase alloc_phase() { return g_phase.load(std::memory_order_relaxed); }

void set_alloc_phase(AllocPhase p) { g_phase.store(p, std::memory_order_relaxed); }

uint64_t steady_state_allocations() {
    uint64_t sum = 0;
    for (int i = 0; i < kNumTags; ++i)
        sum += g_steady_allocs[i].load(std::memory_order_relaxed);
    return sum;
}

uint64_t steady_state_allocations(RegionTag tag) {
    return g_steady_allocs[tag_index(tag)].load(std::memory_order_relaxed);
}

void reset_steady_state_allocations() {
    for (int i = 0; i < kNumTags; ++i) {
        g_steady_allocs[i].store(0, std::memory_order_relaxed);
        g_steady_logged[i].store(false, std::memory_order_relaxed);
    }
}

AllocPhaseScope::AllocPhaseScope(AllocPhase phase, const char* reason)
    : prev_(alloc_phase()), reason_(reason ? reason : "?") {
    IMP_LOG_INFO("alloc phase: entering %d for '%s'", static_cast<int>(phase), reason_);
    set_alloc_phase(phase);
}

AllocPhaseScope::~AllocPhaseScope() {
    set_alloc_phase(prev_);
    IMP_LOG_INFO("alloc phase: leaving '%s'", reason_);
}

namespace {

}  // namespace

// Records a device allocation made while serving. Debug aborts (the call site
// is the bug); release counts + logs once per tag and lets the server keep
// going. Called by Backend::acquire() and by the --wrap interposer, so the
// counter covers allocations that never went through Backend at all.
void note_serving_allocation(RegionTag tag, size_t bytes, const void* site) {
    if (alloc_phase() != AllocPhase::Serving)
        return;
    const int i = tag_index(tag);
    g_steady_allocs[i].fetch_add(1, std::memory_order_relaxed);
    if (!g_steady_logged[i].exchange(true, std::memory_order_relaxed)) {
        IMP_LOG_WARN("I2 violation: %.2f MiB for '%s' while serving (site %p) — this must be "
                     "drawn from a pre-planned pool (docs/MEMORY_ARCHITECTURE.md A3.2)",
                     bytes / (1024.0 * 1024.0), region_tag_name(tag), site);
    }
#ifndef NDEBUG
    IMP_LOG_ERROR("I2 violation (debug build is fatal): %.2f MiB for '%s' while serving",
                  bytes / (1024.0 * 1024.0), region_tag_name(tag));
    std::abort();
#endif
}

namespace {

void guard_serving_phase(size_t bytes, RegionTag tag) { note_serving_allocation(tag, bytes); }

}  // namespace

// --- Region ---

void Region::steal_(Region&& o) noexcept {
    owner_ = o.owner_;
    base_ = o.base_;
    committed_ = o.committed_;
    reserved_ = o.reserved_;
    tag_ = o.tag_;
    o.owner_ = nullptr;
    o.base_ = nullptr;
    o.committed_ = 0;
    o.reserved_ = 0;
}

void Region::reset() {
    if (owner_ && base_)
        owner_->release_(*this);
    owner_ = nullptr;
    base_ = nullptr;
    committed_ = 0;
    reserved_ = 0;
}

// --- Backend ---

Region Backend::make_region_(void* base, size_t committed, size_t reserved, RegionTag tag) {
    Region r;
    r.owner_ = this;
    r.base_ = base;
    r.committed_ = committed;
    r.reserved_ = reserved;
    r.tag_ = tag;
    return r;
}

void Backend::release_(Region& region) {
    do_release(region.base_, region.committed_, region.reserved_, region.tag_);
}

AcquireResult Backend::acquire(size_t bytes, size_t alignment, RegionTag tag) {
    AcquireResult res;
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        res.error = MemError::InvalidArgument;
        return res;
    }
    guard_serving_phase(bytes, tag);

    void* base = nullptr;
    size_t reserved = 0;
    res.error = do_acquire(bytes, alignment, tag, &base, &reserved);
    if (res.error != MemError::Ok)
        return res;
    res.region = make_region_(base, bytes, reserved ? reserved : bytes, tag);
    return res;
}

MemError Backend::do_acquire_growable(size_t, size_t, size_t, RegionTag, void**) {
    return MemError::NotGrowable;
}

void Backend::set_committed_(Region& r, size_t committed) { r.committed_ = committed; }

AcquireResult Backend::acquire_growable(size_t reserve_bytes, size_t initial_commit,
                                        size_t alignment, RegionTag tag) {
    AcquireResult res;
    if (reserve_bytes == 0 || initial_commit > reserve_bytes || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
        res.error = MemError::InvalidArgument;
        return res;
    }
    guard_serving_phase(initial_commit, tag);

    void* base = nullptr;
    res.error = do_acquire_growable(reserve_bytes, initial_commit, alignment, tag, &base);
    if (res.error != MemError::Ok)
        return res;
    res.region = make_region_(base, initial_commit, reserve_bytes, tag);
    return res;
}

// --- CudaMallocBackend ---

namespace {

class CudaMallocBackend final : public Backend {
public:
    MemError commit(Region&, size_t) override { return MemError::NotGrowable; }

    BackendStats stats() const override {
        std::lock_guard<std::mutex> lock(mu_);
        BackendStats s = stats_;
        s.capacity = capacity();
        return s;
    }

    size_t capacity() const override { return vram_budget_bytes(); }

protected:
    MemError do_acquire(size_t bytes, size_t alignment, RegionTag, void** out_base,
                        size_t* out_reserved) override {
        // cudaMalloc's own alignment is 256 B and in practice much larger;
        // anything stricter than that has to be handled by over-allocating,
        // which no current caller needs. Refuse rather than lie about it.
        if (alignment > 256)
            return MemError::InvalidArgument;

        const size_t cap = capacity();
        if (cap > 0) {
            std::lock_guard<std::mutex> lock(mu_);
            if (stats_.live_bytes + bytes > cap)
                return MemError::BudgetExceeded;
        }

        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess || p == nullptr) {
            (void)cudaGetLastError();  // clear the sticky error; the caller decides
            return MemError::OutOfMemory;
        }
        *out_base = p;
        *out_reserved = bytes;
        {
            std::lock_guard<std::mutex> lock(mu_);
            stats_.live_bytes += bytes;
            stats_.reserved_bytes += bytes;
            stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
            stats_.acquire_count++;
        }
        return MemError::Ok;
    }

    void do_release(void* base, size_t committed, size_t reserved, RegionTag) override {
        IMP_CUDA_CHECK_LOG(cudaFree(base));
        std::lock_guard<std::mutex> lock(mu_);
        stats_.live_bytes -= std::min(stats_.live_bytes, committed);
        stats_.reserved_bytes -= std::min(stats_.reserved_bytes, reserved);
        stats_.release_count++;
    }

private:
    mutable std::mutex mu_;
    BackendStats stats_;
};

}  // namespace

Backend& cuda_malloc_backend() {
    static CudaMallocBackend inst;
    return inst;
}

}  // namespace imp
