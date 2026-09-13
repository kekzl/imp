#pragma once

// L1 of the memory architecture (docs/internals/MEMORY.md §A3.1/§A3.2):
// physical acquisition, the allocation-phase guard, and accounting.
//
// This is the ONLY layer that is allowed to talk to the CUDA driver about
// memory. Everything above it (tier allocators, typed handles, subsystems)
// receives Regions and views. What this layer is deliberately NOT responsible
// for: lifetime tiering, sizing policy, or deciding how much anything gets —
// those live in the allocators (§A3.3) and the planner (§A4).
//
// Backend is an interface so tests can substitute FakeBackend (§A6) and run the
// whole allocator stack on host memory, in the CPU-only CI lane, with no GPU.

#include <cstddef>
#include <cstdint>
#include <utility>

namespace imp {

// Tags: the unit of reporting for I7 (capacity and occupancy are separate concepts,
// reported separately). One tag per thing an operator recognizes in --mem-report, not
// one per allocation site.
enum class RegionTag {
    ModelResident,        // T1 arena: weights + the pre-dequant weight caches
    EnginePersistent,     // T2 arena: workspaces, cuBLAS/CUTLASS, graph buffers
    KvBlockPool,          // T3: the global paged KV block group
    SwaBlockPool,         // T3: the dedicated sliding-window block group
    ResidualRing,         // T3: BitDecoding FP16 residual ring
    SsmState,             // T3: per-sequence SSM/GDN conv + h state
    RecurrentSnapshots,   // T3: hybrid prefix-state snapshot store
    ForwardScratch,       // T4: the LIFO scratch stack
    HostStaging,          // T5: pinned/pageable host staging (load only)
    Other,
};

const char* region_tag_name(RegionTag);

enum class MemError {
    Ok = 0,
    OutOfMemory,       // the device (or the fake's capacity) said no
    BudgetExceeded,    // would exceed the installed --vram-budget
    NotGrowable,       // commit() on a backend that cannot grow a region
    InvalidArgument,
};

const char* mem_error_name(MemError);

// Allocation phase (I2), monotonic within a model's lifetime, driven by the engine.
// Serving means warmup is done; asking the driver for memory after that is a defect.
// Debug builds abort on a Serving-phase acquisition; release builds count it, log once
// per tag, and proceed. The counter is the I2 acceptance test and must reach zero.
enum class AllocPhase { Loading, Planning, Serving };

AllocPhase alloc_phase();
void set_alloc_phase(AllocPhase);

// Record a device allocation made while serving. Backend::acquire() calls this itself;
// the --wrap interposer (alloc_interpose.cpp) calls it for allocations that never went
// through Backend, making steady_state_allocations() authoritative rather than
// indicative. `site` is the caller's return address (symbolize with addr2line) or null.
void note_serving_allocation(RegionTag tag, size_t bytes, const void* site = nullptr);

// A commit into a growable region's reservation while serving. Planned by
// construction (the reservation was sized at init), so it is counted apart
// from the I2 violations above and logged once per tag at INFO.
void note_planned_commit(RegionTag tag, size_t bytes);

// Total acquisitions observed while in AllocPhase::Serving, all tags.
uint64_t steady_state_allocations();
// Per-tag breakdown; `tag` indexes the RegionTag enum.
uint64_t steady_state_allocations(RegionTag tag);
void reset_steady_state_allocations();
// Commits into growable reservations while serving (lazy pools), all tags.
uint64_t planned_serving_commits();

// RAII bracket for the one legitimate re-entry into an allocating phase
// after serving has begun: server.model_swap tears the model down and builds
// a new one. Logged on both edges so it cannot be used silently.
class AllocPhaseScope {
public:
    AllocPhaseScope(AllocPhase phase, const char* reason);
    ~AllocPhaseScope();
    AllocPhaseScope(const AllocPhaseScope&) = delete;
    AllocPhaseScope& operator=(const AllocPhaseScope&) = delete;

private:
    AllocPhase prev_;
    const char* reason_;
};

class Backend;

// Region: the only type holding a raw device pointer from the driver. Move-only, RAII:
// the destructor returns it to its backend. A tier allocator owns one (or a few) and
// hands out views; nothing else ever sees one.
// `reserved` >= `committed`; they differ only for growable (VMM) backends, where
// reserved is virtual address range and committed is mapped physical memory. Equal for
// the cudaMalloc backend.
class Region {
public:
    Region() = default;
    ~Region() { reset(); }

    Region(Region&& other) noexcept { steal_(std::move(other)); }
    Region& operator=(Region&& other) noexcept {
        if (this != &other) {
            reset();
            steal_(std::move(other));
        }
        return *this;
    }
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;

    // Return the memory to its backend. Idempotent.
    void reset();

    void* base() const { return base_; }
    size_t committed() const { return committed_; }
    size_t reserved() const { return reserved_; }
    RegionTag tag() const { return tag_; }
    bool valid() const { return base_ != nullptr; }
    explicit operator bool() const { return valid(); }

private:
    friend class Backend;
    void steal_(Region&& o) noexcept;

    Backend* owner_ = nullptr;
    void* base_ = nullptr;
    size_t committed_ = 0;
    size_t reserved_ = 0;
    RegionTag tag_ = RegionTag::Other;
};

struct AcquireResult {
    Region region;
    MemError error = MemError::Ok;
    explicit operator bool() const { return error == MemError::Ok; }
};

struct BackendStats {
    size_t live_bytes = 0;       // currently committed across all live regions
    size_t peak_bytes = 0;       // high-water of live_bytes
    size_t reserved_bytes = 0;   // virtual reservation (== live for cudaMalloc)
    uint64_t acquire_count = 0;
    uint64_t release_count = 0;
    size_t capacity = 0;         // 0 = "whatever the device has"
};

// Backend: physical acquisition. Fails cleanly; never throws, never aborts on
// out-of-memory (I6: exhaustion is a typed, recoverable value, not a crash deep inside a
// kernel launch).
class Backend {
public:
    virtual ~Backend() = default;

    // Acquire `bytes`, aligned to `alignment` (a power of two, >= 256).
    // Consults the phase guard before doing anything.
    AcquireResult acquire(size_t bytes, size_t alignment, RegionTag tag);

    // Reserve reserve_bytes of address space and commit initial_commit of it. The region's
    // base() is then stable for the whole reservation, which is what lets a graph-captured
    // pointer survive growth (I3). Backends that cannot do this return NotGrowable and the
    // caller falls back to a fixed acquire().
    AcquireResult acquire_growable(size_t reserve_bytes, size_t initial_commit, size_t alignment,
                                   RegionTag tag);

    // Grow/shrink a growable region in place, keeping base() stable. Returns NotGrowable on
    // backends that cannot (cudaMalloc). Non-virtual: it counts the growth
    // (note_planned_commit) then dispatches to do_commit(), so a growable pool committing
    // pages on the request path is counted like acquire() is (previously missed, #1649).
    MemError commit(Region& region, size_t new_committed);

    // Commit (or release) one interior range of a growable region. A paged pool does not
    // grow at its end: the KV pool lays blocks out per layer, so growth extends every
    // layer's sub-range at once, and the committed set is multiple interior prefixes, not
    // one; commit() alone can only express that by committing up to the last layer.
    // offset and bytes round OUT to the backend's granularity; callers needing separated
    // sub-ranges must keep them granule-aligned (the KV pool pads per-layer strides for
    // exactly that, which costs only address space).
    MemError commit_range(Region& region, size_t offset, size_t bytes);
    virtual MemError decommit_range(Region& region, size_t offset, size_t bytes);

    virtual BackendStats stats() const = 0;

    // Installed hard cap in bytes (--vram-budget); 0 = uncapped.
    virtual size_t capacity() const = 0;

    // Commit granularity of a growable region: commit_range() rounds out to
    // it. A lazy slab that commits per slot pads its slot stride to this so
    // two slots never share a granule.
    virtual size_t granularity() const { return 256; }

protected:
    // Implementations override these two. `acquire()` wraps do_acquire() with
    // the phase guard and accounting so no backend can forget either.
    virtual MemError do_acquire(size_t bytes, size_t alignment, RegionTag tag, void** out_base,
                                size_t* out_reserved) = 0;
    virtual void do_release(void* base, size_t committed, size_t reserved, RegionTag tag) = 0;

    // Growth. `commit()`/`commit_range()` wrap these with the phase guard, so
    // no backend can forget it - the same argument acquire()/do_acquire() makes
    // above, applied to the entry points that grow rather than acquire.
    virtual MemError do_commit(Region& region, size_t new_committed) = 0;
    virtual MemError do_commit_range(Region& region, size_t offset, size_t bytes);

    // Growable acquisition. Default: unsupported.
    virtual MemError do_acquire_growable(size_t reserve_bytes, size_t initial_commit,
                                         size_t alignment, RegionTag tag, void** out_base);

    // Backends adjust a Region's committed size from commit(); the field is
    // private and Backend is Region's only friend.
    static void set_committed_(Region& r, size_t committed);

    // Backends construct Regions through this — Region's fields are private
    // and Backend is its only friend.
    Region make_region_(void* base, size_t committed, size_t reserved, RegionTag tag);

private:
    friend class Region;
    void release_(Region& region);
};

// The concrete production backend: plain cudaMalloc/cudaFree. Not growable.
// See §A3.1 for why VMM is scoped to the KV pool and gated on a WSL2 spike
// rather than being the default here.
Backend& cuda_malloc_backend();

// The growable backend (CUDA VMM), or nullptr where the device cannot do it. A pointer
// rather than a reference because "no virtual memory management" is a real answer
// callers must fall back from, not a startup failure.
Backend* vmm_backend();

}  // namespace imp
