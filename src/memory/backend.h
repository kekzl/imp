#pragma once

// L1 of the memory architecture (docs/MEMORY_ARCHITECTURE.md §A3.1/§A3.2):
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

// ─────────────────────────────────────────────────────────────────────
// Tags — the unit of reporting for I7 (capacity and occupancy are separate
// concepts with separate reporting). One tag per thing an operator would
// recognise in a `--mem-report` line, not one per allocation site.
// ─────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────
// Allocation phase (I2). Monotonic within a model's lifetime; the engine
// drives it. `Serving` means warmup is done and steady state has begun —
// from that point on, asking the driver for memory is a defect.
//
// Debug builds abort on a Serving-phase acquisition so the offending call
// site is caught in CI. Release builds count it, log once per tag, and
// proceed: a production server must not die over an accounting bug. The
// counter is the I2 acceptance test (criterion 3) and the migration progress
// bar — it starts non-zero and must reach zero.
// ─────────────────────────────────────────────────────────────────────
enum class AllocPhase { Loading, Planning, Serving };

AllocPhase alloc_phase();
void set_alloc_phase(AllocPhase);

// Record a device allocation made while serving. Backend::acquire() calls
// this itself; the --wrap interposer (memory/alloc_interpose.cpp) calls it for
// allocations that never went through Backend at all, which is what makes
// steady_state_allocations() authoritative rather than merely indicative.
// `site` is the caller's return address when available (symbolize with
// addr2line), nullptr otherwise.
void note_serving_allocation(RegionTag tag, size_t bytes, const void* site = nullptr);

// Total acquisitions observed while in AllocPhase::Serving, all tags.
uint64_t steady_state_allocations();
// Per-tag breakdown; `tag` indexes the RegionTag enum.
uint64_t steady_state_allocations(RegionTag tag);
void reset_steady_state_allocations();

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

// ─────────────────────────────────────────────────────────────────────
// Region — the only type in imp that holds a raw device pointer obtained
// from the driver. Move-only, RAII: the destructor returns it to the backend
// that produced it. A tier allocator owns exactly one (or a few) of these and
// hands out views into them; nothing else ever sees one.
//
// `reserved` >= `committed`. They differ only for growable backends (VMM),
// where `reserved` is the virtual address range and `committed` is the
// physical memory currently mapped into it. For the cudaMalloc backend they
// are always equal.
// ─────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────
// Backend — physical acquisition. Fails cleanly; never throws, never aborts
// on an out-of-memory condition (I6: exhaustion is a typed, recoverable
// value, not a crash deep inside a kernel launch).
// ─────────────────────────────────────────────────────────────────────
class Backend {
public:
    virtual ~Backend() = default;

    // Acquire `bytes`, aligned to `alignment` (a power of two, >= 256).
    // Consults the phase guard before doing anything.
    AcquireResult acquire(size_t bytes, size_t alignment, RegionTag tag);

    // Reserve `reserve_bytes` of address space and commit `initial_commit` of
    // it. The region's base() is then stable for the whole reservation, which
    // is what lets a graph-captured pointer survive growth (I3). Backends that
    // cannot do this return MemError::NotGrowable and the caller falls back to
    // a fixed acquire().
    AcquireResult acquire_growable(size_t reserve_bytes, size_t initial_commit, size_t alignment,
                                   RegionTag tag);

    // Grow/shrink a growable region in place, keeping `base()` stable.
    // Returns MemError::NotGrowable on backends that cannot (cudaMalloc).
    virtual MemError commit(Region& region, size_t new_committed) = 0;

    virtual BackendStats stats() const = 0;

    // Installed hard cap in bytes (--vram-budget); 0 = uncapped.
    virtual size_t capacity() const = 0;

protected:
    // Implementations override these two. `acquire()` wraps do_acquire() with
    // the phase guard and accounting so no backend can forget either.
    virtual MemError do_acquire(size_t bytes, size_t alignment, RegionTag tag, void** out_base,
                                size_t* out_reserved) = 0;
    virtual void do_release(void* base, size_t committed, size_t reserved, RegionTag tag) = 0;

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

}  // namespace imp
