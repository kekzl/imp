#pragma once

// FakeBackend — the substitution seam that makes the memory subsystem testable
// without a GPU (docs/internals/MEMORY.md §A6).
//
// imp's CI has no GPU runner; the lane that actually runs is `ctest -L unit`.
// Every allocator, the planner, and all the refcount logic therefore have to be
// exercisable on host memory. FakeBackend hands out ordinary heap memory behind
// the same Backend interface and adds what a test needs and a real device
// cannot give:
//
//   - a bounded capacity, so exhaustion paths are testable without a 32 GiB card
//   - a full journal of every acquire/release/commit, for conservation replay
//   - poison-on-release (0xDE), so use-after-free is a deterministic memcmp
//     failure rather than a GPU fault
//   - injectable failure (fail the n-th acquisition), to drive the rollback
//     paths that are hand-written per call site today
//   - growth simulation, asserting base addresses never move — the host-side
//     proof of I3 that the VMM backend will have to satisfy on device

#include "memory/backend.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace imp {

struct AllocEvent {
    enum class Op { Acquire, Release, Commit, Decommit };
    uint64_t seq = 0;
    Op op = Op::Acquire;
    AllocPhase phase = AllocPhase::Loading;
    RegionTag tag = RegionTag::Other;
    size_t bytes = 0;
    const void* base = nullptr;
};

class FakeBackend final : public Backend {
public:
    // capacity_bytes = 0 means "unbounded" (host heap is the only limit).
    explicit FakeBackend(size_t capacity_bytes = 0, bool growable = true);
    ~FakeBackend() override;

    MemError do_commit(Region& region, size_t new_committed) override;
    BackendStats stats() const override;
    size_t capacity() const override { return capacity_; }

    // ── test controls ────────────────────────────────────────────────
    // Make the n-th (1-based) subsequent acquisition fail with `err`.
    // 0 disables. Consumed on trigger.
    void fail_acquisition(uint64_t nth, MemError err = MemError::OutOfMemory);

    const std::vector<AllocEvent>& journal() const { return journal_; }

    // Σ acquired − Σ released, recomputed from the journal. V1 (conservation)
    // asserts this equals stats().live_bytes after every operation.
    size_t journal_live_bytes() const;

    // Number of regions still live.
    size_t live_regions() const;

    // True if every byte of [base, base+bytes) is the release poison. Lets a
    // test prove a buffer was actually returned rather than merely forgotten.
    // Valid for the most recent kQuarantineDepth released regions (they are
    // poisoned and held, not freed, precisely so this is not a use-after-free)
    // and for the decommitted tail of a growable region.
    static bool is_poisoned(const void* base, size_t bytes);

    static constexpr unsigned char kPoison = 0xDE;
    static constexpr size_t kQuarantineDepth = 16;

protected:
    MemError do_acquire(size_t bytes, size_t alignment, RegionTag tag, void** out_base,
                        size_t* out_reserved) override;
    MemError do_acquire_growable(size_t reserve_bytes, size_t initial_commit, size_t alignment,
                                 RegionTag tag, void** out_base) override;
    void do_release(void* base, size_t committed, size_t reserved, RegionTag tag) override;

private:
    struct Live {
        void* raw = nullptr;      // the malloc'd pointer (base may be aligned up)
        void* base = nullptr;
        size_t committed = 0;
        size_t reserved = 0;
        RegionTag tag = RegionTag::Other;
    };

    void record_(AllocEvent::Op op, RegionTag tag, size_t bytes, const void* base);

    mutable std::mutex mu_;
    size_t capacity_ = 0;
    bool growable_ = true;
    BackendStats stats_;
    std::vector<Live> live_;
    std::vector<Live> quarantine_;  // poisoned, not yet freed (see is_poisoned)
    std::vector<AllocEvent> journal_;
    uint64_t seq_ = 0;
    uint64_t acquire_ordinal_ = 0;
    uint64_t fail_at_ = 0;
    MemError fail_err_ = MemError::OutOfMemory;
};

}  // namespace imp
