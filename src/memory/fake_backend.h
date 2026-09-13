#pragma once

// FakeBackend: the substitution seam that makes the memory subsystem testable without a
// GPU (MEMORY.md A6). imp's CI has no GPU runner (only `ctest -L unit` runs), so every
// allocator, the planner, and refcount logic must be exercisable on host memory.
// FakeBackend hands out heap memory behind the Backend interface, plus what a test needs
// and a real device cannot give: bounded capacity (exhaustion testable without a 32 GiB
// card), a full acquire/release/commit journal (conservation replay), poison-on-release
// (0xDE, use-after-free becomes a deterministic memcmp), injectable failure (drive
// rollback paths), and growth simulation asserting base addresses never move (the
// host-side proof of I3).

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
    // A range commit is modelled as a prefix extension: committed becomes max(committed,
    // offset + bytes). Interior gaps are not modelled, exact for a slab that commits its
    // slots in order and conservative (over-counts) otherwise.
    MemError do_commit_range(Region& region, size_t offset, size_t bytes) override;
    // 4 KiB rather than the VMM backend's 2 MiB, so a test can see stride
    // padding without allocating megabytes per slot.
    static constexpr size_t kGranularity = 4096;
    size_t granularity() const override { return kGranularity; }
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

    // True if every byte of [base, base+bytes) is the release poison: lets a test prove a
    // buffer was actually returned rather than merely forgotten. Valid for the most recent
    // kQuarantineDepth released regions (poisoned and held, not freed, precisely so this
    // isn't a use-after-free) and for the decommitted tail of a growable region.
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
