#pragma once

// T2 slot pool for the conditional graph loop
// (docs/MEMORY_ARCHITECTURE.md §A2/§A3.4, A7 step 5.3).
//
// CudaGraphConditionalRunner::setup() allocated 13 device buffers and 4 pinned
// host buffers, and cleanup() freed them again — once per burst, which the
// --wrap interposer measured as the whole of the remaining steady-state
// allocation traffic (AUDIT B28: 238 of 238 calls after the earlier fixes).
//
// These cannot come from the T4 scratch stack. A Mark rewinds when the forward
// returns, but the graph that baked these addresses in is replayed *later* —
// the stack's discipline is exactly wrong for them (I3). What they need is a
// fixed set of long-lived slots with stable addresses, taken for the length of
// a burst and returned: T3's shape, at T2's lifetime.
//
// One slot is one contiguous device region carved at fixed offsets, plus one
// pinned+mapped host region. Two runners exist at a time in the engine (the
// local one in try_graph_loop_decode and Engine::async_graph_runner_), so the
// pool is small; exceeding it, or exceeding a capacity, falls back to direct
// allocation so a surprising config degrades in throughput rather than failing.
//
// Side effect worth knowing about: because a returned slot keeps its address,
// consecutive bursts see *identical* pointers. Nothing depends on that yet —
// the runner still recaptures — but it is the precondition for reusing a
// captured graph across bursts, which is not in this change's scope.

#include "memory/backend.h"
#include "memory/host_pinned.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace imp {

class GraphSlotPool;

// Size of the sampler scratch each slot carries. Must equal
// compute/sampling.h's SAMPLE_SCRATCH_BYTES; it is restated here rather than
// included because that header pulls in the CUDA sampling surface and this one
// is host-only. cuda_graph.cu static_asserts the two against each other, so a
// change to either side is a compile error rather than a silent overrun.
constexpr size_t kGraphSlotSampleScratchBytes =
    sizeof(int32_t) + 64 * (2 * sizeof(float) + 128 * (sizeof(float) + sizeof(int32_t)));

// The pinned-host half of a slot comes from T5's engine-persistent allocator
// (memory/host_pinned.h). It used to be declared here; it moved out because it
// is a tier, not a detail of this pool — 26 acquisition sites in 11 files were
// waiting for it. Substituting it is still what makes this pool testable in the
// CPU lane rather than needing a GPU (A6).

// Capacities a slot is cut for. Requests beyond these fall back.
struct GraphSlotCaps {
    // Ring buffer + penalty ring length, in tokens.
    int max_steps = 0;
    // Penalty ring = prefix history + max_steps.
    int penalty_slots = 0;
    // Stop token ids.
    int stop_ids = 64;
};

// Pointers into one slot. Everything is device memory except the h_* fields,
// which are pinned host memory; d_ring / d_step_counter_mapped /
// d_burst_done_mapped are the device-side views of the mapped host buffers.
//
// A default-constructed view is all-null, which is what the runner sees when
// the pool declines and it must allocate for itself.
struct GraphSlotView {
    void* sample_scratch = nullptr;  // >= SAMPLE_SCRATCH_BYTES, also holds the token id

    int* position = nullptr;
    int* context_len = nullptr;
    int* step_counter = nullptr;
    int* step_limit = nullptr;
    int* think_limit = nullptr;
    int* think_count = nullptr;
    int* in_think = nullptr;
    int* think_exit_step = nullptr;
    int* content_after_think = nullptr;
    int* penalty_count = nullptr;

    int32_t* stop_ids = nullptr;
    int32_t* penalty_ring = nullptr;

    int32_t* h_ring = nullptr;
    int32_t* d_ring = nullptr;
    int* h_step_counter = nullptr;
    int* d_step_counter_mapped = nullptr;
    int* h_burst_done = nullptr;
    int* d_burst_done_mapped = nullptr;
    int32_t* h_decode_scratch = nullptr;

    bool valid() const { return sample_scratch != nullptr; }
};

// RAII lease on one slot. Move-only: a slot has exactly one holder for the
// length of a burst, which is the property that makes the addresses safe to
// bake into a captured graph.
class GraphSlotLease {
public:
    GraphSlotLease() = default;
    ~GraphSlotLease() { release(); }

    GraphSlotLease(GraphSlotLease&& o) noexcept : pool_(o.pool_), index_(o.index_), view_(o.view_) {
        o.pool_ = nullptr;
        o.index_ = -1;
        o.view_ = {};
    }
    GraphSlotLease& operator=(GraphSlotLease&& o) noexcept {
        if (this != &o) {
            release();
            pool_ = o.pool_;
            index_ = o.index_;
            view_ = o.view_;
            o.pool_ = nullptr;
            o.index_ = -1;
            o.view_ = {};
        }
        return *this;
    }
    GraphSlotLease(const GraphSlotLease&) = delete;
    GraphSlotLease& operator=(const GraphSlotLease&) = delete;

    void release();

    bool valid() const { return pool_ != nullptr && index_ >= 0; }
    const GraphSlotView& view() const { return view_; }

private:
    friend class GraphSlotPool;
    GraphSlotLease(GraphSlotPool* pool, int index, const GraphSlotView& view)
        : pool_(pool), index_(index), view_(view) {}

    GraphSlotPool* pool_ = nullptr;
    int index_ = -1;
    GraphSlotView view_{};
};

class GraphSlotPool {
public:
    GraphSlotPool() = default;
    ~GraphSlotPool();

    GraphSlotPool(const GraphSlotPool&) = delete;
    GraphSlotPool& operator=(const GraphSlotPool&) = delete;

    // Cut `num_slots` slots for `caps`. The device side comes from `backend`
    // as one region; the pinned host side is one allocation from `host`.
    [[nodiscard]] MemError open(Backend& backend, HostPinnedAllocator& host, const GraphSlotCaps& caps,
                                int num_slots);
    // Same, against the CUDA pinned allocator.
    [[nodiscard]] MemError open(Backend& backend, const GraphSlotCaps& caps, int num_slots) {
        return open(backend, cuda_host_pinned_allocator(), caps, num_slots);
    }
    void close();
    bool is_open() const;

    // Take a slot that covers `need`. Returns an invalid lease when the pool
    // is closed, exhausted, or too small — the caller then allocates itself.
    [[nodiscard]] GraphSlotLease acquire(const GraphSlotCaps& need);

    const GraphSlotCaps& caps() const { return caps_; }
    int num_slots() const { return num_slots_; }
    int free_slots() const;
    // Times acquire() declined. A non-zero count in a steady-state server is
    // the signal that the caps or the slot count are wrong, so --mem-report
    // and the I2 soak can both see it.
    uint64_t declines() const;
    // Reason breakdown, for the same readers.
    uint64_t declines_exhausted() const;
    uint64_t declines_too_small() const;

    size_t device_bytes() const { return device_bytes_; }
    size_t host_bytes() const { return host_bytes_; }

private:
    friend class GraphSlotLease;
    void release_(int index);
    GraphSlotView carve_(int index) const;

    mutable std::mutex mu_;
    Region region_;      // device side, one region for all slots
    PinnedBuffer host_;  // pinned+mapped host side, one allocation for all slots
    bool open_ = false;
    GraphSlotCaps caps_{};
    int num_slots_ = 0;
    size_t slot_device_stride_ = 0;
    size_t slot_host_stride_ = 0;
    size_t device_bytes_ = 0;
    size_t host_bytes_ = 0;
    std::vector<bool> in_use_;
    uint64_t declines_exhausted_ = 0;
    uint64_t declines_too_small_ = 0;
};

// The process-global pool, opened by Engine::init next to the T2 arena and for
// the same reason: its tenant is constructed deep inside the decode path with
// no Engine to reach through.
GraphSlotPool& graph_slot_pool();

// Open the global pool for a context of `max_seq_len`. The capacities follow
// from that: a burst cannot be longer than the context, and the penalty ring is
// prefix history + burst length, so twice the context bounds it. Four slots
// covers the two runners that exist at a time with room to spare.
//
// Never fatal — a pool that will not open just means the runner keeps
// allocating for itself, which is what it did before this existed.
void graph_slot_pool_open_for(Backend& backend, int max_seq_len);

}  // namespace imp
