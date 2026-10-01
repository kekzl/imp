#pragma once

#include <cuda_runtime_api.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace imp {

// MemAccount: process-global VRAM accounting harness (Phase-0 audit). VRAMAllocator only
// tracks allocations routed through it; the large consumers (KV pool, NVFP4 decode
// cache, CUTLASS/cuBLAS workspaces, CUDA-graph buffers, weights) call cudaMalloc
// directly and are invisible to it. Three complementary, low-overhead signals close
// the gap:
//   1. CHECKPOINTS: a labeled cudaMemGetInfo snapshot at each init phase; the free-VRAM
//      delta between consecutive checkpoints attributes memory to that phase with FULL
//      coverage (also sees raw cudaMalloc). Ground-truth backbone of the breakdown.
//   2. NOTES: explicit note(pool, +/-bytes) at the big allocation sites, giving
//      current+peak attribution WITHIN a phase (WEIGHTS, KV_BLOCK_POOL,
//      WEIGHT_CACHE_FP16/_FP8/_NVFP4/_CUTLASS_SF, EXEC_WORKSPACES). Anything else lands
//      in the UNTRACKED residual; add a note() when a new consumer grows big.
//   3. SAMPLER: a background thread polling cudaMemGetInfo at high frequency, recording
//      the true device-used PEAK, capturing transient spikes a steady-state snapshot
//      would miss.
// Gated behind diagnostics.vram_audit (default off): disabled, note()/checkpoint() cost
// one relaxed atomic or a single cudaMemGetInfo and the sampler never starts.
class MemAccount {
public:
    static MemAccount& instance();

    // Enable/disable. When disabled, checkpoint()/report() still take a cheap
    // cudaMemGetInfo so the device free/used line is always available, but no
    // history is retained and the sampler never runs.
    void set_enabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    // Append-only file the report() table is mirrored into (in addition to the
    // log). Empty = log only.
    void set_dump_path(std::string path);

    // Per-pool current + peak attribution. pool must be a string literal /
    // stable pointer (stored by value into a small fixed registry by name).
    void note(const char* pool, std::ptrdiff_t delta_bytes);

    // Charge `bytes` to `pool` for one allocation, keyed by its pointer; note_free uncharges it
    // (no-op for an uncharged pointer). For owners that free some of what they noted (#2354).
    void note_alloc(const char* pool, const void* ptr, size_t bytes);
    void note_free(const void* ptr);
    int64_t pool_current(const char* pool) const;

    // Record a labeled device snapshot (cudaMemGetInfo). The delta vs the
    // previous checkpoint is the measured cost of the phase just completed.
    void checkpoint(const char* name);

    // Reset the CUDA allocator high-water marks to their current values, at the
    // Loading->Serving transition, so everything reported afterwards was allocated WHILE
    // SERVING (I2, criterion 3). This is the layer the allocation-phase guard cannot
    // provide: the guard only sees Backend-routed allocations, while these attributes are
    // maintained by the CUDA runtime itself and catch every cudaMallocAsync and captured
    // stream-ordered allocation regardless of module or allocator.
    void arm_steady_state_watermarks();

    // Named, non-imp charges the pool notes cannot see, so the reconciliation residual
    // reports what is genuinely unattributed. Set from Engine::init once known:
    //   context  - CUDA primary context + driver (checkpoint 00_pre_init)
    //   library  - the fixed charge CUDA/cuBLAS/CUTLASS claim on the first forward (A1.5)
    //   arena    - engine-persistent tier reservation (its high-water is what the planner
    //              should eventually use)
    size_t unattributed_bytes() const;
    // The same residual unclamped: negative when the ledger counts more than the device holds.
    int64_t unattributed_signed_bytes() const;

    void set_named_charges(size_t context_bytes, size_t library_bytes, size_t arena_bytes,
                           size_t arena_high_water);

    // Background device-used peak sampler.
    void sampler_start(int interval_us = 2000);
    void sampler_stop();

    // Emit the full audit table: checkpoints + per-pool current/peak + device
    // free/used/peak + reconciliation residual (device_used - sum(pools) =
    // untracked weights/fragmentation). phase_label tags the emission point.
    void report(const char* phase_label);

private:
    MemAccount() = default;
    ~MemAccount();
    MemAccount(const MemAccount&) = delete;
    MemAccount& operator=(const MemAccount&) = delete;

    struct Pool {
        std::string name;
        int64_t current = 0;
        int64_t peak = 0;
        int64_t alloc_count = 0;
    };
    struct Checkpoint {
        std::string name;
        size_t free_bytes = 0;
        size_t used_bytes = 0;  // total - free
    };

    Pool& pool_locked(const char* name);
    void sample_once();  // updates peak_used_ from cudaMemGetInfo

    std::atomic<bool> enabled_{false};
    mutable std::mutex mu_;
    std::vector<Pool> pools_;
    struct Charge {
        const char* pool;
        size_t bytes;
    };
    std::unordered_map<const void*, Charge> charges_;  // note_alloc rows still charged
    std::vector<Checkpoint> checkpoints_;
    std::string dump_path_;
    size_t named_context_ = 0;
    size_t named_library_ = 0;
    size_t named_arena_ = 0;
    size_t named_arena_high_ = 0;

    std::atomic<size_t> peak_used_{0};
    std::atomic<bool> sampler_run_{false};
    std::thread sampler_;
    int sampler_interval_us_ = 2000;
};

// Retires pending stream-ordered frees and returns the default CUDA mempool's unused
// reserved slack to the driver (cudaMemPoolTrimTo). Engine init raises the default
// cudaMallocAsync pool's release threshold to UINT64_MAX so freed blocks are kept for
// reuse, so cudaFreeAsync alone only parks weights-sized memory in the pool; the next
// plain-cudaMalloc path can't see it and OOMs. Call after tearing down anything that
// freed large async allocations. Safe at process exit.
// Returns the bytes handed back to the driver (reserved before - after).
size_t trim_device_mempool();

// Device mempools with release threshold 0, created on first use (null if creation fails):
// a cudaFreeAsync'd block goes back to the driver at the next sync instead of sharing a
// chunk with long-lived allocations. One pool per class Phase 4b (or the FP8 LM head) may free, so a class
// that stays live does not pin the chunks of one that is freed.
enum class ReleasePool { GdnPacks = 0, GdnSources = 1, LmHead = 2 };
cudaMemPool_t release_on_free_pool(ReleasePool which);
// Its reserved bytes (cudaMemPoolAttrReservedMemCurrent), 0 without a pool.
size_t release_on_free_pool_reserved(ReleasePool which);
// cudaMallocAsync, from the release-on-free pool of the innermost live ReleasePoolScope if any.
cudaError_t malloc_async_in_scope(void** ptr, size_t size, cudaStream_t stream);
class ReleasePoolScope {
public:
    explicit ReleasePoolScope(ReleasePool which);
    ~ReleasePoolScope();
    ReleasePoolScope(const ReleasePoolScope&) = delete;
    ReleasePoolScope& operator=(const ReleasePoolScope&) = delete;

private:
    int prev_;
};

// I7: capacity is not occupancy (MEMORY.md). A single "VRAM used" number cannot
// distinguish a KV pool 90% full from one 90% reserved and empty, and every capacity
// question an operator asks needs both halves; each tier reports the capacity it holds
// and what is live inside it. Process-global tiers only; anything owned by an Engine
// (the KV pool) is added by the caller. Cheap enough to call per scrape.
struct MemTierStat {
    const char* tier = "";
    size_t reserved = 0;  // capacity this tier holds
    size_t live = 0;      // in use inside it (== reserved when not tracked)
};

std::vector<MemTierStat> memory_tier_stats();

// Installed --vram-budget and this process's own usage against it, both in bytes.
// own_bytes is the same baseline delta the budget view sizes from, so "is the cap
// respected" is answerable from the same number the planner used, not device-used
// (which also carries the CUDA context and any neighbour process). budget_bytes is 0
// when no budget is installed.
struct MemBudgetStat {
    size_t budget_bytes = 0;
    size_t own_bytes = 0;
    size_t own_peak_bytes = 0;
};

MemBudgetStat memory_budget_stat();

}  // namespace imp
