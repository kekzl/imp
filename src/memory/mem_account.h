#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace imp {

// ─────────────────────────────────────────────────────────────────────
// MemAccount — process-global VRAM accounting harness (Phase-0 audit).
//
// The VRAMAllocator only tracks allocations routed through it (~14 files);
// the large consumers (KV pool, NVFP4 decode cache, CUTLASS/cuBLAS
// workspaces, CUDA-graph buffers, weights) call cudaMalloc directly and are
// invisible to it. MemAccount closes that gap with two complementary,
// low-overhead signals plus a peak sampler:
//
//   1. CHECKPOINTS — a labeled cudaMemGetInfo snapshot at each init phase.
//      The free-VRAM delta between consecutive checkpoints attributes device
//      memory to that phase with FULL coverage (it also sees raw cudaMalloc).
//      This is the ground-truth backbone of the breakdown.
//
//   2. NOTES — explicit note(pool, +/-bytes) calls at the big allocation
//      sites give current + peak attribution WITHIN a phase. Wired pools:
//      WEIGHTS (SafeTensors/GGUF upload), KV_BLOCK_POOL (KV + scale pools),
//      WEIGHT_CACHE_FP16 / _FP8 / _NVFP4 / _CUTLASS_SF (pre-dequant caches,
//      noted as build totals after QuantPipeline::build), EXEC_WORKSPACES
//      (executor workspace estimate). Anything else lands in the UNTRACKED
//      reconciliation residual — add a note() when a new consumer grows big.
//
//   3. SAMPLER — a background thread polling cudaMemGetInfo at high frequency
//      records the true device-used PEAK, capturing transient spikes (the
//      prefill activation / materialized-scores matrix, CUTLASS autotuning
//      transients) that a steady-state snapshot would miss.
//
// Everything is gated behind RuntimeConfig diagnostics.vram_audit (default
// off): when disabled, note()/checkpoint() are ~one relaxed atomic / a single
// cudaMemGetInfo and the sampler thread is never started. Safe to leave the
// note() calls compiled in on the hot path — they are not on it (allocation
// sites only), but the gate keeps them free regardless.
// ─────────────────────────────────────────────────────────────────────
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

    // Record a labeled device snapshot (cudaMemGetInfo). The delta vs the
    // previous checkpoint is the measured cost of the phase just completed.
    void checkpoint(const char* name);

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
    std::vector<Checkpoint> checkpoints_;
    std::string dump_path_;

    std::atomic<size_t> peak_used_{0};
    std::atomic<bool> sampler_run_{false};
    std::thread sampler_;
    int sampler_interval_us_ = 2000;
};

// Retire pending stream-ordered frees and return the default CUDA mempool's
// unused reserved slack to the driver (cudaMemPoolTrimTo). Engine init raises
// the default cudaMallocAsync pool's release threshold to UINT64_MAX so freed
// blocks are kept for re-use, so cudaFreeAsync alone only parks weights-sized
// memory in the pool — the next plain-cudaMalloc path (cudaMemGetInfo-based
// sizing, token-embedding upload) can't see it and OOMs. Call after tearing
// down anything that freed large async allocations (model/context teardown).
// Safe at process exit (guards a torn-down pool, clears sticky errors). Logs
// the reserved delta.
void trim_device_mempool();

}  // namespace imp
