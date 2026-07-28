#pragma once

// The planner (docs/MEMORY_ARCHITECTURE.md §A4).
//
// compute_vram_budget() calls itself "pure computation — just arithmetic". It
// is pure in the functional sense and impure in the useful one: its dominant
// input is a live cudaMemGetInfo reading taken after the weight upload and
// before the weight caches are built. So the KV pool is sized from a number
// that is ~3.9 GiB too optimistic (§A1.5), the cache phases re-derive their own
// budgets from live free VRAM again (#1100), and the engine works around the
// ordering with a physical balloon allocation whose only job is to hide bytes
// from the planner.
//
// plan_memory() has three properties that code does not:
//
//   1. It never queries the device. Its only capacity input is budget_bytes,
//      so the same config yields a byte-identical plan on every boot. That
//      ends the "free VRAM before the upload swings by 1.6 GB between
//      identical invocations -> different auto-batch -> different KV clamp"
//      trap recorded against #1103.
//   2. It is a pure function of a plain struct, so it runs in the CPU-only CI
//      lane with no GPU and no Model.
//   3. It fails at load time with an itemised report and the largest levers,
//      never mid-generation.
//
// This header is deliberately free of CUDA, Model and EngineConfig. The
// adapter that fills PlanInput from those lives in runtime/.

#include "memory/backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Shape facts the plan needs. All sizes in bytes, all counts absolute.
struct ModelShape {
    int n_layers = 0;
    int n_kv_layers = 0;     // attention layers only (hybrids have fewer)
    int n_kv_heads = 0;
    int head_dim = 0;
    // Σ device bytes of the uploaded weights, as the loader will place them.
    size_t weight_bytes = 0;
    // Σ device bytes of the pre-dequant weight caches this checkpoint wants
    // (FP16 / FP8 / NVFP4 / CUTLASS-SF). Computed by the caller from the same
    // predicate the phases use, so the estimate and the build cannot drift.
    size_t weight_cache_bytes = 0;
    // Mandatory subset of weight_cache_bytes: without it, decode falls off the
    // captured graph path entirely. Never traded away for KV.
    size_t mandatory_cache_bytes = 0;
};

struct FeatureSet {
    bool cuda_graphs = true;
    size_t vision_tower_bytes = 0;      // 0 = no --mmproj
    size_t spec_decode_bytes = 0;       // draft + verify staging, at max k
    size_t ssm_state_bytes = 0;         // batch-shaped, computed by the caller
    size_t recurrent_snapshot_bytes = 0;
    size_t residual_ring_bytes = 0;
    // SWA-aware sizing: sliding-window layers hold a fixed live span instead
    // of full context. 0/0 = feature off.
    int swa_live_tokens = 0;
    int n_swa_layers = 0;
};

struct ConcurrencyLimits {
    int max_batch_size = 1;
    int max_seq_len = 0;
    int kv_block_size = 16;
    // K+V bytes of one block for ONE layer, packing- and scale-aware.
    // Single source: kv_block_bytes_per_layer() in runtime/vram_budget.h.
    size_t kv_block_bytes_per_layer = 0;
    // Floor: the pool must hold at least this many tokens or long requests are
    // rejected at admission while /v1/models still advertises max_seq_len.
    int min_kv_tokens = 0;
};

// The fixed charge that is not imp's memory and cannot be planned away.
// A1.5 measured ~3.9 GiB claimed by CUDA/cuBLAS/CUTLASS on the first forward
// pass, invariant to batch and context. Carrying it as an explicit, named
// input is the difference between a plan and a guess.
struct LibraryReserve {
    size_t bytes = 0;
    const char* source = "unset";
};

struct PlanInput {
    ModelShape model;
    FeatureSet features;
    ConcurrencyLimits limits;
    LibraryReserve library;
    // --vram-budget, or the device total. The plan fits this or fails.
    size_t budget_bytes = 0;
    // CUDA primary context + driver overhead (measured: 1679.6 MiB on this
    // WSL2/WDDM box). Gone before imp allocates anything.
    size_t context_bytes = 0;
    // Executor forward-scratch high-water. From a recorded warmup measurement
    // when available; the caller's conservative estimate otherwise.
    size_t forward_scratch_bytes = 0;
    // Workspaces, cuBLAS/CUTLASS scratch, graph buffers.
    size_t engine_persistent_bytes = 0;
};

struct KvPlan {
    int blocks = 0;
    int blocks_per_seq = 0;
    int swa_blocks = 0;
    size_t bytes = 0;
    size_t swa_bytes = 0;
    // True when the pool holds less than the min_kv_tokens floor: requests
    // longer than the pool are rejected at admission. Loud, not silent.
    bool below_floor = false;
};

struct PlanLine {
    const char* name = "";
    RegionTag tag = RegionTag::Other;
    size_t bytes = 0;
};

struct MemoryPlan {
    size_t model_resident = 0;      // weights + weight caches
    size_t engine_persistent = 0;
    size_t forward_scratch = 0;
    KvPlan kv;
    std::vector<PlanLine> pools;    // SWA group, SSM state, residual ring, ...
    size_t library_reserve = 0;
    size_t context_reserve = 0;

    size_t total() const;
    // Every line item, largest first — the body of --mem-report.
    std::vector<PlanLine> lines() const;
};

struct PlanLever {
    std::string change;   // "runtime.max_seq_len 4096 -> 2048"
    size_t frees = 0;
};

struct PlanFailure {
    size_t requested = 0;
    size_t budget = 0;
    size_t over_by = 0;
    std::vector<PlanLine> lines;
    std::vector<PlanLever> levers;
    // Operator-facing message: the itemisation plus the three largest levers.
    std::string report() const;
};

struct PlanResult {
    bool ok = false;
    MemoryPlan plan;
    PlanFailure failure;
    explicit operator bool() const { return ok; }
};

// Pure. Never touches the device. Deterministic for a given input.
PlanResult plan_memory(const PlanInput& in);

}  // namespace imp
