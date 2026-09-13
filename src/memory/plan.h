#pragma once

// The planner (MEMORY.md A4). compute_vram_budget() is pure in the functional sense and
// impure in the useful one: its dominant input is a live cudaMemGetInfo reading taken
// after weight upload and before the weight caches are built, ~3.9 GiB too optimistic
// (A1.5), and the cache phases still re-derive their own budgets from live free VRAM
// (#1100). The physical-balloon workaround for the ordering is gone (AUDIT B62, the
// guarantee is a planned floor now), but the live re-derivation it papered over remains.
// plan_memory() has three properties code alone does not:
//   1. Never queries the device: its only capacity input is budget_bytes, so the same
//      config yields a byte-identical plan every boot (ends the free-VRAM-swings-between-
//      identical-boots trap, #1103).
//   2. Pure function of a plain struct: runs in the CPU-only CI lane, no GPU, no Model.
//   3. Fails at load time with an itemised report and the largest levers, never
//      mid-generation.
// Deliberately free of CUDA, Model and EngineConfig; the adapter filling PlanInput from
// those lives in runtime/.

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

// The fixed charge that is not imp's memory and cannot be planned away: ~3.9 GiB claimed
// by CUDA/cuBLAS/CUTLASS on the first forward pass, invariant to batch and context
// (A1.5). Carrying it as an explicit, named input is the difference between a plan and a
// guess.
struct LibraryReserve {
    size_t bytes = 0;
    const char* source = "unset";
};

// The first-forward library claim varies by config (mmq_q8_imma's per-weight s8 planes
// used to be miscounted here; they are planned separately now, VRAMBudget::
// imma_plane_bytes). kMeasuredLibraryReserveBytes stays as the cold-start floor for
// models with no recorded measurement: over-reserving costs KV (absorbed partly by the
// 10% reserve floor), under-reserving spills the card. Re-measure after a driver or CUDA
// bump; imp.conf vram.library_reserve_mb overrides it per host.
constexpr size_t kMeasuredLibraryReserveBytes = 3900ull * 1024 * 1024;

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
    size_t model_resident = 0;      // weights + the MANDATORY weight caches
    // The rest of the weight-cache demand, granted from the residual after model-resident
    // charges, only above the one-sequence KV floor: it is the tier the engine itself trades
    // away when VRAM is short, so committing it whole made the plan reject configurations
    // that serve fine (AUDIT B69).
    size_t optional_caches = 0;
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

// The largest max_batch_size in [1, in.limits.max_batch_size] the plan accepts,
// shrinking the batch-shaped part of features.ssm_state_bytes by ssm_bytes_per_slot per
// dropped slot (any fixed remainder stays). 0 when not even one slot fits. Pure.
int plan_fitting_batch(const PlanInput& in, size_t ssm_bytes_per_slot);

}  // namespace imp
