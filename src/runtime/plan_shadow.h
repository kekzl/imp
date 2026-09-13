#pragma once

// A7 step 2b (docs/internals/MEMORY.md): run plan_memory() alongside
// compute_vram_budget(), log both, attribute the difference, without
// applying anything.
//
// Both sides are fed the same demand figures (weight-cache, SSM footprint,
// workspace estimate); what is compared is the allocation policy: residual
// distribution, admission-floor honoring, and charges the old pass cannot
// see. Migrating the demand estimates themselves is step 6.
//
// Plain scalars, not Model/EngineConfig: keeps this header CUDA-free and
// dependency-free, testable in the CPU lane.

#include "memory/plan.h"

#include <cstddef>
#include <string>

namespace imp {

struct VRAMBudget;

struct ShadowPlanProbe {
    // Free VRAM the engine has left to distribute at budget time. Weights and
    // the CUDA context are already spent, so they are NOT charged again.
    size_t distributable_bytes = 0;

    // Demand, taken from the live budget pass so both sides see one number.
    size_t weight_cache_demand = 0;
    size_t mandatory_cache_bytes = 0;
    size_t ssm_state_bytes = 0;
    size_t engine_persistent_bytes = 0;
    // Pools the plan reads from FeatureSet: server.recurrent_snapshot_mb
    // (256 MiB default) is cudaMalloc'd AFTER the KV pool is sized, so the
    // plan must account for it or size the KV pool over memory another
    // tenant is about to take.
    size_t recurrent_snapshot_bytes = 0;
    size_t spec_decode_bytes = 0;
    size_t residual_ring_bytes = 0;

    // The charge the old pass cannot see (A1.5). 0 disables it.
    size_t library_reserve_bytes = 0;

    // Geometry.
    int n_kv_layers = 0;
    int n_swa_layers = 0;
    int swa_live_tokens = 0;
    int max_batch_size = 1;
    int max_seq_len = 0;
    int kv_block_size = 16;
    int min_kv_tokens = 0;
    size_t kv_block_bytes_per_layer = 0;

    // Set for the things this probe does NOT model yet, so the report says so
    // instead of quietly implying full coverage.
    bool workspace_estimate_available = false;
    bool vision_tower_unmodelled = false;
};

// Build the PlanInput. Pure.
PlanInput shadow_plan_input(const ShadowPlanProbe& probe);

// Human-readable comparison: what the live budget chose, what the plan would
// choose, and the attribution of the gap. Pure: returns the text, does not log.
std::string shadow_plan_report(const ShadowPlanProbe& probe, const PlanResult& shadow,
                               int live_kv_blocks);

}  // namespace imp
