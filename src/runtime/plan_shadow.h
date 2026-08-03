#pragma once

// A7 step 2b (docs/MEMORY_ARCHITECTURE.md): run plan_memory() alongside
// compute_vram_budget(), log both, and attribute the difference — WITHOUT
// applying anything.
//
// The comparison is deliberately narrow. Both sides are fed the SAME demand
// figures (weight-cache estimate, SSM footprint, workspace estimate), which
// the old pass already computes and now publishes. What is compared is
// therefore the *allocation policy*: how the residual is distributed, whether
// the admission floor is honoured, and what the plan charges that the old pass
// cannot see. Migrating the demand estimates themselves is step 6.
//
// The probe carries plain scalars rather than Model/EngineConfig on purpose:
// this header stays CUDA-free and dependency-free, so the whole thing is
// testable in the CPU lane.

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
// choose, and the attribution of the gap. Pure — returns the text, does not log.
std::string shadow_plan_report(const ShadowPlanProbe& probe, const PlanResult& shadow,
                               int live_kv_blocks);

}  // namespace imp
