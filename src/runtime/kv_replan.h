#pragma once

#include "runtime/plan_shadow.h"
#include "runtime/vram_budget.h"

#include <cstddef>

namespace imp {

// KV sizing inputs kept from init, to re-size at the library reserve warmup measured (#2436).
// Armed only when init charged the constant: no cached measurement, no operator pin.
struct KvReplanInput {
    bool armed = false;
    ShadowPlanProbe probe;                 // init probe; library_reserve_bytes = the charge
    int plan_blocks = 0;                   // blocks the init plan applied, before the commit pct
    int live_blocks = 0;                   // live pass kv_max_blocks
    int live_blocks_before_fit_clamp = 0;  // 0 = the post-weight fit clamp did not bind
    int live_blocks_pre_floor = 0;         // live pass before the min_kv_tokens floor
    size_t live_reserve_other_bytes = 0;   // live reserve before the library floor
    size_t per_block_bytes = 0;            // K+V of one block over every attention layer
    int ceiling_blocks = 0;                // the pool's ceiling at init, 0 = fixed pool
    int initial_pct = 100;                 // kv_cache.growable_initial_pct
};

struct KvReplan {
    bool apply = false;
    int plan_blocks = 0;     // plan at the measured reserve
    int commit_blocks = 0;   // what the pool commits up front at that plan
    int ceiling_blocks = 0;  // growable ceiling at the measured reserve, >= the init ceiling
};

// The plan, commit and ceiling a start that had found `measured` cached would have built.
// Never smaller than init; no-op unless armed and the measurement is below the charge.
KvReplan replan_kv_for_measured_reserve(const KvReplanInput& in, size_t measured);

// Commit a growable pool makes up front: pct of the plan, at least 16 blocks (init rule).
int kv_initial_commit_blocks(int plan_blocks, int initial_pct);

// Init half, after the plan: arms only for an accepted plan that charged the constant.
void kv_replan_arm(KvReplanInput& r, const ShadowPlanProbe& probe, const VRAMBudget& live, int plan_blocks,
                   bool plan_ok, bool constant_charged);
// Init half, after the pool is built: a pool that needed the halving retry stays as built.
void kv_replan_set_pool(KvReplanInput& r, size_t per_block_bytes, int ceiling_planned, int ceiling_built);

}  // namespace imp
