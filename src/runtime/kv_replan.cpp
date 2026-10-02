#include "runtime/kv_replan.h"

#include "memory/plan.h"

#include <algorithm>

namespace imp {

int kv_initial_commit_blocks(int plan_blocks, int initial_pct) {
    const int pct = std::clamp(initial_pct, 1, 100);
    if (pct >= 100)
        return plan_blocks;
    return std::min(plan_blocks, std::max(16, plan_blocks / 100 * pct));
}

// Live pass ceiling at a smaller reserve: only the post-weight fit clamp depends on it, 1 block
// per per_block_bytes freed, up to what the clamp cut (vram_budget.cpp, KV clamped X -> Y).
static int live_blocks_at(const KvReplanInput& in, size_t charged, size_t measured) {
    if (in.live_blocks_before_fit_clamp <= 0 || in.per_block_bytes == 0)
        return in.live_blocks;
    const size_t before = std::max(in.live_reserve_other_bytes, charged);
    const size_t after = std::max(in.live_reserve_other_bytes, measured);
    const size_t freed = before > after ? before - after : 0;
    const long long fit = static_cast<long long>(in.live_blocks_pre_floor) +
                          static_cast<long long>(freed / in.per_block_bytes);
    const long long capped = std::min<long long>(fit, in.live_blocks_before_fit_clamp);
    return std::max(in.live_blocks, static_cast<int>(capped));
}

KvReplan replan_kv_for_measured_reserve(const KvReplanInput& in, size_t measured) {
    KvReplan out;
    const size_t charged = in.probe.library_reserve_bytes;
    if (!in.armed || measured >= charged)
        return out;
    ShadowPlanProbe probe = in.probe;
    probe.library_reserve_bytes = measured;
    const int live = live_blocks_at(in, charged, measured);
    if (probe.kv_growable_ceiling_blocks > 0)
        probe.kv_growable_ceiling_blocks = live;
    const PlanResult plan = plan_memory(shadow_plan_input(probe));
    out.plan_blocks = std::max(in.plan_blocks, plan.ok ? plan.plan.kv.blocks : 0);
    out.commit_blocks = kv_initial_commit_blocks(out.plan_blocks, in.initial_pct);
    out.ceiling_blocks = in.ceiling_blocks > 0 ? std::max({in.ceiling_blocks, out.plan_blocks, live}) : 0;
    out.apply = out.plan_blocks > in.plan_blocks || out.ceiling_blocks > in.ceiling_blocks;
    return out;
}

void kv_replan_arm(KvReplanInput& r, const ShadowPlanProbe& probe, const VRAMBudget& live, int plan_blocks,
                   bool plan_ok, bool constant_charged) {
    r = KvReplanInput{};
    r.armed = plan_ok && constant_charged;
    r.probe = probe;
    r.plan_blocks = plan_blocks;
    r.live_blocks = live.kv_max_blocks;
    r.live_blocks_before_fit_clamp = live.kv_blocks_before_fit_clamp;
    r.live_blocks_pre_floor = live.kv_blocks_pre_floor;
    r.live_reserve_other_bytes = live.reserve_other_bytes;
}

void kv_replan_set_pool(KvReplanInput& r, size_t per_block_bytes, int ceiling_planned, int ceiling_built) {
    r.per_block_bytes = per_block_bytes;
    r.ceiling_blocks = ceiling_built;
    if (ceiling_built != ceiling_planned)
        r.armed = false;
}

}  // namespace imp
