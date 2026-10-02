// KV re-sizing at the library reserve warmup measured (#2436). CPU-only: plan_memory() is pure,
// and the live-pass correction is integer arithmetic on what compute_vram_budget() recorded.

#include <gtest/gtest.h>

#include "memory/plan.h"
#include "runtime/kv_replan.h"

#include <algorithm>

using namespace imp;

namespace {

constexpr size_t kMiB = 1024ull * 1024;
constexpr size_t kKiB = 1024ull;

// Qwen3.8-27B-NVFP4 first start, 2026-10-02 (main 339d2891): distributable 9951 MiB, 16 KV layers,
// 18 KiB per block-layer (NVFP4 K+V with scales, 288 KiB per block), live pass clamped
// 229376 -> 13599 blocks with the reserve floored 1629 -> 3900 MiB.
KvReplanInput qwen38_27b_first_start() {
    KvReplanInput in;
    in.armed = true;
    ShadowPlanProbe& p = in.probe;
    p.distributable_bytes = 9951 * kMiB;
    p.weight_cache_demand = 2815 * kMiB;
    p.mandatory_cache_bytes = 1602 * kMiB;  // CUTLASS SF planes, budget log "sf=1601.6"
    p.ssm_state_bytes = 2226 * kMiB;
    p.recurrent_snapshot_bytes = 238 * kMiB;
    p.spec_decode_bytes = 159 * kMiB;
    p.engine_persistent_bytes = 139 * kMiB;
    p.library_reserve_bytes = 3900 * kMiB;
    p.n_kv_layers = 16;
    p.max_batch_size = 28;
    p.max_seq_len = 131072;
    p.kv_block_size = 16;
    p.kv_block_bytes_per_layer = 18 * kKiB;
    p.kv_growable_ceiling_blocks = 13599;
    in.plan_blocks = plan_memory(shadow_plan_input(p)).plan.kv.blocks;
    in.live_blocks = 13599;
    in.live_blocks_before_fit_clamp = 229376;
    in.live_blocks_pre_floor = 13599;
    in.live_reserve_other_bytes = 1629 * kMiB;
    in.per_block_bytes = 288 * kKiB;
    in.ceiling_blocks = 13599;
    in.initial_pct = 25;
    return in;
}

int plan_blocks_at(ShadowPlanProbe p, size_t reserve) {
    p.library_reserve_bytes = reserve;
    const PlanResult r = plan_memory(shadow_plan_input(p));
    return r.ok ? r.plan.kv.blocks : 0;
}

}  // namespace

// The plan a start that found the measurement cached builds: same probe, measured charge.
TEST(KvReplan, PlansWhatACachedMeasurementPlans) {
    const KvReplanInput in = qwen38_27b_first_start();
    const KvReplan r = replan_kv_for_measured_reserve(in, 1930 * kMiB);
    ASSERT_TRUE(r.apply);
    EXPECT_EQ(r.plan_blocks, plan_blocks_at(in.probe, 1930 * kMiB));
    EXPECT_EQ(in.plan_blocks, 5998) << "fixture reproduces the logged first-start plan";
    EXPECT_NEAR(r.plan_blocks, 8688, 1) << "and the logged plan at the cached 1930 MiB (MiB-rounded inputs)";
    EXPECT_EQ(r.commit_blocks, kv_initial_commit_blocks(r.plan_blocks, 25));
}

// 1970 MiB less reserve / 288 KiB per block = 7004 blocks: 13599 -> 20603, the second start's ceiling.
TEST(KvReplan, CeilingGainsTheFreedReserveInBlocks) {
    const KvReplan r = replan_kv_for_measured_reserve(qwen38_27b_first_start(), 1930 * kMiB);
    EXPECT_EQ(r.ceiling_blocks, 20603);
}

// The live reserve never drops below its other floors (1629 MiB): only 3900 - 1629 MiB is freed.
TEST(KvReplan, LiveReserveKeepsItsOtherFloor) {
    const KvReplan r = replan_kv_for_measured_reserve(qwen38_27b_first_start(), 1000 * kMiB);
    EXPECT_EQ(r.ceiling_blocks, 13599 + static_cast<int>((3900 - 1629) * kMiB / (288 * kKiB)));
}

// The fit clamp only lifts up to what it cut.
TEST(KvReplan, CeilingStopsAtTheUnclampedDemand) {
    KvReplanInput in = qwen38_27b_first_start();
    in.live_blocks_before_fit_clamp = 15000;
    EXPECT_EQ(replan_kv_for_measured_reserve(in, 1930 * kMiB).ceiling_blocks, 15000);
}

// A live pass bound by demand, not VRAM, has nothing to gain from the reserve.
TEST(KvReplan, DemandBoundLivePassKeepsItsCeiling) {
    KvReplanInput in = qwen38_27b_first_start();
    in.live_blocks_before_fit_clamp = 0;
    const KvReplan r = replan_kv_for_measured_reserve(in, 1930 * kMiB);
    EXPECT_EQ(r.ceiling_blocks, std::max(13599, r.plan_blocks));
}

TEST(KvReplan, NoOpWhenNotArmedOrNotOverCharged) {
    KvReplanInput in = qwen38_27b_first_start();
    EXPECT_FALSE(replan_kv_for_measured_reserve(in, 3900 * kMiB).apply) << "measurement equals the charge";
    EXPECT_FALSE(replan_kv_for_measured_reserve(in, 4500 * kMiB).apply) << "under-charge never shrinks";
    in.armed = false;
    EXPECT_FALSE(replan_kv_for_measured_reserve(in, 1930 * kMiB).apply);
}

// A fixed pool has no ceiling to raise; the plan may still say more.
TEST(KvReplan, FixedPoolKeepsCeilingZero) {
    KvReplanInput in = qwen38_27b_first_start();
    in.ceiling_blocks = 0;
    EXPECT_EQ(replan_kv_for_measured_reserve(in, 1930 * kMiB).ceiling_blocks, 0);
}

TEST(KvReplan, InitialCommitFollowsTheInitRule) {
    EXPECT_EQ(kv_initial_commit_blocks(8688, 25), 2150);
    EXPECT_EQ(kv_initial_commit_blocks(40, 25), 16);
    EXPECT_EQ(kv_initial_commit_blocks(10, 25), 10);
    EXPECT_EQ(kv_initial_commit_blocks(8688, 100), 8688);
}

TEST(KvReplan, ArmsOnlyForAnAcceptedPlanThatChargedTheConstant) {
    const KvReplanInput base = qwen38_27b_first_start();
    VRAMBudget live;
    live.kv_max_blocks = 13599;
    live.kv_blocks_before_fit_clamp = 229376;
    live.kv_blocks_pre_floor = 13599;
    live.reserve_other_bytes = 1629 * kMiB;
    KvReplanInput r;
    kv_replan_arm(r, base.probe, live, base.plan_blocks, /*plan_ok=*/true, /*constant_charged=*/true);
    EXPECT_TRUE(r.armed);
    EXPECT_EQ(r.live_blocks_before_fit_clamp, 229376);
    kv_replan_set_pool(r, 288 * kKiB, 13599, 13599);
    EXPECT_TRUE(r.armed);
    kv_replan_set_pool(r, 288 * kKiB, 13599, 6799);
    EXPECT_FALSE(r.armed) << "a pool that needed the halving retry stays as built";
    kv_replan_arm(r, base.probe, live, base.plan_blocks, /*plan_ok=*/false, /*constant_charged=*/true);
    EXPECT_FALSE(r.armed);
    kv_replan_arm(r, base.probe, live, base.plan_blocks, /*plan_ok=*/true, /*constant_charged=*/false);
    EXPECT_FALSE(r.armed) << "a cached or pinned reserve is already the measurement";
}
