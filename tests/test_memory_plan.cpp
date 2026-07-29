// The planner (docs/MEMORY_ARCHITECTURE.md A4), invariants V7 (determinism)
// and V8 (sufficiency).
//
// CPU-only by construction: plan_memory() never queries the device, takes no
// Model and no EngineConfig, and is a pure function of a plain struct. That is
// the property under test as much as any individual number — the thing it
// replaces, compute_vram_budget(), is driven by a live cudaMemGetInfo reading
// and therefore cannot be tested at all without a GPU.

#include <gtest/gtest.h>

#include "memory/plan.h"

#include <random>
#include <string>

using namespace imp;

namespace {

constexpr size_t kMiB = 1024ull * 1024;
constexpr size_t kGiB = 1024ull * kMiB;

// Roughly the measured dense reference config: Qwen3-4B Q8_0, 36 layers,
// n_kv_heads 8, head_dim 128, block 16 -> 8*128*2*2 = 4096 B per block-layer.
PlanInput dense_input() {
    PlanInput in;
    in.model.n_layers = 36;
    in.model.n_kv_layers = 36;
    in.model.n_kv_heads = 8;
    in.model.head_dim = 128;
    in.model.weight_bytes = 4076 * kMiB;
    in.model.weight_cache_bytes = 2158 * kMiB;
    in.model.mandatory_cache_bytes = 0;

    in.limits.max_batch_size = 8;
    in.limits.max_seq_len = 4096;
    in.limits.kv_block_size = 16;
    in.limits.kv_block_bytes_per_layer = 16ull * 8 * 128 * 2 * 2;
    in.limits.min_kv_tokens = 16384;

    in.engine_persistent_bytes = 397 * kMiB;
    in.forward_scratch_bytes = 64 * kMiB;
    in.context_bytes = 1680 * kMiB;
    in.library = LibraryReserve{3900 * kMiB, "measured 2026-07-28"};
    in.budget_bytes = 32607 * kMiB;
    return in;
}

size_t kv_bytes_for(const PlanInput& in, int blocks) {
    return static_cast<size_t>(blocks) * in.limits.kv_block_bytes_per_layer *
           static_cast<size_t>(in.model.n_kv_layers);
}

}  // namespace

// ── The plan is a plan, not a measurement ─────────────────────────────

TEST(MemoryPlan, FitsTheDenseReferenceConfig) {
    auto res = plan_memory(dense_input());
    ASSERT_TRUE(res) << res.failure.report();
    EXPECT_LE(res.plan.total(), dense_input().budget_bytes);
    EXPECT_GT(res.plan.kv.blocks, 0);
    EXPECT_FALSE(res.plan.kv.below_floor);
}

TEST(MemoryPlan, ChargesTheLibraryReserveAsAFirstClassLineItem) {
    // A1.5: ~3.9 GiB is claimed on the first forward pass, after the old
    // planner was already done. If the plan does not charge it, the KV pool is
    // sized from a number that much too optimistic.
    //
    // The comparison only says anything where the residual actually binds. At
    // the full 32 GiB budget KV stops at what it needs (2048 blocks) with room
    // to spare, so both arms would be identical and the test would pass while
    // proving nothing — squeeze the budget until KV is residual-bound.
    auto in = dense_input();
    in.budget_bytes = 16 * kGiB;
    auto with = plan_memory(in);
    ASSERT_TRUE(with);

    in.library.bytes = 0;
    auto without = plan_memory(in);
    ASSERT_TRUE(without);

    EXPECT_GT(without.plan.kv.blocks, with.plan.kv.blocks)
        << "ignoring the library reserve must visibly over-allocate KV — that is the bug";
    EXPECT_EQ(with.plan.library_reserve, 3900 * kMiB);

    bool named = false;
    for (const auto& l : with.plan.lines())
        if (std::string(l.name).find("library") != std::string::npos)
            named = true;
    EXPECT_TRUE(named) << "I7: it has to be attributed, not buried in a residual";
}

TEST(MemoryPlan, KvTakesTheComputedResidualAndNeverMoreThanItNeeds) {
    auto in = dense_input();
    auto res = plan_memory(in);
    ASSERT_TRUE(res);

    // 8 seqs x 4096 tokens at block 16 = 2048 blocks. The card can hold more,
    // so the plan must stop at what is actually needed rather than filling.
    EXPECT_EQ(res.plan.kv.blocks_per_seq, 256);
    EXPECT_EQ(res.plan.kv.blocks, 256 * 8);
    EXPECT_EQ(res.plan.kv.bytes, kv_bytes_for(in, 256 * 8));
}

TEST(MemoryPlan, KvShrinksToTheResidualWhenTheBudgetIsTight) {
    // 16 GiB leaves a residual between the admission floor (1024 blocks) and
    // what 8x4096 wants (2048), so the pool is genuinely residual-bound.
    auto in = dense_input();
    in.budget_bytes = 16 * kGiB;
    auto res = plan_memory(in);
    ASSERT_TRUE(res) << res.failure.report();

    EXPECT_LT(res.plan.kv.blocks, res.plan.kv.blocks_per_seq * 8)
        << "a tight budget must reduce the pool, not overcommit it";
    EXPECT_FALSE(res.plan.kv.below_floor);
    EXPECT_LE(res.plan.total(), in.budget_bytes);
}

TEST(MemoryPlan, ATooTightBudgetFailsInsteadOfServingAnUnusablePool) {
    // 12 GiB: the fixed charges alone (context + library + weights + caches +
    // workspaces) leave a residual of a few MiB. The old planner clamped KV to
    // its 16-block floor and served anyway; every longer prompt then came back
    // cancelled. Failing at load is the point of I4.
    auto in = dense_input();
    in.budget_bytes = 12 * kGiB;
    auto res = plan_memory(in);
    ASSERT_FALSE(res.ok);
    EXPECT_TRUE(res.plan.kv.below_floor);
    EXPECT_NE(res.failure.report().find("Cannot fit"), std::string::npos);
}

// ── V7: determinism ───────────────────────────────────────────────────

TEST(MemoryPlan, IsDeterministicAcrossRandomisedConfigs) {
    std::mt19937 rng(4242);
    for (int i = 0; i < 1000; ++i) {
        PlanInput in = dense_input();
        in.limits.max_batch_size = 1 + static_cast<int>(rng() % 32);
        in.limits.max_seq_len = 512 * (1 + static_cast<int>(rng() % 64));
        in.limits.kv_block_size = (rng() % 2) ? 16 : 32;
        in.model.weight_bytes = (1 + rng() % 24) * kGiB;
        in.model.weight_cache_bytes = (rng() % 4) * kGiB;
        in.budget_bytes = (8 + rng() % 25) * kGiB;
        in.library.bytes = (rng() % 5) * kGiB;

        auto a = plan_memory(in);
        auto b = plan_memory(in);
        ASSERT_EQ(a.ok, b.ok) << "i=" << i;
        ASSERT_EQ(a.plan.total(), b.plan.total()) << "i=" << i;
        ASSERT_EQ(a.plan.kv.blocks, b.plan.kv.blocks) << "i=" << i;
        ASSERT_EQ(a.plan.model_resident, b.plan.model_resident) << "i=" << i;
        if (!a.ok)
            ASSERT_EQ(a.failure.over_by, b.failure.over_by) << "i=" << i;
    }
}

// ── V8: the plan is sufficient — it never promises what does not fit ──

TEST(MemoryPlan, NeverProducesAnOverBudgetPlan) {
    std::mt19937 rng(99);
    for (int i = 0; i < 2000; ++i) {
        PlanInput in = dense_input();
        in.limits.max_batch_size = 1 + static_cast<int>(rng() % 64);
        in.limits.max_seq_len = 1024 * (1 + static_cast<int>(rng() % 128));
        in.model.weight_bytes = (rng() % 30) * kGiB;
        in.model.weight_cache_bytes = (rng() % 8) * kGiB;
        in.features.ssm_state_bytes = (rng() % 512) * kMiB;
        in.features.vision_tower_bytes = (rng() % 2000) * kMiB;
        in.budget_bytes = (4 + rng() % 29) * kGiB;

        auto res = plan_memory(in);
        if (res.ok) {
            ASSERT_LE(res.plan.total(), in.budget_bytes)
                << "i=" << i << ": a successful plan that does not fit is the whole failure mode";
        } else {
            ASSERT_GT(res.failure.requested, 0u) << "i=" << i;
            ASSERT_EQ(res.failure.budget, in.budget_bytes) << "i=" << i;
        }
    }
}

// ── Failure is a load-time report, not a mid-generation surprise ──────

TEST(MemoryPlan, FailsWithAnItemisedReportAndActionableLevers) {
    auto in = dense_input();
    in.model.weight_bytes = 28 * kGiB;  // will not fit alongside the fixed charges
    auto res = plan_memory(in);
    ASSERT_FALSE(res.ok);

    EXPECT_GT(res.failure.over_by, 0u);
    EXPECT_FALSE(res.failure.lines.empty());
    ASSERT_FALSE(res.failure.levers.empty()) << "a failure the operator cannot act on is useless";
    EXPECT_LE(res.failure.levers.size(), 3u);

    const std::string report = res.failure.report();
    EXPECT_NE(report.find("Cannot fit"), std::string::npos);
    EXPECT_NE(report.find("over by"), std::string::npos);
    EXPECT_NE(report.find("levers"), std::string::npos);
    // Named line items, not a single number (I7).
    EXPECT_NE(report.find("model weights"), std::string::npos);
    EXPECT_NE(report.find("library reserve"), std::string::npos);

    // The levers are sorted by how much they actually free.
    for (size_t i = 1; i < res.failure.levers.size(); ++i)
        EXPECT_GE(res.failure.levers[i - 1].frees, res.failure.levers[i].frees);
}

TEST(MemoryPlan, RefusesToServeAPoolBelowTheAdmissionFloor) {
    // Observed on Qwen3.6-35B-A3B-NVFP4 at --max-batch 64: KV collapsed to 16
    // blocks = 512 tokens and every longer prompt came back cancelled with no
    // hint why, while /v1/models kept advertising max_seq_len. Failing at load
    // is strictly better than serving a config that cannot answer.
    auto in = dense_input();
    in.model.weight_bytes = 24 * kGiB;
    in.limits.min_kv_tokens = 16384;

    auto res = plan_memory(in);
    ASSERT_FALSE(res.ok) << "a pool below the floor must not be reported as a working plan";
    EXPECT_GT(res.failure.over_by, 0u);
}

// ── SWA layers are charged batch-shaped, not context-shaped ───────────

TEST(MemoryPlan, SwaLayersAreChargedTheirWindowNotTheContext) {
    auto in = dense_input();
    in.features.n_swa_layers = 24;      // 24 of 36 are sliding-window
    in.features.swa_live_tokens = 1024;

    auto res = plan_memory(in);
    ASSERT_TRUE(res) << res.failure.report();
    EXPECT_GT(res.plan.kv.swa_blocks, 0);
    EXPECT_GT(res.plan.kv.swa_bytes, 0u);

    // The global pool now covers 12 layers, so it must be markedly cheaper
    // than the all-global plan for the same context.
    auto all_global = plan_memory(dense_input());
    ASSERT_TRUE(all_global);
    EXPECT_LT(res.plan.kv.bytes, all_global.plan.kv.bytes)
        << "SWA sizing has to actually reduce the context-scaled part";
}

TEST(MemoryPlan, SwaSizingIsIgnoredWhenTheLayerCountIsNonsensical) {
    auto in = dense_input();
    in.features.n_swa_layers = 999;  // more SWA layers than the model has
    in.features.swa_live_tokens = 1024;
    auto res = plan_memory(in);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.plan.kv.swa_blocks, 0) << "a bad hint must degrade to the safe plan";
}

// ── Edges ─────────────────────────────────────────────────────────────

TEST(MemoryPlan, ZeroBudgetFailsCleanly) {
    auto in = dense_input();
    in.budget_bytes = 0;
    auto res = plan_memory(in);
    EXPECT_FALSE(res.ok);
    EXPECT_EQ(res.plan.kv.blocks, 0);
}

TEST(MemoryPlan, NoKvLayersIsNotADivideByZero) {
    auto in = dense_input();
    in.model.n_kv_layers = 0;
    in.limits.kv_block_bytes_per_layer = 0;
    in.limits.min_kv_tokens = 0;
    auto res = plan_memory(in);
    EXPECT_EQ(res.plan.kv.blocks, 0);
    EXPECT_EQ(res.plan.kv.bytes, 0u);
}

TEST(MemoryPlan, TotalEqualsTheSumOfItsLines) {
    auto in = dense_input();
    in.features.ssm_state_bytes = 128 * kMiB;
    in.features.residual_ring_bytes = 64 * kMiB;
    in.features.vision_tower_bytes = 1610 * kMiB;
    auto res = plan_memory(in);
    ASSERT_TRUE(res) << res.failure.report();

    size_t sum = 0;
    for (const auto& l : res.plan.lines())
        sum += l.bytes;
    EXPECT_EQ(sum, res.plan.total())
        << "criterion 6 is >=95% accounted; a plan that does not add up cannot get there";
}

// ── A7 step 2b: the shadow plan run next to the live budget ───────────

#include "runtime/plan_shadow.h"

namespace {

// The measured dense reference point, expressed the way the engine sees it at
// budget time: weights and context are already spent, so what is left is the
// distributable residual.
ShadowPlanProbe dense_probe() {
    ShadowPlanProbe p;
    p.distributable_bytes = 22290 * kMiB;  // logged by the live pass on this config
    p.weight_cache_demand = 2158 * kMiB;
    p.ssm_state_bytes = 0;
    p.engine_persistent_bytes = 397 * kMiB;
    p.workspace_estimate_available = true;
    p.library_reserve_bytes = kMeasuredLibraryReserveBytes;
    p.n_kv_layers = 36;
    p.max_batch_size = 8;
    p.max_seq_len = 4096;
    p.kv_block_size = 16;
    p.min_kv_tokens = 16384;
    p.kv_block_bytes_per_layer = 16ull * 8 * 128 * 2 * 2;
    return p;
}

}  // namespace

TEST(ShadowPlan, DoesNotChargeWeightsOrContextASecondTime) {
    // At budget time the weights and the CUDA context are already resident.
    // Charging them again would make the shadow plan reject configurations the
    // engine is happily running.
    const auto in = shadow_plan_input(dense_probe());
    EXPECT_EQ(in.model.weight_bytes, 0u);
    EXPECT_EQ(in.context_bytes, 0u);
    EXPECT_EQ(in.budget_bytes, dense_probe().distributable_bytes);
}

TEST(ShadowPlan, FeedsTheLivePassOwnDemandFigureSoThePoliciesAreComparable) {
    auto p = dense_probe();
    p.weight_cache_demand = 1234 * kMiB;
    const auto in = shadow_plan_input(p);
    EXPECT_EQ(in.model.weight_cache_bytes, 1234u * kMiB)
        << "the comparison is of allocation policy, not of demand estimation";
}

TEST(ShadowPlan, TheLibraryReserveIsWhatSeparatesItFromTheLivePass) {
    auto with = plan_memory(shadow_plan_input(dense_probe()));
    ASSERT_TRUE(with) << with.failure.report();

    auto p = dense_probe();
    p.library_reserve_bytes = 0;
    auto without = plan_memory(shadow_plan_input(p));
    ASSERT_TRUE(without);

    EXPECT_LE(with.plan.kv.blocks, without.plan.kv.blocks);
    EXPECT_LE(with.plan.total(), dense_probe().distributable_bytes);
}

TEST(ShadowPlan, ReportSaysWhatItDoesNotModel) {
    auto p = dense_probe();
    p.workspace_estimate_available = false;
    p.vision_tower_unmodelled = true;
    const auto res = plan_memory(shadow_plan_input(p));
    const std::string r = shadow_plan_report(p, res, /*live_kv_blocks=*/2048);

    EXPECT_NE(r.find("shadow plan"), std::string::npos);
    EXPECT_NE(r.find("NOT applied"), std::string::npos);
    EXPECT_NE(r.find("workspaces not modelled"), std::string::npos)
        << "a probe that hides its gaps is worse than no probe";
    EXPECT_NE(r.find("vision tower not modelled"), std::string::npos);
    EXPECT_NE(r.find("forward scratch not modelled"), std::string::npos);
}

TEST(ShadowPlan, ReportShowsBothKvDecisions) {
    const auto p = dense_probe();
    const auto res = plan_memory(shadow_plan_input(p));
    ASSERT_TRUE(res);
    const std::string r = shadow_plan_report(p, res, /*live_kv_blocks=*/4096);
    EXPECT_NE(r.find("live 4096 blocks"), std::string::npos);
    EXPECT_NE(r.find("library reserve"), std::string::npos);
}

TEST(ShadowPlan, ReportsARejectionWithTheFullFailureText) {
    auto p = dense_probe();
    p.distributable_bytes = 6 * 1024 * kMiB;  // not enough for the floor
    const auto res = plan_memory(shadow_plan_input(p));
    ASSERT_FALSE(res.ok);
    const std::string r = shadow_plan_report(p, res, /*live_kv_blocks=*/16);
    EXPECT_NE(r.find("plan REJECTS"), std::string::npos);
    EXPECT_NE(r.find("Cannot fit"), std::string::npos)
        << "the operator needs the itemisation, not just the verdict";
}
