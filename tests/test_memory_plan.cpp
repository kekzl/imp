// The planner (docs/internals/MEMORY.md A4), invariants V7 (determinism)
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
        if (!a.ok) {
            ASSERT_EQ(a.failure.over_by, b.failure.over_by) << "i=" << i;
        }
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

namespace {

// The shadow probe of Qwen3.8-27B-NVFP4 at runtime.max_batch_size=64
// (2026-09-08 server log): distributable 9683 MiB after the weights, SSM/GDN
// state 4968 MiB for 64 slots, library reserve 3900, mandatory caches 1602,
// engine-persistent 949. The plan rejected (over by 1735 MiB), the live pass
// never charged the state, and the pool probe read 528 GB/s: spilled.
PlanInput hybrid_64_slots() {
    PlanInput in;
    in.model.n_layers = 64;
    in.model.n_kv_layers = 16;
    in.model.n_kv_heads = 4;
    in.model.head_dim = 256;
    in.model.weight_bytes = 0;  // already uploaded: the budget is post-weight
    in.model.weight_cache_bytes = 1602 * kMiB;
    in.model.mandatory_cache_bytes = 1602 * kMiB;
    in.features.ssm_state_bytes = 64ull * 77 * kMiB + 40 * kMiB;  // 4968 MiB
    in.limits.max_batch_size = 64;
    in.limits.max_seq_len = 4096;
    in.limits.kv_block_size = 16;
    in.limits.kv_block_bytes_per_layer = 16ull * 4 * 256 * 2 / 2 + 1024;  // NVFP4 + scales
    in.limits.min_kv_tokens = 16384;
    in.engine_persistent_bytes = 949 * kMiB;
    in.library = LibraryReserve{3900 * kMiB, "measured 2026-09-08"};
    in.budget_bytes = 9683 * kMiB;
    return in;
}

const PlanLever* batch_lever(const PlanResult& res) {
    for (const auto& lv : res.failure.levers)
        if (lv.change.rfind("runtime.max_batch_size", 0) == 0)
            return &lv;
    return nullptr;
}

}  // namespace

TEST(MemoryPlan, BatchLeverCountsTheBatchShapedSsmState) {
    const auto in = hybrid_64_slots();
    const auto res = plan_memory(in);
    ASSERT_FALSE(res.ok);
    const PlanLever* lv = batch_lever(res);
    ASSERT_NE(lv, nullptr) << "the batch lever must be offered when the state is the overrun";
    // Was "runtime.max_batch_size 64 -> 63 frees 0 MiB": the KV pool was
    // already at its floor and the per-slot state was not counted.
    EXPECT_GE(lv->frees, in.features.ssm_state_bytes / 64);
}

TEST(MemoryPlan, FittingBatchIsTheLargestThePlanAccepts) {
    const auto in = hybrid_64_slots();
    const size_t per_slot = in.features.ssm_state_bytes / 64;
    const int fit = plan_fitting_batch(in, per_slot);
    ASSERT_GE(fit, 1);
    ASSERT_LT(fit, 64);
    auto at = [&](int b) {
        PlanInput t = in;
        t.limits.max_batch_size = b;
        t.features.ssm_state_bytes = in.features.ssm_state_bytes - per_slot * static_cast<size_t>(64 - b);
        return plan_memory(t).ok;
    };
    EXPECT_TRUE(at(fit));
    EXPECT_FALSE(at(fit + 1));
    // A configuration that fits keeps its batch.
    EXPECT_EQ(plan_fitting_batch(dense_input(), 0), 8);
    // Nothing fits: says so instead of inventing a slot.
    PlanInput none = in;
    none.budget_bytes = 1 * kMiB;
    EXPECT_EQ(plan_fitting_batch(none, per_slot), 0);
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

// ── Every FeatureSet field the plan reads must reach a line ───────────

TEST(MemoryPlan, EveryFeatureFieldReachesALine) {
    // plan_memory() reads five FeatureSet byte fields. Four of them were
    // written by nobody: the only caller (shadow_plan_input) filled
    // ssm_state_bytes and left the rest at 0, so the 256 MiB recurrent
    // snapshot store, the speculative staging and the vision tower were
    // charged to nothing and taken AFTER the KV pool was sized. A field that
    // cannot appear as its own line cannot be reconciled against the log
    // either, which is what made the 5088 MiB state invisible (MEMORY.md D14).
    //
    // Distinct values, so a field folded into another line cannot pass by
    // coincidence.
    auto in = dense_input();
    in.features.ssm_state_bytes = 101 * kMiB;
    in.features.recurrent_snapshot_bytes = 102 * kMiB;
    in.features.residual_ring_bytes = 103 * kMiB;
    in.features.spec_decode_bytes = 104 * kMiB;
    in.features.vision_tower_bytes = 105 * kMiB;

    auto res = plan_memory(in);
    ASSERT_TRUE(res) << res.failure.report();

    const auto lines = res.plan.lines();
    auto line_with = [&](size_t bytes) -> const PlanLine* {
        for (const auto& l : lines)
            if (l.bytes == bytes)
                return &l;
        return nullptr;
    };
    EXPECT_NE(line_with(101 * kMiB), nullptr) << "features.ssm_state_bytes";
    EXPECT_NE(line_with(102 * kMiB), nullptr) << "features.recurrent_snapshot_bytes";
    EXPECT_NE(line_with(103 * kMiB), nullptr) << "features.residual_ring_bytes";
    EXPECT_NE(line_with(104 * kMiB), nullptr) << "features.spec_decode_bytes";
    EXPECT_NE(line_with(105 * kMiB), nullptr) << "features.vision_tower_bytes";
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

// ── The SSM/GDN state byte formula, one copy ──────────────────────────

#include "memory/ssm_state_size.h"

namespace {

// Qwen3.8-27B-NVFP4 (/home/kekz/models/Qwen3.8-27B-NVFP4-vllm/config.json):
// 64 layers of which 48 are linear_attention, linear_conv_kernel_dim 4,
// linear_num_value_heads 48 x linear_value_head_dim 128 -> ssm_inner_size 6144,
// linear_key_head_dim 128 -> ssm_state_size, linear_num_key_heads 16 -> groups.
// conv_channels = 6144 + 2 * 16 * 128 = 10240.
SsmStateGeometry qwen38_gdn(QType h_dtype) {
    return SsmStateGeometry{/*n_ssm_layers=*/48,   /*conv_channels=*/10240, /*conv_kernel=*/4,
                            /*n_heads=*/48,        /*head_dim=*/128,        /*state_size=*/128,
                            h_dtype};
}

}  // namespace

TEST(SsmStatePool, PinsTheQwen38GeometryTheAllocatorTakes) {
    // The two copies of this formula disagreed: vram_budget.cpp charged
    // conv_channels * (conv_kernel - 1) * 4 unaligned, ssm_state.cu allocated
    // align256(conv_channels * conv_kernel * 4) + align256(h). 4968 MiB planned
    // against 5088 MiB taken at 64 slots (MEMORY.md D14) - short in the
    // direction that oversubscribes the card.
    const auto g = qwen38_gdn(QType::F16);
    EXPECT_EQ(ssm_conv_bytes_per_layer(g), 10240ull * 4 * 4);          // already 256-aligned
    EXPECT_EQ(ssm_h_bytes_per_layer(g), 48ull * 128 * 128 * 2);        // already 256-aligned
    EXPECT_EQ(ssm_bytes_per_slot(g), 48ull * (163840 + 1572864));      // 79.5 MiB
    EXPECT_EQ(ssm_pool_bytes(g, 64, 0), 5088ull * kMiB);
    EXPECT_EQ(ssm_pool_bytes(g, 41, 0), 41ull * 79 * kMiB + 41 * kMiB / 2);

    // The retired charge, so a revert to it cannot pass unnoticed.
    const size_t old_charge = 48ull * 64 *
                              (10240ull * (4 - 1) * sizeof(float) + 48ull * 128 * 128 * 2);
    EXPECT_GT(ssm_pool_bytes(g, 64, 0), old_charge)
        << "the plan must never charge less than the allocator takes";

    // Reserved verify slots are priced with the pool, not on top of it.
    EXPECT_EQ(ssm_pool_bytes(g, 41, 3), ssm_pool_bytes(g, 44, 0));

    // F32 h state doubles the dominant term.
    EXPECT_EQ(ssm_h_bytes_per_layer(qwen38_gdn(QType::F32)), 2 * ssm_h_bytes_per_layer(g));
}

TEST(SsmStatePool, FailureMessageNamesSlotsLayersFreeAndTheLever) {
    // `Failed to allocate SSM state pool (5335154688 bytes)` was the whole
    // message: no slot count, no free figure, no knob.
    const std::string m = ssm_pool_failure_message(ssm_pool_bytes(qwen38_gdn(QType::F16), 64, 0),
                                                  /*slots=*/64, /*reserved_slots=*/3,
                                                  /*n_ssm_layers=*/48, /*free_bytes=*/312 * kMiB);
    EXPECT_NE(m.find("5088 MiB"), std::string::npos) << m;
    EXPECT_NE(m.find("67 slots"), std::string::npos) << m;
    EXPECT_NE(m.find("64 live + 3 reserved"), std::string::npos) << m;
    EXPECT_NE(m.find("48 layers"), std::string::npos) << m;
    EXPECT_NE(m.find("312 MiB free"), std::string::npos) << m;
    EXPECT_NE(m.find("runtime.max_batch_size"), std::string::npos)
        << "a refusal without the lever is a stack trace with better grammar: " << m;
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

// The hybrid the ceiling line exists for: 48 GDN layers of per-slot recurrent
// state against 16 attention layers of paged KV (Qwen3.8-27B-NVFP4, block 16),
// after the #1958 clamp took the batch from 64 to 41 slots.
ShadowPlanProbe hybrid_probe() {
    ShadowPlanProbe p;
    p.distributable_bytes = 9760 * kMiB;
    p.weight_cache_demand = 1602 * kMiB;
    p.mandatory_cache_bytes = 1602 * kMiB;
    p.ssm_state_bytes = 41ull * 77 * kMiB + 26 * kMiB;  // 41 slots x 77.625 MiB
    p.engine_persistent_bytes = 949 * kMiB;
    p.workspace_estimate_available = true;
    p.library_reserve_bytes = 3900 * kMiB;
    p.n_kv_layers = 16;
    p.max_batch_size = 41;
    p.max_seq_len = 4096;
    p.kv_block_size = 16;
    p.min_kv_tokens = 4096;
    p.kv_block_bytes_per_layer = 16ull * 4 * 256 * 2 / 2 + 1024;  // NVFP4 + scales
    return p;
}

}  // namespace

// ── E1: the report states BOTH pool ceilings ──────────────────────────

TEST(MemoryPlan, KvSeqCeilingIsBlocksOverBlocksPerSeq) {
    // The KV pool's ceiling in sequences is what the plan already knows and
    // never said: blocks / blocks_per_seq. On the hybrid the two pools are
    // sized by different rules (state is a fixed pre-charge per slot, KV takes
    // the residual), so nothing forces them to agree - and when they disagree
    // the state was bought for slots the KV pool cannot serve.
    const auto res = plan_memory(shadow_plan_input(hybrid_probe()));
    ASSERT_TRUE(res) << res.failure.report();
    ASSERT_GT(res.plan.kv.blocks_per_seq, 0);
    const int kv_seqs = res.plan.kv.blocks / res.plan.kv.blocks_per_seq;
    const int recurrent_seqs = hybrid_probe().max_batch_size;
    EXPECT_EQ(res.plan.kv.blocks_per_seq, 4096 / 16);
    // Either the batch fits both pools, or the ceiling line has to warn.
    const int fit = plan_fitting_batch(shadow_plan_input(hybrid_probe()),
                                       hybrid_probe().ssm_state_bytes / 41);
    EXPECT_TRUE(fit <= kv_seqs || kv_seqs < recurrent_seqs)
        << "a batch the plan accepts (" << fit << ") above the KV ceiling (" << kv_seqs
        << ") must be reported, not left to the first long request";
}

TEST(ShadowPlan, ReportStatesBothPoolCeilings) {
    const auto p = hybrid_probe();
    const auto res = plan_memory(shadow_plan_input(p));
    ASSERT_TRUE(res) << res.failure.report();
    const std::string r = shadow_plan_report(p, res, /*live_kv_blocks=*/896);

    const int kv_seqs = res.plan.kv.blocks / res.plan.kv.blocks_per_seq;
    EXPECT_NE(r.find("ceiling: recurrent 41 seqs"), std::string::npos)
        << "startup must REPORT the recurrent ceiling, not only the KV one:\n" << r;
    EXPECT_NE(r.find("KV " + std::to_string(kv_seqs) + " seqs at max_seq_len 4096"), std::string::npos) << r;
    EXPECT_NE(r.find("(256 blocks/seq)"), std::string::npos) << r;
    ASSERT_LT(kv_seqs, 41) << "fixture no longer reproduces the over-subscribed shape";
    EXPECT_NE(r.find("WARN"), std::string::npos)
        << "state paid for 41 slots the KV pool cannot serve at full context, silently:\n" << r;
}

TEST(ShadowPlan, NoCeilingWarningWhenBothPoolsServeTheBatch) {
    // Same hybrid, a batch both pools serve. The warning has to be a statement
    // about THIS configuration, not about hybrids.
    auto p = hybrid_probe();
    p.max_batch_size = 3;
    p.ssm_state_bytes = 3ull * 77 * kMiB + 2 * kMiB;
    const auto res = plan_memory(shadow_plan_input(p));
    ASSERT_TRUE(res) << res.failure.report();
    const std::string r = shadow_plan_report(p, res, /*live_kv_blocks=*/768);
    EXPECT_NE(r.find("ceiling: recurrent 3 seqs, KV 3 seqs"), std::string::npos) << r;
    EXPECT_EQ(r.find("WARN"), std::string::npos)
        << "a plan whose pools agree must not cry wolf:\n" << r;

    // A dense pool holding fewer full-context sequences than max_batch_size is
    // ordinary continuous batching: no per-slot pre-charge, nothing wasted.
    auto d = dense_probe();
    d.max_batch_size = 64;
    const auto dres = plan_memory(shadow_plan_input(d));
    ASSERT_TRUE(dres) << dres.failure.report();
    const std::string dr = shadow_plan_report(d, dres, /*live_kv_blocks=*/2048);
    EXPECT_NE(dr.find("ceiling: recurrent 64 seqs"), std::string::npos) << dr;
    EXPECT_EQ(dr.find("WARN"), std::string::npos) << "no state was bought per slot:\n" << dr;
}

TEST(ShadowPlan, ARejectedPlanStillStatesTheCeilings) {
    // The rejected branch is exactly where the operator has to see which of the
    // two ceilings binds - the failure text names bytes, not sequences.
    auto p = hybrid_probe();
    p.distributable_bytes = 9683 * kMiB;  // 19 MiB short, the pre-clamp shape
    const auto res = plan_memory(shadow_plan_input(p));
    ASSERT_FALSE(res.ok);
    const std::string r = shadow_plan_report(p, res, /*live_kv_blocks=*/16);
    EXPECT_NE(r.find("plan REJECTS"), std::string::npos) << r;
    EXPECT_NE(r.find("ceiling: recurrent 41 seqs"), std::string::npos)
        << "a rejection without the ceilings leaves the operator guessing which pool to cut:\n"
        << r;
}

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

    // The report used to say "shadow plan … computed, NOT applied". It is
    // applied now — the KV block count it prints is the one the pool uses — so
    // the wording must not claim otherwise (AUDIT B69).
    EXPECT_NE(r.find("memory plan"), std::string::npos);
    EXPECT_NE(r.find("APPLIED"), std::string::npos);
    EXPECT_EQ(r.find("NOT applied"), std::string::npos)
        << "the report still describes itself as a shadow";
    EXPECT_NE(r.find("workspaces not modelled"), std::string::npos)
        << "a probe that hides its gaps is worse than no probe";
    // The note says the shadow plan does not model the tower — it no longer claims
    // the size is unknowable, because since F-12 the arena reserves it from a
    // pre-load probe. Match on what stayed true, not on the old reason.
    EXPECT_NE(r.find("vision tower not in the shadow plan"), std::string::npos);
    EXPECT_NE(r.find("forward scratch not modelled"), std::string::npos);
}

TEST(ShadowPlan, ReportShowsBothKvDecisions) {
    const auto p = dense_probe();
    const auto res = plan_memory(shadow_plan_input(p));
    ASSERT_TRUE(res);
    const std::string r = shadow_plan_report(p, res, /*live_kv_blocks=*/4096);
    EXPECT_NE(r.find("live pass 4096 blocks"), std::string::npos);
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
