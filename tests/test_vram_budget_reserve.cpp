// Q4_K sources aren't nvfp4_beneficial, so compute_vram_budget's heuristic weight-cache
// estimate is ~0 while the source-aware StoragePlanner routes them to the FP16 cache. Before
// the fix the KV backstop sized to available-minus-heuristic, ate all post-weight VRAM, built
// 0 cache tensors, and decode fell back to on-the-fly dequant (11 tok/s). Budget must reserve
// the planner projection (capped so one full max_seq_len sequence fits) before KV fills the rest.

#include <gtest/gtest.h>

#include "compute/nvfp4_sfatom_quant.h"
#include "compute/mmq_q8_imma.h"
#include "memory/plan.h"  // kMeasuredLibraryReserveBytes, plan_memory
#include "runtime/plan_shadow.h"
#include "memory/vram_query.h"
#include "model/model.h"
#include "model/model_config.h"
#include "exec/storage_planner.h"
#include "exec/moe_nvfp4_cache_rule.h"
#include "runtime/engine_config.h"
#include "runtime/vram_budget.h"

#include <algorithm>
#include <cuda_fp16.h>

#include "test_cuda_skip.h"
#include "vram_budget_fixture.h"

// Shape-tuned scenarios (tight budgets, floors that must fire) were sized against a 3900 MiB library
// charge, the cold-start constant until 2026-10-31; pinned so they test their path, not the constant.
constexpr int kFixtureLibraryMiB = 3900;

using namespace imp;
using namespace imp::vram_fixture;

TEST(VramBudgetReserve, PlannerDemandKeepsKvPoolFromEatingWeightCacheRoom) {
    SKIP_IF_NO_CUDA();  // compute_vram_budget queries total VRAM

    // Attention Q6_K (nvfp4-beneficial, small heuristic footprint), FFN Q4_K
    // (heuristic-invisible, planner routes FP16) — the Ornith-35B shape.
    Model m;
    fill_model(m, QType::Q6_K, QType::Q4_K);

    EngineConfig config;
    config.max_seq_len = 32768;
    config.max_batch_size = 8;
    config.use_nvfp4_decode = 2;   // mode 2: reserve floor is the flat 512 MiB
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;
    // Isolated from the library charge (#1109 floors reserve_bytes at the GEMM libraries' first
    // forward claim, its own test below): folding it in made this assert the sum of two unrelated
    // policies, which went red for three PRs (AUDIT B63).
    config.library_reserve_mb = 0;

    const int n_kv_layers = 32;
    const int head_dim = 128;
    const size_t GiB = 1024ull * 1024 * 1024;
    const size_t free_vram = 8 * GiB;

    VRAMBudget budget = compute_vram_budget(m, config, n_kv_layers, head_dim, free_vram);

    // per-block cost as compute_vram_budget derives it (F16, block size 16):
    const size_t per_block = 16ull * 8 * 128 * 2 /*K+V*/ * 2 /*bytes*/ * n_kv_layers;
    const size_t kv_bytes = static_cast<size_t>(budget.kv_max_blocks) * per_block;
    const int blocks_per_seq = 32768 / 16;

    // The FFN FP16 demand here is ~8.6 GiB (> available), so the reservation
    // is capped at available minus one full sequence — the KV pool must come
    // out at roughly one max_seq_len sequence, not fill all of free_vram.
    EXPECT_GE(budget.kv_max_blocks, blocks_per_seq)
        << "one full max_seq_len sequence must always fit in the KV pool";
    EXPECT_LE(kv_bytes, free_vram - 3 * GiB)
        << "KV pool ate the weight-cache room again (heuristic-only backstop)";
}

// #963 follow-up: when KV is cheap (hybrid, few attention layers), the auto floor must cover
// the full advertised context plus StreamingLLM headroom, not stop at the flat 16384-token
// cap. The old floor let a max_seq_len=17408 hybrid's 16k prompt fill the pool to 94%,
// tripping the >90% streaming valve (graphs off, windowed attention) on a request that fits.
TEST(VramBudgetReserve, CheapKvFloorCoversFullMaxSeqLen) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::Q6_K, QType::Q6_K);
    m.config_.n_kv_heads = 2;  // hybrid-class cheap KV (Qwen3.6-35B: 2 heads)

    EngineConfig config;
    config.max_seq_len = 17408;
    config.max_batch_size = 1;
    config.use_nvfp4_decode = 2;
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;

    const int n_kv_layers = 10;  // 10 attention layers out of 40 (GDN hybrid)
    const int head_dim = 256;
    const size_t GiB = 1024ull * 1024 * 1024;

    VRAMBudget budget = compute_vram_budget(m, config, n_kv_layers, head_dim, 8 * GiB);

    // Full coverage + 12.5% streaming headroom ≈ 383 MiB here (well under the
    // 1 GiB cheap-KV bound) — the pool must hold max_seq_len plus headroom so
    // a full-length request never trips the >90% streaming valve.
    const int bs = 16;  // kKVBlockSize (config.kv_block_size unset)
    const int full_cov_tok = config.max_seq_len + config.max_seq_len / 8;
    EXPECT_GE(budget.kv_max_blocks * bs, full_cov_tok)
        << "cheap-KV floor stopped below the advertised context again";
}

TEST(VramBudgetReserve, BeneficialSourcesKeepFullKvPool) {
    SKIP_IF_NO_CUDA();

    // All-Q6_K model: heuristic covers the demand, planner projection stays
    // in the same ballpark — the reservation must NOT fire and shrink KV.
    Model m;
    fill_model(m, QType::Q6_K, QType::Q6_K);

    EngineConfig config;
    config.max_seq_len = 32768;
    config.max_batch_size = 8;
    config.use_nvfp4_decode = 2;
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;
    // Isolated from the library charge (#1109 floors reserve_bytes at the GEMM libraries' first
    // forward claim, its own test below): folding it in made this assert the sum of two unrelated
    // policies, which went red for three PRs (AUDIT B63).
    config.library_reserve_mb = 0;

    const size_t GiB = 1024ull * 1024 * 1024;
    VRAMBudget with_planner = compute_vram_budget(m, config, 32, 128, 8 * GiB);

    // Q6_K routes to NVFP4 tier (heuristic ~= planner), so KV keeps most of post-reserve VRAM.
    // #1103: used to assert an absolute kv_bytes >= 4 GiB, silently encoding the old 512 MiB
    // mode-2 safety reserve, which planned KV below what VRAMAllocator's can_allocate() permits
    // (free >= bytes + 5% of total for >=16 MiB), so caches failed mid-build. With the reserve
    // corrected, express the property relative to distributable VRAM instead of an absolute number.
    const size_t per_block = 16ull * 8 * 128 * 2 * 2 * 32;
    const size_t kv_bytes = static_cast<size_t>(with_planner.kv_max_blocks) * per_block;
    const size_t distributable = 8 * GiB - with_planner.reserve_bytes;
    EXPECT_GE(kv_bytes, distributable / 2) << "reservation must not fire for heuristic-covered sources "
                                           << "(kv=" << kv_bytes / (1024 * 1024) << " MiB of "
                                           << distributable / (1024 * 1024) << " MiB distributable)";
}

// F16 weights keep both the heuristic and planner projection at zero, isolating the
// kv_fraction/reserve_floor_pct knobs' effect; max_seq_len is small enough that neither the
// min-KV floor nor the target_blocks clamp rewrites the fraction result.

namespace {

EngineConfig knob_config() {
    EngineConfig config;
    config.max_seq_len = 2048;
    config.max_batch_size = 8;
    config.use_nvfp4_decode = 0;  // FP16_ONLY strategy, reserve floor applies
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;
    return config;
}

}  // namespace

TEST(VramBudgetReserve, VramKnobDefaultsArePinned) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::F16, QType::F16);

    const size_t GiB = 1024ull * 1024 * 1024;
    EngineConfig def = knob_config();
    EngineConfig expl = knob_config();
    expl.kv_fraction = 1.0f;
    expl.vram_reserve_floor_pct = 0;

    VRAMBudget a = compute_vram_budget(m, def, 32, 128, 8 * GiB);
    VRAMBudget b = compute_vram_budget(m, expl, 32, 128, 8 * GiB);
    EXPECT_EQ(a.reserve_bytes, b.reserve_bytes);
    EXPECT_EQ(a.kv_cache_bytes, b.kv_cache_bytes);
    EXPECT_EQ(a.kv_max_blocks, b.kv_max_blocks);
}

TEST(VramBudgetReserve, KvFractionScalesKvPool) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::F16, QType::F16);

    const size_t GiB = 1024ull * 1024 * 1024;
    EngineConfig cfg_08 = knob_config();
    EngineConfig cfg_04 = knob_config();
    cfg_08.kv_fraction = 0.8f;
    cfg_04.kv_fraction = 0.4f;

    VRAMBudget b08 = compute_vram_budget(m, cfg_08, 32, 128, 8 * GiB);
    VRAMBudget b04 = compute_vram_budget(m, cfg_04, 32, 128, 8 * GiB);
    // Same `available` in both runs: halving kv_fraction halves the bytes target
    // (kv_max_blocks can converge downstream via the fraction-independent physical-fit backstop
    // / min-KV floor, but the bytes target is the knob's own contract).
    EXPECT_EQ(b04.kv_cache_bytes * 2, b08.kv_cache_bytes);
    EXPECT_LE(b04.kv_max_blocks, b08.kv_max_blocks);
}

TEST(VramBudgetReserve, ReserveFloorPctScalesReserve) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::F16, QType::F16);

    const size_t GiB = 1024ull * 1024 * 1024;
    EngineConfig cfg_00 = knob_config();
    EngineConfig cfg_10 = knob_config();
    EngineConfig cfg_20 = knob_config();
    cfg_10.vram_reserve_floor_pct = 10;
    cfg_20.vram_reserve_floor_pct = 20;
    cfg_00.library_reserve_mb = 0;
    // reserve_bytes = max(pct floor, library charge); the library charge (its own test,
    // ReserveIsFlooredAtTheLibraryCharge) wins on any real card, which is exactly what silently
    // broke this test in #1109 (AUDIT B63).
    cfg_10.library_reserve_mb = 0;
    cfg_20.library_reserve_mb = 0;

    size_t total_vram = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(nullptr, &total_vram));
    ASSERT_GT(total_vram, 0u);

    VRAMBudget b00 = compute_vram_budget(m, cfg_00, 32, 128, 8 * GiB);
    VRAMBudget b10 = compute_vram_budget(m, cfg_10, 32, 128, 8 * GiB);
    VRAMBudget b20 = compute_vram_budget(m, cfg_20, 32, 128, 8 * GiB);
    // Default (B-29f): no policy floor, the reserve is the measured allocator headroom.
    EXPECT_EQ(b00.reserve_bytes, std::max<size_t>(512ull * 1024 * 1024, vram_allocator_headroom(total_vram)));
    EXPECT_GE(b10.reserve_bytes, b00.reserve_bytes);
    // Feature reserve here is 512 MiB (graphs off) — the floor dominates on
    // any real card, so the reserve must equal vram_reserve_floor(total, pct).
    EXPECT_EQ(b10.reserve_bytes,
              std::max<size_t>(512ull * 1024 * 1024, vram_reserve_floor(total_vram, 10)));
    EXPECT_EQ(b20.reserve_bytes,
              std::max<size_t>(512ull * 1024 * 1024, vram_reserve_floor(total_vram, 20)));
    EXPECT_GE(b20.reserve_bytes, b10.reserve_bytes);
}

// #1765: transient init headroom folded into the weight-cache estimate (phase3 reserve,
// max(total/10,floor)+margin) must not be charged against KV in the shadow plan; fed whole,
// the plan granted optional caches everything above the one-sequence floor and collapsed KV
// to 128 blocks with GiBs free. Engine now feeds estimate-transient; old feeding kept as the
// contrast arm so the assertion can't pass vacuously.
TEST(VramBudgetReserve, TransientReserveDoesNotStarveTheKvPlan) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_prequant_model(m);

    EngineConfig config;
    config.library_reserve_mb = kFixtureLibraryMiB;
    config.max_seq_len = 4096;
    config.max_batch_size = 32;
    config.use_nvfp4_decode = 2;
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;

    // Sized so BOTH demands fit, but residual-minus-floor is SMALLER than
    // the transient reserve: the pre-fix feeding must visibly starve KV
    // while the fixed feeding still grants the full want.
    const size_t free_vram = 8704ull * 1024 * 1024;
    VRAMBudget live = compute_vram_budget(m, config, /*n_kv_layers=*/8, /*head_dim=*/128, free_vram);
    ASSERT_GT(live.weight_cache_transient_bytes, 0u)
        << "native-NVFP4 branch did not fold a transient reserve — the fixture no longer "
           "exercises the #1765 shape";
    ASSERT_LT(live.weight_cache_transient_bytes, live.weight_cache_estimate_bytes);

    ShadowPlanProbe probe;
    probe.distributable_bytes = free_vram;
    probe.mandatory_cache_bytes = live.mandatory_sf_bytes + live.mandatory_moe_bytes;
    probe.ssm_state_bytes = live.ssm_footprint_bytes;
    probe.library_reserve_bytes = size_t{kFixtureLibraryMiB} << 20;
    probe.n_kv_layers = 8;
    probe.max_batch_size = 32;
    probe.max_seq_len = 4096;
    probe.kv_block_size = 16;
    probe.min_kv_tokens = config.min_kv_tokens;
    probe.kv_block_bytes_per_layer =
        kv_block_bytes_per_layer(config.kv_cache_dtype, 16, m.config().n_kv_heads, 128);

    const int want_blocks = ((4096 + 15) / 16) * 32;

    // The engine's feeding since the fix: steady-state demand only.
    probe.weight_cache_demand = live.weight_cache_estimate_bytes - live.weight_cache_transient_bytes;
    const PlanResult fixed = plan_memory(shadow_plan_input(probe));
    ASSERT_TRUE(fixed.ok);
    EXPECT_EQ(fixed.plan.kv.blocks, want_blocks)
        << "KV plan collapsed toward the one-sequence floor with " << fixed.plan.kv.blocks
        << " blocks against a want of " << want_blocks;

    // Contrast arm: the pre-fix feeding reproduces the collapse.
    probe.weight_cache_demand = live.weight_cache_estimate_bytes;
    const PlanResult old_way = plan_memory(shadow_plan_input(probe));
    if (old_way.ok) {
        EXPECT_LT(old_way.plan.kv.blocks, want_blocks)
            << "the transient charge no longer starves the plan — if this is deliberate, "
               "the estimate/transient split may be removable";
    }
}

TEST(VramBudgetReserve, PreallocCoversDemandAndFreesKv) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_prequant_model(m);

    EngineConfig config;
    config.library_reserve_mb = kFixtureLibraryMiB;
    config.max_seq_len = 32768;
    config.max_batch_size = 8;
    config.use_nvfp4_decode = 2;
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;

    const size_t GiB = 1024ull * 1024 * 1024;
    NativeCacheDemand d = compute_native_cache_demand(m);
    ASSERT_GT(d.total(), 0u);

    // The floors are stated by the PLAN now, not conditional on bytes some
    // caller physically pre-held (AUDIT B62 — the balloon is gone). They are
    // the measured demand, unconditionally, on every call.
    VRAMBudget b = compute_vram_budget(m, config, 2, 128, 4 * GiB);
    EXPECT_EQ(b.mandatory_sf_bytes, d.sf_bytes);
    EXPECT_EQ(b.mandatory_moe_bytes, d.moe_slab_bytes);

    // And they do not depend on how much VRAM the call is told about: a floor
    // that shrank with free VRAM would be exactly the live-free-derived number
    // it exists to override.
    VRAMBudget tight = compute_vram_budget(m, config, 2, 128, 2 * GiB);
    EXPECT_EQ(tight.mandatory_sf_bytes, d.sf_bytes);
    EXPECT_EQ(tight.mandatory_moe_bytes, d.moe_slab_bytes);

    // KV is charged the full demand — nothing is hidden from free_vram, so a
    // roomier card still gets the larger pool.
    EXPECT_GE(b.kv_max_blocks, tight.kv_max_blocks);
}

// The floor that #1109 added, and the test it should have come with. Its
// absence is why three tests in this file encoded the pre-#1109 arithmetic and
// stayed red for three PRs: CI has no GPU runner, so nothing ran them.
TEST(VramBudgetReserve, ReserveIsFlooredAtTheLibraryCharge) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::F16, QType::F16);
    const size_t GiB = 1024ull * 1024 * 1024;

    // Deliberately VRAM-bound: knob_config()'s 2048-token context makes the pool demand-bound
    // (512 blocks either way), which would let a reserve assertion pass for the wrong reason. At
    // 32k the pool is sized by what's left, which is what the charge is supposed to move.
    EngineConfig uncharged;
    uncharged.max_seq_len = 32768;
    uncharged.max_batch_size = 8;
    uncharged.use_nvfp4_decode = 0;
    uncharged.use_cuda_graphs = false;
    uncharged.kv_cache_dtype = QType::F16;
    uncharged.library_reserve_mb = 0;
    EngineConfig charged = uncharged;
    charged.library_reserve_mb = 6000;

    VRAMBudget b_none = compute_vram_budget(m, uncharged, 32, 128, 8 * GiB);
    VRAMBudget b_chg = compute_vram_budget(m, charged, 32, 128, 8 * GiB);

    // The charge is a floor, not an addend: it replaces the reserve when larger.
    EXPECT_EQ(b_chg.reserve_bytes, 6000ull << 20);
    EXPECT_GT(b_chg.reserve_bytes, b_none.reserve_bytes);

    // And it has to come out of the elastic tier, or the plan would hand out
    // bytes the first forward is about to take (the #1109 failure: budget
    // 16000 -> peak 22584 MiB, i.e. --vram-budget changed nothing).
    EXPECT_LT(b_chg.kv_max_blocks, b_none.kv_max_blocks)
        << "the library charge must reduce the KV pool, not be added on top of it";

    // The default (-1) charges the measured constant rather than nothing.
    EngineConfig dflt = uncharged;
    dflt.library_reserve_mb = -1;
    VRAMBudget b_dflt = compute_vram_budget(m, dflt, 32, 128, 8 * GiB);
    EXPECT_GE(b_dflt.reserve_bytes, kMeasuredLibraryReserveBytes)
        << "an unset library_reserve_mb must still charge the measured constant";

    // Exposed on its own: phase 3-moe charges it against live free VRAM (max'd into
    // reserve_bytes it is lost whenever the allocator headroom is larger).
    EXPECT_EQ(b_none.library_reserve_bytes, 0u);
    EXPECT_EQ(b_chg.library_reserve_bytes, 6000ull << 20);
    EXPECT_EQ(b_dflt.library_reserve_bytes, kMeasuredLibraryReserveBytes);
}

// V8 plan sufficiency ("no tier exceeded"): replaces a wrong live<=plan assertion that
// passed on five VRAM-generous shapes but failed on a bench-sized shape (live hands out 224
// KV blocks where the plan affords 56 - different questions; since #1135 the pool takes the
// PLAN's number, so live being more generous costs nothing). What must hold: a plan reporting
// ok must fit inside its own budget, or everything downstream of it is furniture.
TEST(VramBudgetReserve, PlanNeverExceedsTheBudgetItWasGiven) {
    SKIP_IF_NO_CUDA();

    struct Case {
        const char* name;
        QType attn;
        QType ffn;
        int max_seq_len;
        int max_batch;
        int n_kv_layers;
        int head_dim;
        size_t free_vram_gib;
        int use_nvfp4_decode;
    };
    const Case cases[] = {
        {"dense Q8 roomy", QType::Q8_0, QType::Q8_0, 8192, 4, 32, 128, 20, 0},
        {"dense Q8 tight", QType::Q8_0, QType::Q8_0, 32768, 8, 32, 128, 6, 0},
        {"Q6_K mode-2", QType::Q6_K, QType::Q6_K, 32768, 1, 36, 128, 16, 2},
        {"Q4_K wide batch", QType::Q4_K, QType::Q4_K, 4096, 16, 40, 128, 12, 0},
        {"cheap KV hybrid", QType::Q6_K, QType::Q6_K, 17408, 2, 10, 256, 8, 2},
        // The bench regime: small advertised context, batch 1. The shape the
        // five above could not see.
        {"bench-sized ctx", QType::Q8_0, QType::Q8_0, 896, 1, 36, 128, 20, 2},
        // Cannot fit by construction: the library reserve alone exceeds this
        // budget. Present so the sufficiency assertion above has a case that can
        // actually fail — see the comment at the end of this test.
        {"impossible", QType::Q8_0, QType::Q8_0, 32768, 8, 32, 128, 2, 0},
    };

    int checked = 0;
    int refused = 0;
    for (const Case& c : cases) {
        Model m;
        fill_model(m, c.attn, c.ffn);

        EngineConfig config;
        config.library_reserve_mb = kFixtureLibraryMiB;
        config.max_seq_len = c.max_seq_len;
        config.max_batch_size = c.max_batch;
        config.use_nvfp4_decode = c.use_nvfp4_decode;
        config.use_cuda_graphs = false;
        config.kv_cache_dtype = QType::F16;

        const size_t free_vram = c.free_vram_gib * 1024ull * 1024 * 1024;
        VRAMBudget live = compute_vram_budget(m, config, c.n_kv_layers, c.head_dim, free_vram);

        ShadowPlanProbe probe;
        probe.distributable_bytes = free_vram;
        probe.weight_cache_demand = live.weight_cache_estimate_bytes;
        probe.mandatory_cache_bytes = live.mandatory_sf_bytes + live.mandatory_moe_bytes;
        probe.ssm_state_bytes = live.ssm_footprint_bytes;
        probe.library_reserve_bytes = size_t{kFixtureLibraryMiB} << 20;
        probe.n_kv_layers = c.n_kv_layers;
        probe.max_batch_size = c.max_batch;
        probe.max_seq_len = c.max_seq_len;
        probe.kv_block_size = 16;
        probe.min_kv_tokens = config.min_kv_tokens;
        probe.kv_block_bytes_per_layer =
            kv_block_bytes_per_layer(config.kv_cache_dtype, 16, m.config().n_kv_heads, c.head_dim);

        const PlanResult plan = plan_memory(shadow_plan_input(probe));
        if (plan.ok) {
            ++checked;
            EXPECT_LE(plan.plan.total(), free_vram)
                << c.name << ": the plan reports ok while asking for "
                << (plan.plan.total() >> 20) << " MiB of a " << (free_vram >> 20) << " MiB budget";
        } else {
            ++refused;
            // A refusal must have a reason attached, or the operator gets "no"
            // with nothing to act on.
            EXPECT_GT(plan.failure.over_by + (plan.plan.kv.below_floor ? 1u : 0u), 0u)
                << c.name << ": refused without either an overrun or a floor violation";
        }
    }

    EXPECT_GE(checked, 3) << "only " << checked
                          << " of 7 cases produced a plan — the rest were refused, so the "
                             "sufficiency half of this test is not measuring what it claims";
    // Without a case that genuinely cannot fit, the fit assertion is a tautology (every plan
    // already passed the check it asserts). Mutation-verified: removing the fit check inside
    // plan_memory() left this green until the impossible case was added.
    EXPECT_GE(refused, 1) << "no case exercised the refusal path";
}

// Live pass sizes KV from kv_fraction of free VRAM and clamps; the plan grants
// blocks_per_seq*batch out of a computed residual. At a small advertised context they differ
// by 4x, neither wrong - answers to different questions. The pool takes the plan's (B69).
TEST(VramBudgetReserve, LivePassMayExceedThePlanAtSmallContexts) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::Q8_0, QType::Q8_0);

    EngineConfig config;
    config.max_seq_len = 896;  // the --bench shape
    config.max_batch_size = 1;
    config.use_nvfp4_decode = 2;
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;

    const size_t free_vram = 20ull * 1024 * 1024 * 1024;
    VRAMBudget live = compute_vram_budget(m, config, 36, 128, free_vram);

    ShadowPlanProbe probe;
    probe.distributable_bytes = free_vram;
    probe.weight_cache_demand = live.weight_cache_estimate_bytes;
    probe.mandatory_cache_bytes = live.mandatory_sf_bytes + live.mandatory_moe_bytes;
    probe.ssm_state_bytes = live.ssm_footprint_bytes;
    probe.library_reserve_bytes = kMeasuredLibraryReserveBytes;
    probe.n_kv_layers = 36;
    probe.max_batch_size = 1;
    probe.max_seq_len = 896;
    probe.kv_block_size = 16;
    probe.min_kv_tokens = config.min_kv_tokens;
    probe.kv_block_bytes_per_layer =
        kv_block_bytes_per_layer(config.kv_cache_dtype, 16, m.config().n_kv_heads, 128);

    const PlanResult plan = plan_memory(shadow_plan_input(probe));
    ASSERT_TRUE(plan.ok);

    // The plan grants exactly the advertised need; the live pass grants more.
    EXPECT_EQ(plan.plan.kv.blocks, (896 + 15) / 16)
        << "the plan should grant blocks_per_seq x batch and nothing for reuse";
    EXPECT_GT(live.kv_max_blocks, plan.plan.kv.blocks)
        << "if this ever stops holding, the two passes have converged at small "
           "contexts and the note in B69 needs revisiting";
}

TEST(VramBudgetReserve, VramKnobsAreClamped) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::F16, QType::F16);

    const size_t GiB = 1024ull * 1024 * 1024;
    EngineConfig silly = knob_config();
    silly.kv_fraction = 5.0f;           // clamped to 1.0
    silly.vram_reserve_floor_pct = -5;  // clamped to 0 → the allocator headroom

    EngineConfig max_sane = knob_config();
    max_sane.kv_fraction = 1.0f;
    max_sane.vram_reserve_floor_pct = 0;

    VRAMBudget a = compute_vram_budget(m, silly, 32, 128, 8 * GiB);
    VRAMBudget b = compute_vram_budget(m, max_sane, 32, 128, 8 * GiB);
    EXPECT_EQ(a.reserve_bytes, b.reserve_bytes);
    EXPECT_EQ(a.kv_cache_bytes, b.kv_cache_bytes);
    EXPECT_EQ(a.kv_max_blocks, b.kv_max_blocks);
}

// #1103: budget and VRAMAllocator must agree on headroom. Mode 2 skips the 10% reserve-floor
// POLICY to fit larger caches, but the allocator's 5% is not policy - can_allocate() refuses
// any >=16 MiB allocation leaving less free than that. A plan below it can't execute: KV was
// sized against 512 MiB assumed headroom while caches needed 1630 MiB and failed mid-build.
TEST(VramBudgetReserve, ReserveNeverUndercutsTheAllocatorHeadroom) {
    SKIP_IF_NO_CUDA();

    size_t total_vram = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(nullptr, &total_vram));
    ASSERT_GT(total_vram, 0u);
    const size_t hard_floor = vram_allocator_headroom(total_vram);

    Model m;
    fill_model(m, QType::Q6_K, QType::Q6_K);

    const size_t GiB = 1024ull * 1024 * 1024;
    for (int mode : {0, 1, 2}) {
        EngineConfig config;
        config.max_seq_len = 32768;
        config.max_batch_size = 8;
        config.use_nvfp4_decode = mode;  // mode 2 is the one that skipped the floor
        config.use_cuda_graphs = false;
        config.kv_cache_dtype = QType::F16;

        VRAMBudget b = compute_vram_budget(m, config, 32, 128, 8 * GiB);
        EXPECT_GE(b.reserve_bytes, hard_floor)
            << "use_nvfp4_decode=" << mode << ": plan reserves less than the allocator will ever leave free";
    }
}

// #1747: the divergence log printed kv_max_blocks under "live pass would have said", but
// kv_max_blocks may already be raised by the min_kv_tokens rescue floor, so a floored start
// reported a config lower bound as a live-pass reading. Pre-floor figure is kept so the log
// can name both; mutation "assign after the raise" breaks this.
TEST(VramBudgetReserve, PreFloorBlocksSurviveTheMinKvTokensRaise) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::Q8_0, QType::Q8_0);

    EngineConfig config;
    config.library_reserve_mb = kFixtureLibraryMiB;
    config.max_seq_len = 32768;
    config.max_batch_size = 1;
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;
    config.min_kv_tokens = 16384;  // 1024 blocks at block size 16

    // 8 GiB is chosen, not arbitrary: sizing lands at 618 blocks there and the floor raises it
    // to 1024; at 9 GiB+ the floor never fires and the test would pass proving nothing. Assertion
    // is that the floor fired at all.
    const size_t tight = 8ull * 1024 * 1024 * 1024;
    VRAMBudget b = compute_vram_budget(m, config, 36, 128, tight);

    ASSERT_LT(b.kv_blocks_pre_floor, b.kv_max_blocks)
        << "the floor did not fire at this shape, so nothing here is being tested: pre="
        << b.kv_blocks_pre_floor << " final=" << b.kv_max_blocks;
    EXPECT_EQ(b.kv_max_blocks, config.min_kv_tokens / 16)
        << "the raise should land exactly on the configured floor";
    EXPECT_GT(b.kv_blocks_pre_floor, 0) << "pre-floor figure was never recorded";

    // The point of the field. Assigning it after the raise instead of before
    // makes these two equal, which is the state the divergence log used to
    // report as "live pass would have said" (#1747).
    EXPECT_NE(b.kv_blocks_pre_floor, b.kv_max_blocks);
}

// #1899: mmq_q8_imma keeps a lazy s8+(alpha,beta) copy of every Q8_0 weight on its first
// prefill; nothing charged it, so it grew into whatever the KV pool left free (5612 MiB cold
// on Qwen3-8B-Q8_0, 7942 MiB with library_reserve_mb=12000, same model/tree). Budget now
// charges the exact allocator arithmetic and caps the allocator at the charge.
TEST(VramBudgetReserve, ImmaPlanesAreChargedForQ8Weights) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::Q8_0, QType::Q8_0);

    EngineConfig config;
    config.max_seq_len = 8192;
    config.max_batch_size = 32;  // VRAM binds the pool, not blocks_per_seq * batch
    config.use_cuda_graphs = false;
    config.use_nvfp4_decode = 1;  // the overlay branch that routes Q8_0 to IMMA
    config.kv_cache_dtype = QType::F16;
    config.library_reserve_mb = 0;

    // What the allocator will take, spelled out rather than delegated: this is
    // the assertion that keeps planner and allocator from drifting apart.
    size_t expect = 0;
    for (int i = 0; i < m.config().n_layers; ++i) {
        const auto& L = m.layer(i);
        for (const Tensor* w : {&L.wq, &L.w_gate, &L.w_up, &L.w_down}) {
            const size_t elems = static_cast<size_t>(w->shape[0]) * w->shape[1];
            expect += elems + elems / 32 * 2 * sizeof(__half);
            EXPECT_EQ(imma_q8_plane_bytes(w->shape[0], w->shape[1]), elems + elems / 32 * 2 * sizeof(__half));
        }
    }

    const size_t GiB = 1024ull * 1024 * 1024;
    VRAMBudget on = compute_vram_budget(m, config, 32, 128, 24 * GiB, 0, 0, nullptr, 0,
                                        /*q8_imma_prefill=*/true, /*moe_imma_prefill=*/true);
    EXPECT_EQ(on.imma_plane_bytes, expect) << "the plan must charge exactly what mmq_q8_imma allocates";

    // q8_imma_prefill=false: no planes are built, so charging them would
    // hand the KV pool ~5 GiB less for nothing.
    VRAMBudget off = compute_vram_budget(m, config, 32, 128, 24 * GiB, 0, 0, nullptr, 0,
                                         /*q8_imma_prefill=*/false, /*moe_imma_prefill=*/false);
    EXPECT_EQ(off.imma_plane_bytes, 0u);
    EXPECT_GT(off.kv_max_blocks, on.kv_max_blocks)
        << "the charge must come out of the KV pool — that is the whole point";
}

TEST(VramBudgetReserve, ImmaPlaneChargeYieldsToTheKvGuarantee) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::Q8_0, QType::Q8_0);

    EngineConfig config;
    config.max_seq_len = 4096;
    config.max_batch_size = 1;
    config.use_cuda_graphs = false;
    config.use_nvfp4_decode = 1;
    config.kv_cache_dtype = QType::F16;
    config.library_reserve_mb = 0;
    config.min_kv_tokens = 8192;  // 512 blocks at block size 16

    // Tight enough that the full plane demand cannot coexist with the floor:
    // uncapped, the plan would take the pool down to it.
    const size_t GiB = 1024ull * 1024 * 1024;
    VRAMBudget b = compute_vram_budget(m, config, 32, 128, 7 * GiB, 0, 0, nullptr, 0,
                                       /*q8_imma_prefill=*/true, /*moe_imma_prefill=*/true);

    size_t demand = 0;
    for (int i = 0; i < m.config().n_layers; ++i) {
        const auto& L = m.layer(i);
        for (const Tensor* w : {&L.wq, &L.w_gate, &L.w_up, &L.w_down})
            demand += imma_q8_plane_bytes(w->shape[0], w->shape[1]);
    }
    ASSERT_GT(demand, 7 * GiB / 2) << "shape no longer tight enough to test the cap";
    EXPECT_LT(b.imma_plane_bytes, demand) << "the cap did not fire";
    EXPECT_GE(b.kv_max_blocks, config.min_kv_tokens / 16) << "the planes were allowed to eat the KV floor";
}
