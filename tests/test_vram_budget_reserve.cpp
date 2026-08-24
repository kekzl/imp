// KV-pool sizing vs weight-cache demand (Ornith-35B Q4_K_M regression).
//
// Q4_K sources are not nvfp4_beneficial, so the heuristic weight-cache
// estimate in compute_vram_budget is ~0 while the StoragePlanner (source-
// aware) routes them to the FP16 cache. Before the fix the KV backstop
// sized the pool to available-minus-heuristic — i.e. it ate all post-weight
// VRAM, phases 1/3 built 0 cache tensors, and decode fell back to
// on-the-fly dequant (11 tok/s instead of a cached decode path).
//
// The budget must reserve the planner projection (capped so one full
// max_seq_len sequence always fits) before the KV pool fills the rest.

#include <gtest/gtest.h>

#include "compute/gemm_cutlass_sm120.h"
#include "memory/plan.h"  // kMeasuredLibraryReserveBytes, plan_memory
#include "runtime/plan_shadow.h"
#include "memory/vram_query.h"
#include "model/model.h"
#include "model/model_config.h"
#include "runtime/engine.h"
#include "runtime/storage_planner.h"
#include "runtime/vram_budget.h"

#include <algorithm>
#include <cuda_fp16.h>

#include "test_cuda_skip.h"

using namespace imp;

namespace {

Tensor make_weight(TensorKind kind, int64_t rows, int64_t cols, QType qt, uintptr_t sentinel) {
    Tensor t;
    t.data = reinterpret_cast<void*>(sentinel);
    t.qtype = qt;
    t.ndim = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.on_device = false;
    t.kind = kind;
    return t;
}

// 32 layers; attention weights of `attn_qt`, FFN weights of `ffn_qt`
// (Model is non-copyable — populate the caller's instance).
void fill_model(Model& m, QType attn_qt, QType ffn_qt) {
    m.config_.n_layers = 32;
    m.config_.n_kv_heads = 8;
    for (int i = 0; i < 32; ++i) {
        TransformerLayer L;
        uintptr_t base = static_cast<uintptr_t>(i) * 100 + 1;
        L.wq = make_weight(TensorKind::WQ, 4096, 4096, attn_qt, base + 0);
        L.w_gate = make_weight(TensorKind::W_GATE, 11008, 4096, ffn_qt, base + 1);
        L.w_up = make_weight(TensorKind::W_UP, 11008, 4096, ffn_qt, base + 2);
        L.w_down = make_weight(TensorKind::W_DOWN, 4096, 11008, ffn_qt, base + 3);
        m.layers_.push_back(std::move(L));
    }
}

}  // namespace

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
    // Isolate this test from the library charge (#1109 floors reserve_bytes at
    // what cuBLAS/CUTLASS claim on the first forward). That charge has its own
    // test below; folding it in here made this one assert the sum of two
    // unrelated policies, which is how it went red for three PRs (AUDIT B63).
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

// #963 follow-up: when KV is cheap (hybrid — few attention layers), the
// auto floor must cover the full advertised context plus the StreamingLLM
// headroom, not stop at the flat 16384-token cap. With the old floor a
// max_seq_len=17408 hybrid got a pool a 16k prompt fills to 94%, tripping
// the >90% streaming valve (graphs off, windowed attention) on a request
// that fits outright.
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
    // Isolate this test from the library charge (#1109 floors reserve_bytes at
    // what cuBLAS/CUTLASS claim on the first forward). That charge has its own
    // test below; folding it in here made this one assert the sum of two
    // unrelated policies, which is how it went red for three PRs (AUDIT B63).
    config.library_reserve_mb = 0;

    const size_t GiB = 1024ull * 1024 * 1024;
    VRAMBudget with_planner = compute_vram_budget(m, config, 32, 128, 8 * GiB);

    // Q6_K routes to the NVFP4 tier: heuristic ≈ planner, so KV keeps most of
    // the post-reserve VRAM (well above the one-sequence floor).
    //
    // Threshold restated 2026-07-28 (#1103). This used to assert an absolute
    // `kv_bytes >= 4 GiB`, which silently encoded the old mode-2 safety reserve
    // of 512 MiB. That reserve was wrong: it planned KV down to a level the
    // VRAMAllocator refuses to allocate against (it requires free >= bytes +
    // 5% of total for anything >=16 MiB), so the plan could not be executed and
    // the caches it starved failed mid-build. With the reserve corrected the
    // absolute number is unreachable in this synthetic 8 GiB scenario. The
    // property under test is unchanged — the planner-driven weight-cache
    // reservation must not fire — so express it relative to what is actually
    // distributable, which is what "KV keeps most of it" always meant.
    const size_t per_block = 16ull * 8 * 128 * 2 * 2 * 32;
    const size_t kv_bytes = static_cast<size_t>(with_planner.kv_max_blocks) * per_block;
    const size_t distributable = 8 * GiB - with_planner.reserve_bytes;
    EXPECT_GE(kv_bytes, distributable / 2) << "reservation must not fire for heuristic-covered sources "
                                           << "(kv=" << kv_bytes / (1024 * 1024) << " MiB of "
                                           << distributable / (1024 * 1024) << " MiB distributable)";
}

// --- imp.conf [vram] knobs (kv_fraction / reserve_floor_pct) ---
//
// F16 weights keep both the heuristic and the planner projection at zero, so
// the KV math is undisturbed by weight-cache reservations — the knobs' effect
// is exactly observable. max_seq_len is small enough that neither the
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
    expl.kv_fraction = 0.8f;
    expl.vram_reserve_floor_pct = 10;

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
    cfg_04.kv_fraction = 0.4f;

    VRAMBudget b08 = compute_vram_budget(m, cfg_08, 32, 128, 8 * GiB);
    VRAMBudget b04 = compute_vram_budget(m, cfg_04, 32, 128, 8 * GiB);
    // Same `available` in both runs — halving the fraction halves the pool
    // target. (kv_max_blocks can converge to the same value downstream via
    // the physical-fit backstop / min-KV floor, which are fraction-
    // independent — the bytes target is the knob's contract.)
    EXPECT_EQ(b04.kv_cache_bytes * 2, b08.kv_cache_bytes);
    EXPECT_LE(b04.kv_max_blocks, b08.kv_max_blocks);
}

TEST(VramBudgetReserve, ReserveFloorPctScalesReserve) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::F16, QType::F16);

    const size_t GiB = 1024ull * 1024 * 1024;
    EngineConfig cfg_10 = knob_config();
    EngineConfig cfg_20 = knob_config();
    cfg_20.vram_reserve_floor_pct = 20;
    // The floor percentage is what this test is about; the library charge is a
    // separate floor with its own test (ReserveIsFlooredAtTheLibraryCharge).
    // With both live, reserve_bytes is max(pct floor, library charge) and the
    // library charge wins on any real card — which is exactly what silently
    // broke this test in #1109 (AUDIT B63).
    cfg_10.library_reserve_mb = 0;
    cfg_20.library_reserve_mb = 0;

    size_t total_vram = 0;
    vram_budget_mem_get_info(nullptr, &total_vram);
    ASSERT_GT(total_vram, 0u);

    VRAMBudget b10 = compute_vram_budget(m, cfg_10, 32, 128, 8 * GiB);
    VRAMBudget b20 = compute_vram_budget(m, cfg_20, 32, 128, 8 * GiB);
    // Feature reserve here is 512 MiB (graphs off) — the floor dominates on
    // any real card, so the reserve must equal vram_reserve_floor(total, pct).
    EXPECT_EQ(b10.reserve_bytes,
              std::max<size_t>(512ull * 1024 * 1024, vram_reserve_floor(total_vram, 10)));
    EXPECT_EQ(b20.reserve_bytes,
              std::max<size_t>(512ull * 1024 * 1024, vram_reserve_floor(total_vram, 20)));
    EXPECT_GE(b20.reserve_bytes, b10.reserve_bytes);
}

// --- Mandatory native-NVFP4 decode-cache demand + balloon prealloc ---
//
// compute_native_cache_demand must mirror phase 3b's SfAtom slab sizing
// exactly (cutlass_nvfp4_sf_size + 256-byte per-entry alignment; contiguous
// expert groups as ONE entry) and include the GDN/SSM projections that
// phase 0b registers — the old inline elems/16 heuristic omitted them and
// was not an upper bound under SfAtom padding.

namespace {

// Fake prequant hybrid: 2 layers, dense attention proj + GDN projections +
// 4-expert MoE (per-expert 2D views, the SafeTensors prequant layout).
// Shapes store the PACKED byte dim (K/2), mirroring the NVFP4 wire format.
void fill_prequant_model(Model& m) {
    m.config_.n_layers = 2;
    m.config_.n_kv_heads = 8;
    m.config_.n_experts = 4;
    m.config_.is_nvfp4_prequant = true;
    uintptr_t next = 1;
    for (int i = 0; i < 2; ++i) {
        TransformerLayer L;
        L.wq = make_weight(TensorKind::WQ, 1024, 512, QType::INT8, next++);
        L.ssm_in = make_weight(TensorKind::WQ, 2048, 1024, QType::INT8, next++);
        L.gdn_gate = make_weight(TensorKind::WQ, 512, 1024, QType::INT8, next++);
        for (int e = 0; e < 4; ++e) {
            L.expert_w_up.push_back(make_weight(TensorKind::EXPERT_UP, 768, 512, QType::INT8, next++));
            L.expert_w_down.push_back(
                make_weight(TensorKind::EXPERT_DOWN, 256, 768, QType::INT8, next++));
        }
        m.layers_.push_back(std::move(L));
    }
}

size_t sf_entry(int n, int k_packed) {
    constexpr size_t kSfAlign = 256;
    size_t sz = imp::cutlass_nvfp4_sf_size(n, k_packed * 2);
    return (sz + kSfAlign - 1) / kSfAlign * kSfAlign;
}

size_t sf_group(int ne, int n, int k_packed) {
    constexpr size_t kSfAlign = 256;
    size_t sz = static_cast<size_t>(ne) * imp::cutlass_nvfp4_sf_size(n, k_packed * 2);
    return (sz + kSfAlign - 1) / kSfAlign * kSfAlign;
}

size_t moe_slab(int ne, int n, int k_packed) {
    int64_t k = static_cast<int64_t>(k_packed) * 2;
    return static_cast<size_t>(ne) * n * k_packed + static_cast<size_t>(ne) * n * (k / 16) +
           static_cast<size_t>(ne) * sizeof(float);
}

}  // namespace

TEST(VramBudgetReserve, NativeCacheDemandMatchesPhase3Sizing) {
    Model m;
    fill_prequant_model(m);

    NativeCacheDemand d = compute_native_cache_demand(m);

    const size_t expected_sf = 2 * (sf_entry(1024, 512)      // wq
                                    + sf_entry(2048, 1024)   // ssm_in (GDN hybrid!)
                                    + sf_entry(512, 1024)    // gdn_gate
                                    + sf_group(4, 768, 512)  // expert up group
                                    + sf_group(4, 256, 768)  // expert down group
                                   );
    EXPECT_EQ(d.sf_bytes, expected_sf);

    const size_t expected_slab = std::max(moe_slab(4, 768, 512), moe_slab(4, 256, 768));
    EXPECT_EQ(d.moe_slab_bytes, expected_slab);
    EXPECT_EQ(d.total(), expected_sf + expected_slab);
}

TEST(VramBudgetReserve, NativeCacheDemandZeroForNonPrequant) {
    Model m;
    fill_model(m, QType::Q6_K, QType::Q6_K);

    NativeCacheDemand d = compute_native_cache_demand(m);
    EXPECT_EQ(d.sf_bytes, 0u);
    EXPECT_EQ(d.moe_slab_bytes, 0u);
}

TEST(VramBudgetReserve, PreallocCoversDemandAndFreesKv) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_prequant_model(m);

    EngineConfig config;
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

    // A VRAM-BOUND pool, deliberately: knob_config()'s 2048-token context makes
    // the pool demand-bound (512 blocks either way), and an assertion about the
    // reserve would then pass for the wrong reason. At 32k the pool is sized by
    // what is left, which is what the charge is supposed to move.
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
}

// V8 — plan sufficiency, which is what A7 actually asked for ("assert no tier is
// exceeded"). This replaced an assertion of mine that was simply wrong:
// `live <= plan`. It read well, it passed on five VRAM-generous shapes, and the
// bench-sized shape below disproves it — the live pass hands out 224 KV blocks
// where the plan affords 56, because the two answer different questions. Since
// #1135 the pool takes the PLAN's number, so the live pass being more generous
// costs nothing and is not a violation of anything.
//
// What must hold is the plan's own contract: a plan that reports ok must fit
// inside the budget it was handed. If that ever fails, every number downstream
// of it is furniture.
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
        probe.library_reserve_bytes = kMeasuredLibraryReserveBytes;
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
    // Without a case that genuinely cannot fit, the EXPECT_LE above is a
    // tautology: every plan it sees has already passed the very check it is
    // asserting. Mutation-verified — removing the fit check inside plan_memory()
    // left this test green until the impossible case was added.
    EXPECT_GE(refused, 1) << "no case exercised the refusal path";
}

// The divergence the old assertion mistook for a bug, pinned so nobody restores
// it. The live pass sizes KV from kv_fraction of what is free and clamps; the
// plan grants blocks_per_seq x batch out of a computed residual. At a small
// advertised context those answers differ by 4x, and neither is wrong — they are
// answers to different questions. The pool takes the plan's (B69).
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
    silly.kv_fraction = 5.0f;            // clamped to 0.95
    silly.vram_reserve_floor_pct = -5;   // clamped to 0 → 512 MiB feature base

    EngineConfig max_sane = knob_config();
    max_sane.kv_fraction = 0.95f;
    max_sane.vram_reserve_floor_pct = 0;

    VRAMBudget a = compute_vram_budget(m, silly, 32, 128, 8 * GiB);
    VRAMBudget b = compute_vram_budget(m, max_sane, 32, 128, 8 * GiB);
    EXPECT_EQ(a.reserve_bytes, b.reserve_bytes);
    EXPECT_EQ(a.kv_cache_bytes, b.kv_cache_bytes);
    EXPECT_EQ(a.kv_max_blocks, b.kv_max_blocks);
}

// #1103: the budget and the VRAMAllocator must agree on the headroom. Mode 2
// deliberately skips the 10% reserve-floor POLICY to fit larger weight caches,
// but the allocator's 5% is not policy — can_allocate() refuses any allocation
// >=16 MiB that would leave less free than that. A plan below it cannot be
// executed: the KV pool was sized against 512 MiB of assumed headroom while
// every cache allocation needed 1630 MiB, so the caches failed mid-build.
TEST(VramBudgetReserve, ReserveNeverUndercutsTheAllocatorHeadroom) {
    SKIP_IF_NO_CUDA();

    size_t total_vram = 0;
    vram_budget_mem_get_info(nullptr, &total_vram);
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

// kv_block_bytes_per_layer (#942): the single source for KV-size estimates.
// The pre-upload expert-offload reserve used to multiply by raw dtype_size(),
// which returns 0 for NVFP4/MXFP4_KV (zeroing the KV headroom) and 1 byte/elem
// for INT4 (2x the packed size), and never counted scale overhead.
TEST(VramBudgetReserve, KvBlockBytesPerLayerIsPackingAndScaleAware) {
    constexpr int bs = 16, nkv = 8, hd = 128;
    const size_t elems = static_cast<size_t>(bs) * nkv * hd;

    // Plain dtypes: elems * elem_size * 2 (K+V), no scale overhead.
    EXPECT_EQ(kv_block_bytes_per_layer(QType::F16, bs, nkv, hd), elems * 2 * 2);
    EXPECT_EQ(kv_block_bytes_per_layer(QType::FP8_E4M3, bs, nkv, hd), elems * 1 * 2);

    // INT8: 1 byte/elem + per-token half scale.
    EXPECT_EQ(kv_block_bytes_per_layer(QType::INT8, bs, nkv, hd),
              (elems + static_cast<size_t>(bs) * nkv * sizeof(half)) * 2);

    // INT4: packed 2 elems/byte + per-token half scale — NOT dtype_size()'s
    // 1 byte/elem.
    EXPECT_EQ(kv_block_bytes_per_layer(QType::INT4, bs, nkv, hd),
              (elems / 2 + static_cast<size_t>(bs) * nkv * sizeof(half)) * 2);

    // NVFP4 / MXFP4_KV: packed + per-16-element-group scales; raw dtype_size()
    // returns 0 here, which is exactly the #942 failure mode.
    const size_t packed4 = (elems / 2 + static_cast<size_t>(bs) * nkv * (hd / 16)) * 2;
    EXPECT_EQ(kv_block_bytes_per_layer(QType::NVFP4, bs, nkv, hd), packed4);
    EXPECT_EQ(kv_block_bytes_per_layer(QType::MXFP4_KV, bs, nkv, hd), packed4);
    EXPECT_GT(kv_block_bytes_per_layer(QType::NVFP4, bs, nkv, hd), 0u);
}

// #1747: the divergence log printed kv_max_blocks under "live pass would have
// said", and kv_max_blocks may already have been raised by the min_kv_tokens
// rescue floor. A start that hit the floor therefore reported a lower bound of
// the CONFIGURATION as if it were a reading of what the live pass sized. The
// pre-floor figure is kept so the log can name both; this pins that it is the
// unfloored one, which is what the mutation "assign after the raise" breaks.
TEST(VramBudgetReserve, PreFloorBlocksSurviveTheMinKvTokensRaise) {
    SKIP_IF_NO_CUDA();

    Model m;
    fill_model(m, QType::Q8_0, QType::Q8_0);

    EngineConfig config;
    config.max_seq_len = 32768;
    config.max_batch_size = 1;
    config.use_cuda_graphs = false;
    config.kv_cache_dtype = QType::F16;
    config.min_kv_tokens = 16384;  // 1024 blocks at block size 16

    // 8 GiB is chosen, not arbitrary: the sizing lands at 618 blocks there and
    // the floor raises it to 1024. At 9 GiB and above the floor never fires and
    // this test would pass while proving nothing, which is why the assertion
    // below is an ASSERT on the floor having fired at all.
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
