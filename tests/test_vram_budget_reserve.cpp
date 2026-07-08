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

#include "memory/vram_query.h"
#include "model/model.h"
#include "model/model_config.h"
#include "runtime/engine.h"
#include "runtime/storage_planner.h"
#include "runtime/vram_budget.h"

#include <algorithm>

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

    const size_t GiB = 1024ull * 1024 * 1024;
    VRAMBudget with_planner = compute_vram_budget(m, config, 32, 128, 8 * GiB);

    // Q6_K routes to the NVFP4 tier: heuristic ≈ planner, so KV keeps most of
    // the post-reserve VRAM (well above the one-sequence floor).
    const size_t per_block = 16ull * 8 * 128 * 2 * 2 * 32;
    const size_t kv_bytes = static_cast<size_t>(with_planner.kv_max_blocks) * per_block;
    EXPECT_GE(kv_bytes, 4 * GiB) << "reservation must not fire for heuristic-covered sources";
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
