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

#include "model/model.h"
#include "model/model_config.h"
#include "runtime/engine.h"
#include "runtime/storage_planner.h"
#include "runtime/vram_budget.h"

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
