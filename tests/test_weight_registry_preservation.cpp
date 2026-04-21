#include "runtime/storage_planner.h"
#include "model/model_config.h"
#include "model/model.h"
#include "imp/tensor_kind.h"
#include "imp/storage_tier.h"

#include <gtest/gtest.h>
#include <cstdint>

using namespace imp;

// ---------------------------------------------------------------------------
// Helper: build a minimal Tensor descriptor (host ptr, no GPU memory needed)
// ---------------------------------------------------------------------------

static Tensor make_tensor_stub(TensorKind kind, int64_t rows, int64_t cols,
                               uintptr_t ptr_sentinel) {
    Tensor t;
    t.data       = reinterpret_cast<void*>(ptr_sentinel);
    t.dtype      = DType::FP16;
    t.ndim       = 2;
    t.shape[0]   = rows;
    t.shape[1]   = cols;
    t.on_device  = false;
    t.kind       = kind;
    return t;
}

// ---------------------------------------------------------------------------
// Critical regression test: NVFP4-only mode must NOT downgrade FP16-only kinds
// This guards against the d0e9b03 bug class where SSM_IN/SSM_OUT were
// incorrectly assigned NVFP4 tier in NVFP4-preferring mode.
// ---------------------------------------------------------------------------

TEST(WeightRegistryPreservation, NVFP4ModeDoesNotDowngradeSSMInOut) {
    Model m;
    m.config_.n_layers = 2;

    for (int i = 0; i < 2; ++i) {
        TransformerLayer L;
        // WQ is ALL_QUANT / required_floor=NVFP4 → should be assigned NVFP4
        L.wq = make_tensor_stub(TensorKind::WQ,
                                4096, 4096,
                                static_cast<uintptr_t>(i * 100 + 1));
        // SSM_IN is FP16_ONLY / required_floor=FP16 → must stay FP16
        L.ssm_in = make_tensor_stub(TensorKind::SSM_IN,
                                    4096, 4096,
                                    static_cast<uintptr_t>(i * 100 + 2));
        // SSM_OUT is FP16_ONLY / required_floor=FP16 → must stay FP16
        L.ssm_out = make_tensor_stub(TensorKind::SSM_OUT,
                                     4096, 4096,
                                     static_cast<uintptr_t>(i * 100 + 3));
        m.layers_.push_back(std::move(L));
    }

    PlanHints hints;
    hints.prefer_nvfp4_decode  = true;
    hints.vram_budget_bytes    = size_t{100} * 1024 * 1024 * 1024;  // 100 GiB — generous

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    int wq_count = 0, ssm_count = 0;
    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::WQ) {
            EXPECT_EQ(e.tier, StorageTier::NVFP4)
                << "WQ should be NVFP4 under prefer_nvfp4_decode";
            wq_count++;
        } else if (e.kind == TensorKind::SSM_IN || e.kind == TensorKind::SSM_OUT) {
            EXPECT_EQ(e.tier, StorageTier::FP16)
                << "SSM_IN/OUT must remain FP16 even in NVFP4 mode "
                   "(regression test for d0e9b03 bug class)";
            ssm_count++;
        }
    }
    EXPECT_EQ(wq_count, 2)  << "expected 2 WQ entries (one per layer)";
    EXPECT_EQ(ssm_count, 4) << "expected 4 SSM entries (ssm_in + ssm_out × 2 layers)";
}

// ---------------------------------------------------------------------------
// Budget constraint: a budget of 1 byte must cause plan.failed
// ---------------------------------------------------------------------------

TEST(StoragePlanner, TinyBudgetReturnsFailure) {
    // Two layers, each with a WQ (4096×4096) = 4096*4096*2 = 32 MiB at FP16,
    // or 4096*4096/2 + 4096*4096/16 ≈ 9 MiB at NVFP4 (required_floor).
    // Either way, a 1-byte budget must fail.
    Model m;
    m.config_.n_layers = 2;

    for (int i = 0; i < 2; ++i) {
        TransformerLayer L;
        L.wq = make_tensor_stub(TensorKind::WQ,
                                4096, 4096,
                                static_cast<uintptr_t>(i * 100 + 1));
        // Add an FFN weight as well so there's more to compress
        L.w_down = make_tensor_stub(TensorKind::W_DOWN,
                                    4096, 4096,
                                    static_cast<uintptr_t>(i * 100 + 2));
        m.layers_.push_back(std::move(L));
    }

    PlanHints hints;
    hints.prefer_nvfp4_decode = false;
    hints.vram_budget_bytes   = 1;  // absurdly small

    StoragePlan plan = plan_storage(m, m.config_, hints);
    EXPECT_TRUE(plan.failed)
        << "plan should fail when budget is 1 byte and model is several MiB";
    EXPECT_FALSE(plan.failure_reason.empty())
        << "failed plan should carry a non-empty failure_reason";
}

// ---------------------------------------------------------------------------
// Generous budget: initial tier selection is preserved unchanged
// ---------------------------------------------------------------------------

TEST(StoragePlanner, GenerousBudgetPreservesInitialTiers) {
    // With no budget pressure and no hints, tensors should land at required_floor.
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    // WQ → required_floor = NVFP4 (ALL_QUANT, floor=NVFP4)
    L.wq = make_tensor_stub(TensorKind::WQ, 2048, 2048, 0x1000);
    // SSM_IN → required_floor = FP16 (FP16_ONLY)
    L.ssm_in = make_tensor_stub(TensorKind::SSM_IN, 2048, 2048, 0x2000);
    // W_GATE → required_floor = NVFP4
    L.w_gate = make_tensor_stub(TensorKind::W_GATE, 2048, 2048, 0x3000);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.prefer_nvfp4_decode = false;
    hints.vram_budget_bytes   = size_t{100} * 1024 * 1024 * 1024;  // 100 GiB

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::WQ) {
            EXPECT_EQ(e.tier, StorageTier::NVFP4)
                << "WQ required_floor is NVFP4";
        } else if (e.kind == TensorKind::SSM_IN) {
            EXPECT_EQ(e.tier, StorageTier::FP16)
                << "SSM_IN required_floor is FP16";
        } else if (e.kind == TensorKind::W_GATE) {
            EXPECT_EQ(e.tier, StorageTier::NVFP4)
                << "W_GATE required_floor is NVFP4";
        }
    }
    EXPECT_EQ(static_cast<int>(plan.entries.size()), 3);
}

// ---------------------------------------------------------------------------
// dual_path hint: attention uses FP8, FFN uses NVFP4
// ---------------------------------------------------------------------------

TEST(StoragePlanner, DualPathHintRoutesCorrectly) {
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    L.wq     = make_tensor_stub(TensorKind::WQ,     1024, 1024, 0x1000);
    L.wo     = make_tensor_stub(TensorKind::WO,     1024, 1024, 0x2000);
    L.w_gate = make_tensor_stub(TensorKind::W_GATE, 1024, 1024, 0x3000);
    L.w_up   = make_tensor_stub(TensorKind::W_UP,   1024, 1024, 0x4000);
    L.w_down = make_tensor_stub(TensorKind::W_DOWN, 1024, 1024, 0x5000);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.dual_path_attn_fp8_ffn_nvfp4 = true;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    for (const auto& e : plan.entries) {
        switch (e.kind) {
            case TensorKind::WQ:
            case TensorKind::WO:
                EXPECT_EQ(e.tier, StorageTier::FP8)
                    << "attention projections should be FP8 under dual_path hint";
                break;
            case TensorKind::W_GATE:
            case TensorKind::W_UP:
            case TensorKind::W_DOWN:
                EXPECT_EQ(e.tier, StorageTier::NVFP4)
                    << "FFN projections should be NVFP4 under dual_path hint";
                break;
            default:
                break;
        }
    }
    EXPECT_EQ(static_cast<int>(plan.entries.size()), 5);
}

// ---------------------------------------------------------------------------
// Byte accounting: projected_vram_bytes matches sum of entry bytes
// ---------------------------------------------------------------------------

TEST(StoragePlanner, ProjectedVRAMMatchesEntrySum) {
    Model m;
    m.config_.n_layers = 3;

    for (int i = 0; i < 3; ++i) {
        TransformerLayer L;
        L.wq     = make_tensor_stub(TensorKind::WQ,    512, 512, static_cast<uintptr_t>(0x1000 + i));
        L.ssm_in = make_tensor_stub(TensorKind::SSM_IN, 512, 512, static_cast<uintptr_t>(0x2000 + i));
        m.layers_.push_back(std::move(L));
    }

    PlanHints hints;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    size_t manual_sum = 0;
    for (const auto& e : plan.entries) manual_sum += static_cast<size_t>(e.bytes);

    EXPECT_EQ(plan.projected_vram_bytes, manual_sum);
}
