#include "runtime/storage_planner.h"
#include "model/model_config.h"
#include "model/model.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"

#include <gtest/gtest.h>
#include <cstdint>

using namespace imp;

// ---------------------------------------------------------------------------
// Helper: build a minimal Tensor descriptor (host ptr, no GPU memory needed)
// ---------------------------------------------------------------------------

static Tensor make_tensor_stub(TensorKind kind, int64_t rows, int64_t cols, uintptr_t ptr_sentinel,
                               QType source_qtype = QType::Q6_K) {
    Tensor t;
    t.data = reinterpret_cast<void*>(ptr_sentinel);
    // Default Q6_K matches the canonical nvfp4-beneficial source used by
    // production benches (Qwen3-14B Q6_K etc.). Tests can override to
    // exercise source-qtype-aware capability refinement
    // (see Phase 5 PR #1 commit 5.1.1).
    t.qtype = source_qtype;
    t.ndim = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.on_device = false;
    t.kind = kind;
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
        L.wq = make_tensor_stub(TensorKind::WQ, 4096, 4096, static_cast<uintptr_t>(i * 100 + 1));
        // SSM_IN is FP16_ONLY / required_floor=FP16 → must stay FP16
        L.ssm_in = make_tensor_stub(TensorKind::SSM_IN, 4096, 4096, static_cast<uintptr_t>(i * 100 + 2));
        // SSM_OUT is FP16_ONLY / required_floor=FP16 → must stay FP16
        L.ssm_out = make_tensor_stub(TensorKind::SSM_OUT, 4096, 4096, static_cast<uintptr_t>(i * 100 + 3));
        m.layers_.push_back(std::move(L));
    }

    PlanHints hints;
    hints.prefer_nvfp4_decode = true;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;  // 100 GiB — generous

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    int wq_count = 0, ssm_count = 0;
    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::WQ) {
            EXPECT_EQ(e.tier, StorageTier::NVFP4) << "WQ should be NVFP4 under prefer_nvfp4_decode";
            wq_count++;
        } else if (e.kind == TensorKind::SSM_IN || e.kind == TensorKind::SSM_OUT) {
            EXPECT_EQ(e.tier, StorageTier::FP16) << "SSM_IN/OUT must remain FP16 even in NVFP4 mode "
                                                    "(regression test for d0e9b03 bug class)";
            ssm_count++;
        }
    }
    EXPECT_EQ(wq_count, 2) << "expected 2 WQ entries (one per layer)";
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
        L.wq = make_tensor_stub(TensorKind::WQ, 4096, 4096, static_cast<uintptr_t>(i * 100 + 1));
        // Add an FFN weight as well so there's more to compress
        L.w_down = make_tensor_stub(TensorKind::W_DOWN, 4096, 4096, static_cast<uintptr_t>(i * 100 + 2));
        m.layers_.push_back(std::move(L));
    }

    PlanHints hints;
    hints.prefer_nvfp4_decode = false;
    hints.vram_budget_bytes = 1;  // absurdly small

    StoragePlan plan = plan_storage(m, m.config_, hints);
    EXPECT_TRUE(plan.failed) << "plan should fail when budget is 1 byte and model is several MiB";
    EXPECT_FALSE(plan.failure_reason.empty()) << "failed plan should carry a non-empty failure_reason";
}

// ---------------------------------------------------------------------------
// Native-NVFP4 sources are priced at their INCREMENTAL cost (#1765): the
// decode cache borrows the resident source storage (Phase 0b registers
// zero-copy), so a plan that routes them to NVFP4 must not charge the full
// tier bytes. Before the fix, every native-checkpoint load projected the
// whole model as new demand and the budget check failed on every start,
// making a real insufficiency indistinguishable from the normal case.
// ---------------------------------------------------------------------------

TEST(StoragePlanner, NativeNvfp4SourcesCostNothingAtNvfp4Tier) {
    Model m;
    m.config_.n_layers = 2;

    for (int i = 0; i < 2; ++i) {
        TransformerLayer L;
        L.wq = make_tensor_stub(TensorKind::WQ, 4096, 4096, static_cast<uintptr_t>(i * 100 + 1),
                                QType::NVFP4);
        L.w_down = make_tensor_stub(TensorKind::W_DOWN, 4096, 4096,
                                    static_cast<uintptr_t>(i * 100 + 2), QType::NVFP4);
        m.layers_.push_back(std::move(L));
    }
    // The F16 token embedding is the resident upload itself (Phase 1 builds
    // FP16 caches only for dequantable sources) - it must not be charged, or
    // its 2.4 GiB alone re-fails the budget check on the real checkpoint.
    m.tok_emb_ = make_tensor_stub(TensorKind::TOK_EMBED, 8192, 4096,
                                  static_cast<uintptr_t>(9001), QType::F16);

    PlanHints hints;
    hints.prefer_nvfp4_decode = true;
    // Contrast pin: full-tier pricing of these 4 tensors is ~37.7 MB
    // (4 * (4096*4096/2 + 4096*4096/16)), far above this budget - the
    // pre-fix pricing MUST fail here, which TinyBudgetReturnsFailure shows
    // for allocating (Q6_K) sources. Zero-copy sources must fit.
    hints.vram_budget_bytes = size_t{1} * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    EXPECT_FALSE(plan.failed) << plan.failure_reason;
    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::TOK_EMBED) {
            EXPECT_EQ(e.tier, StorageTier::FP16);
            EXPECT_EQ(e.bytes, 0) << "F16-source embedding at FP16 tier is the resident upload";
            continue;
        }
        EXPECT_EQ(e.tier, StorageTier::NVFP4);
        EXPECT_EQ(e.bytes, 0) << "NVFP4-source entry at NVFP4 tier must be zero-copy";
    }
    EXPECT_EQ(plan.projected_vram_bytes, 0u);
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
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;  // 100 GiB

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::WQ) {
            EXPECT_EQ(e.tier, StorageTier::NVFP4) << "WQ required_floor is NVFP4";
        } else if (e.kind == TensorKind::SSM_IN) {
            EXPECT_EQ(e.tier, StorageTier::FP16) << "SSM_IN required_floor is FP16";
        } else if (e.kind == TensorKind::W_GATE) {
            EXPECT_EQ(e.tier, StorageTier::NVFP4) << "W_GATE required_floor is NVFP4";
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
    L.wq = make_tensor_stub(TensorKind::WQ, 1024, 1024, 0x1000);
    L.wo = make_tensor_stub(TensorKind::WO, 1024, 1024, 0x2000);
    L.w_gate = make_tensor_stub(TensorKind::W_GATE, 1024, 1024, 0x3000);
    L.w_up = make_tensor_stub(TensorKind::W_UP, 1024, 1024, 0x4000);
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
// GDN_GATE is intentionally excluded from overlay enumeration: the GDN scan
// kernel consumes the raw weight pointer, never through gemm_dispatch, so an
// overlay copy would burn VRAM without a consumer. Locking this decision
// against accidental re-introduction.
// ---------------------------------------------------------------------------

TEST(StoragePlanner, GDNGateIsNotEnumeratedForOverlay) {
    Model m;
    m.config_.n_layers = 2;

    for (int i = 0; i < 2; ++i) {
        TransformerLayer L;
        L.gdn_gate = make_tensor_stub(TensorKind::GDN_GATE, 4096, 4096, static_cast<uintptr_t>(i * 100 + 1));
        m.layers_.push_back(std::move(L));
    }

    PlanHints hints;
    hints.vram_budget_bytes = size_t{10} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    for (const auto& e : plan.entries) {
        EXPECT_NE(e.kind, TensorKind::GDN_GATE)
            << "GDN_GATE must not appear in overlay plan — see storage_planner.cpp";
    }
}

// ---------------------------------------------------------------------------
// Byte accounting: projected_vram_bytes matches sum of entry bytes
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared-expert FFN enumeration (Nemotron / DeepSeek / Qwen3.5-MoE style).
// A layer with BOTH regular FFN and a shared-expert FFN must produce plan
// entries for BOTH — otherwise the storage flip silently drops the shared
// projection and inference produces garbage.
// ---------------------------------------------------------------------------

TEST(StoragePlanner, EnumeratesSharedExpertFFN) {
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    // Regular FFN (3 tensors)
    L.w_gate = make_tensor_stub(TensorKind::W_GATE, 1024, 1024, 0x1000);
    L.w_up = make_tensor_stub(TensorKind::W_UP, 1024, 1024, 0x2000);
    L.w_down = make_tensor_stub(TensorKind::W_DOWN, 1024, 1024, 0x3000);
    // Shared-expert FFN (3 additional tensors)
    L.w_gate_shared = make_tensor_stub(TensorKind::W_GATE, 1024, 1024, 0x4000);
    L.w_up_shared = make_tensor_stub(TensorKind::W_UP, 1024, 1024, 0x5000);
    L.w_down_shared = make_tensor_stub(TensorKind::W_DOWN, 1024, 1024, 0x6000);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.vram_budget_bytes = size_t{10} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    int gate_count = 0, up_count = 0, down_count = 0;
    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::W_GATE)
            gate_count++;
        if (e.kind == TensorKind::W_UP)
            up_count++;
        if (e.kind == TensorKind::W_DOWN)
            down_count++;
    }
    EXPECT_EQ(gate_count, 2) << "expected 2 W_GATE entries (regular + shared)";
    EXPECT_EQ(up_count, 2) << "expected 2 W_UP entries (regular + shared)";
    EXPECT_EQ(down_count, 2) << "expected 2 W_DOWN entries (regular + shared)";
}

// ---------------------------------------------------------------------------
// Top-level tensors (tok_emb, out_proj / LM head) must be enumerated.
// For NVFP4-prequant models (Qwen3-Coder-30B), out_proj has a choice of tier
// and would silently vanish from the plan if not enumerated.
// ---------------------------------------------------------------------------

TEST(StoragePlanner, EnumeratesTopLevelEmbeddingsAndLMHead) {
    Model m;
    m.config_.n_layers = 0;  // no layers — only top-level tensors

    m.tok_emb_ = make_tensor_stub(TensorKind::TOK_EMBED, 128256, 4096, 0x1000);
    m.out_proj_ = make_tensor_stub(TensorKind::LM_HEAD, 128256, 4096, 0x2000);

    PlanHints hints;
    hints.vram_budget_bytes = size_t{10} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    int tok_count = 0, lm_count = 0;
    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::TOK_EMBED)
            tok_count++;
        if (e.kind == TensorKind::LM_HEAD)
            lm_count++;
    }
    EXPECT_EQ(tok_count, 1) << "expected 1 TOK_EMBED entry";
    EXPECT_EQ(lm_count, 1) << "expected 1 LM_HEAD entry";
}

TEST(StoragePlanner, ProjectedVRAMMatchesEntrySum) {
    Model m;
    m.config_.n_layers = 3;

    for (int i = 0; i < 3; ++i) {
        TransformerLayer L;
        L.wq = make_tensor_stub(TensorKind::WQ, 512, 512, static_cast<uintptr_t>(0x1000 + i));
        L.ssm_in = make_tensor_stub(TensorKind::SSM_IN, 512, 512, static_cast<uintptr_t>(0x2000 + i));
        m.layers_.push_back(std::move(L));
    }

    PlanHints hints;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    size_t manual_sum = 0;
    for (const auto& e : plan.entries)
        manual_sum += static_cast<size_t>(e.bytes);

    EXPECT_EQ(plan.projected_vram_bytes, manual_sum);
}

// ---------------------------------------------------------------------------
// Phase 5 PR #1 — Commit 5.1.1 regression tests:
// source-qtype-aware capability refinement
//
// Closes the 2026-05-24 Q4_K_M cache coverage gap by ensuring the planner
// doesn't propose NVFP4 for sub-5-bit-source weights (where NVFP4 is a
// representation change at similar bit-width, no compression win).
// ---------------------------------------------------------------------------

TEST(StoragePlanner, Q4KSourceDoesNotPickNVFP4UnderPreferHint) {
    // Q4_K-source W_GATE: even with prefer_nvfp4_decode=true, NVFP4 must NOT
    // be picked (effective_capabilities strips NVFP4 from supported, raises
    // floor to FP16). This is the structural fix for the Gemma-3-12B Q4_K_M
    // bug where the runtime hit zero cache coverage.
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    L.w_gate = make_tensor_stub(TensorKind::W_GATE, 2048, 2048, 0x1000, QType::Q4_K);
    L.w_up = make_tensor_stub(TensorKind::W_UP, 2048, 2048, 0x2000, QType::Q4_K);
    L.w_down = make_tensor_stub(TensorKind::W_DOWN, 2048, 2048, 0x3000, QType::Q4_K);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.prefer_nvfp4_decode = true;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    int gate_count = 0;
    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::W_GATE || e.kind == TensorKind::W_UP ||
            e.kind == TensorKind::W_DOWN) {
            EXPECT_EQ(e.tier, StorageTier::FP16)
                << "Q4_K-source FFN weights must land on FP16, NOT NVFP4 — "
                   "representation change at same bit-width is a quality risk "
                   "(regression test for 2026-05-24 Q4_K coverage-gap bug)";
            gate_count++;
        }
    }
    EXPECT_EQ(gate_count, 3) << "expected 3 FFN entries (gate, up, down)";
}

TEST(StoragePlanner, Q6KSourcePreservesNVFP4UnderPreferHint) {
    // Q6_K-source WQ: with prefer_nvfp4_decode=true, NVFP4 IS the right pick
    // (>5.5 bits → compression win). Sanity check that the refinement only
    // strips NVFP4 for sub-5-bit sources.
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    L.wq = make_tensor_stub(TensorKind::WQ, 2048, 2048, 0x1000, QType::Q6_K);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.prefer_nvfp4_decode = true;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    bool found_wq = false;
    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::WQ) {
            EXPECT_EQ(e.tier, StorageTier::NVFP4)
                << "Q6_K-source WQ should land on NVFP4 under prefer_nvfp4_decode";
            found_wq = true;
        }
    }
    EXPECT_TRUE(found_wq) << "WQ entry should be present in plan";
}

TEST(StoragePlanner, F16SourceDoesNotPickNVFP4) {
    // F16-source weights (typical: LM head, embeddings) do NOT get NVFP4
    // overlay — matches runtime nvfp4_beneficial() policy. Documented as
    // "no NVFP4 conversion for F16 sources" since the runtime has no
    // dequant→quant path for raw FP16 weights anyway.
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    L.wq = make_tensor_stub(TensorKind::WQ, 2048, 2048, 0x1000, QType::F16);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.prefer_nvfp4_decode = true;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::WQ) {
            EXPECT_NE(e.tier, StorageTier::NVFP4)
                << "F16-source WQ must NOT pick NVFP4 — runtime has no FP16→NVFP4 quant path";
        }
    }
}

TEST(StoragePlanner, NativeNVFP4SourceMandatesNVFP4Tier) {
    // Native NVFP4 weight: the "overlay" IS the source storage. Tier must
    // be NVFP4 (or CUTLASS_NVFP4); no other tier is valid.
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    L.wq = make_tensor_stub(TensorKind::WQ, 2048, 2048, 0x1000, QType::NVFP4);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    for (const auto& e : plan.entries) {
        if (e.kind == TensorKind::WQ) {
            EXPECT_TRUE(e.tier == StorageTier::NVFP4 || e.tier == StorageTier::CUTLASS_NVFP4)
                << "NVFP4-source WQ must stay NVFP4 (or CUTLASS_NVFP4 layout)";
            EXPECT_EQ(e.source_qtype, QType::NVFP4) << "source_qtype must be propagated to Entry";
        }
    }
}

TEST(StoragePlanner, EntryCarriesSourceQtype) {
    // Sanity: plan.Entry.source_qtype must be populated from the input
    // tensor's qtype. The Phase-5 budget-downgrade loop relies on this for
    // re-querying effective_capabilities per entry.
    Model m;
    m.config_.n_layers = 1;

    TransformerLayer L;
    L.wq = make_tensor_stub(TensorKind::WQ, 512, 512, 0x1000, QType::Q6_K);
    L.wk = make_tensor_stub(TensorKind::WK, 512, 512, 0x2000, QType::Q8_0);
    L.w_gate = make_tensor_stub(TensorKind::W_GATE, 512, 512, 0x3000, QType::Q4_K);
    m.layers_.push_back(std::move(L));

    PlanHints hints;
    hints.vram_budget_bytes = size_t{100} * 1024 * 1024 * 1024;

    StoragePlan plan = plan_storage(m, m.config_, hints);
    ASSERT_FALSE(plan.failed) << plan.failure_reason;

    int seen_q6k = 0, seen_q80 = 0, seen_q4k = 0;
    for (const auto& e : plan.entries) {
        if (e.source_qtype == QType::Q6_K)
            ++seen_q6k;
        else if (e.source_qtype == QType::Q8_0)
            ++seen_q80;
        else if (e.source_qtype == QType::Q4_K)
            ++seen_q4k;
    }
    EXPECT_EQ(seen_q6k, 1);
    EXPECT_EQ(seen_q80, 1);
    EXPECT_EQ(seen_q4k, 1);
}
