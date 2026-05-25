#include "model/tensor_kind_table.h"
#include "core/storage_tier.h"

#include <gtest/gtest.h>

using namespace imp;

TEST(TensorKindTable, EveryKindHasEntry) {
    for (int i = 0; i < static_cast<int>(TensorKind::_COUNT); ++i) {
        auto k = static_cast<TensorKind>(i);
        const auto& cap = capabilities_of(k);
        EXPECT_NE(cap.supported, TierMask{0})
            << "kind " << tensor_kind_name(k) << " has empty supported mask";
    }
}

TEST(TensorKindTable, RequiredFloorIsInSupported) {
    for (int i = 0; i < static_cast<int>(TensorKind::_COUNT); ++i) {
        auto k = static_cast<TensorKind>(i);
        const auto& cap = capabilities_of(k);
        EXPECT_TRUE(mask_contains(cap.supported, cap.required_floor))
            << "kind " << tensor_kind_name(k) << " floor not in supported mask";
    }
}

TEST(TensorKindTable, GDNTensorsAreFP16Only) {
    for (auto k : {TensorKind::SSM_IN, TensorKind::SSM_OUT, TensorKind::CONV1D_W, TensorKind::CONV1D_B,
                   TensorKind::BETA, TensorKind::ALPHA, TensorKind::SSM_GROUP_NORM}) {
        const auto& cap = capabilities_of(k);
        EXPECT_EQ(cap.supported, mask(StorageTier::FP16))
            << "GDN kind " << tensor_kind_name(k) << " must be FP16-only (no quantized replacement exists)";
        EXPECT_EQ(cap.required_floor, StorageTier::FP16);
    }
}

TEST(TensorKindTable, NormsAreFP32Only) {
    for (auto k :
         {TensorKind::ATTN_NORM, TensorKind::FFN_NORM, TensorKind::POST_ATTN_NORM, TensorKind::POST_FFN_NORM,
          TensorKind::QK_NORM_Q, TensorKind::QK_NORM_K, TensorKind::A_LOG, TensorKind::DT_BIAS}) {
        const auto& cap = capabilities_of(k);
        EXPECT_EQ(cap.supported, mask(StorageTier::FP32));
    }
}

TEST(TensorKindTable, AttentionProjectionsSupportAllQuantTiers) {
    for (auto k : {TensorKind::WQ, TensorKind::WK, TensorKind::WV, TensorKind::WO, TensorKind::W_GATE,
                   TensorKind::W_UP, TensorKind::W_DOWN}) {
        const auto& cap = capabilities_of(k);
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::FP16));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::FP8));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::NVFP4));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::CUTLASS_NVFP4));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::MXFP4))
            << "kind " << tensor_kind_name(k) << " must support MXFP4 for fused QKV path";
    }
}

// ---------------------------------------------------------------------------
// Phase 5 PR #1 — Commit 5.1.1: effective_capabilities(kind, qtype)
// ---------------------------------------------------------------------------

TEST(EffectiveCapabilities, SubFiveBitSourcesStripNVFP4) {
    // Q4_K, Q4_0, Q5_0, Q5_1, Q3_K, Q2_K are < 5.5 bits/elem → NVFP4 overlay
    // is no compression win, possible quality risk. Must be stripped from
    // supported mask; floor raised to FP16.
    for (auto qtype :
         {QType::Q4_K, QType::Q4_0, QType::Q5_0, QType::Q5_1, QType::Q3_K, QType::Q2_K, QType::Q4_1}) {
        auto cap = effective_capabilities(TensorKind::W_GATE, qtype);
        EXPECT_FALSE(mask_contains(cap.supported, StorageTier::NVFP4))
            << "qtype " << qtype_name(qtype) << " (sub-5-bit) must NOT support NVFP4 overlay";
        EXPECT_FALSE(mask_contains(cap.supported, StorageTier::CUTLASS_NVFP4))
            << "qtype " << qtype_name(qtype) << " must NOT support CUTLASS_NVFP4 overlay";
        EXPECT_FALSE(mask_contains(cap.supported, StorageTier::MXFP4))
            << "qtype " << qtype_name(qtype) << " must NOT support MXFP4 overlay";
        EXPECT_EQ(cap.required_floor, StorageTier::FP16)
            << "qtype " << qtype_name(qtype) << " floor must be FP16 after NVFP4 strip";
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::FP16))
            << "FP16 must remain in supported as the floor";
    }
}

TEST(EffectiveCapabilities, NVFP4BeneficialSourcesPreserveNVFP4) {
    // Q5_K, Q6_K, Q8_0, Q8_K are >= 5.5 bits/elem → NVFP4 IS a compression
    // win. Capabilities should be unchanged from the kind defaults.
    for (auto qtype : {QType::Q5_K, QType::Q6_K, QType::Q8_0, QType::Q8_K}) {
        auto cap = effective_capabilities(TensorKind::W_GATE, qtype);
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::NVFP4))
            << "qtype " << qtype_name(qtype) << " (>=5.5-bit) must support NVFP4 overlay";
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::FP16));
        EXPECT_EQ(cap.required_floor, StorageTier::NVFP4)
            << "qtype " << qtype_name(qtype) << " floor stays at NVFP4 (compression win available)";
    }
}

TEST(EffectiveCapabilities, NativeNVFP4Source) {
    // Native NVFP4 source: only NVFP4-family tiers are valid. The "overlay"
    // IS the source storage.
    auto cap = effective_capabilities(TensorKind::W_GATE, QType::NVFP4);
    EXPECT_TRUE(mask_contains(cap.supported, StorageTier::NVFP4));
    EXPECT_TRUE(mask_contains(cap.supported, StorageTier::CUTLASS_NVFP4));
    EXPECT_FALSE(mask_contains(cap.supported, StorageTier::FP16))
        << "native NVFP4 should not propose FP16 (no dequant path expected)";
    EXPECT_EQ(cap.required_floor, StorageTier::NVFP4);
}

TEST(EffectiveCapabilities, NativeMXFP4Source) {
    auto cap = effective_capabilities(TensorKind::W_GATE, QType::MXFP4);
    EXPECT_TRUE(mask_contains(cap.supported, StorageTier::MXFP4));
    EXPECT_FALSE(mask_contains(cap.supported, StorageTier::FP16));
    EXPECT_FALSE(mask_contains(cap.supported, StorageTier::NVFP4));
    EXPECT_EQ(cap.required_floor, StorageTier::MXFP4);
}

TEST(EffectiveCapabilities, F16SourceStripsNVFP4) {
    // F16 source: matches runtime nvfp4_beneficial() policy — no NVFP4 path
    // exists for raw FP16 weights. Used by LM head / embeddings.
    auto cap = effective_capabilities(TensorKind::WQ, QType::F16);
    EXPECT_FALSE(mask_contains(cap.supported, StorageTier::NVFP4));
    EXPECT_EQ(cap.required_floor, StorageTier::FP16);
}

TEST(EffectiveCapabilities, FP16OnlyKindsAreUnaffected) {
    // FP16_ONLY kinds (SSM_IN, SSM_OUT, TOK_EMBED, LM_HEAD) already lack
    // NVFP4 in their supported mask — refinement should be a no-op.
    for (auto kind : {TensorKind::SSM_IN, TensorKind::SSM_OUT, TensorKind::TOK_EMBED,
                      TensorKind::LM_HEAD}) {
        for (auto qtype : {QType::F16, QType::Q4_K, QType::Q6_K, QType::Q8_0}) {
            auto cap = effective_capabilities(kind, qtype);
            EXPECT_EQ(cap.supported, mask(StorageTier::FP16))
                << "kind=" << tensor_kind_name(kind) << " qtype=" << qtype_name(qtype)
                << " — FP16_ONLY kind should stay FP16_ONLY after refinement";
            EXPECT_EQ(cap.required_floor, StorageTier::FP16);
        }
    }
}

TEST(EffectiveCapabilities, FloorAlwaysInSupported) {
    // For every (kind, qtype) combination, required_floor must remain in
    // supported mask after refinement.
    for (int k = 0; k < static_cast<int>(TensorKind::_COUNT); ++k) {
        auto kind = static_cast<TensorKind>(k);
        for (auto qtype : {QType::F16, QType::Q4_K, QType::Q5_K, QType::Q6_K, QType::Q8_0,
                           QType::NVFP4, QType::MXFP4, QType::F32}) {
            auto cap = effective_capabilities(kind, qtype);
            EXPECT_TRUE(mask_contains(cap.supported, cap.required_floor))
                << "kind=" << tensor_kind_name(kind) << " qtype=" << qtype_name(qtype)
                << " — required_floor must remain in supported after refinement";
        }
    }
}

TEST(TensorKindTable, EveryKindHasName) {
    for (int i = 1; i < static_cast<int>(TensorKind::_COUNT); ++i) {
        auto k = static_cast<TensorKind>(i);
        const char* name = tensor_kind_name(k);
        EXPECT_NE(std::string(name), "UNKNOWN")
            << "TensorKind index " << i << " returns UNKNOWN — add a case to tensor_kind_name.cpp";
    }
}

TEST(TensorKindTable, GDNProjectionsAreFP16Only) {
    for (auto k : {TensorKind::GDN_ALPHA, TensorKind::GDN_BETA, TensorKind::GDN_ALPHA_BETA_PACKED,
                   TensorKind::GDN_INPUT_PACKED}) {
        const auto& cap = capabilities_of(k);
        EXPECT_EQ(cap.supported, mask(StorageTier::FP16))
            << "GDN kind " << tensor_kind_name(k)
            << " must be FP16-only (delta-rule projections, no quantized path)";
        EXPECT_EQ(cap.required_floor, StorageTier::FP16);
    }
}
