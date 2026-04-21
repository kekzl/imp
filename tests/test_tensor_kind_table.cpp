#include "model/tensor_kind_table.h"
#include "imp/storage_tier.h"

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
            << "kind " << tensor_kind_name(k)
            << " floor not in supported mask";
    }
}

TEST(TensorKindTable, GDNTensorsAreFP16Only) {
    for (auto k : {TensorKind::SSM_IN, TensorKind::SSM_OUT,
                   TensorKind::CONV1D_W, TensorKind::CONV1D_B,
                   TensorKind::BETA, TensorKind::ALPHA,
                   TensorKind::SSM_GROUP_NORM}) {
        const auto& cap = capabilities_of(k);
        EXPECT_EQ(cap.supported, mask(StorageTier::FP16))
            << "GDN kind " << tensor_kind_name(k)
            << " must be FP16-only (no quantized replacement exists)";
        EXPECT_EQ(cap.required_floor, StorageTier::FP16);
    }
}

TEST(TensorKindTable, NormsAreFP32Only) {
    for (auto k : {TensorKind::ATTN_NORM, TensorKind::FFN_NORM,
                   TensorKind::POST_ATTN_NORM, TensorKind::POST_FFN_NORM,
                   TensorKind::QK_NORM_Q, TensorKind::QK_NORM_K,
                   TensorKind::A_LOG, TensorKind::DT_BIAS}) {
        const auto& cap = capabilities_of(k);
        EXPECT_EQ(cap.supported, mask(StorageTier::FP32));
    }
}

TEST(TensorKindTable, AttentionProjectionsSupportAllQuantTiers) {
    for (auto k : {TensorKind::WQ, TensorKind::WK, TensorKind::WV, TensorKind::WO,
                   TensorKind::W_GATE, TensorKind::W_UP, TensorKind::W_DOWN}) {
        const auto& cap = capabilities_of(k);
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::FP16));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::FP8));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::NVFP4));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::CUTLASS_NVFP4));
        EXPECT_TRUE(mask_contains(cap.supported, StorageTier::MXFP4))
            << "kind " << tensor_kind_name(k) << " must support MXFP4 for fused QKV path";
    }
}
