// Host-only cases of tests/test_nvfp4_compressed_tensors_ref.cu, in test-core (CPU lane): no CUDA call.
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace imp {
namespace {

// Test 5 (F2): Modelopt path defensively zeros NaN/+Inf weight_scale_2.
TEST(NvFP4PromoteWeightScale2, ModeloptNaNZeroes) {
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(std::nanf(""), /*is_llm_compressor=*/false, &zeroed);
    EXPECT_EQ(r, 0.0f);
    EXPECT_TRUE(zeroed);
    EXPECT_FALSE(std::isnan(r));
}

TEST(NvFP4PromoteWeightScale2, ModeloptPlusInfZeroes) {
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(std::numeric_limits<float>::infinity(),
                                           /*is_llm_compressor=*/false, &zeroed);
    EXPECT_EQ(r, 0.0f);
    EXPECT_TRUE(zeroed);
}

TEST(NvFP4PromoteWeightScale2, ModeloptNegInfZeroes) {
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(-std::numeric_limits<float>::infinity(),
                                           /*is_llm_compressor=*/false, &zeroed);
    EXPECT_EQ(r, 0.0f);
    EXPECT_TRUE(zeroed);
}

TEST(NvFP4PromoteWeightScale2, ModeloptZeroIsZeroFlagged) {
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(0.0f, /*is_llm_compressor=*/false, &zeroed);
    EXPECT_EQ(r, 0.0f);
    EXPECT_TRUE(zeroed);  // Modelopt zero is a legitimate "null layer" but we still flag for diagnostic
}

TEST(NvFP4PromoteWeightScale2, ModeloptFinitePassesThrough) {
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(0.125f, /*is_llm_compressor=*/false, &zeroed);
    EXPECT_FLOAT_EQ(r, 0.125f);
    EXPECT_FALSE(zeroed);
}

TEST(NvFP4PromoteWeightScale2, LlmCompressorNaNZeroes) {
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(std::nanf(""), /*is_llm_compressor=*/true, &zeroed);
    EXPECT_EQ(r, 0.0f);
    EXPECT_TRUE(zeroed);
}

TEST(NvFP4PromoteWeightScale2, LlmCompressorZeroNoInf) {
    // The bug PR #113 first guarded: 1/0 = +Inf without the defensive zeroing.
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(0.0f, /*is_llm_compressor=*/true, &zeroed);
    EXPECT_EQ(r, 0.0f);
    EXPECT_FALSE(std::isinf(r));
    EXPECT_TRUE(zeroed);
}

TEST(NvFP4PromoteWeightScale2, LlmCompressorTinyNonFiniteFlip) {
    // Subnormal where 1/x overflows to +Inf — must be defensively zeroed.
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(std::numeric_limits<float>::denorm_min(),
                                           /*is_llm_compressor=*/true, &zeroed);
    EXPECT_EQ(r, 0.0f);
    EXPECT_FALSE(std::isinf(r));
    EXPECT_TRUE(zeroed);
}

TEST(NvFP4PromoteWeightScale2, LlmCompressorFiniteFlipsCorrectly) {
    bool zeroed = false;
    float r = nvfp4_promote_weight_scale_2(8.0f, /*is_llm_compressor=*/true, &zeroed);
    EXPECT_FLOAT_EQ(r, 0.125f);
    EXPECT_FALSE(zeroed);
}

// ---- F8: weight_scale dtype validation ----

TEST(NvFP4ValidateWeightScaleDtype, AcceptsFp8E4m3) {
    std::string err;
    EXPECT_TRUE(nvfp4_validate_weight_scale_dtype(QType::FP8_E4M3, &err)) << err;
    EXPECT_TRUE(err.empty());
}

TEST(NvFP4ValidateWeightScaleDtype, RejectsUInt8MxFP4Crossroute) {
    // INT8 maps from the wire 'U8'/'I8' string MXFP4/GPTQ ship weight_scale (UE8M0) bytes in.
    // Cross-misrouting (loader misclassifies MXFP4 as NVFP4) would read UE8M0 bytes as E4M3.
    std::string err;
    EXPECT_FALSE(nvfp4_validate_weight_scale_dtype(QType::INT8, &err));
    EXPECT_FALSE(err.empty());
}

TEST(NvFP4ValidateWeightScaleDtype, RejectsFp8E5m2) {
    // E5M2 is range-extended FP8 used for activations, never weight scales
    // in the compressed-tensors spec. Reject defensively.
    std::string err;
    EXPECT_FALSE(nvfp4_validate_weight_scale_dtype(QType::FP8_E5M2, &err));
    EXPECT_FALSE(err.empty());
}

TEST(NvFP4ValidateWeightScaleDtype, RejectsF16) {
    // Some pipelines emit weight_scale as FP16 directly — that's not the
    // compressed-tensors spec.
    std::string err;
    EXPECT_FALSE(nvfp4_validate_weight_scale_dtype(QType::F16, &err));
    EXPECT_FALSE(err.empty());
}

TEST(NvFP4ValidateWeightScaleDtype, RejectsNone) {
    // Sentinel "no qtype set" must never accidentally pass.
    std::string err;
    EXPECT_FALSE(nvfp4_validate_weight_scale_dtype(QType::NONE, &err));
    EXPECT_FALSE(err.empty());
}

// ---- F6: weight_packed vs weight_scale shape validation ----

TEST(NvFP4ValidatePackedScaleShapes, AcceptsTypicalQwen3) {
    // Qwen3-MoE q_proj: logical [4096, 2048]. Packed = [4096, 1024].
    // weight_scale = [4096, 128] (2048/16).
    std::string err;
    EXPECT_TRUE(nvfp4_validate_packed_scale_shapes(4096, 1024, 4096, 128, &err)) << err;
}

TEST(NvFP4ValidatePackedScaleShapes, AcceptsSmallGemma4Expert) {
    // Gemma-4 expert dim: 256 rows × packed-K 1408 → scale [256, 176].
    std::string err;
    EXPECT_TRUE(nvfp4_validate_packed_scale_shapes(256, 1408, 256, 176, &err)) << err;
}

TEST(NvFP4ValidatePackedScaleShapes, RejectsOuterDimMismatch) {
    // Transposed weight_scale would put N as inner dim and K as outer.
    std::string err;
    EXPECT_FALSE(nvfp4_validate_packed_scale_shapes(4096, 1024, 1024, 4096, &err));
    EXPECT_FALSE(err.empty());
}

TEST(NvFP4ValidatePackedScaleShapes, RejectsGroupSize8) {
    // group_size=8: scale would be [N, K/8] = [4096, 256]. Not the spec.
    std::string err;
    EXPECT_FALSE(nvfp4_validate_packed_scale_shapes(4096, 1024, 4096, 256, &err));
    EXPECT_FALSE(err.empty());
}

TEST(NvFP4ValidatePackedScaleShapes, RejectsGroupSize32) {
    // group_size=32: scale = [N, K/32] = [4096, 64].
    std::string err;
    EXPECT_FALSE(nvfp4_validate_packed_scale_shapes(4096, 1024, 4096, 64, &err));
    EXPECT_FALSE(err.empty());
}

TEST(NvFP4ValidatePackedScaleShapes, RejectsZeroScaleInner) {
    std::string err;
    EXPECT_FALSE(nvfp4_validate_packed_scale_shapes(4096, 1024, 4096, 0, &err));
    EXPECT_FALSE(err.empty());
}

TEST(NvFP4ValidatePackedScaleShapes, AcceptsTinyTestShape) {
    // [16, 64] packed inner with [16, 8] scale inner — used by the F1 baseline test.
    std::string err;
    EXPECT_TRUE(nvfp4_validate_packed_scale_shapes(16, 64, 16, 8, &err)) << err;
}

}  // namespace
}  // namespace imp
