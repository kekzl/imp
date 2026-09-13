// Reference test for compressed-tensors NVFP4 dequant, starting from the ON-DISK format
// directly (not imp's own quantizer/dequantizer roundtrip, which can't catch a paired
// sign-flip/nibble-order bug or a missing weight_scale_2 factor).
// Tolerance 1e-2 FP16: FMA-order divergence between the sequential ref and imp's
// parallel-warp accumulator dominates over K=128; 1e-5 is unrealistic.

#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace imp {
namespace {

// Pure-host reference E2M1 -> FP32 decode following the OCP NVFP4 spec.
// Magnitude set: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
// Bit layout: bit3=sign, bits2:0=magnitude index.
float e2m1_nibble_to_f32_ref(uint8_t nibble) {
    static const float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = magnitudes[nibble & 0x07];
    return (nibble & 0x08) ? -mag : mag;
}

// Pure-host FP8 E4M3-fn->FP32 decode, independently derived from the spec (mirrors device's
// fp8_e4m3_to_float_fast) rather than testing the device kernel against a rephrased copy
// of itself.
float fp8_e4m3_to_f32_ref(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 3) & 0x0F;
    uint32_t man = bits & 0x07;
    float magnitude;
    if (exp == 0) {
        magnitude = static_cast<float>(man) / 512.0f;
    } else {
        // (1 + m/8) * 2^(exp-7), bias=7
        magnitude = (1.0f + static_cast<float>(man) / 8.0f) * std::ldexp(1.0f, static_cast<int>(exp) - 7);
    }
    return sign ? -magnitude : magnitude;
}

// Encode an FP32 in [0, 448] into FP8 E4M3 bits. Test-only — bit-exact for
// values that ARE in the FP8 E4M3 representable set; near-misses round to
// the nearest FP8. Used only to construct test fixtures with known scales.
uint8_t f32_to_fp8_e4m3_bits(float val) {
    uint32_t sign = (val < 0.0f) ? 1u : 0u;
    float abs_val = std::fabs(val);
    if (abs_val == 0.0f)
        return static_cast<uint8_t>(sign << 7);
    if (abs_val < 1.0f / 512.0f)
        return static_cast<uint8_t>(sign << 7);  // round to ±0
    int exp;
    float m = std::frexp(abs_val, &exp);  // m in [0.5, 1)
    // Want (1 + man/8) * 2^(e-7) ≈ abs_val, where exp_field = e (in [1,15]).
    int exp_field = exp - 1 + 7;  // because frexp yields m*2^e, m in [.5,1) → 2^(e-1) is the leading bit
    if (exp_field <= 0)
        exp_field = 0;
    if (exp_field > 15)
        exp_field = 15;
    float scaled = abs_val / std::ldexp(1.0f, exp_field - 7);
    int man = static_cast<int>(std::round((scaled - 1.0f) * 8.0f));
    if (man < 0)
        man = 0;
    if (man > 7)
        man = 7;
    if (exp_field == 15 && man == 7)
        man = 6;  // saturate NaN slot to max normal
    return static_cast<uint8_t>((sign << 7) | ((exp_field & 0x0F) << 3) | (man & 0x07));
}

class NvFP4CompressedTensorsRef : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

// Test 1: Mixed-magnitude weights, FP8 scale = 1.0 (bits 0x38), tensor_scale = 1.0.
// All factors at unity: any spec-deviation in nibble decode or alignment is exposed.
TEST_F(NvFP4CompressedTensorsRef, BaselineUnityScales) {
    constexpr int N = 64;
    constexpr int K = 128;
    static_assert(K % 16 == 0, "K must be multiple of group_size=16");

    const int n_mb = K / 16;

    // Build weight_packed deterministically: 2 nibbles per byte, low=even k, high=odd k.
    std::vector<uint8_t> h_packed(N * K / 2);
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < K / 2; ++kb) {
            uint8_t nib_low = static_cast<uint8_t>(((n + 2 * kb) * 5u) & 0x0F);
            uint8_t nib_high = static_cast<uint8_t>(((n + 2 * kb + 1) * 5u) & 0x0F);
            h_packed[n * (K / 2) + kb] = static_cast<uint8_t>((nib_high << 4) | nib_low);
        }
    }

    // weight_scale = all 1.0f (FP8 E4M3 bits = 0x38 = exp=7, man=0).
    constexpr uint8_t kFp8Bits_one = 0x38;
    std::vector<uint8_t> h_scale_e4m3(N * n_mb, kFp8Bits_one);
    ASSERT_NEAR(fp8_e4m3_to_f32_ref(kFp8Bits_one), 1.0f, 0.0f);

    const float tensor_scale = 1.0f;

    // Activation X: linspace [-1, 1] over K.
    std::vector<half> h_x(K);
    for (int k = 0; k < K; ++k)
        h_x[k] = __float2half(-1.0f + 2.0f * k / static_cast<float>(K - 1));

    // ---- Pure-host reference Y[n] = sum_k W[n,k] * X[k] ----
    std::vector<float> y_ref_f32(N, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int kb = k / 2;
            uint8_t byte = h_packed[n * (K / 2) + kb];
            uint8_t nibble = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            float w = e2m1_nibble_to_f32_ref(nibble);
            float scale_e4m3 = fp8_e4m3_to_f32_ref(h_scale_e4m3[n * n_mb + (k / 16)]);
            float x = __half2float(h_x[k]);
            y_ref_f32[n] += w * scale_e4m3 * tensor_scale * x;
        }
    }

    // ---- imp path via gemv_nvfp4_kpar ----
    void *d_packed = nullptr, *d_scale = nullptr, *d_x = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_packed, h_packed.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_scale, h_scale_e4m3.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(half)), cudaSuccess);
    cudaMemcpy(d_packed, h_packed.data(), h_packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale_e4m3.data(), h_scale_e4m3.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, N * sizeof(half));

    NvFP4QuantResult A;
    A.packed_data = d_packed;
    A.micro_scales = d_scale;
    A.tensor_scale = tensor_scale;
    A.N = N;
    A.K = K;

    gemv_nvfp4_kpar(A, reinterpret_cast<const half*>(d_x), reinterpret_cast<half*>(d_y), N, K, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<half> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_packed);
    cudaFree(d_scale);
    cudaFree(d_x);
    cudaFree(d_y);

    // Compare: max-abs-diff should be < 1e-2 (FP16 FMA-order noise dominates).
    float max_abs_diff = 0.0f;
    int worst_idx = -1;
    for (int n = 0; n < N; ++n) {
        float imp_v = __half2float(h_y[n]);
        float diff = std::fabs(imp_v - y_ref_f32[n]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            worst_idx = n;
        }
    }
    EXPECT_LT(max_abs_diff, 1e-2f) << "Compressed-tensors NVFP4 GEMV diverges from spec reference. "
                                   << "Worst row n=" << worst_idx
                                   << " imp=" << __half2float(h_y[worst_idx])
                                   << " ref=" << y_ref_f32[worst_idx]
                                   << " (max_abs_diff=" << max_abs_diff << ").";
}

// Test 2: Per-block scales vary (FP8 != 1.0) and tensor_scale != 1.0.
// Stresses two-level scaling: a sign-flipped or missing factor would break.
TEST_F(NvFP4CompressedTensorsRef, TwoLevelScalingVaryingPerBlock) {
    constexpr int N = 32;
    constexpr int K = 128;
    const int n_mb = K / 16;

    std::vector<uint8_t> h_packed(N * K / 2);
    for (size_t i = 0; i < h_packed.size(); ++i)
        h_packed[i] = static_cast<uint8_t>((i * 11u + 3u) & 0xFF);

    // Per-block scales cycle through {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.25}.
    static const float scale_values[8] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 0.25f};
    std::vector<uint8_t> h_scale_e4m3(N * n_mb);
    for (size_t i = 0; i < h_scale_e4m3.size(); ++i)
        h_scale_e4m3[i] = f32_to_fp8_e4m3_bits(scale_values[i % 8]);
    // An all-zero micro-scale (16-element group with amax=0) is legal on-wire and must zero
    // that group's contribution, not decode as subnormal or fall through to the tensor scale.
    // Untested before: every other cycled value is non-zero.
    ASSERT_GE(h_scale_e4m3.size(), 2u * static_cast<size_t>(n_mb));
    h_scale_e4m3[static_cast<size_t>(n_mb)] = 0x00;      // row 1, first group
    h_scale_e4m3[static_cast<size_t>(n_mb) + 1] = 0x00;  // row 1, second group
    ASSERT_EQ(fp8_e4m3_to_f32_ref(0x00), 0.0f);

    const float tensor_scale = 0.125f;  // Non-trivial multiplier — cannot be silently dropped.

    std::vector<half> h_x(K);
    for (int k = 0; k < K; ++k)
        h_x[k] = __float2half(0.5f * std::sin(0.1f * k));

    std::vector<float> y_ref_f32(N, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            uint8_t byte = h_packed[n * (K / 2) + (k / 2)];
            uint8_t nibble = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            float w = e2m1_nibble_to_f32_ref(nibble);
            float scale_e4m3 = fp8_e4m3_to_f32_ref(h_scale_e4m3[n * n_mb + (k / 16)]);
            float x = __half2float(h_x[k]);
            y_ref_f32[n] += w * scale_e4m3 * tensor_scale * x;
        }
    }

    void *d_packed = nullptr, *d_scale = nullptr, *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_packed, h_packed.size());
    cudaMalloc(&d_scale, h_scale_e4m3.size());
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_packed, h_packed.data(), h_packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale_e4m3.data(), h_scale_e4m3.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, N * sizeof(half));

    NvFP4QuantResult A;
    A.packed_data = d_packed;
    A.micro_scales = d_scale;
    A.tensor_scale = tensor_scale;
    A.N = N;
    A.K = K;

    gemv_nvfp4_kpar(A, reinterpret_cast<const half*>(d_x), reinterpret_cast<half*>(d_y), N, K, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    // Row 1 (k<32) draws only from the two zero-scale groups, so it must be EXACTLY zero.
    // Full-K comparison can't catch this: falling through to tensor scale (0x00 as 1.0) moves it
    // by 0.186 (caught), but exp==0-as-normal moves it 0.0029 and subnormal misread 0.00036,
    // both under the 1e-2 the FP16 output quantum forces over K=128.
    std::vector<half> h_x_head = h_x;
    for (int k = 32; k < K; ++k)
        h_x_head[k] = __float2half(0.0f);
    cudaMemcpy(d_x, h_x_head.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, N * sizeof(half));
    gemv_nvfp4_kpar(A, reinterpret_cast<const half*>(d_x), reinterpret_cast<half*>(d_y), N, K, stream_);
    cudaStreamSynchronize(stream_);
    std::vector<half> h_y_head(N);
    cudaMemcpy(h_y_head.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_packed);
    cudaFree(d_scale);
    cudaFree(d_x);
    cudaFree(d_y);

    float max_abs_diff = 0.0f;
    for (int n = 0; n < N; ++n) {
        float diff = std::fabs(__half2float(h_y[n]) - y_ref_f32[n]);
        if (diff > max_abs_diff)
            max_abs_diff = diff;
    }
    EXPECT_LT(max_abs_diff, 1e-2f);

    EXPECT_EQ(__half2float(h_y_head[1]), 0.0f)
        << "row 1's first two micro-scales are 0x00, so its whole k < 32 contribution must vanish; "
        << "got " << __half2float(h_y_head[1]);
    // Not vacuous: rows with live scales still move under the same activation.
    bool other_row_moved = false;
    for (int n = 0; n < N; ++n)
        if (n != 1 && __half2float(h_y_head[n]) != 0.0f)
            other_row_moved = true;
    EXPECT_TRUE(other_row_moved) << "the k < 32 activation produced an all-zero output, so the "
                                    "row-1 check above proves nothing";
}

// Test 3: Zero tensor_scale → output must be exactly zero (defensive zeroing
// as PR #113 / new F2 fix). Validates that promoted_scale=0 actually produces
// 0 GEMV output, not NaN/Inf, regardless of garbage in weight_packed.
TEST_F(NvFP4CompressedTensorsRef, ZeroTensorScaleProducesZeroOutput) {
    constexpr int N = 16;
    constexpr int K = 64;
    const int n_mb = K / 16;

    std::vector<uint8_t> h_packed(N * K / 2);
    for (size_t i = 0; i < h_packed.size(); ++i)
        h_packed[i] = 0xFF;  // every nibble = -6.0 (max negative magnitude)

    std::vector<uint8_t> h_scale_e4m3(N * n_mb, f32_to_fp8_e4m3_bits(64.0f));  // big scale
    const float tensor_scale = 0.0f;  // zeroes everything

    std::vector<half> h_x(K, __float2half(1.0f));

    void *d_packed = nullptr, *d_scale = nullptr, *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_packed, h_packed.size());
    cudaMalloc(&d_scale, h_scale_e4m3.size());
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_packed, h_packed.data(), h_packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale_e4m3.data(), h_scale_e4m3.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, N * sizeof(half));

    NvFP4QuantResult A;
    A.packed_data = d_packed;
    A.micro_scales = d_scale;
    A.tensor_scale = tensor_scale;
    A.N = N;
    A.K = K;

    gemv_nvfp4_kpar(A, reinterpret_cast<const half*>(d_x), reinterpret_cast<half*>(d_y), N, K, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_packed);
    cudaFree(d_scale);
    cudaFree(d_x);
    cudaFree(d_y);

    for (int n = 0; n < N; ++n) {
        float v = __half2float(h_y[n]);
        EXPECT_EQ(v, 0.0f) << "Row " << n << " has non-zero output " << v
                           << " despite tensor_scale=0; layer-zero defensive path is broken.";
        EXPECT_FALSE(std::isnan(v)) << "Row " << n << " is NaN under tensor_scale=0.";
        EXPECT_FALSE(std::isinf(v)) << "Row " << n << " is Inf under tensor_scale=0.";
    }
}

// Test 4: Negative weights → output sign is correct.
// Catches a sign-bit drop in the nibble decode path (which the HW PTX cvt
// instruction handles, but a future SW fallback must too).
TEST_F(NvFP4CompressedTensorsRef, NegativeWeightsSignPreserved) {
    constexpr int N = 8;
    constexpr int K = 32;
    const int n_mb = K / 16;

    // Every nibble = 0x0A → sign=1, magnitude index=2 → -1.0
    std::vector<uint8_t> h_packed(N * K / 2, 0xAA);

    std::vector<uint8_t> h_scale_e4m3(N * n_mb, f32_to_fp8_e4m3_bits(1.0f));
    const float tensor_scale = 1.0f;

    std::vector<half> h_x(K, __float2half(1.0f));

    void *d_packed = nullptr, *d_scale = nullptr, *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_packed, h_packed.size());
    cudaMalloc(&d_scale, h_scale_e4m3.size());
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_packed, h_packed.data(), h_packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale_e4m3.data(), h_scale_e4m3.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, N * sizeof(half));

    NvFP4QuantResult A;
    A.packed_data = d_packed;
    A.micro_scales = d_scale;
    A.tensor_scale = tensor_scale;
    A.N = N;
    A.K = K;

    gemv_nvfp4_kpar(A, reinterpret_cast<const half*>(d_x), reinterpret_cast<half*>(d_y), N, K, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_packed);
    cudaFree(d_scale);
    cudaFree(d_x);
    cudaFree(d_y);

    // Each row: K=32 elements × -1.0 × 1.0 × 1.0 = -32.
    const float expected = -static_cast<float>(K);
    for (int n = 0; n < N; ++n) {
        float v = __half2float(h_y[n]);
        EXPECT_NEAR(v, expected, 1e-2f) << "Row " << n << " expected " << expected << " got " << v
                                        << " — sign bit dropped in nibble decode?";
    }
}

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
