#include <gtest/gtest.h>
#include "compute/nvfp4_quant_ref.h"
#include "core/tensor.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>

namespace imp {
namespace {

// FP16->NVFP4(FP8 UE4M3 scale)->FP16 round-trip. E2M1's 8 magnitudes plus per-16-element
// scale give a worst-case ~12.5% quantization step per group; RMSE over Gaussian input
// should be a few percent of std.
TEST(Nvfp4QuantRefTest, RoundTripGaussian1024) {
    constexpr int N = 1024;

    // Generate Gaussian input (deterministic seed for stability).
    std::vector<half> h_input(N);
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float input_ss = 0.0f;
    for (int i = 0; i < N; ++i) {
        float v = dist(rng);
        h_input[i] = __float2half(v);
        input_ss += v * v;
    }
    float input_rms = std::sqrt(input_ss / N);

    half* d_input = nullptr;
    uint8_t* d_nvfp4 = nullptr;
    uint8_t* d_sf = nullptr;
    half* d_output = nullptr;

    ASSERT_EQ(cudaMalloc(&d_input, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_nvfp4, N / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sf, (N + 15) / 16), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, N * sizeof(half)), cudaSuccess);

    cudaMemcpy(d_input, h_input.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    nvfp4_quant_linear_fp16(d_input, d_nvfp4, d_sf, N, 0);
    nvfp4_dequant_linear_fp16(d_nvfp4, d_sf, d_output, N, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_output(N);
    cudaMemcpy(h_output.data(), d_output, N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_nvfp4);
    cudaFree(d_sf);
    cudaFree(d_output);

    // Error metric: RMSE as fraction of input RMS.
    float err_ss = 0.0f;
    for (int i = 0; i < N; ++i) {
        float a = __half2float(h_input[i]);
        float b = __half2float(h_output[i]);
        float d = a - b;
        err_ss += d * d;
    }
    float err_rms = std::sqrt(err_ss / N);
    float rel_err_rms = err_rms / input_rms;

    // E2M1 with per-16-elem scale: expected RMSE ≈ 2-6% of input RMS for
    // Gaussian data. Fail if > 15% (something fundamentally wrong).
    EXPECT_LT(rel_err_rms, 0.15f) << "NVFP4 round-trip RMSE " << (rel_err_rms * 100.0f)
                                  << "% of input RMS — quant math likely wrong.";

    // Informational — not a failure condition.
    std::cout << "[  INFO    ] NVFP4 round-trip relative RMSE: " << (rel_err_rms * 100.0f) << "%"
              << std::endl;
}

// Representable-value invariance: if every input is already exactly in
// the E2M1 value set scaled by a representable FP8 scale, the round-trip
// should be bit-exact (modulo FP16 rounding).
TEST(Nvfp4QuantRefTest, RepresentableValuesAreBitExact) {
    constexpr int N = 16;  // one group

    // E2M1 magnitudes × scale 1.0 → all representable.
    std::vector<half> h_input(N);
    float vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int i = 0; i < N; ++i) {
        float v = vals[i % 8];
        if (i >= 8)
            v = -v;
        h_input[i] = __float2half(v);
    }

    half* d_input = nullptr;
    uint8_t* d_nvfp4 = nullptr;
    uint8_t* d_sf = nullptr;
    half* d_output = nullptr;

    ASSERT_EQ(cudaMalloc(&d_input, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_nvfp4, N / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sf, 1), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, N * sizeof(half)), cudaSuccess);

    cudaMemcpy(d_input, h_input.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    nvfp4_quant_linear_fp16(d_input, d_nvfp4, d_sf, N, 0);
    nvfp4_dequant_linear_fp16(d_nvfp4, d_sf, d_output, N, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_output(N);
    cudaMemcpy(h_output.data(), d_output, N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_nvfp4);
    cudaFree(d_sf);
    cudaFree(d_output);

    // Each value should round-trip within FP16 tolerance.
    for (int i = 0; i < N; ++i) {
        float a = __half2float(h_input[i]);
        float b = __half2float(h_output[i]);
        EXPECT_NEAR(a, b, 0.01f) << "idx " << i << ": in=" << a << " out=" << b;
    }
}

// quantize_fp16_to_nvfp4 (Modelopt layout, value = e2m1 * FP8 micro * tensor_scale) stores
// tensor_scale = absmax / 6; a wrong divisor cancels in its own round trip (M-NV1), so the dequant
// uses the host scale. Blocks are e2m1 * 0.5 * {1, 1/2, 1/4, 1/8}: every step is exact.
TEST(Nvfp4QuantRefTest, Fp16ToNvfp4TensorScaleIsAbsmaxOverSix) {
    constexpr int N = 2;
    constexpr int K = 64;  // 8 micro-blocks of 16
    constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    constexpr float kBlockMult[4] = {1.0f, 0.5f, 0.25f, 0.125f};

    std::vector<half> h_input(static_cast<size_t>(N) * K);
    float absmax = 0.0f;
    for (size_t i = 0; i < h_input.size(); ++i) {
        const size_t j = i % 16;
        float v = kE2M1[j % 8] * kBlockMult[(i / 16) % 4] * 0.5f;
        if (j >= 8)
            v = -v;
        h_input[i] = __float2half(v);
        absmax = std::max(absmax, std::fabs(__half2float(h_input[i])));
    }

    half* d_input = nullptr;
    half* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, h_input.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, h_input.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half), cudaMemcpyHostToDevice),
              cudaSuccess);

    const int64_t shape[2] = {N, K};
    Tensor t(d_input, QType::F16, 2, shape, /*on_device=*/true);
    NvFP4Owned q;
    quantize_fp16_to_nvfp4(t, q, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // calibrate_nvfp4_scales runs the same host float division: exact compare.
    const float want_scale = absmax / 6.0f;
    EXPECT_EQ(q.tensor_scale, want_scale);

    // NOLINTNEXTLINE(cppcoreguidelines-slicing): the view of the owner
    NvFP4QuantResult view = q;
    view.tensor_scale = want_scale;
    dequantize_nvfp4_to_fp16(view, d_output, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<half> h_output(h_input.size());
    ASSERT_EQ(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_input);
    cudaFree(d_output);

    int mismatches = 0;
    for (size_t i = 0; i < h_input.size(); ++i) {
        const float a = __half2float(h_input[i]);
        const float b = __half2float(h_output[i]);
        if (a != b && mismatches++ < 4)
            ADD_FAILURE() << "idx " << i << ": in=" << a << " out=" << b;
    }
    EXPECT_EQ(mismatches, 0);
}

}  // namespace
}  // namespace imp
