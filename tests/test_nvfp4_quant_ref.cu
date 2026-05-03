#include <gtest/gtest.h>
#include "compute/nvfp4_quant_ref.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <cmath>

namespace imp {
namespace {

// Round-trip validation: FP16 → NVFP4 (with FP8 UE4M3 scale) → FP16.
// Measures relative error. E2M1 with 8 representable magnitudes
// plus per-16-element scale should give worst-case ~1/8 = 12.5%
// quantization step per group; averaged over a Gaussian input
// distribution the RMSE should be a few percent of the std.
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

}  // namespace
}  // namespace imp
