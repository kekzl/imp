#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/gemm.h"

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace imp {
namespace {

// =========================================================================
// Fused SwiGLU + Q8_1 quantize produces valid blocks
// =========================================================================

TEST(GemmDP4ATest, FusedSwiGLUQuantize) {
    constexpr int K = 256;
    srand(42);

    std::vector<float> h_gate(K), h_up(K);
    for (auto& v : h_gate)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_up)
        v = (rand() % 200 - 100) / 100.0f;

    // CPU reference: SwiGLU then manual Q8_1 check
    std::vector<float> act_ref(K);
    for (int i = 0; i < K; i++) {
        float g = h_gate[i];
        float silu_g = g / (1.0f + expf(-g));
        act_ref[i] = silu_g * h_up[i];
    }

    half *d_gate, *d_up;
    block_q8_1* d_q8_out;
    float* d_d8;
    int n_blocks = K / 32;
    cudaMalloc(&d_gate, K * sizeof(half));
    cudaMalloc(&d_up, K * sizeof(half));
    cudaMalloc(&d_q8_out, n_blocks * sizeof(block_q8_1));
    cudaMalloc(&d_d8, n_blocks * sizeof(float));

    std::vector<half> hg(K), hu(K);
    for (int i = 0; i < K; i++) {
        hg[i] = __float2half(h_gate[i]);
        hu[i] = __float2half(h_up[i]);
    }
    cudaMemcpy(d_gate, hg.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, hu.data(), K * sizeof(half), cudaMemcpyHostToDevice);

    swiglu_quantize_q8_1(d_gate, d_up, d_q8_out, d_d8, K, nullptr);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "SwiGLU+Q8_1 kernel failed";

    // Validate: dequantize Q8_1 blocks and compare against CPU SwiGLU reference
    std::vector<block_q8_1> h_q8(n_blocks);
    cudaMemcpy(h_q8.data(), d_q8_out, n_blocks * sizeof(block_q8_1), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int b = 0; b < n_blocks; b++) {
        float scale = __half2float(h_q8[b].d);
        for (int i = 0; i < 32; i++) {
            float deq = scale * h_q8[b].qs[i];
            float ref = act_ref[b * 32 + i];
            max_err = fmaxf(max_err, fabsf(deq - ref));
        }
    }
    // Q8_1 quantization introduces ~1-2% error
    EXPECT_LT(max_err, 0.15f) << "SwiGLU+Q8_1 dequant max error too large";

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_q8_out);
    cudaFree(d_d8);
}

}  // namespace
}  // namespace imp
