// dequant_awq4 kernel vs awq::dequant4_host (#2205): bit-exact FP16 on a Qwen-shaped projection.

#include "quant/dequant_awq.h"
#include "test_cuda_skip.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace imp {
namespace {

TEST(DequantAwqGpu, KernelBitEqualToHostReference) {
    SKIP_IF_NO_CUDA();
    // K=896 N=4864 group 128: Qwen2.5-0.5B gate_proj, 7 groups, grid not a multiple of the block.
    const int K = 896, N = 4864, G = 128, groups = K / G;
    std::mt19937 rng(2205);
    std::vector<int32_t> qweight(static_cast<size_t>(K) * N / 8), qzeros(static_cast<size_t>(groups) * N / 8);
    for (auto& v : qweight)
        v = static_cast<int32_t>(rng());
    for (auto& v : qzeros)
        v = static_cast<int32_t>(rng());
    std::uniform_real_distribution<float> sd(-0.05f, 0.05f);
    std::vector<__half> scales(static_cast<size_t>(groups) * N);
    for (auto& s : scales)
        s = __float2half_rn(sd(rng));

    std::vector<__half> want(static_cast<size_t>(N) * K);
    awq::dequant4_host(want.data(), qweight.data(), qzeros.data(), scales.data(), N, K, G);

    int32_t *d_qw = nullptr, *d_qz = nullptr;
    half *d_sc = nullptr, *d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_qw, qweight.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_qz, qzeros.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, scales.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, want.size() * 2), cudaSuccess);
    cudaMemcpy(d_qw, qweight.data(), qweight.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, qzeros.data(), qzeros.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice);
    dequant_awq4(d_out, d_qw, d_qz, d_sc, N, K, G, nullptr);
    std::vector<__half> got(want.size());
    ASSERT_EQ(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_qw);
    cudaFree(d_qz);
    cudaFree(d_sc);
    cudaFree(d_out);

    size_t mismatches = 0;
    for (size_t i = 0; i < want.size(); ++i)
        mismatches += std::memcmp(&got[i], &want[i], sizeof(__half)) != 0;
    EXPECT_EQ(mismatches, 0u);
}

}  // namespace
}  // namespace imp
