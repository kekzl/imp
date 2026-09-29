// dequant_gptq4 kernel vs gptq::dequant4_host (#2249): bit-exact FP16 on a Qwen-shaped projection.

#include "quant/dequant_gptq.h"
#include "test_cuda_skip.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace imp {
namespace {

TEST(DequantGptqGpu, KernelBitEqualToHostReference) {
    SKIP_IF_NO_CUDA();
    // K=896 N=4864 group 128: Qwen2.5-0.5B gate_proj, 7 groups; v1 and v2, with and without g_idx.
    const int K = 896, N = 4864, G = 128, groups = K / G;
    std::mt19937 rng(2249);
    std::vector<int32_t> qweight(static_cast<size_t>(K / 8) * N), qzeros(static_cast<size_t>(groups) * N / 8);
    for (auto& v : qweight)
        v = static_cast<int32_t>(rng());
    for (auto& v : qzeros)
        v = static_cast<int32_t>(rng());
    std::uniform_real_distribution<float> sd(-0.05f, 0.05f);
    std::vector<__half> scales(static_cast<size_t>(groups) * N);
    for (auto& s : scales)
        s = __float2half_rn(sd(rng));
    std::vector<int32_t> g_idx(K);
    for (int k = 0; k < K; ++k)
        g_idx[k] = static_cast<int32_t>(rng() % groups);

    int32_t *d_qw = nullptr, *d_qz = nullptr, *d_gi = nullptr;
    half *d_sc = nullptr, *d_out = nullptr;
    const size_t out_n = static_cast<size_t>(N) * K;
    ASSERT_EQ(cudaMalloc(&d_qw, qweight.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_qz, qzeros.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_gi, g_idx.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, scales.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, out_n * 2), cudaSuccess);
    cudaMemcpy(d_qw, qweight.data(), qweight.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, qzeros.data(), qzeros.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gi, g_idx.data(), g_idx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice);

    for (gptq::ZeroFormat fmt : {gptq::ZeroFormat::V1, gptq::ZeroFormat::V2})
        for (bool use_gidx : {false, true}) {
            std::vector<__half> want(out_n), got(out_n);
            gptq::dequant4_host(want.data(), qweight.data(), qzeros.data(), scales.data(),
                                use_gidx ? g_idx.data() : nullptr, N, K, G, fmt);
            dequant_gptq4(d_out, d_qw, d_qz, d_sc, use_gidx ? d_gi : nullptr, N, K, G, fmt, nullptr);
            ASSERT_EQ(cudaMemcpy(got.data(), d_out, out_n * 2, cudaMemcpyDeviceToHost), cudaSuccess);
            size_t mismatches = 0;
            for (size_t i = 0; i < out_n; ++i)
                mismatches += std::memcmp(&got[i], &want[i], sizeof(__half)) != 0;
            EXPECT_EQ(mismatches, 0u) << "fmt " << static_cast<int>(fmt) << " g_idx " << use_gidx;
        }
    cudaFree(d_qw);
    cudaFree(d_qz);
    cudaFree(d_gi);
    cudaFree(d_sc);
    cudaFree(d_out);
}

}  // namespace
}  // namespace imp
