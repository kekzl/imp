// Timing cases of tests/test_mmq_q8_imma_tail.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "scoped_engine_arena.h"

#include "test_mmq_q8_imma_tail_util.cuh"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the IMMA activation scratch

namespace {

// #2416 bench: dense Q8_0 IMMA at Qwen3-8B q_o (N = K = 4096) around the 3-wave boundary (480 / 512 /
// 1024 tiles on 170 SMs). Two activation buffers alternate so every call pays the memoized activation
// quantize. Prints us per GEMM; no speed assert.
TEST(MmqQ8Imma, LongPromptBench) {
    const int N = 4096, K = 4096;
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, 5);
    uint8_t* d_w = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    for (const int M : {1920, 2048, 4096}) {
        std::vector<__half> x((size_t)M * K);
        std::mt19937 rng(M);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : x)
            v = __float2half(nd(rng));
        __half *d_x[2] = {nullptr, nullptr}, *d_out = nullptr;
        for (auto& p : d_x) {
            ASSERT_EQ(cudaMalloc(&p, x.size() * sizeof(__half)), cudaSuccess);
            cudaMemcpy(p, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
        }
        ASSERT_EQ(cudaMalloc(&d_out, (size_t)M * N * sizeof(__half)), cudaSuccess);
        for (int i = 0; i < 20; i++)
            ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x[i & 1], d_out, M, N, K, nullptr, 0.0f, false));
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        std::vector<float> ms(7);
        for (auto& m : ms) {
            cudaEventRecord(t0);
            for (int i = 0; i < 20; i++)
                (void)mmq_q8_imma_gemm(d_w, d_x[i & 1], d_out, M, N, K, nullptr, 0.0f, false);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            cudaEventElapsedTime(&m, t0, t1);
        }
        std::sort(ms.begin(), ms.end());
        printf("Q8LongPrompt M=%d N=%d K=%d: %.1f us per GEMM incl. act quantize (median of 7 x 20)\n", M, N,
               K, 1000.0 * ms[3] / 20);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        for (auto* p : d_x)
            cudaFree(p);
        cudaFree(d_out);
    }
    cudaFree(d_w);
    mmq_q8_imma_release_all();
}

}  // namespace
}  // namespace imp
