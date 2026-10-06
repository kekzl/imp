// #2416: dense Q8_0 IMMA wave tail (BM=32 for a last wave under half full): per-row bit identity
// across M, and the q_o long-prompt bench. Own file: test_mmq_q8_imma.cu sits at the function-size gate.
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

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the IMMA activation scratch
namespace {

constexpr int kBlk = 32;
constexpr int kBlockBytes = 34;  // half d + 32 s8

void gen_q8_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    ASSERT_EQ(K % kBlk, 0);
    const int bpr = K / kBlk;
    W.resize(static_cast<size_t>(N) * bpr * kBlockBytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    for (size_t b = 0; b < W.size() / kBlockBytes; ++b) {
        uint8_t* bp = W.data() + b * kBlockBytes;
        __half d = __float2half(dd(rng));
        std::memcpy(bp, &d, 2);
        for (int i = 0; i < 32; ++i)
            bp[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(qd(rng)));
    }
}

// #2416 wave tail: at N=4096 (32 column tiles) M=2048 runs rows 1920..2047 as BM=32 tiles (512 tiles =
// 3 x 170 + 2), M=4096 runs them inside the BM=128 grid. A prompt row must keep its bits (#2152).
TEST(MmqQ8Imma, WaveTailRowsBitIdenticalAcrossM) {
    const int N = 4096, K = 1024, M_small = 2048, M_big = 4096;
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, 77);
    std::vector<__half> x((size_t)M_big * K);
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x)
        v = __float2half(nd(rng));
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_xs = nullptr, *d_a = nullptr, *d_b = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_xs, (size_t)M_small * K * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_a, (size_t)M_small * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, (size_t)M_big * N * sizeof(__half)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, x.data(), (size_t)M_small * K * sizeof(__half), cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_xs, d_a, M_small, N, K, nullptr, 0.0f, false));
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_b, M_big, N, K, nullptr, 0.0f, false));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint16_t> a((size_t)M_small * N), b((size_t)M_small * N);
    cudaMemcpy(a.data(), d_a, a.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, b.size() * 2, cudaMemcpyDeviceToHost);  // first M_small rows
    size_t diff_tail = 0, diff_head = 0;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i])
            (i >= (size_t)1920 * N ? diff_tail : diff_head)++;
    EXPECT_EQ(diff_head, 0u) << "BM=128 rows differ between M=2048 and M=4096";
    EXPECT_EQ(diff_tail, 0u) << "tail rows 1920..2047 (BM=32 at M=2048) differ from BM=128 at M=4096";
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_xs);
    cudaFree(d_a);
    cudaFree(d_b);
    mmq_q8_imma_release_all();
}

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
