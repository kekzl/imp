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

// #2416 128-K step: N=1024 M=128 is a dense small grid (8 tiles, 64-K BM=32 kernel), M=4096 runs the 128-K
// BM=128 kernel; first 128 rows must match bit for bit (#2152). beta 1 = __hadd2(C, beta-0 tile).
TEST(MmqQ8Imma, K128StepRowsBitIdenticalToK64) {
    const int N = 1024, K = 1024, M_small = 128, M_big = 4096;
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, 91);
    std::vector<__half> x((size_t)M_big * K), c((size_t)M_big * N);
    std::mt19937 rng(91);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x)
        v = __float2half(nd(rng));
    for (auto& v : c)
        v = __float2half(nd(rng));
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_a, (size_t)M_small * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, (size_t)M_big * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, c.size() * sizeof(__half)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), c.size() * sizeof(__half), cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_a, M_small, N, K, nullptr, 0.0f, false));
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_b, M_big, N, K, nullptr, 0.0f, false));
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_c, M_big, N, K, nullptr, 1.0f, false));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> a((size_t)M_small * N), b((size_t)M_big * N), cb((size_t)M_big * N);
    cudaMemcpy(a.data(), d_a, a.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, b.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(cb.data(), d_c, cb.size() * 2, cudaMemcpyDeviceToHost);
    size_t diff_rows = 0, diff_beta = 0;
    for (size_t i = 0; i < a.size(); i++)
        diff_rows += std::memcmp(&a[i], &b[i], 2) != 0;
    for (size_t i = 0; i < b.size(); i++) {
        const __half want = __hadd(c[i], b[i]);
        diff_beta += std::memcmp(&want, &cb[i], 2) != 0;
    }
    EXPECT_EQ(diff_rows, 0u) << "128-K BM=128 rows differ from the 64-K BM=32 kernel";
    EXPECT_EQ(diff_beta, 0u) << "beta 1 is not __hadd2(C, beta-0 result)";
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    mmq_q8_imma_release_all();
}

// #2416 beta 1 with the wave tail (o_proj shape, M=2048: 15 BM=128 row-blocks + BM=32 rows 1920..2047):
// every element equals __hadd(C, beta-0 result).
TEST(MmqQ8Imma, WaveTailBeta1IsHaddOfBeta0) {
    const int N = 4096, K = 1024, M = 2048;
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, 31);
    std::vector<__half> x((size_t)M * K), c((size_t)M * N);
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x)
        v = __float2half(nd(rng));
    for (auto& v : c)
        v = __float2half(nd(rng));
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_b = nullptr, *d_c = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, c.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, c.size() * sizeof(__half)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), c.size() * sizeof(__half), cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_b, M, N, K, nullptr, 0.0f, false));
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_c, M, N, K, nullptr, 1.0f, false));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> b(c.size()), cb(c.size());
    cudaMemcpy(b.data(), d_b, b.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(cb.data(), d_c, cb.size() * 2, cudaMemcpyDeviceToHost);
    size_t diff_head = 0, diff_tail = 0;
    for (size_t i = 0; i < c.size(); i++) {
        const __half want = __hadd(c[i], b[i]);
        if (std::memcmp(&want, &cb[i], 2) != 0)
            (i >= (size_t)1920 * N ? diff_tail : diff_head)++;
    }
    EXPECT_EQ(diff_head, 0u) << "BM=128 rows: beta 1 != C + beta-0";
    EXPECT_EQ(diff_tail, 0u) << "BM=32 tail rows: beta 1 != C + beta-0";
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_c);
    mmq_q8_imma_release_all();
}

// #2416 q/k/v quantize reuse: reuse_act skips only a quantize of the same (x, M, K) since the last
// mmq_imma_act_forget(). Same pointer, new content after a forget must re-quantize (a workspace reused
// across layers served layer-1 activations otherwise).
TEST(MmqQ8Imma, ReuseActHonoursForget) {
    const int N = 1024, K = 1024, M = 256;
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, 13);
    std::vector<__half> x1((size_t)M * K), x2((size_t)M * K);
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x1)
        v = __float2half(nd(rng));
    for (auto& v : x2)
        v = __float2half(nd(rng));
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_ref = nullptr, *d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x1.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ref, (size_t)M * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, (size_t)M * N * sizeof(__half)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    std::vector<uint16_t> ref((size_t)M * N), got((size_t)M * N);
    // reference: x2 quantized fresh
    cudaMemcpy(d_x, x2.data(), x2.size() * sizeof(__half), cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_ref, M, N, K, nullptr, 0.0f, false));
    // x1 quantized, then the same pointer overwritten with x2 behind a forget
    cudaMemcpy(d_x, x1.data(), x1.size() * sizeof(__half), cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr, 0.0f, false));
    cudaMemcpy(d_x, x2.data(), x2.size() * sizeof(__half), cudaMemcpyHostToDevice);
    mmq_imma_act_forget();
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr, 0.0f, false, /*reuse_act=*/true));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(ref.data(), d_ref, ref.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost);
    EXPECT_EQ(std::memcmp(ref.data(), got.data(), ref.size() * 2), 0) << "reuse after forget served stale x1";
    // a second call on unchanged x2 with reuse_act reads the quantize just written: same bits
    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr, 0.0f, false, /*reuse_act=*/true));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost);
    EXPECT_EQ(std::memcmp(ref.data(), got.data(), ref.size() * 2), 0) << "reuse on unchanged x differs";
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_ref);
    cudaFree(d_out);
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

// Short prompts: M=2/40/64 at N=4096 (32 BN=128 tiles, under one wave) run the BN=32 kernel; the same
// rows inside M=4096 run BN=128 / k128. Bit for bit, beta 0 and beta 1 (#2152).
TEST(MmqQ8Imma, NarrowTileShortPromptRowsBitIdentical) {
    const int N = 4096, K = 1024, M_big = 4096;
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, 91);
    std::vector<__half> x((size_t)M_big * K), c((size_t)M_big * N);
    std::mt19937 rng(91);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x)
        v = __float2half(nd(rng));
    for (auto& v : c)
        v = __float2half(nd(rng));
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_s = nullptr, *d_b = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, (size_t)64 * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, (size_t)M_big * N * sizeof(__half)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    for (float beta : {0.0f, 1.0f}) {
        cudaMemcpy(d_b, c.data(), c.size() * sizeof(__half), cudaMemcpyHostToDevice);
        ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_b, M_big, N, K, nullptr, beta, false));
        std::vector<uint16_t> big((size_t)64 * N);
        ASSERT_EQ(cudaMemcpy(big.data(), d_b, big.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
        for (int M : {2, 40, 64}) {
            cudaMemcpy(d_s, c.data(), (size_t)M * N * sizeof(__half), cudaMemcpyHostToDevice);
            ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_s, M, N, K, nullptr, beta, false));
            std::vector<uint16_t> small((size_t)M * N);
            ASSERT_EQ(cudaMemcpy(small.data(), d_s, small.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
            size_t diff = 0;
            for (size_t i = 0; i < small.size(); i++)
                diff += small[i] != big[i];
            EXPECT_EQ(diff, 0u) << "M=" << M << " beta=" << beta << " rows differ from the M=4096 grid";
        }
    }
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_s);
    cudaFree(d_b);
    mmq_q8_imma_release_all();
}

}  // namespace
}  // namespace imp
