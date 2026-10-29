// gemm_q45k_f16_tile (fused Q4_K / Q5_K prompt GEMM, ROADMAP row 123) against dequant_gpu + gemm_f16_tile on
// the same tile config: the decode is the dequant_q45k_v2 math, so the outputs must match bit for bit.
// Plus an L2-flushed timing table vs the raw IMMA tiles on Qwen3.8-27B shapes (informational).

#include "compute/gemm_f16_tile.h"
#include "compute/gemm_q45k_f16_tile.h"
#include "compute/mmq_q8_imma.h"
#include "core/process_diag.h"
#include "core/qtype.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;

namespace {

// Random Q4_K / Q5_K blocks: any scale/qh/qs bytes, d and dmin finite fp16 in [1e-3, 4e-3) (the min term
// varies per sub-block, so a dropped dmin or swapped scale/min nibble shows).
std::vector<uint8_t> random_kq(bool q5, int N, int K, uint32_t seed) {
    const int bytes = q5 ? 176 : 144;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dd(1e-3f, 4e-3f);
    const size_t blocks = static_cast<size_t>(N) * (K / 256);
    std::vector<uint8_t> w(blocks * bytes);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = w.data() + (b * bytes);
        for (int i = 4; i < bytes; ++i)
            p[i] = static_cast<uint8_t>(rng());
        const half d = __float2half(dd(rng));
        const half m = __float2half(dd(rng));
        std::memcpy(p, &d, 2);
        std::memcpy(p + 2, &m, 2);
    }
    return w;
}

__global__ void fill_act_kernel(half* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f);
    }
}

struct KqBufs {
    int M = 0, N = 0, K = 0;
    bool q5 = false;
    uint8_t* wq = nullptr;
    half *a = nullptr, *w16 = nullptr, *ref = nullptr, *own = nullptr;
    void alloc(bool q5k, int m, int n, int k, cudaStream_t s) {
        q5 = q5k;
        M = m;
        N = n;
        K = k;
        const auto host = random_kq(q5, N, K, 0x45u + N + K + (q5 ? 1 : 0));
        cudaMalloc(&wq, host.size());
        cudaMemcpy(wq, host.data(), host.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&a, size_t(M) * K * 2);
        cudaMalloc(&w16, size_t(N) * K * 2);
        cudaMalloc(&ref, size_t(M) * N * 2);
        cudaMalloc(&own, size_t(M) * N * 2);
        fill_act_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u);
        fill_act_kernel<<<1024, 256, 0, s>>>(ref, size_t(M) * N, 0x99u);
        cudaMemcpyAsync(own, ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s);
    }
    QType qtype() const { return q5 ? QType::Q5_K : QType::Q4_K; }
    void release() const {
        cudaFree(wq);
        cudaFree(a);
        cudaFree(w16);
        cudaFree(ref);
        cudaFree(own);
    }
};

size_t count_diff(const KqBufs& b, size_t* nonzero) {
    std::vector<uint16_t> r(size_t(b.M) * b.N), o(r.size());
    cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
    size_t diff = 0;
    *nonzero = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        diff += r[i] != o[i] ? 1 : 0;
        *nonzero += (r[i] & 0x7fff) != 0 ? 1 : 0;
    }
    return diff;
}

class GemmQ45kTileTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int, bool, float, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

TEST_P(GemmQ45kTileTest, BitEqualToDequantPlusF16Tile) {
    const auto [q5, M, N, K, acc16, beta, variant] = GetParam();
    KqBufs b;
    b.alloc(q5, M, N, K, s_);
    dequant_gpu(b.wq, b.w16, b.qtype(), N, K, s_);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_TRUE(gemm_q45k_f16_tile(q5, b.a, b.wq, b.own, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    size_t nonzero = 0;
    const size_t diff = count_diff(b, &nonzero);
    std::fprintf(stderr,
                 "[q45k tile] %s M=%d N=%d K=%d acc16=%d beta=%.0f v%d: %zu / %d differ, %zu nonzero\n",
                 q5 ? "Q5_K" : "Q4_K", M, N, K, acc16 ? 1 : 0, beta, variant, diff, M * N, nonzero);
    EXPECT_EQ(diff, 0u);
    EXPECT_GT(nonzero, size_t(M) * N / 2);
    b.release();
}

// variant: 1 128x128, 2 256x128 (fp16 accumulate, same k order in every tile); fp32 accumulate has one
// config.
INSTANTIATE_TEST_SUITE_P(Shapes, GemmQ45kTileTest,
                         ::testing::Values(std::make_tuple(true, 128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(true, 200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(true, 300, 1032, 1024, true, 1.0f, 1),
                                           std::make_tuple(true, 512, 1024, 5120, true, 0.0f, 2),
                                           std::make_tuple(true, 257, 520, 2048, false, 0.0f, -1),
                                           std::make_tuple(false, 128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(false, 200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(false, 300, 1032, 1024, true, 1.0f, 2),
                                           std::make_tuple(false, 512, 1024, 5120, true, 0.0f, 1),
                                           std::make_tuple(false, 130, 264, 512, false, 1.0f, -1)));

// Repeat runs on a multi-wave shape stay bit-identical (the skeleton's release fence, #2913).
TEST(GemmQ45kRepeat, BitIdenticalAcrossRuns) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (int variant : {1, 2}) {
        KqBufs b;
        b.alloc(true, 1536, 5120, 5120, s);
        dequant_gpu(b.wq, b.w16, QType::Q5_K, b.N, b.K, s);
        ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
        int bad = 0;
        for (int it = 0; it < 10; ++it) {
            ASSERT_TRUE(
                gemm_q45k_f16_tile(true, b.a, b.wq, b.own, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            size_t nz = 0;
            bad += count_diff(b, &nz) != 0 ? 1 : 0;
        }
        std::fprintf(stderr, "[q45k tile] repeat 1536x5120x5120 v%d: %d / 10 runs differ\n", variant, bad);
        EXPECT_EQ(bad, 0);
        b.release();
    }
    cudaStreamDestroy(s);
}

// Timing with a 256 MiB memset before each call (W of 17408x5120 fits the 96 MB L2 otherwise): fused
// 128x128 / 256x128 vs the raw IMMA tile the uncached route runs at M > 32 (activation quantize included).
TEST(GemmQ45kTileBench, FlushedVsImma) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    void* flush = nullptr;
    const size_t fbytes = size_t(256) << 20;
    cudaMalloc(&flush, fbytes);
    auto time_us = [&](auto&& fn) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::vector<float> v;
        for (int i = 0; i < 25; ++i) {
            cudaMemsetAsync(flush, i, fbytes, s);
            cudaEventRecord(e0, s);
            fn();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (i >= 5)
                v.push_back(1000.0f * ms);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {6144, 5120},
                             {10240, 5120}, {5120, 6144},  {1024, 5120}};
    std::fprintf(stderr, "%4s %5s %6s %6s %9s %9s %9s %9s %9s\n", "fmt", "M", "N", "K", "imma_us", "128x128",
                 "256x128", "auto", "fp32acc");
    for (bool q5 : {true, false})
        for (const auto& nk : shapes)
            for (int M : {128, 512, 1024, 2048}) {
                KqBufs b;
                b.alloc(q5, M, nk[0], nk[1], s);
                const float t_imma = time_us([&] {
                    (void)(q5 ? mmq_gguf_imma_smallm_gemm(b.wq, QType::Q5_K, b.a, b.own, M, b.N, b.K, s)
                              : mmq_q4k_imma_gemm(b.wq, b.a, b.own, M, b.N, b.K, s));
                });
                float t[4] = {};
                for (int v = 0; v < 4; ++v)
                    t[v] = time_us([&] {
                        (void)gemm_q45k_f16_tile(q5, b.a, b.wq, b.own, M, b.N, b.K, 1.0f, 0.0f, v < 3, s,
                                                 v == 0 ? 1 : (v == 1 ? 2 : -1));
                    });
                ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
                std::fprintf(stderr, "%4s %5d %6d %6d %9.1f %9.1f %9.1f %9.1f %9.1f\n", q5 ? "Q5_K" : "Q4_K",
                             M, b.N, b.K, t_imma, t[0], t[1], t[2], t[3]);
                b.release();
            }
    cudaFree(flush);
    cudaStreamDestroy(s);
}
