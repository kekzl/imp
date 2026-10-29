// gemm_q6k_f16_tile (fused Q6_K prompt GEMM, ROADMAP row 123) against dequant_gpu + gemm_f16_tile on the
// same tile config: the decode is the dequant_q6k_v2 math, so the outputs must match bit for bit.
// Plus a timing table vs dequant + cuBLASLt gemm() on Qwen3-14B shapes (informational).

#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_q6k_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;

namespace {

constexpr int kQ6 = 210;

// Random Q6_K blocks: any ql/qh/scale bytes, d a finite fp16 in [1e-3, 4e-3).
std::vector<uint8_t> random_q6k(int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dd(1e-3f, 4e-3f);
    const size_t blocks = static_cast<size_t>(N) * (K / 256);
    std::vector<uint8_t> w(blocks * kQ6);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = w.data() + (b * kQ6);
        for (int i = 0; i < 208; ++i)
            p[i] = static_cast<uint8_t>(rng());
        const half d = __float2half(dd(rng));
        std::memcpy(p + 208, &d, 2);
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

struct Q6Bufs {
    int M = 0, N = 0, K = 0;
    uint8_t* wq = nullptr;
    half *a = nullptr, *w16 = nullptr, *ref = nullptr, *own = nullptr;
    void alloc(int m, int n, int k, cudaStream_t s) {
        M = m;
        N = n;
        K = k;
        const auto host = random_q6k(N, K, 0x6u + N + K);
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
    void release() const {
        cudaFree(wq);
        cudaFree(a);
        cudaFree(w16);
        cudaFree(ref);
        cudaFree(own);
    }
};

class GemmQ6kRepeatTest : public ::testing::TestWithParam<int> {};

class GemmQ6kTileTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool, float, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

TEST_P(GemmQ6kTileTest, BitEqualToDequantPlusF16Tile) {
    const auto [M, N, K, acc16, beta, variant] = GetParam();
    Q6Bufs b;
    b.alloc(M, N, K, s_);
    dequant_gpu(b.wq, b.w16, QType::Q6_K, N, K, s_);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_TRUE(gemm_q6k_f16_tile(b.a, b.wq, b.own, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<uint16_t> r(size_t(M) * N), o(size_t(M) * N);
    cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
    size_t diff = 0, nonzero = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        diff += r[i] != o[i] ? 1 : 0;
        nonzero += (r[i] & 0x7fff) != 0 ? 1 : 0;
    }
    std::fprintf(stderr, "[q6k tile] M=%d N=%d K=%d acc16=%d beta=%.0f v%d: %zu / %zu differ, %zu nonzero\n",
                 M, N, K, acc16 ? 1 : 0, beta, variant, diff, r.size(), nonzero);
    EXPECT_EQ(diff, 0u);
    EXPECT_GT(nonzero, r.size() / 2);
    b.release();
}

// variant: 1 128x128, 2 256x128 (fp16 accumulate; every tile sums k in the same order, so the 128x256
// f16 tile is the reference for both); the fp32 accumulator has one config (variant ignored).
INSTANTIATE_TEST_SUITE_P(Shapes, GemmQ6kTileTest,
                         ::testing::Values(std::make_tuple(128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 1),
                                           std::make_tuple(512, 1024, 5120, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 2),
                                           std::make_tuple(512, 1024, 5120, true, 0.0f, 2),
                                           std::make_tuple(257, 520, 2048, false, 0.0f, -1),
                                           std::make_tuple(130, 264, 512, false, 1.0f, -1)));

// Repeat runs on a multi-wave shape stay bit-identical to the reference: without the release fence in the
// MMA loop the decoders overwrote a slot behind its last LDSM (19/20 runs differed, max |d| 106).
TEST_P(GemmQ6kRepeatTest, BitIdenticalAcrossRuns) {
    const int variant = GetParam();
    cudaStream_t s;
    cudaStreamCreate(&s);
    Q6Bufs b;
    b.alloc(1536, 5120, 5120, s);
    dequant_gpu(b.wq, b.w16, QType::Q6_K, b.N, b.K, s);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    std::vector<uint16_t> r(size_t(b.M) * b.N), o(r.size());
    cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int it = 0; it < 10; ++it) {
        ASSERT_TRUE(gemm_q6k_f16_tile(b.a, b.wq, b.own, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
        bad += r != o ? 1 : 0;
    }
    std::fprintf(stderr, "[q6k tile] repeat 1536x5120x5120 v%d: %d / 10 runs differ\n", variant, bad);
    EXPECT_EQ(bad, 0);
    b.release();
    cudaStreamDestroy(s);
}

INSTANTIATE_TEST_SUITE_P(Variants, GemmQ6kRepeatTest, ::testing::Values(1, 2));

// Timing: fused vs dequant + gemm() (cuBLASLt or the f16 tile route, as the prompt path runs today).
TEST(GemmQ6kTileBench, VsDequantPlusGemm) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    process_diag_set_cublas_fp16_acc(true);
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {5120, 5120}, {1024, 5120}};
    std::fprintf(stderr, "%6s %6s %6s %10s %9s %9s %9s\n", "M", "N", "K", "dq+gemm_us", "dq_us", "128x128",
                 "256x128");
    for (const auto& nk : shapes)
        for (int M : {512, 2048}) {
            Q6Bufs b;
            b.alloc(M, nk[0], nk[1], s);
            auto time_us = [&](auto&& fn) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                for (int i = 0; i < 10; ++i)
                    fn();
                cudaEventRecord(e0, s);
                for (int i = 0; i < 20; ++i)
                    fn();
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms / 20;
            };
            int64_t as[2] = {M, b.K}, ws[2] = {b.N, b.K}, ds[2] = {M, b.N};
            Tensor ta(b.a, QType::F16, 2, as, true), tw(b.w16, QType::F16, 2, ws, true),
                td(b.ref, QType::F16, 2, ds, true);
            const double t_dq = time_us([&] { dequant_gpu(b.wq, b.w16, QType::Q6_K, b.N, b.K, s); });
            const double t_lib = time_us([&] {
                dequant_gpu(b.wq, b.w16, QType::Q6_K, b.N, b.K, s);
                gemm(ta, tw, td, 1.0f, 0.0f, s);
            });
            const double t_128 = time_us(
                [&] { (void)gemm_q6k_f16_tile(b.a, b.wq, b.own, M, b.N, b.K, 1.0f, 0.0f, true, s, 1); });
            const double t_256 = time_us(
                [&] { (void)gemm_q6k_f16_tile(b.a, b.wq, b.own, M, b.N, b.K, 1.0f, 0.0f, true, s, 2); });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            std::fprintf(stderr, "%6d %6d %6d %10.1f %9.1f %9.1f %9.1f\n", M, b.N, b.K, t_lib, t_dq, t_128,
                         t_256);
            b.release();
        }
    process_diag_set_cublas_fp16_acc(false);
    cudaStreamDestroy(s);
}
