// gemm_mxfp4_f16_tile (fused MXFP4 W4A16 prompt GEMM, ROADMAP row 123) against dequant_mxfp4_linear_to_fp16
// + gemm_f16_tile: same fp16 W bits, same per-element k order, so the outputs must match bit for bit.
// Plus a repeat test and a timing table on Qwen3.5-4B shapes (informational).

#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_mxfp4_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "quant/mxfp4_sfatom.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

using namespace imp;

namespace {

__global__ void fill_act_kernel(half* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f);
    }
}

struct Mx4Bufs {
    int M = 0, N = 0, K = 0;
    uint8_t *w = nullptr, *sc = nullptr;
    half *a = nullptr, *w16 = nullptr, *ref = nullptr, *own = nullptr;
    // Any E2M1 bytes; UE8M0 exponents 2^-16 .. 2^1, varied per 32-k group.
    void alloc(int m, int n, int k, cudaStream_t s) {
        M = m;
        N = n;
        K = k;
        std::mt19937 rng(0x4u + n + k);
        std::vector<uint8_t> hw(size_t(N) * (K / 2)), hs(size_t(N) * (K / 32));
        for (auto& b : hw)
            b = static_cast<uint8_t>(rng());
        for (auto& e : hs)
            e = static_cast<uint8_t>(111 + (rng() % 18));
        cudaMalloc(&w, hw.size());
        cudaMalloc(&sc, hs.size());
        cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(sc, hs.data(), hs.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&a, size_t(M) * K * 2);
        cudaMalloc(&w16, size_t(N) * K * 2);
        cudaMalloc(&ref, size_t(M) * N * 2);
        cudaMalloc(&own, size_t(M) * N * 2);
        fill_act_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u);
        fill_act_kernel<<<1024, 256, 0, s>>>(ref, size_t(M) * N, 0x99u);
        cudaMemcpyAsync(own, ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s);
        dequant_mxfp4_linear_to_fp16(w, sc, N, K, w16, s);
    }
    void release() const {
        cudaFree(w);
        cudaFree(sc);
        cudaFree(a);
        cudaFree(w16);
        cudaFree(ref);
        cudaFree(own);
    }
};

size_t count_diff(const half* x, const half* y, size_t n, size_t* nonzero) {
    std::vector<uint16_t> a(n), b(n);
    cudaMemcpy(a.data(), x, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), y, n * 2, cudaMemcpyDeviceToHost);
    size_t d = 0, nz = 0;
    for (size_t i = 0; i < n; ++i) {
        d += a[i] != b[i] ? 1 : 0;
        nz += (a[i] & 0x7fff) != 0 ? 1 : 0;
    }
    if (nonzero != nullptr)
        *nonzero = nz;
    return d;
}

class GemmMxfp4TileTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool, float, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

class GemmMxfp4RepeatTest : public ::testing::TestWithParam<int> {};

}  // namespace

TEST_P(GemmMxfp4TileTest, BitEqualToDequantPlusF16Tile) {
    const auto [M, N, K, acc16, beta, variant] = GetParam();
    Mx4Bufs b;
    b.alloc(M, N, K, s_);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_TRUE(gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    size_t nonzero = 0;
    const size_t diff = count_diff(b.ref, b.own, size_t(M) * N, &nonzero);
    std::fprintf(stderr, "[mxfp4 tile] M=%d N=%d K=%d acc16=%d beta=%.0f v%d: %zu differ, %zu nonzero\n", M,
                 N, K, acc16 ? 1 : 0, beta, variant, diff, nonzero);
    EXPECT_EQ(diff, 0u);
    EXPECT_GT(nonzero, size_t(M) * N / 2);
    b.release();
}

// variant: 1 128x128, 2 256x128 (fp16 accumulate; every tile sums k in the same order); fp32: one config.
INSTANTIATE_TEST_SUITE_P(Shapes, GemmMxfp4TileTest,
                         ::testing::Values(std::make_tuple(128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 1),
                                           std::make_tuple(512, 1024, 2560, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 2),
                                           std::make_tuple(512, 2048, 2560, true, 0.0f, 2),
                                           std::make_tuple(257, 520, 2048, false, 0.0f, -1),
                                           std::make_tuple(130, 264, 512, false, 1.0f, -1)));

// Repeat runs on a multi-wave shape stay bit-identical to the reference (the slot-release race of #2913).
TEST_P(GemmMxfp4RepeatTest, BitIdenticalAcrossRuns) {
    const int variant = GetParam();
    cudaStream_t s;
    cudaStreamCreate(&s);
    Mx4Bufs b;
    b.alloc(1536, 9216, 2560, s);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
    int bad = 0;
    for (int it = 0; it < 10; ++it) {
        ASSERT_TRUE(gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        bad += count_diff(b.ref, b.own, size_t(b.M) * b.N, nullptr) != 0 ? 1 : 0;
    }
    std::fprintf(stderr, "[mxfp4 tile] repeat 1536x9216x2560 v%d: %d / 10 runs differ\n", variant, bad);
    EXPECT_EQ(bad, 0);
    b.release();
    cudaStreamDestroy(s);
}

INSTANTIATE_TEST_SUITE_P(Variants, GemmMxfp4RepeatTest, ::testing::Values(1, 2));

// Timing: fused vs dequant + gemm() (the W4A16 scratch path of executor_gemm_dispatch.cpp).
TEST(GemmMxfp4TileBench, VsDequantPlusGemm) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    process_diag_set_cublas_fp16_acc(true);
    const int shapes[][2] = {{9216, 2560}, {2560, 9216}, {2560, 2560}, {1024, 2560}};
    std::fprintf(stderr, "%6s %6s %6s %10s %9s %9s %9s\n", "M", "N", "K", "dq+gemm_us", "dq_us", "128x128",
                 "256x128");
    for (const auto& nk : shapes)
        for (int M : {512, 2048, 4096}) {
            Mx4Bufs b;
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
            const double t_dq = time_us([&] { dequant_mxfp4_linear_to_fp16(b.w, b.sc, b.N, b.K, b.w16, s); });
            const double t_lib = time_us([&] {
                dequant_mxfp4_linear_to_fp16(b.w, b.sc, b.N, b.K, b.w16, s);
                gemm(ta, tw, td, 1.0f, 0.0f, s);
            });
            const double t_128 = time_us([&] {
                (void)gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, M, b.N, b.K, 1.0f, 0.0f, true, s, 1);
            });
            const double t_256 = time_us([&] {
                (void)gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, M, b.N, b.K, 1.0f, 0.0f, true, s, 2);
            });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            std::fprintf(stderr, "%6d %6d %6d %10.1f %9.1f %9.1f %9.1f\n", M, b.N, b.K, t_lib, t_dq, t_128,
                         t_256);
            b.release();
        }
    process_diag_set_cublas_fp16_acc(false);
    cudaStreamDestroy(s);
}
