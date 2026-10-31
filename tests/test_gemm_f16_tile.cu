// gemm_f16_tile (own FP16 GEMM, ROADMAP row 123) against cuBLASLt gemm() at the same accumulator
// precision, plus a timing table on Qwen3-14B / Qwen3.8 projection shapes (informational).

#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace imp;

namespace {

__global__ void fill_f16_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

struct Bufs {
    half *a = nullptr, *w = nullptr, *ref = nullptr, *own = nullptr;
    int M = 0, N = 0, K = 0;
    void alloc(int m, int n, int k, cudaStream_t s) {
        M = m;
        N = n;
        K = k;
        cudaMalloc(&a, size_t(M) * K * 2);
        cudaMalloc(&w, size_t(N) * K * 2);
        cudaMalloc(&ref, size_t(M) * N * 2);
        cudaMalloc(&own, size_t(M) * N * 2);
        fill_f16_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u, 1.0f);
        fill_f16_kernel<<<1024, 256, 0, s>>>(w, size_t(N) * K, 0x77u, 0.05f);
        fill_f16_kernel<<<1024, 256, 0, s>>>(ref, size_t(M) * N, 0x99u, 1.0f);
        cudaMemcpyAsync(own, ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s);
    }
    void release() {
        cudaFree(a);
        cudaFree(w);
        cudaFree(ref);
        cudaFree(own);
    }
    void lib(float beta, cudaStream_t s) const {
        int64_t as[2] = {M, K}, ws[2] = {N, K}, ds[2] = {M, N};
        Tensor ta(a, QType::F16, 2, as, true), tw(w, QType::F16, 2, ws, true),
            td(ref, QType::F16, 2, ds, true);
        gemm(ta, tw, td, 1.0f, beta, s);
    }
};

class GemmF16TileTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool, float>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override {
        process_diag_set_cublas_fp16_acc(false);
        cudaStreamDestroy(s_);
    }
    cudaStream_t s_ = nullptr;
};

}  // namespace

// fp32 accumulate: within fp16 rounding of cuBLASLt; fp16 accumulate: both sum in fp16, so the
// bound is the fp16 accumulation error over K (relative to the output scale).
TEST_P(GemmF16TileTest, MatchesCublasLt) {
    const auto [M, N, K, acc16, beta] = GetParam();
    process_diag_set_cublas_fp16_acc(acc16);
    Bufs b;
    b.alloc(M, N, K, s_);  // own == ref == the beta input here
    half* d_in = nullptr;
    cudaMalloc(&d_in, size_t(M) * N * 2);
    cudaMemcpyAsync(d_in, b.ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s_);
    b.lib(beta, s_);
    std::vector<half> r(size_t(M) * N), o(size_t(M) * N);
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
    for (int v = 0; v < (acc16 ? 2 : 1); ++v) {
        cudaMemcpyAsync(b.own, d_in, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s_);
        ASSERT_TRUE(gemm_f16_tile(b.a, b.w, b.own, M, N, K, 1.0f, beta, acc16, s_, v)) << "variant " << v;
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_diff = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < r.size(); ++i) {
            const double x = __half2float(r[i]), y = __half2float(o[i]);
            bad += std::isfinite(y) ? 0 : 1;
            max_ref = std::max(max_ref, std::fabs(x));
            max_diff = std::max(max_diff, std::fabs(x - y));
        }
        std::fprintf(stderr,
                     "[f16 tile v%d] M=%d N=%d K=%d acc16=%d beta=%.0f max|ref|=%.4g max|diff|=%.4g\n", v, M,
                     N, K, acc16 ? 1 : 0, beta, max_ref, max_diff);
        EXPECT_EQ(bad, 0u) << "variant " << v;
        EXPECT_LE(max_diff, (acc16 ? 1e-2 : 2e-3) * max_ref + 1e-3) << "variant " << v;
    }
    cudaFree(d_in);
    b.release();
}

INSTANTIATE_TEST_SUITE_P(Shapes, GemmF16TileTest,
                         ::testing::Values(std::make_tuple(128, 128, 64, false, 0.0f),
                                           std::make_tuple(200, 384, 512, false, 0.0f),
                                           std::make_tuple(200, 384, 512, false, 1.0f),
                                           std::make_tuple(512, 1024, 5120, false, 0.0f),
                                           std::make_tuple(512, 1024, 5120, true, 0.0f),
                                           std::make_tuple(1000, 2056, 1024, true, 1.0f)));

// gemm_f16_splitk (cluster split-K, fp32 accumulate) against cuBLASLt fp32 accumulate on MoE router shapes
// (Gemma-4 2048x128x2816, Qwen3-30B 2048x128x2048, Qwen3.6 4096x256x2048) and ragged edges; timing printed.
TEST(GemmF16SplitK, MatchesCublasLt) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][3] = {{2048, 128, 2816}, {2048, 128, 2048}, {4096, 256, 2048},
                             {200, 64, 192},    {33, 8, 64},       {1000, 136, 704}};
    for (const auto& sh : shapes) {
        Bufs b;
        b.alloc(sh[0], sh[1], sh[2], s);
        b.lib(0.0f, s);
        const size_t ws_bytes = gemm_f16_splitk_workspace_bytes(b.M, b.N);
        void* ws = nullptr;
        cudaMalloc(&ws, ws_bytes);
        ASSERT_TRUE(gemm_f16_splitk(b.a, b.w, b.own, b.M, b.N, b.K, 1.0f, ws, ws_bytes, s));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        std::vector<half> r(size_t(b.M) * b.N), o(size_t(b.M) * b.N);
        cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_diff = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < r.size(); ++i) {
            const double x = __half2float(r[i]), y = __half2float(o[i]);
            bad += std::isfinite(y) ? 0 : 1;
            max_ref = std::max(max_ref, std::fabs(x));
            max_diff = std::max(max_diff, std::fabs(x - y));
        }
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float t[2] = {0.0f, 0.0f};
        for (int k = 0; k < 2; ++k) {
            for (int i = 0; i < 60; ++i) {
                if (i == 20)
                    cudaEventRecord(e0, s);
                if (k == 0)
                    b.lib(0.0f, s);
                else
                    (void)gemm_f16_splitk(b.a, b.w, b.own, b.M, b.N, b.K, 1.0f, ws, ws_bytes, s);
            }
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t[k], e0, e1);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        std::fprintf(stderr,
                     "[f16 splitk] M=%d N=%d K=%d max|ref|=%.4g max|diff|=%.4g lt %.1f us own %.1f us\n", b.M,
                     b.N, b.K, max_ref, max_diff, 1000.0 * t[0] / 40, 1000.0 * t[1] / 40);
        EXPECT_EQ(bad, 0u);
        EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << b.M << "x" << b.N << "x" << b.K;
        cudaFree(ws);
        b.release();
    }
    cudaStreamDestroy(s);
}

// Timing vs cuBLASLt gemm() at fp16 accumulate (the default prefill route).
TEST(GemmF16TileBench, VsCublasLt) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    process_diag_set_cublas_fp16_acc(true);
    // {2688, 4096}: Nemotron-3-Nano Mamba out_proj (176 tiles at M 2048, one wave + 6 CTAs).
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {5120, 5120}, {1024, 5120}, {2688, 4096}};
    std::fprintf(stderr, "%6s %6s %6s %9s %9s %9s %9s %9s %8s\n", "M", "N", "K", "lt_us", "v0", "v1", "v2",
                 "v3", "best_TF");
    for (const auto& nk : shapes)
        for (int M : {512, 2048, 4096}) {
            Bufs b;
            b.alloc(M, nk[0], nk[1], s);
            auto time_us = [&](auto&& fn) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                for (int i = 0; i < 20; ++i)
                    fn();
                cudaEventRecord(e0, s);
                for (int i = 0; i < 40; ++i)
                    fn();
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms / 40;
            };
            const double t_lt = time_us([&] { b.lib(0.0f, s); });
            double t_v[2];
            for (int v = 0; v < 2; ++v)
                t_v[v] = time_us(
                    [&] { (void)gemm_f16_tile(b.a, b.w, b.own, b.M, b.N, b.K, 1.0f, 0.0f, true, s, v); });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const double best = *std::min_element(t_v, t_v + 2);
            std::fprintf(stderr, "%6d %6d %6d %9.1f %9.1f %9.1f %8.0f\n", b.M, b.N, b.K, t_lt, t_v[0], t_v[1],
                         2.0 * b.M * b.N * double(b.K) / (best * 1e-6) / 1e12);
            b.release();
        }
    process_diag_set_cublas_fp16_acc(false);
    cudaStreamDestroy(s);
}
