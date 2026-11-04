// Streamed FP16 FP8 row-scale GEMM (gemv_fp8_rowscale_f16.cu): bit identity against the split-K twin and
// the isolated sweep behind its dispatch (docs/audit/fp8_proj_stream_2026_11_04/).
#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace imp {
namespace {

class FP8ProjStreamTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

constexpr int kProjVariants[] = {0, 2};  // default dispatch, streamed kernel only

// Streamed FP16 kernel (variant 2) and the default dispatch (0) vs the split-K kernels (1): every output bit
// equal, Flash-Next GDN / attention and drafter shapes, a ragged M, every pass split (1..40 rows); K 2304
// (K % 512 != 0) is refused by the streamed kernel and falls back.
TEST_F(FP8ProjStreamTest, StreamedMatchesSplitK) {
    constexpr int NMAX = 40;
    const std::pair<int, int> shapes[] = {
        {2560, 6144}, {10240, 2560}, {516, 2560}, {4100, 4096}, {1024, 2304}};
    for (auto [M, K] : shapes) {
        std::vector<half> w(static_cast<size_t>(M) * K), x(static_cast<size_t>(NMAX) * K);
        uint32_t s = 7u + static_cast<uint32_t>(M);
        auto rnd = [&s] {
            s = s * 1664525u + 1013904223u;
            return static_cast<float>(s >> 8) / 16777216.0f - 0.5f;
        };
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = __float2half(0.02f * static_cast<float>(1 + (i / K) % 9) * rnd());
        for (auto& v : x)
            v = __float2half(4.0f * rnd());
        void *d_w = nullptr, *d_q = nullptr, *d_x = nullptr, *d_a = nullptr, *d_b = nullptr;
        float* d_s = nullptr;
        const size_t y_bytes = static_cast<size_t>(NMAX) * M * sizeof(half);
        ASSERT_EQ(cudaMalloc(&d_w, w.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_q, w.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_a, y_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_b, y_bytes), cudaSuccess);
        cudaMemcpy(d_w, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
        quantize_fp8_rows_async(d_w, d_q, M, K, d_s, stream_);
        const half* hx = static_cast<const half*>(d_x);
        std::vector<uint16_t> a(static_cast<size_t>(NMAX) * M), b(a.size());
        for (int n : {1, 2, 4, 8, 12, 16, 17, 29, 32, 40}) {
            cudaMemset(d_a, 0xff, y_bytes);
            ASSERT_TRUE(
                gemv_fp8_rowscale_f16_tuned(d_q, d_s, hx, static_cast<half*>(d_a), M, K, n, 1, stream_));
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            cudaMemcpy(a.data(), d_a, static_cast<size_t>(n) * M * sizeof(half), cudaMemcpyDeviceToHost);
            for (int v : kProjVariants) {
                cudaMemset(d_b, 0xff, y_bytes);  // NaN fill: an unwritten output fails below
                if (v == 2 && K % 512 != 0) {
                    EXPECT_FALSE(gemv_fp8_rowscale_f16_tuned(d_q, d_s, hx, static_cast<half*>(d_b), M, K, n,
                                                             v, stream_));
                    continue;
                }
                ASSERT_TRUE(
                    gemv_fp8_rowscale_f16_tuned(d_q, d_s, hx, static_cast<half*>(d_b), M, K, n, v, stream_))
                    << "variant " << v;
                ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
                cudaMemcpy(b.data(), d_b, static_cast<size_t>(n) * M * sizeof(half), cudaMemcpyDeviceToHost);
                size_t bad = 0;
                for (size_t i = 0; i < static_cast<size_t>(n) * M; ++i)
                    bad += a[i] != b[i];
                EXPECT_EQ(bad, 0u) << "M=" << M << " K=" << K << " n=" << n << " variant " << v;
            }
        }
        for (void* p : {d_w, d_q, d_x, d_a, d_b})
            cudaFree(p);
        cudaFree(d_s);
    }
}

// FP16 projection sweep (IMP_FP8_PROJ_SWEEP=1): us/call per variant over a rotating weight pool > L2
// (decode reads each projection once per step), split-K (variant 1) vs the default dispatch (0).
TEST_F(FP8ProjStreamTest, Sweep) {
    if (getenv("IMP_FP8_PROJ_SWEEP") == nullptr)
        GTEST_SKIP() << "set IMP_FP8_PROJ_SWEEP=1 to run the projection sweep";
    const int variants[] = {1, 0};
    // Flash-Next GDN in / gate / out, attention q / o, DSpark drafter (Qwen3-8B) q / gate-up / down / lm_head
    const std::pair<int, int> shapes[] = {{10240, 2560}, {6144, 2560},  {2560, 6144},  {12288, 2560},
                                          {4096, 4096},  {24576, 4096}, {4096, 12288}, {151936, 4096}};
    constexpr size_t kPoolBytes = size_t{320} << 20;  // > 96 MB L2 by 3.3x
    constexpr int kMaxRows = 32;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (auto [M, K] : shapes) {
        const size_t wb = static_cast<size_t>(M) * K;
        const int copies = static_cast<int>((kPoolBytes + wb - 1) / wb);
        void* d_pool = nullptr;
        void *d_w = nullptr, *d_x = nullptr, *d_y = nullptr;
        float* d_s = nullptr;
        ASSERT_EQ(cudaMalloc(&d_pool, wb * copies), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_w, wb * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(kMaxRows) * K * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(kMaxRows) * M * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
        cudaMemset(d_w, 0x2a, wb * sizeof(half));
        cudaMemset(d_x, 0x2a, static_cast<size_t>(kMaxRows) * K * sizeof(half));
        auto* pool = static_cast<uint8_t*>(d_pool);
        quantize_fp8_rows_async(d_w, pool, M, K, d_s, stream_);
        for (int c = 1; c < copies; ++c)
            cudaMemcpyAsync(pool + (wb * c), pool, wb, cudaMemcpyDeviceToDevice, stream_);
        cudaFree(d_w);
        const half* hx = static_cast<const half*>(d_x);
        auto* hy = static_cast<half*>(d_y);
        for (int n : {2, 4, 8, 12, 16, 32}) {
            int it = 0;
            auto run = [&](int v) {
                return gemv_fp8_rowscale_f16_tuned(pool + (wb * (it++ % copies)), d_s, hx, hy, M, K, n, v,
                                                   stream_);
            };
            for (int i = 0; i < 1000; ++i)  // clock ramp
                for (int v : variants)
                    ASSERT_TRUE(run(v)) << "variant " << v;
            // Variants alternate per rep: a host-side stall hits both, best of 11 drops it.
            double best[2] = {1e30, 1e30};
            for (int r = 0; r < 11; ++r)
                for (int vi = 0; vi < 2; ++vi) {
                    float ms = 0.0f;
                    cudaEventRecord(t0, stream_);
                    for (int i = 0; i < 200; ++i)
                        (void)run(variants[vi]);  // timing loop
                    cudaEventRecord(t1, stream_);
                    ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
                    cudaEventElapsedTime(&ms, t0, t1);
                    best[vi] = std::min(best[vi], ms * 1000.0 / 200);
                }
            for (int vi = 0; vi < 2; ++vi)
                printf("[ PROJ     ] M=%d K=%d rows=%d variant=%d: %.2f us/call, %.0f GB/s\n", M, K, n,
                       variants[vi], best[vi], static_cast<double>(wb) / best[vi] / 1e3);
        }
        cudaFree(d_pool);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_s);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

}  // namespace
}  // namespace imp
