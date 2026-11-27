// Streamed FP16 FP8 row-scale GEMM (gemv_fp8_rowscale_f16.cu): bit identity against the split-K twin and
// the isolated sweep behind its dispatch
// (https://github.com/kekzl/imp/tree/d53d594b/docs/audit/fp8_proj_stream_2026_11_04).
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
#include "test_fp8_proj_stream_util.cuh"

namespace imp {
namespace {

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

}  // namespace
}  // namespace imp
