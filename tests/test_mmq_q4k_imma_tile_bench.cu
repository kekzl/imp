// Phase 2B: mmq_q4k_imma_tile(X_s8,x_scale,x_rowsum,W_s8,alpha,beta,out) vs FP32 reference
// GEMM over dequantized inputs, within INT8/FP16 quantization noise.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/mmq_q4k_imma_tile.h"
#include "test_mmq_q4k_imma_tile_util.cuh"

namespace imp {
namespace {

// CPU reference: out[m, n] = Σ_sub x_scale[m, sub] · (α[n, sub] · Σ X·W + β[n, sub] · Σ X)
// Equivalent to the kernel's math, computed at FP32 precision.
void cpu_reference(const std::vector<int8_t>& X, const std::vector<__half>& x_scale,
                   const std::vector<float>& x_rowsum, const std::vector<int8_t>& W,
                   const std::vector<__half>& alpha, const std::vector<__half>& beta,
                   std::vector<float>& out, int M, int N, int K) {
    const int subs = K / kSub;
    out.assign(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int s = 0; s < subs; ++s) {
                int32_t sumi = 0;
                for (int k = 0; k < kSub; ++k) {
                    sumi += static_cast<int32_t>(X[static_cast<size_t>(m) * K + s * kSub + k]) *
                            static_cast<int32_t>(W[static_cast<size_t>(n) * K + s * kSub + k]);
                }
                float a = __half2float(alpha[static_cast<size_t>(n) * subs + s]);
                float b = __half2float(beta[static_cast<size_t>(n) * subs + s]);
                float xs = __half2float(x_scale[static_cast<size_t>(m) * subs + s]);
                float xrs = x_rowsum[static_cast<size_t>(m) * subs + s];
                acc += xs * (a * static_cast<float>(sumi) + b * xrs);
            }
            out[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

void run_correctness(int M, int N, int K, unsigned seed, float max_abs_tol, float max_rel_tol) {
    SCOPED_TRACE(testing::Message() << "M=" << M << " N=" << N << " K=" << K << " seed=" << seed);
    ASSERT_EQ(M % 16, 0);
    ASSERT_EQ(N % 8, 0);
    ASSERT_EQ(K % kSub, 0);

    std::vector<int8_t> hW;
    std::vector<__half> halpha, hbeta;
    gen_random_w_sym_s8(hW, halpha, hbeta, N, K, seed);

    std::vector<int8_t> hX;
    std::vector<__half> hx_scale;
    std::vector<float> hx_rowsum;
    gen_random_x_s8(hX, hx_scale, hx_rowsum, M, K, seed + 1);

    std::vector<float> ref;
    cpu_reference(hX, hx_scale, hx_rowsum, hW, halpha, hbeta, ref, M, N, K);

    int8_t *dX = nullptr, *dW = nullptr;
    __half *dxscale = nullptr, *dalpha = nullptr, *dbeta = nullptr;
    float* dxrowsum = nullptr;
    __half* dout = nullptr;
    const int subs = K / kSub;
    cudaMalloc(&dX, hX.size());
    cudaMalloc(&dW, hW.size());
    cudaMalloc(&dxscale, hx_scale.size() * sizeof(__half));
    cudaMalloc(&dxrowsum, hx_rowsum.size() * sizeof(float));
    cudaMalloc(&dalpha, halpha.size() * sizeof(__half));
    cudaMalloc(&dbeta, hbeta.size() * sizeof(__half));
    cudaMalloc(&dout, static_cast<size_t>(M) * N * sizeof(__half));
    cudaMemcpy(dX, hX.data(), hX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dxscale, hx_scale.data(), hx_scale.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dxrowsum, hx_rowsum.data(), hx_rowsum.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, halpha.data(), halpha.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, hbeta.data(), hbeta.size() * sizeof(__half), cudaMemcpyHostToDevice);

    mmq_q4k_imma_tile(dX, dxscale, dxrowsum, dW, dalpha, dbeta, dout, M, N, K, nullptr);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<__half> hout(static_cast<size_t>(M) * N);
    cudaMemcpy(hout.data(), dout, hout.size() * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_abs = 0.0f, max_rel = 0.0f;
    int worst_m = 0, worst_n = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float got = __half2float(hout[static_cast<size_t>(m) * N + n]);
            float want = ref[static_cast<size_t>(m) * N + n];
            float ae = std::fabs(got - want);
            if (ae > max_abs) {
                max_abs = ae;
                worst_m = m;
                worst_n = n;
            }
            float re = std::fabs(want) > 1e-3f ? ae / std::fabs(want) : 0.0f;
            max_rel = std::max(max_rel, re);
        }
    }
    std::fprintf(stderr, "[q4k-imma-tile M=%d N=%d K=%d] max_abs=%.4f max_rel=%.4f at (%d,%d)\n", M, N, K,
                 max_abs, max_rel, worst_m, worst_n);
    EXPECT_LT(max_abs, max_abs_tol)
        << "absolute error > tol at (" << worst_m << ", " << worst_n << ")";
    EXPECT_LT(max_rel, max_rel_tol) << "relative error > tol";

    cudaFree(dX);
    cudaFree(dW);
    cudaFree(dxscale);
    cudaFree(dxrowsum);
    cudaFree(dalpha);
    cudaFree(dbeta);
    cudaFree(dout);
    (void)subs;
}

TEST(MmqQ4kImmaTile, CorrectnessTiny) {
    // Smallest config that fills one CTA (BLOCK_M=64, BLOCK_N=32):
    // one block, one sub-block of K.
    run_correctness(/*M=*/64, /*N=*/32, /*K=*/32, /*seed=*/3, /*abs=*/2.0f, /*rel=*/0.02f);
}

TEST(MmqQ4kImmaTile, CorrectnessMultiSub) {
    // K=256 → 8 sub-blocks; tests the cross-sub-block accumulator.
    run_correctness(/*M=*/64, /*N=*/32, /*K=*/256, /*seed=*/7, /*abs=*/8.0f, /*rel=*/0.02f);
}

TEST(MmqQ4kImmaTile, CorrectnessMultiTile) {
    // Multiple CTAs per dim. Sweeps the grid-launch logic.
    run_correctness(/*M=*/128, /*N=*/64, /*K=*/128, /*seed=*/13, /*abs=*/6.0f, /*rel=*/0.02f);
}

TEST(MmqQ4kImmaTile, CorrectnessFFNLikeShape) {
    // FFN-like dimensions: K = 512 (16 sub-blocks), M = 128, N = 64.
    // Exercises the cross-sub-block float accumulator at scale.
    run_correctness(/*M=*/128, /*N=*/64, /*K=*/512, /*seed=*/37, /*abs=*/40.0f, /*rel=*/0.02f);
}

}  // namespace
}  // namespace imp
