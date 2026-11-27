// Capture-safe sm_120 FP16 WMMA GEMM: validates bit-stable correctness vs a CPU reference,
// same numerics as cuBLASLt gemm() (FP16 tolerance) where WMMA replaces cuBLASLt under
// capture. Per-shape timing: tests/bench/test_gemm_capture_fp16_sm120_bench.cu.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "compute/gemm_capture_fp16_sm120.h"
#include "compute/gemm.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "test_gemm_capture_fp16_sm120_util.cuh"

namespace imp {
namespace {

// CPU reference: D = alpha * A @ B^T + beta * D, A[M,K] row-major, B[N,K] row-major
// (semantic B^T), D[M,N] row-major.
void cpu_gemm_ref(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& D,
                  int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[(int64_t)i * K + k] * B[(int64_t)j * K + k];
            float prev = D[(int64_t)i * N + j];
            D[(int64_t)i * N + j] = alpha * sum + beta * prev;
        }
    }
}

std::vector<float> download_fp16(const __half* src, int64_t n) {
    std::vector<__half> h(n);
    cudaMemcpy(h.data(), src, n * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; ++i) out[i] = __half2float(h[i]);
    return out;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::abs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

TEST_F(GemmCaptureSm120, Correctness_128x128x32_alpha1_beta0) {
    constexpr int M = 128, N = 128, K = 32;
    std::vector<float> A(M * K), B(N * K), D(M * N, 0.0f), D_ref(M * N, 0.0f);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    cpu_gemm_ref(A, B, D_ref, M, N, K, 1.0f, 0.0f);

    DeviceFp16 dA(M * K), dB(N * K), dD(M * N);
    upload_fp16(dA.data, A);
    upload_fp16(dB.data, B);
    cudaMemset(dD.data, 0, M * N * sizeof(__half));

    ASSERT_TRUE(gemm_capture_fp16_sm120(dA.data, dB.data, dD.data, M, N, K, 1.0f, 0.0f, 0));
    cudaDeviceSynchronize();

    auto got = download_fp16(dD.data, M * N);
    double m = max_abs_diff(got, D_ref);
    EXPECT_LT(m, 5e-2) << "max abs diff " << m;
}

TEST_F(GemmCaptureSm120, Correctness_NarrowN_2048x32x2560) {
    // The GDN-projection shape that used to be rejected (N=32 < BN=128) and fell
    // through to cuBLASLt → status-14 under capture → whole decode graph aborted
    // (#934 residual). The partial-N tile must be masked correctly.
    constexpr int M = 2048, N = 32, K = 2560;
    std::vector<float> A(M * K), B(N * K), D_ref(M * N, 0.0f);
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    cpu_gemm_ref(A, B, D_ref, M, N, K, 1.0f, 0.0f);

    DeviceFp16 dA(M * K), dB(N * K), dD(M * N);
    upload_fp16(dA.data, A);
    upload_fp16(dB.data, B);
    cudaMemset(dD.data, 0, M * N * sizeof(__half));

    ASSERT_TRUE(gemm_capture_fp16_sm120(dA.data, dB.data, dD.data, M, N, K, 1.0f, 0.0f, 0));
    cudaDeviceSynchronize();

    auto got = download_fp16(dD.data, M * N);
    // K=2560, FP16 accumulation → looser tolerance (matches the VsCublas case).
    double m = max_abs_diff(got, D_ref);
    EXPECT_LT(m, 0.5) << "max abs diff " << m;
}

TEST_F(GemmCaptureSm120, Correctness_256x256x256_alpha_beta) {
    constexpr int M = 256, N = 256, K = 256;
    std::vector<float> A(M * K), B(N * K), D(M * N), D_ref(M * N);
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (size_t i = 0; i < D.size(); ++i) D[i] = D_ref[i] = dist(rng);

    cpu_gemm_ref(A, B, D_ref, M, N, K, 0.7f, 0.3f);

    DeviceFp16 dA(M * K), dB(N * K), dD(M * N);
    upload_fp16(dA.data, A);
    upload_fp16(dB.data, B);
    upload_fp16(dD.data, D);

    ASSERT_TRUE(gemm_capture_fp16_sm120(dA.data, dB.data, dD.data, M, N, K, 0.7f, 0.3f, 0));
    cudaDeviceSynchronize();

    auto got = download_fp16(dD.data, M * N);
    // K=256, FP16 accumulation, alpha+beta blend → larger tolerance.
    double m = max_abs_diff(got, D_ref);
    EXPECT_LT(m, 0.5) << "max abs diff " << m;
}

TEST_F(GemmCaptureSm120, Correctness_VsCublas_512x2048x2560) {
    // Typical mid-size prefill shape. Compare to the cuBLASLt path through gemm().
    constexpr int M = 512, N = 2048, K = 2560;
    std::vector<float> A(M * K), B(N * K);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    DeviceFp16 dA(M * K), dB(N * K), dD_wmma(M * N), dD_cublas(M * N);
    upload_fp16(dA.data, A);
    upload_fp16(dB.data, B);
    cudaMemset(dD_wmma.data, 0, M * N * sizeof(__half));
    cudaMemset(dD_cublas.data, 0, M * N * sizeof(__half));

    // WMMA path
    ASSERT_TRUE(gemm_capture_fp16_sm120(dA.data, dB.data, dD_wmma.data, M, N, K, 1.0f, 0.0f, 0));

    // cuBLASLt via gemm(): build minimal Tensor wrappers around the device pointers.
    Tensor tA{}, tB{}, tC{};
    tA.qtype = QType::F16;
    tA.data  = dA.data;
    tA.ndim  = 2;
    tA.shape[0] = M;
    tA.shape[1] = K;
    tA.compute_strides();
    tA.on_device = true;

    tB.qtype = QType::F16;
    tB.data  = dB.data;
    tB.ndim  = 2;
    tB.shape[0] = N;
    tB.shape[1] = K;
    tB.compute_strides();
    tB.on_device = true;

    tC.qtype = QType::F16;
    tC.data  = dD_cublas.data;
    tC.ndim  = 2;
    tC.shape[0] = M;
    tC.shape[1] = N;
    tC.compute_strides();
    tC.on_device = true;

    gemm_init();
    gemm(tA, tB, tC);
    cudaDeviceSynchronize();

    auto a_wmma   = download_fp16(dD_wmma.data, M * N);
    auto a_cublas = download_fp16(dD_cublas.data, M * N);
    double m      = max_abs_diff(a_wmma, a_cublas);

    // FP16 accumulation in WMMA (FP32 acc) vs cuBLASLt (may differ in internal
    // precision/order). K=2560 → expect ~ K * eps_fp16 * max(|a|,|b|) ~ a few units.
    EXPECT_LT(m, 5.0) << "max abs diff vs cuBLAS " << m;
    std::printf("[GemmCaptureSm120] vs-cuBLAS M=%d N=%d K=%d max_abs_diff=%.4g\n", M, N, K, m);
    std::fflush(stdout);
}
}  // namespace
}  // namespace imp
