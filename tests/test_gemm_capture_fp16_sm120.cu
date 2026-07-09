// =============================================================================
// test_gemm_capture_fp16_sm120.cu — correctness + microbench for the
// capture-safe sm_120 FP16 WMMA GEMM kernel.
// =============================================================================
//
// Validates:
//   1. Bit-stable correctness vs CPU reference for representative shapes.
//   2. Same numerics as cuBLAS gemm() (within FP16 tolerance) for the cases
//      where the WMMA path replaces cuBLASLt under capture.
//   3. Per-shape kernel timing alongside cuBLASLt for cross-validation.
//      Bench result is printed; not gated, so a regression doesn't fail the
//      suite but is visible to the operator.
//
// =============================================================================

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

namespace imp {
namespace {

// CPU reference: D = alpha * A @ B^T + beta * D
//   A [M, K] row-major
//   B [N, K] row-major  (semantic B^T)
//   D [M, N] row-major
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

struct DeviceFp16 {
    __half* data = nullptr;
    int64_t numel = 0;
    explicit DeviceFp16(int64_t n) : numel(n) { cudaMalloc(&data, n * sizeof(__half)); }
    ~DeviceFp16() {
        if (data) cudaFree(data);
    }
    DeviceFp16(const DeviceFp16&) = delete;
    DeviceFp16& operator=(const DeviceFp16&) = delete;
};

void upload_fp16(__half* dst, const std::vector<float>& src) {
    std::vector<__half> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = __float2half(src[i]);
    cudaMemcpy(dst, h.data(), src.size() * sizeof(__half), cudaMemcpyHostToDevice);
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

class GemmCaptureSm120 : public ::testing::Test {
protected:
    void SetUp() override {
        if (!capture_gemm_fp16_sm120_available()) GTEST_SKIP() << "Not sm_120+";
    }
};

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
    // Typical mid-size prefill shape. Compare to cuBLAS path through gemm().
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

    // cuBLAS via gemm() — build minimal Tensor wrappers around the device pointers.
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

// ---------------------------------------------------------------------------
// Microbench
// ---------------------------------------------------------------------------

struct ShapeBench {
    int M, N, K;
};

double bench_wmma(const __half* A, const __half* B, __half* D, int M, int N, int K, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Warmup
    for (int i = 0; i < 3; ++i)
        gemm_capture_fp16_sm120(A, B, D, M, N, K, 1.0f, 0.0f, 0);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        gemm_capture_fp16_sm120(A, B, D, M, N, K, 1.0f, 0.0f, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / iters;
}

double bench_cublas(__half* A, __half* B, __half* D, int M, int N, int K, int iters) {
    Tensor tA{}, tB{}, tC{};
    tA.qtype       = QType::F16;
    tA.data        = A;
    tA.ndim        = 2;
    tA.shape[0]    = M;
    tA.shape[1]    = K;
    tA.compute_strides();
    tA.on_device = true;
    tB.qtype     = QType::F16;
    tB.data      = B;
    tB.ndim      = 2;
    tB.shape[0]  = N;
    tB.shape[1]  = K;
    tB.compute_strides();
    tB.on_device = true;
    tC.qtype     = QType::F16;
    tC.data      = D;
    tC.ndim      = 2;
    tC.shape[0]  = M;
    tC.shape[1]  = N;
    tC.compute_strides();
    tC.on_device = true;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 3; ++i) gemm(tA, tB, tC);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) gemm(tA, tB, tC);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / iters;
}

TEST_F(GemmCaptureSm120, BenchVsCublasLt_ProductionShapes) {
    gemm_init();

    // Shapes representative of NVFP4 MoE prefill FP16 GEMMs that hit this kernel
    // under capture. M = prefill batch (512 typical), K = hidden dim, N varies.
    std::vector<ShapeBench> shapes = {
        {512, 4096, 2560},     // medium proj-like
        {512, 8192, 2560},     // larger
        {512, 2048, 4096},     // narrower N, wider K
        {512, 8192, 4096},     // Gemma-4-ish hidden
        {512, 16384, 4096},    // big N tile
        {256, 4096, 2560},     // smaller batch
        {1024, 4096, 2560},    // larger batch
        {512, 128, 2560},      // tiny N (still ≥ BN=128)
    };

    std::printf("\n=== gemm_capture_fp16_sm120 vs cuBLASLt (FP16×FP16→FP16) ===\n");
    std::printf("%6s %6s %6s | %10s %10s %10s | %9s %9s\n", "M", "N", "K", "WMMA µs",
                "cuBLAS µs", "ratio", "WMMA TF", "cuBLAS TF");

    constexpr int iters = 50;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    for (auto s : shapes) {
        std::vector<float> A(s.M * s.K), B(s.N * s.K);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        DeviceFp16 dA(s.M * s.K), dB(s.N * s.K), dD(s.M * s.N);
        upload_fp16(dA.data, A);
        upload_fp16(dB.data, B);
        cudaMemset(dD.data, 0, s.M * s.N * sizeof(__half));

        double ms_wmma = bench_wmma(dA.data, dB.data, dD.data, s.M, s.N, s.K, iters);
        double ms_cub  = bench_cublas(dA.data, dB.data, dD.data, s.M, s.N, s.K, iters);

        double flops    = 2.0 * (double)s.M * s.N * s.K;
        double tflop_w  = flops / (ms_wmma * 1e-3) / 1e12;
        double tflop_c  = flops / (ms_cub * 1e-3) / 1e12;

        std::printf("%6d %6d %6d | %10.2f %10.2f %10.3fx | %9.1f %9.1f\n", s.M, s.N, s.K,
                    ms_wmma * 1e3, ms_cub * 1e3, ms_wmma / ms_cub, tflop_w, tflop_c);
        std::fflush(stdout);
    }
    std::printf("=== end bench ===\n\n");
}

}  // namespace
}  // namespace imp
