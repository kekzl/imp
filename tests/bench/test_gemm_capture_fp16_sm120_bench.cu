// Timing cases of tests/test_gemm_capture_fp16_sm120.cu, built into test-bench (make test-bench), not into
// the default GPU run.
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
        // timing loop: the launch status is not under test
        (void)gemm_capture_fp16_sm120(A, B, D, M, N, K, 1.0f, 0.0f, 0);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        // timing loop: the launch status is not under test
        (void)gemm_capture_fp16_sm120(A, B, D, M, N, K, 1.0f, 0.0f, 0);
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
    tA.qtype = QType::F16;
    tA.data = A;
    tA.ndim = 2;
    tA.shape[0] = M;
    tA.shape[1] = K;
    tA.compute_strides();
    tA.on_device = true;
    tB.qtype = QType::F16;
    tB.data = B;
    tB.ndim = 2;
    tB.shape[0] = N;
    tB.shape[1] = K;
    tB.compute_strides();
    tB.on_device = true;
    tC.qtype = QType::F16;
    tC.data = D;
    tC.ndim = 2;
    tC.shape[0] = M;
    tC.shape[1] = N;
    tC.compute_strides();
    tC.on_device = true;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 3; ++i)
        gemm(tA, tB, tC);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        gemm(tA, tB, tC);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / iters;
}

// Manual A/B print, no assertion: run with --gtest_also_run_disabled_tests.
TEST_F(GemmCaptureSm120, DISABLED_BenchVsCublasLt_ProductionShapes) {
    gemm_init();

    // Shapes representative of NVFP4 MoE prefill FP16 GEMMs that hit this kernel
    // under capture. M = prefill batch (512 typical), K = hidden dim, N varies.
    std::vector<ShapeBench> shapes = {
        {512, 4096, 2560},   // medium proj-like
        {512, 8192, 2560},   // larger
        {512, 2048, 4096},   // narrower N, wider K
        {512, 8192, 4096},   // Gemma-4-ish hidden
        {512, 16384, 4096},  // big N tile
        {256, 4096, 2560},   // smaller batch
        {1024, 4096, 2560},  // larger batch
        {512, 128, 2560},    // tiny N (still ≥ BN=128)
    };

    std::printf("\n=== gemm_capture_fp16_sm120 vs cuBLASLt (FP16×FP16→FP16) ===\n");
    std::printf("%6s %6s %6s | %10s %10s %10s | %9s %9s\n", "M", "N", "K", "WMMA µs", "cuBLAS µs", "ratio",
                "WMMA TF", "cuBLAS TF");

    constexpr int iters = 50;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    for (auto s : shapes) {
        std::vector<float> A(s.M * s.K), B(s.N * s.K);
        for (auto& v : A)
            v = dist(rng);
        for (auto& v : B)
            v = dist(rng);

        DeviceFp16 dA(s.M * s.K), dB(s.N * s.K), dD(s.M * s.N);
        upload_fp16(dA.data, A);
        upload_fp16(dB.data, B);
        cudaMemset(dD.data, 0, s.M * s.N * sizeof(__half));

        double ms_wmma = bench_wmma(dA.data, dB.data, dD.data, s.M, s.N, s.K, iters);
        double ms_cub = bench_cublas(dA.data, dB.data, dD.data, s.M, s.N, s.K, iters);

        double flops = 2.0 * (double)s.M * s.N * s.K;
        double tflop_w = flops / (ms_wmma * 1e-3) / 1e12;
        double tflop_c = flops / (ms_cub * 1e-3) / 1e12;

        std::printf("%6d %6d %6d | %10.2f %10.2f %10.3fx | %9.1f %9.1f\n", s.M, s.N, s.K, ms_wmma * 1e3,
                    ms_cub * 1e3, ms_wmma / ms_cub, tflop_w, tflop_c);
        std::fflush(stdout);
    }
    std::printf("=== end bench ===\n\n");
}

}  // namespace
}  // namespace imp
