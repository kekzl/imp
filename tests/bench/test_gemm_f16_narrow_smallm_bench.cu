// Timing cases of tests/test_gemm_f16_narrow_smallm.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gemm.h"
#include "compute/gemm_f16_narrow_smallm.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace imp;
#include "test_gemm_f16_narrow_smallm_util.cuh"

// Informational: per-launch wall time inside a replayed CUDA graph (the kernel's actual
// regime) vs the two gemm() calls (cuBLASLt) it replaces. No threshold (timing anchors flake in
// the full suite); WSL2 host launch cost would dominate an eager loop.
TEST(GemmF16NarrowSmallM, BenchQwen38Shape) {
    Case c(32, 48, 48, 5120, 11u);
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    const int64_t a_shape[2] = {32, 5120}, w_shape[2] = {48, 5120}, c_shape[2] = {32, 48};
    Tensor tA(c.dA.p, QType::F16, 2, a_shape, true), tW0(c.dW0.p, QType::F16, 2, w_shape, true),
        tW1(c.dW1.p, QType::F16, 2, w_shape, true), tC0(c.dC0.p, QType::F16, 2, c_shape, true),
        tC1(c.dC1.p, QType::F16, 2, c_shape, true);
    auto cublas_pair = [&]() {
        gemm(tA, tW0, tC0, 1.0f, 0.0f, s);
        gemm(tA, tW1, tC1, 1.0f, 0.0f, s);
    };
    // Warm both paths past the idle downclock (>1 s of work).
    cudaEvent_t w0, w1;
    cudaEventCreate(&w0);
    cudaEventCreate(&w1);
    cudaEventRecord(w0, s);
    for (int i = 0; i < 20000; ++i) {
        ASSERT_TRUE(c.run(s));
        if ((i % 10) == 0)
            cublas_pair();
    }
    cudaEventRecord(w1, s);
    ASSERT_EQ(cudaEventSynchronize(w1), cudaSuccess);
    float warm_ms = 0.0f;
    cudaEventElapsedTime(&warm_ms, w0, w1);
    printf("  warmup: %.0f ms\n", warm_ms);

    constexpr int kPerGraph = 64;
    auto time_graph = [&](const char* tag, auto&& body) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
        for (int i = 0; i < kPerGraph; ++i)
            body();
        ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
        for (int i = 0; i < 20; ++i)
            ASSERT_EQ(cudaGraphLaunch(ge, s), cudaSuccess);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int reps = 50;
        cudaEventRecord(e0, s);
        for (int i = 0; i < reps; ++i)
            cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        ASSERT_EQ(cudaEventSynchronize(e1), cudaSuccess);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        printf("  %s: %.2f us per launch in graph (%d launches x %d replays, L2-resident)\n", tag,
               ms * 1000.0f / (reps * kPerGraph), kPerGraph, reps);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    };
    time_graph("narrow kernel (alpha+beta)", [&]() { c.run(s); });
    gemm_set_lt_capture_allowed(true);
    time_graph("cuBLAS gemm() x2 (alpha, beta)", cublas_pair);
    gemm_set_lt_capture_allowed(false);

    // Numerics against the path it replaces: both accumulate in FP32 and round
    // once to FP16, in different summation orders. Report how many outputs
    // differ and by how much (1 FP16 ulp is the expected class).
    cublas_pair();
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    const auto ref0 = c.out(0), ref1 = c.out(1);
    ASSERT_TRUE(c.run(s));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    const auto got0 = c.out(0), got1 = c.out(1);
    size_t n_diff = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < ref0.size(); ++i) {
        for (int p = 0; p < 2; ++p) {
            const float r = __half2float(p ? ref1[i] : ref0[i]);
            const float g = __half2float(p ? got1[i] : got0[i]);
            if (r != g) {
                ++n_diff;
                max_abs = std::max(max_abs, static_cast<double>(std::fabs(r - g)));
                max_rel = std::max(max_rel,
                                   static_cast<double>(std::fabs(r - g) / std::max(std::fabs(r), 1e-3f)));
            }
            EXPECT_NEAR(g, r, 1e-2 + 4e-3 * std::fabs(r)) << "pair " << p << " at " << i;
        }
    }
    printf("  vs cuBLAS: %zu of %zu outputs differ, max |diff| %.3e, max rel %.3e\n", n_diff, 2 * ref0.size(),
           max_abs, max_rel);
    cudaStreamDestroy(s);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
