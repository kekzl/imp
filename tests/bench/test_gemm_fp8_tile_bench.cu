// Timing cases of tests/test_gemm_fp8_tile.cu, built into test-bench (make test-bench), not into the default
// GPU run.
#include "compute/gemm.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace imp;

#include "test_gemm_fp8_tile_util.cuh"

// Cold-L2 timing on Qwen3-8B prefill projections: each call follows a 256 MiB memset flush, median of 31
// timed calls after >= 1 s of warm-up. DISABLED_ so no lane runs it: --gtest_also_run_disabled_tests
// --gtest_filter=*Fp8GemmTileBench*.
TEST(Fp8GemmTileBench, DISABLED_VsCublasLt) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    constexpr size_t kFlushBytes = size_t(256) << 20;
    void* flush = nullptr;
    ASSERT_EQ(cudaMalloc(&flush, kFlushBytes), cudaSuccess);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    // Median device time of one call in us; the flush runs outside the event pair.
    auto median_us = [&](auto&& fn) {
        auto one = [&]() {
            cudaMemsetAsync(flush, 0x5a, kFlushBytes, s);
            cudaEventRecord(e0, s);
            fn();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1000.0 * ms;
        };
        double t = 0.0;
        while (t < 1e6)  // warm for >= 1 s
            t += one();
        std::vector<double> v;
        for (int i = 0; i < 31; ++i)
            v.push_back(one());
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    // Qwen3-8B: q/o 4096x4096, k/v 1024x4096, gate or up 12288x4096 (and the fused 24576), down 4096x12288.
    const int nk[][2] = {{4096, 4096}, {1024, 4096}, {12288, 4096}, {24576, 4096}, {4096, 12288}};
    const int ms[] = {512, 2048};
    std::fprintf(stderr, "%6s %6s %6s %9s %9s %9s %9s %9s %9s %9s %9s %9s %7s\n", "M", "N", "K", "cublasLt",
                 "t128", "t256", "sync", "sk", "hybrid", "t64", "t64sk", "auto", "auto_TF");
    for (const auto& s_nk : nk) {
        for (int M : ms) {
            Operands o;
            make_operands(o, M, s_nk[0], s_nk[1]);
            const double t_ref = median_us([&] { run_ref(o, 0.0f, s); });
            double t_v[kVariants];
            for (int i = 0; i < kVariants; ++i)
                t_v[i] = median_us([&] { (void)run_own(o, 0.0f, variant_of(i), s); });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const double tf = 2.0 * o.M * o.N * double(o.K) / (t_v[kVariants - 1] * 1e-6) / 1e12;
            std::fprintf(stderr, "%6d %6d %6d %9.1f", o.M, o.N, o.K, t_ref);
            for (double t : t_v)
                std::fprintf(stderr, " %9.1f", t);
            std::fprintf(stderr, " %7.0f\n", tf);
            free_operands(o);
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(flush);
    cudaStreamDestroy(s);
}
