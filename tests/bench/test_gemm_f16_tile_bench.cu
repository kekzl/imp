// Timing cases of tests/test_gemm_f16_tile.cu, built into test-bench (make test-bench), not into the default
// GPU run.
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

#include "test_gemm_f16_tile_util.cuh"

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
