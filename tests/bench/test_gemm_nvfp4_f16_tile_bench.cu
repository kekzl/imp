// Timing cases of tests/test_gemm_nvfp4_f16_tile.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_grouped_nvfp4_f16_tile.h"
#include "compute/gemm_nvfp4_f16_tile.h"
#include "core/tensor.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace imp;

#include "test_gemm_nvfp4_f16_tile_util.cuh"

// Timing on Flint attention shapes (hidden 2048, q 4096, kv 512): fused W4A16 vs gemm_f16_tile on an FP16 W.
TEST(GemmNvfp4TileBench, FlintAttentionShapes) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][2] = {{4096, 2048}, {512, 2048}, {2048, 4096}};
    std::fprintf(stderr, "%6s %6s %6s %10s %10s\n", "M", "N", "K", "f16_us", "nvfp4_us");
    for (const auto& nk : shapes)
        for (int M : {512, 2048}) {
            Nv4Bufs b;
            b.alloc(M, nk[0], nk[1], 0x30, 0x48, s);
            auto time_us = [&](auto&& fn) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                for (int i = 0; i < 10; ++i)
                    fn();
                cudaEventRecord(e0, s);
                for (int i = 0; i < 20; ++i)
                    fn();
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms / 20;
            };
            const double t16 = time_us(
                [&] { (void)gemm_f16_tile(b.a, b.w16, b.ref, M, b.N, b.K, 1.0f, 0.0f, false, s, -1); });
            const double t4 = time_us([&] {
                (void)gemm_nvfp4_f16_tile(b.a, b.w, b.sc, b.own, M, b.N, b.K, 1.0f, 0.0f, false, s, -1);
            });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            std::fprintf(stderr, "%6d %6d %6d %10.1f %10.1f\n", M, b.N, b.K, t16, t4);
            b.release();
        }
    cudaStreamDestroy(s);
}
