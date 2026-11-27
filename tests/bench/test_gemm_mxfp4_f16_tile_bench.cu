// Timing cases of tests/test_gemm_mxfp4_f16_tile.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_mxfp4_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "quant/mxfp4_sfatom.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

using namespace imp;

#include "test_gemm_mxfp4_f16_tile_util.cuh"

// Timing: fused vs dequant + gemm() (the W4A16 scratch path of executor_gemm_dispatch.cpp).
TEST(GemmMxfp4TileBench, VsDequantPlusGemm) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    process_diag_set_cublas_fp16_acc(true);
    const int shapes[][2] = {{9216, 2560}, {2560, 9216}, {2560, 2560}, {1024, 2560}};
    std::fprintf(stderr, "%6s %6s %6s %10s %9s %9s %9s\n", "M", "N", "K", "dq+gemm_us", "dq_us", "128x128",
                 "256x128");
    for (const auto& nk : shapes)
        for (int M : {512, 2048, 4096}) {
            Mx4Bufs b;
            b.alloc(M, nk[0], nk[1], s);
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
            int64_t as[2] = {M, b.K}, ws[2] = {b.N, b.K}, ds[2] = {M, b.N};
            Tensor ta(b.a, QType::F16, 2, as, true), tw(b.w16, QType::F16, 2, ws, true),
                td(b.ref, QType::F16, 2, ds, true);
            const double t_dq = time_us([&] { dequant_mxfp4_linear_to_fp16(b.w, b.sc, b.N, b.K, b.w16, s); });
            const double t_lib = time_us([&] {
                dequant_mxfp4_linear_to_fp16(b.w, b.sc, b.N, b.K, b.w16, s);
                gemm(ta, tw, td, 1.0f, 0.0f, s);
            });
            const double t_128 = time_us([&] {
                (void)gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, M, b.N, b.K, 1.0f, 0.0f, true, s, 1);
            });
            const double t_256 = time_us([&] {
                (void)gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, M, b.N, b.K, 1.0f, 0.0f, true, s, 2);
            });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            std::fprintf(stderr, "%6d %6d %6d %10.1f %9.1f %9.1f %9.1f\n", M, b.N, b.K, t_lib, t_dq, t_128,
                         t_256);
            b.release();
        }
    process_diag_set_cublas_fp16_acc(false);
    cudaStreamDestroy(s);
}
