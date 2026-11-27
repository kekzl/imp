// Timing cases of tests/test_gemm_mxfp8_tile.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include "quant/fp8_quant.h"
#include "quant/mxfp8_sfatom.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace imp;

#include "test_gemm_mxfp8_tile_util.cuh"

// Timing on the Qwen3.6-35B GDN projections (qkv, z, out) in the FP8 row-scale view, the prefill
// default, per variant; 4 weight copies rotate out of L2.
TEST(Mxfp8GemmTileBench, Variants) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][2] = {{8192, 2048}, {4096, 2048}, {2048, 4096}};
    const int ms[] = {512, 2048, 4096};
    std::fprintf(stderr, "%6s %6s %6s %8s %8s %8s %8s %8s %7s\n", "M", "N", "K", "t128", "t256", "sync", "sk",
                 "auto", "auto_TF");
    for (const auto& nk : shapes) {
        for (int M : ms) {
            Operands o;
            make_operands(o, M, nk[0], nk[1], true, s);
            constexpr int kCopies = 4;
            std::vector<void*> wd(kCopies), wsf(kCopies);
            for (int c = 0; c < kCopies; ++c) {
                cudaMalloc(&wd[c], o.cw.data_bytes);
                cudaMalloc(&wsf[c], o.cw.sf_bytes);
                cudaMemcpyAsync(wd[c], o.cw.data, o.cw.data_bytes, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(wsf[c], o.cw.scale_factors, o.cw.sf_bytes, cudaMemcpyDeviceToDevice, s);
            }
            auto time_us = [&](auto&& fn) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                for (int i = 0; i < 40; ++i)
                    fn(i % kCopies);
                constexpr int kIters = 40;
                cudaEventRecord(e0, s);
                for (int i = 0; i < kIters; ++i)
                    fn(i % kCopies);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms_total = 0.0f;
                cudaEventElapsedTime(&ms_total, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms_total / kIters;
            };
            double t_v[kVariants];
            for (int v = 0; v < kVariants; ++v)
                t_v[v] = time_us([&](int c) {
                    (void)gemm_mxfp8_tile(o.act, o.act_sf, wd[c], wsf[c], o.cw.col_scale, o.y_own, o.M, o.N,
                                          o.K, s, v < 4 ? v : -1, o.ws, o.ws_bytes);
                });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const double tf = 2.0 * o.M * o.N * double(o.K) / (t_v[4] * 1e-6) / 1e12;
            std::fprintf(stderr, "%6d %6d %6d", o.M, o.N, o.K);
            for (double t : t_v)
                std::fprintf(stderr, " %8.1f", t);
            std::fprintf(stderr, " %7.0f\n", tf);
            for (int c = 0; c < kCopies; ++c) {
                cudaFree(wd[c]);
                cudaFree(wsf[c]);
            }
            free_operands(o);
        }
    }
    cudaStreamDestroy(s);
}
