// Timing cases of tests/test_nvfp4_gemm_tile.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include "compute/gemm.h"
#include "compute/nvfp4_sfatom_quant.h"
#include "core/tensor.h"
#include "nvfp4_ref_gemm.cuh"
#include "quant/nvfp4_gemm_tile.h"
#include "compute/nvfp4_sfatom_owned.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace imp;

#include "test_nvfp4_gemm_tile_util.cuh"

// Timing on Qwen3.8-27B projection shapes, operands from DRAM (L2 flushed before each call).
TEST(Nvfp4GemmTileBench, VsCublasLt) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {6144, 5120},
                             {5120, 6144},  {1024, 5120},  {12288, 5120}};
    const int ms[] = {512, 2048};
    constexpr size_t kFlushBytes = size_t(256) << 20;  // > 2 x the 96 MB L2
    void* flush = nullptr;
    ASSERT_EQ(cudaMalloc(&flush, kFlushBytes), cudaSuccess);
    constexpr int kVariants =
        7;  // variants 0..3, 5 (128x64), 6 (128x64 split), then -1 (auto, the dispatch default)
    std::fprintf(stderr, "%6s %6s %6s %8s %8s %8s %8s %8s %8s %8s %8s %7s\n", "M", "N", "K", "lt", "t128",
                 "t256", "sync", "sk", "t64", "t64sk", "auto", "best_TF");
    for (const auto& nk : shapes) {
        for (int M : ms) {
            Operands o;
            make_operands(o, M, nk[0], nk[1], s);
            // One W copy; the L2 is flushed before every timed call, so W and A come from DRAM as in a
            // prefill pass (rotating 4 copies kept small shapes L2-resident and hid the M < 2048 gap).
            constexpr int kCopies = 1;
            const size_t wb = size_t(o.N) * o.K / 2;
            const size_t sb = o.cw.sf_bytes;
            std::vector<void*> wd(kCopies), wsf(kCopies);
            for (int c = 0; c < kCopies; ++c) {
                cudaMalloc(&wd[c], wb);
                cudaMalloc(&wsf[c], sb);
                cudaMemcpyAsync(wd[c], o.cw.data, wb, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(wsf[c], o.cw.scale_factors, sb, cudaMemcpyDeviceToDevice, s);
            }
            // >= 1 s back to back (clocks ramp ~1 s), then the median of 31 single calls, each after a flush.
            auto time_us = [&](auto&& fn) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, s);
                for (float warm = 0.0f; warm < 1000.0f;) {
                    for (int i = 0; i < 20; ++i)
                        fn(0);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&warm, e0, e1);
                }
                std::vector<float> t(31);
                for (size_t i = 0; i < t.size(); ++i) {
                    cudaMemsetAsync(flush, static_cast<int>(i), kFlushBytes, s);
                    cudaEventRecord(e0, s);
                    fn(0);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&t[i], e0, e1);
                }
                std::nth_element(t.begin(), t.begin() + 15, t.end());
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * t[15];
            };
            const double t_lt = time_us([&](int c) {
                (void)gemm_nvfp4_cublaslt(o.act, o.act_sf, wd[c], wsf[c], o.cw.tensor_scale, o.y_ref, o.M,
                                          o.N, o.K, s);
            });
            double t_v[kVariants];
            for (int v = 0; v < kVariants; ++v)
                t_v[v] = time_us([&](int c) {
                    (void)gemm_nvfp4_tile(o.act, o.act_sf, wd[c], wsf[c], o.cw.tensor_scale, o.y_own, o.M,
                                          o.N, o.K, s, v < 4 ? v : (v < 6 ? v + 1 : -1), o.ws, o.ws_bytes);
                });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            double best = 1e30;
            for (int v = 0; v < kVariants; ++v)
                if (v != 2)  // sync is the alignment fallback, not a contender
                    best = std::min(best, t_v[v]);
            const double tf = 2.0 * o.M * o.N * double(o.K) / (best * 1e-6) / 1e12;
            std::fprintf(stderr, "%6d %6d %6d %8.1f", o.M, o.N, o.K, t_lt);
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
    cudaFree(flush);
    cudaStreamDestroy(s);
}
