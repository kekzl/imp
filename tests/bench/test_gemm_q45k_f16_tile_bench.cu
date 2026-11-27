// Timing cases of tests/test_gemm_q45k_f16_tile.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_q45k_f16_tile.h"
#include "compute/mmq_q8_imma.h"
#include "core/process_diag.h"
#include "core/qtype.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;

#include "test_gemm_q45k_f16_tile_util.cuh"

// Timing with a 256 MiB memset before each call (W of 17408x5120 fits the 96 MB L2 otherwise): fused
// 128x128 / 256x128 vs the raw IMMA tile the uncached route runs at M > 32 (activation quantize included).
TEST(GemmQ45kTileBench, FlushedVsImma) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    void* flush = nullptr;
    const size_t fbytes = size_t(256) << 20;
    cudaMalloc(&flush, fbytes);
    auto time_us = [&](auto&& fn) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::vector<float> v;
        for (int i = 0; i < 25; ++i) {
            cudaMemsetAsync(flush, i, fbytes, s);
            cudaEventRecord(e0, s);
            fn();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (i >= 5)
                v.push_back(1000.0f * ms);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {6144, 5120},
                             {10240, 5120}, {5120, 6144},  {1024, 5120}};
    std::fprintf(stderr, "%4s %5s %6s %6s %9s %9s %9s %9s %9s\n", "fmt", "M", "N", "K", "imma_us", "128x128",
                 "256x128", "auto", "fp32acc");
    for (bool q5 : {true, false})
        for (const auto& nk : shapes)
            for (int M : {128, 512, 1024, 2048}) {
                KqBufs b;
                b.alloc(q5, M, nk[0], nk[1], s);
                const float t_imma = time_us([&] {
                    (void)(q5 ? mmq_gguf_imma_smallm_gemm(b.wq, QType::Q5_K, b.a, b.own, M, b.N, b.K, s)
                              : mmq_q4k_imma_gemm(b.wq, b.a, b.own, M, b.N, b.K, s));
                });
                float t[4] = {};
                for (int v = 0; v < 4; ++v)
                    t[v] = time_us([&] {
                        (void)gemm_q45k_f16_tile(q5, b.a, b.wq, b.own, M, b.N, b.K, 1.0f, 0.0f, v < 3, s,
                                                 v == 0 ? 1 : (v == 1 ? 2 : -1));
                    });
                ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
                std::fprintf(stderr, "%4s %5d %6d %6d %9.1f %9.1f %9.1f %9.1f %9.1f\n", q5 ? "Q5_K" : "Q4_K",
                             M, b.N, b.K, t_imma, t[0], t[1], t[2], t[3]);
                b.release();
            }
    cudaFree(flush);
    cudaStreamDestroy(s);
}
