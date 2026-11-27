// Timing cases of tests/test_gemm_grouped_nvfp4_tile.cu, built into test-bench (make test-bench), not into
// the default GPU run.
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "core/tensor.h"
#include "nvfp4_fp64_ref.h"
#include "compute/cutlass_nvfp4_owned.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "test_gemm_grouped_nvfp4_tile_util.cuh"

namespace imp {

// Qwen3-30B-A3B experts: 128 x (N=768, K=2048) gate/up and (N=2048, K=768) down; rows per
// expert from a skewed draw around pp512 (avg 32) and pp4096 (avg 256) routing.
TEST_F(GroupedNvfp4TileTest, Bench) {
    std::fprintf(stderr, "%6s %6s %6s %10s %8s\n", "avg_M", "N", "K", "own_us", "TFLOP/s");
    for (int avg : {32, 256})
        for (const auto& nk : {std::pair<int, int>{768, 2048}, std::pair<int, int>{2048, 768}}) {
            std::mt19937 gen(7);
            std::gamma_distribution<float> dist(2.0f, avg / 2.0f);
            std::vector<int> m(128);
            for (int& v : m)
                v = std::max(1, static_cast<int>(dist(gen)));
            Grouped g;
            g.build(m, nk.first, nk.second, s_);
            void* d = nullptr;
            cudaMalloc(&d, size_t(g.M_total) * g.N * sizeof(half));
            auto time_us = [&](auto&& fn) {
                for (int i = 0; i < 10; ++i)
                    fn();
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, s_);
                for (int i = 0; i < 20; ++i)
                    fn();
                cudaEventRecord(e1, s_);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms / 20;
            };
            const double t_own = time_us(
                [&] { (void)gemm_grouped_nvfp4_tile(g.ne, g.N, g.K, g.M_total, g.args(d, true), s_); });
            ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
            std::fprintf(stderr, "%6d %6d %6d %10.1f %8.1f\n", avg, g.N, g.K, t_own,
                         2.0 * g.M_total * g.N * g.K / (t_own * 1e6));
            cudaFree(d);
            g.release();
        }
}

}  // namespace imp
