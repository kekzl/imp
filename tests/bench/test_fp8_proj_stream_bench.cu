// Timing cases of tests/test_fp8_proj_stream.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "test_fp8_proj_stream_util.cuh"

namespace imp {
namespace {

// FP16 projection sweep (IMP_FP8_PROJ_SWEEP=1): us/call per variant over a rotating weight pool > L2
// (decode reads each projection once per step), split-K (variant 1) vs the default dispatch (0).
TEST_F(FP8ProjStreamTest, Sweep) {
    if (getenv("IMP_FP8_PROJ_SWEEP") == nullptr)
        GTEST_SKIP() << "set IMP_FP8_PROJ_SWEEP=1 to run the projection sweep";
    const int variants[] = {1, 0};
    // Flash-Next GDN in / gate / out, attention q / o, DSpark drafter (Qwen3-8B) q / gate-up / down / lm_head
    const std::pair<int, int> shapes[] = {{10240, 2560}, {6144, 2560},  {2560, 6144},  {12288, 2560},
                                          {4096, 4096},  {24576, 4096}, {4096, 12288}, {151936, 4096}};
    constexpr size_t kPoolBytes = size_t{320} << 20;  // > 96 MB L2 by 3.3x
    constexpr int kMaxRows = 32;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (auto [M, K] : shapes) {
        const size_t wb = static_cast<size_t>(M) * K;
        const int copies = static_cast<int>((kPoolBytes + wb - 1) / wb);
        void* d_pool = nullptr;
        void *d_w = nullptr, *d_x = nullptr, *d_y = nullptr;
        float* d_s = nullptr;
        ASSERT_EQ(cudaMalloc(&d_pool, wb * copies), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_w, wb * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(kMaxRows) * K * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(kMaxRows) * M * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
        cudaMemset(d_w, 0x2a, wb * sizeof(half));
        cudaMemset(d_x, 0x2a, static_cast<size_t>(kMaxRows) * K * sizeof(half));
        auto* pool = static_cast<uint8_t*>(d_pool);
        quantize_fp8_rows_async(d_w, pool, M, K, d_s, stream_);
        for (int c = 1; c < copies; ++c)
            cudaMemcpyAsync(pool + (wb * c), pool, wb, cudaMemcpyDeviceToDevice, stream_);
        cudaFree(d_w);
        const half* hx = static_cast<const half*>(d_x);
        auto* hy = static_cast<half*>(d_y);
        for (int n : {2, 4, 8, 12, 16, 32}) {
            int it = 0;
            auto run = [&](int v) {
                return gemv_fp8_rowscale_f16_tuned(pool + (wb * (it++ % copies)), d_s, hx, hy, M, K, n, v,
                                                   stream_);
            };
            for (int i = 0; i < 1000; ++i)  // clock ramp
                for (int v : variants)
                    ASSERT_TRUE(run(v)) << "variant " << v;
            // Variants alternate per rep: a host-side stall hits both, best of 11 drops it.
            double best[2] = {1e30, 1e30};
            for (int r = 0; r < 11; ++r)
                for (int vi = 0; vi < 2; ++vi) {
                    float ms = 0.0f;
                    cudaEventRecord(t0, stream_);
                    for (int i = 0; i < 200; ++i)
                        (void)run(variants[vi]);  // timing loop
                    cudaEventRecord(t1, stream_);
                    ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
                    cudaEventElapsedTime(&ms, t0, t1);
                    best[vi] = std::min(best[vi], ms * 1000.0 / 200);
                }
            for (int vi = 0; vi < 2; ++vi)
                printf("[ PROJ     ] M=%d K=%d rows=%d variant=%d: %.2f us/call, %.0f GB/s\n", M, K, n,
                       variants[vi], best[vi], static_cast<double>(wb) / best[vi] / 1e3);
        }
        cudaFree(d_pool);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_s);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

}  // namespace
}  // namespace imp
