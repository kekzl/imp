// Timing cases of tests/test_mmq_q4k_imma_tile_bench.cu, built into test-bench (make test-bench), not into
// the default GPU run.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/mmq_q4k_imma_tile.h"

#include "test_mmq_q4k_imma_tile_util.cuh"

namespace imp {
namespace {

// Phase 2B.1 informational baseline, no perf assertion: prints kernel time and TOPS.
// 1-warp-per-CTA tile (BLOCK_M=16,BLOCK_N=8) under-utilises each SM; Phase 2B.2 is the
// next real lever.
TEST(MmqQ4kImmaTile, BenchSweep) {
    struct Shape {
        int M, N, K;
    };
    // Production FFN shapes (Qwen3-32B FFN ~5120, Gemma-3-12B ~3072). Large M/N is mandatory
    // for Phase 2B.3's BLOCK_M=64/BLOCK_N=32 tile: M=512 N=256 gives 64 CTAs, ~0.4 CTAs/SM,
    // and regresses vs Phase 2B.2.
    Shape shapes[] = {
        {512, 512, 2048},   {1024, 512, 2048},  {2048, 512, 2048},
        {4096, 1024, 2048}, {2048, 2048, 2048}, {4096, 4096, 2048},
    };

    std::fprintf(stderr,
                 "\n[q4k-imma-tile Phase 2B.3 bench, 4 warps/CTA × WRM·WRN=2·2 "
                 "(64×32 output), 2-stage cp.async]\n");
    std::fprintf(stderr, "  %4s %4s %5s  %10s  %10s\n", "M", "N", "K", "ms/rep", "TOPS");

    for (auto sh : shapes) {
        std::vector<int8_t> hW;
        std::vector<__half> halpha, hbeta;
        gen_random_w_sym_s8(hW, halpha, hbeta, sh.N, sh.K, 71);

        std::vector<int8_t> hX;
        std::vector<__half> hx_scale;
        std::vector<float> hx_rowsum;
        gen_random_x_s8(hX, hx_scale, hx_rowsum, sh.M, sh.K, 73);

        int8_t *dX = nullptr, *dW = nullptr;
        __half *dxscale = nullptr, *dalpha = nullptr, *dbeta = nullptr;
        float* dxrowsum = nullptr;
        __half* dout = nullptr;
        cudaMalloc(&dX, hX.size());
        cudaMalloc(&dW, hW.size());
        cudaMalloc(&dxscale, hx_scale.size() * sizeof(__half));
        cudaMalloc(&dxrowsum, hx_rowsum.size() * sizeof(float));
        cudaMalloc(&dalpha, halpha.size() * sizeof(__half));
        cudaMalloc(&dbeta, hbeta.size() * sizeof(__half));
        cudaMalloc(&dout, static_cast<size_t>(sh.M) * sh.N * sizeof(__half));
        cudaMemcpy(dX, hX.data(), hX.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dxscale, hx_scale.data(), hx_scale.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dxrowsum, hx_rowsum.data(), hx_rowsum.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dalpha, halpha.data(), halpha.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dbeta, hbeta.data(), hbeta.size() * sizeof(__half), cudaMemcpyHostToDevice);

        for (int w = 0; w < 3; ++w) {
            mmq_q4k_imma_tile(dX, dxscale, dxrowsum, dW, dalpha, dbeta, dout, sh.M, sh.N, sh.K, nullptr);
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        constexpr int kReps = 20;
        float total_ms = 0.0f;
        for (int r = 0; r < kReps; ++r) {
            cudaEventRecord(start);
            mmq_q4k_imma_tile(dX, dxscale, dxrowsum, dW, dalpha, dbeta, dout, sh.M, sh.N, sh.K, nullptr);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        float ms_per_rep = total_ms / kReps;
        double ops = 2.0 * sh.M * sh.N * sh.K;
        double tops = ops / (ms_per_rep * 1e-3) / 1e12;
        std::fprintf(stderr, "  %4d %4d %5d  %10.4f  %10.3f\n", sh.M, sh.N, sh.K, ms_per_rep, tops);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dX);
        cudaFree(dW);
        cudaFree(dxscale);
        cudaFree(dxrowsum);
        cudaFree(dalpha);
        cudaFree(dbeta);
        cudaFree(dout);
    }
}

}  // namespace
}  // namespace imp
