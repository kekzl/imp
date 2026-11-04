// Batched MTP draft head (lm_head_rows_argmax): FP8 per-row head vs the NVFP4 draft copy over 1-4 rows at
// the Qwen3.8-27B head shape [248320 x 5120]. Checks the NVFP4 multi-row GEMV keeps every row's bits at any
// row count, prices both heads per pass (printed, the step-price input).

#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

namespace imp {
namespace {

__global__ void fill_head_kernel(half* w, int64_t n, uint32_t seed) {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        uint32_t s = static_cast<uint32_t>(i) * 2654435761u + seed;
        s ^= s >> 13;
        s *= 0x5bd1e995u;
        s ^= s >> 15;
        w[i] = __float2half((static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.05f);
    }
}

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

TEST(LmHeadDraftRows, Nvfp4MultiRowIsRowCountInvariantAndPriced) {
    if (!gpu_available())
        GTEST_SKIP() << "no GPU";
    constexpr int V = 248320, K = 5120, kMaxRows = 4, kIters = 20;
    cudaStream_t st = nullptr;
    ASSERT_EQ(cudaStreamCreate(&st), cudaSuccess);
    half* d_w = nullptr;
    void* d_q8 = nullptr;
    float *d_s8 = nullptr, *d_y = nullptr, *d_y1 = nullptr;
    half* d_x = nullptr;
    const size_t n_w = static_cast<size_t>(V) * K;
    ASSERT_EQ(cudaMalloc(&d_w, n_w * sizeof(half)), cudaSuccess);
    fill_head_kernel<<<4096, 256, 0, st>>>(d_w, static_cast<int64_t>(n_w), 7u);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_q8, n_w), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s8, V * sizeof(float)), cudaSuccess);
    quantize_fp8_rows_async(d_w, d_q8, V, K, d_s8, st);
    NvFP4QuantResult q4{};
    int64_t shp[2] = {V, K};
    Tensor wt(d_w, QType::F16, 2, shp, /*on_device=*/true);
    quantize_fp16_to_nvfp4(wt, q4, st);
    ASSERT_EQ(cudaStreamSynchronize(st), cudaSuccess);
    ASSERT_NE(q4.packed_data, nullptr);
    cudaFree(d_w);

    std::vector<half> hx(static_cast<size_t>(kMaxRows) * K);
    uint32_t s = 11u;
    for (auto& v : hx) {
        s = s * 1664525u + 1013904223u;
        v = __float2half((static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 4.0f);
    }
    const size_t y_elems = static_cast<size_t>(kMaxRows) * V;
    ASSERT_EQ(cudaMalloc(&d_x, hx.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, y_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y1, y_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, hx.data(), hx.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    // Row bits: rows run alone (MR 1) equal the same rows in one MR 2/3/4 launch.
    for (int r = 0; r < kMaxRows; ++r)
        gemv_nvfp4_kpar_batched_fp32(q4, d_x + static_cast<size_t>(r) * K, d_y1 + static_cast<size_t>(r) * V,
                                     V, K, 1, st);
    std::vector<float> one(y_elems), many(y_elems);
    ASSERT_EQ(cudaMemcpyAsync(one.data(), d_y1, y_elems * sizeof(float), cudaMemcpyDeviceToHost, st),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(st), cudaSuccess);
    for (int rows = 2; rows <= kMaxRows; ++rows) {
        ASSERT_EQ(cudaMemsetAsync(d_y, 0xff, y_elems * sizeof(float), st), cudaSuccess);
        gemv_nvfp4_kpar_batched_fp32(q4, d_x, d_y, V, K, rows, st);
        ASSERT_EQ(cudaMemcpyAsync(many.data(), d_y, y_elems * sizeof(float), cudaMemcpyDeviceToHost, st),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(st), cudaSuccess);
        EXPECT_EQ(0, std::memcmp(one.data(), many.data(), static_cast<size_t>(rows) * V * sizeof(float)))
            << "rows=" << rows << ": NVFP4 multi-row bits differ from one row per launch";
    }

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    ASSERT_EQ(cudaEventCreate(&e0), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&e1), cudaSuccess);
    auto time_us = [&](auto&& launch) {
        launch();  // warm
        cudaEventRecord(e0, st);
        for (int i = 0; i < kIters; ++i)
            launch();
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        return 1000.0 * ms / kIters;
    };
    const double fp8_mib = static_cast<double>(n_w) / (1024.0 * 1024.0);
    const double nv4_mib = static_cast<double>(n_w / 2 + n_w / 16) / (1024.0 * 1024.0);
    for (int rows = 1; rows <= kMaxRows; ++rows) {
        bool ok = true;
        const double t8 = time_us(
            [&] { ok = gemv_fp8_rowscale_fp32(d_q8, d_s8, d_x, d_y, V, K, rows, st) && ok; });
        const double t4 = time_us([&] { gemv_nvfp4_kpar_batched_fp32(q4, d_x, d_y, V, K, rows, st); });
        EXPECT_TRUE(ok);
        printf("[draft head] rows=%d fp8 %.1f us (%.0f GB/s)  nvfp4 %.1f us (%.0f GB/s)\n", rows, t8,
               fp8_mib * 1.048576e6 / (t8 * 1e3), t4, nv4_mib * 1.048576e6 / (t4 * 1e3));
    }
    EXPECT_EQ(cudaStreamSynchronize(st), cudaSuccess);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    free_nvfp4_result(q4);
    for (void* p : {static_cast<void*>(d_q8), static_cast<void*>(d_s8), static_cast<void*>(d_x),
                    static_cast<void*>(d_y), static_cast<void*>(d_y1)})
        cudaFree(p);
    cudaStreamDestroy(st);
}

}  // namespace
}  // namespace imp
