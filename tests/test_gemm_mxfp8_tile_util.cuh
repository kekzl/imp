#pragma once
// Shared by tests/test_gemm_mxfp8_tile.cu and tests/bench/test_gemm_mxfp8_tile_bench.cu.
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

namespace {

__global__ void fill_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

struct Operands {
    int M = 0, N = 0, K = 0;
    CutlassMxFP8Weight cw;
    void* act = nullptr;
    void* act_sf = nullptr;
    float* row_scales = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    half* y_own = nullptr;
};

// rows = true: W as the FP8 row-scale view (E4M3 rows, SFB = power of two, col_scale residual).
void make_operands(Operands& o, int M, int N, int K, bool rows, cudaStream_t s) {
    o.M = M;
    o.N = N;
    o.K = K;
    half* w = nullptr;
    half* x = nullptr;
    ASSERT_EQ(cudaMalloc(&w, size_t(N) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&x, size_t(M) * K * sizeof(half)), cudaSuccess);
    fill_kernel<<<1024, 256, 0, s>>>(w, size_t(N) * K, 0x1234u, 0.08f);
    fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M) * K, 0x9876u, 2.0f);
    o.cw.N = N;
    o.cw.K = K;
    o.cw.data_bytes = size_t(N) * K;
    o.cw.sf_bytes = cutlass_mxfp8_sf_size(N, K);
    ASSERT_EQ(cudaMalloc(&o.cw.data, o.cw.data_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.cw.scale_factors, o.cw.sf_bytes), cudaSuccess);
    if (rows) {
        float* col = nullptr;
        ASSERT_EQ(cudaMalloc(&o.row_scales, size_t(N) * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&col, size_t(N) * sizeof(float)), cudaSuccess);
        quantize_fp8_rows_async(w, o.cw.data, N, K, o.row_scales, s);
        mxfp8_sf_from_row_scales(o.row_scales, o.cw.scale_factors, col, N, K, s);
        o.cw.col_scale = col;
    } else {
        quantize_fp16_to_mxfp8_cutlass(w, o.cw.data, o.cw.scale_factors, N, K, s);
    }
    ASSERT_EQ(cudaMalloc(&o.act, size_t(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.act_sf, cutlass_mxfp8_sf_size(M, K)), cudaSuccess);
    quantize_fp16_to_mxfp8_cutlass(x, o.act, o.act_sf, M, K, s);
    o.ws_bytes = gemm_tile_streamk_workspace_bytes();
    ASSERT_EQ(cudaMalloc(&o.ws, o.ws_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_own, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    cudaFree(w);
    cudaFree(x);
}

void free_operands(Operands& o) {
    for (void* p : {o.cw.data, o.cw.scale_factors, static_cast<void*>(const_cast<float*>(o.cw.col_scale)),
                    static_cast<void*>(o.row_scales), o.act, o.act_sf, o.ws, static_cast<void*>(o.y_own)})
        cudaFree(p);
}

constexpr int kVariants = 5;  // 0 128x128 PW, 1 128x256, 2 sync, 3 stream-K, then -1 (auto)

}  // namespace
