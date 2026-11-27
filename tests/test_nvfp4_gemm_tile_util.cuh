#pragma once
// Shared by tests/test_nvfp4_gemm_tile.cu and tests/bench/test_nvfp4_gemm_tile_bench.cu.
#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "nvfp4_ref_gemm.cuh"
#include "quant/nvfp4_gemm_tile.h"
#include "compute/cutlass_nvfp4_owned.h"
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

namespace {

struct Operands {
    int M = 0, N = 0, K = 0;
    NvFP4Owned qr;
    NvFP4SfAtomOwned cw;
    void* act = nullptr;
    void* act_sf = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    half* y_ref = nullptr;
    half* y_own = nullptr;
};

__global__ void fill_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

void make_operands(Operands& o, int M, int N, int K, cudaStream_t s) {
    o.M = M;
    o.N = N;
    o.K = K;
    half* w = nullptr;
    half* x = nullptr;
    ASSERT_EQ(cudaMalloc(&w, size_t(N) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&x, size_t(M) * K * sizeof(half)), cudaSuccess);
    fill_kernel<<<1024, 256, 0, s>>>(w, size_t(N) * K, 0x1234u, 0.08f);
    fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M) * K, 0x9876u, 2.0f);
    int64_t wshape[2] = {N, K};
    Tensor wt(w, QType::F16, 2, wshape, true);
    quantize_fp16_to_nvfp4(wt, o.qr, s);
    convert_nvfp4_to_sfatom(o.qr, o.cw, s);
    ASSERT_EQ(cudaMalloc(&o.act, size_t(M) * K / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.act_sf, nvfp4_sfatom_size(M, K)), cudaSuccess);
    cudaMemsetAsync(o.act_sf, 0, nvfp4_sfatom_size(M, K), s);
    quantize_fp16_to_nvfp4_sfatom(x, o.act, o.act_sf, M, K, s);
    // Stream-K need: 4 KiB flags + 170 x 64 KiB partials (the 5090 SM count) as the floor.
    o.ws_bytes = std::max<size_t>(gemm_tile_streamk_workspace_bytes(), 4096 + (size_t(170) << 16));
    ASSERT_EQ(cudaMalloc(&o.ws, std::max<size_t>(o.ws_bytes, 1)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_ref, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_own, size_t(M) * N * sizeof(half)), cudaSuccess);
    cudaStreamSynchronize(s);
    cudaFree(w);
    cudaFree(x);
}

void free_operands(Operands& o) {
    o.cw = NvFP4SfAtomOwned{};
    o.qr = NvFP4Owned{};
    cudaFree(o.act);
    cudaFree(o.act_sf);
    cudaFree(o.ws);
    cudaFree(o.y_ref);
    cudaFree(o.y_own);
}

}  // namespace
