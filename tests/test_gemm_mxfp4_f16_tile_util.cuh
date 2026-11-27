#pragma once
// Shared by tests/test_gemm_mxfp4_f16_tile.cu and tests/bench/test_gemm_mxfp4_f16_tile_bench.cu.
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

namespace {

__global__ void fill_act_kernel(half* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f);
    }
}

struct Mx4Bufs {
    int M = 0, N = 0, K = 0;
    uint8_t *w = nullptr, *sc = nullptr;
    half *a = nullptr, *w16 = nullptr, *ref = nullptr, *own = nullptr;
    // Any E2M1 bytes; UE8M0 exponents 2^-16 .. 2^1, varied per 32-k group.
    void alloc(int m, int n, int k, cudaStream_t s) {
        M = m;
        N = n;
        K = k;
        std::mt19937 rng(0x4u + n + k);
        std::vector<uint8_t> hw(size_t(N) * (K / 2)), hs(size_t(N) * (K / 32));
        for (auto& b : hw)
            b = static_cast<uint8_t>(rng());
        for (auto& e : hs)
            e = static_cast<uint8_t>(111 + (rng() % 18));
        cudaMalloc(&w, hw.size());
        cudaMalloc(&sc, hs.size());
        cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(sc, hs.data(), hs.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&a, size_t(M) * K * 2);
        cudaMalloc(&w16, size_t(N) * K * 2);
        cudaMalloc(&ref, size_t(M) * N * 2);
        cudaMalloc(&own, size_t(M) * N * 2);
        fill_act_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u);
        fill_act_kernel<<<1024, 256, 0, s>>>(ref, size_t(M) * N, 0x99u);
        cudaMemcpyAsync(own, ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s);
        dequant_mxfp4_linear_to_fp16(w, sc, N, K, w16, s);
    }
    void release() const {
        cudaFree(w);
        cudaFree(sc);
        cudaFree(a);
        cudaFree(w16);
        cudaFree(ref);
        cudaFree(own);
    }
};

}  // namespace
