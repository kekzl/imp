#pragma once
// Shared by tests/test_gemm_q6k_f16_tile.cu and tests/bench/test_gemm_q6k_f16_tile_bench.cu.
#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_q6k_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kQ6 = 210;

// Random Q6_K blocks: any ql/qh/scale bytes, d a finite fp16 in [1e-3, 4e-3).
std::vector<uint8_t> random_q6k(int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dd(1e-3f, 4e-3f);
    const size_t blocks = static_cast<size_t>(N) * (K / 256);
    std::vector<uint8_t> w(blocks * kQ6);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = w.data() + (b * kQ6);
        for (int i = 0; i < 208; ++i)
            p[i] = static_cast<uint8_t>(rng());
        const half d = __float2half(dd(rng));
        std::memcpy(p + 208, &d, 2);
    }
    return w;
}

__global__ void fill_act_kernel(half* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f);
    }
}

struct Q6Bufs {
    int M = 0, N = 0, K = 0;
    uint8_t* wq = nullptr;
    half *a = nullptr, *w16 = nullptr, *ref = nullptr, *own = nullptr;
    void alloc(int m, int n, int k, cudaStream_t s) {
        M = m;
        N = n;
        K = k;
        const auto host = random_q6k(N, K, 0x6u + N + K);
        cudaMalloc(&wq, host.size());
        cudaMemcpy(wq, host.data(), host.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&a, size_t(M) * K * 2);
        cudaMalloc(&w16, size_t(N) * K * 2);
        cudaMalloc(&ref, size_t(M) * N * 2);
        cudaMalloc(&own, size_t(M) * N * 2);
        fill_act_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u);
        fill_act_kernel<<<1024, 256, 0, s>>>(ref, size_t(M) * N, 0x99u);
        cudaMemcpyAsync(own, ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s);
    }
    void release() const {
        cudaFree(wq);
        cudaFree(a);
        cudaFree(w16);
        cudaFree(ref);
        cudaFree(own);
    }
};

}  // namespace
