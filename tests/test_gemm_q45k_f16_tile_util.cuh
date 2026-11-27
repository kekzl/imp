#pragma once
// Shared by tests/test_gemm_q45k_f16_tile.cu and tests/bench/test_gemm_q45k_f16_tile_bench.cu.
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

namespace {

// Random Q4_K / Q5_K blocks: any scale/qh/qs bytes, d and dmin finite fp16 in [1e-3, 4e-3) (the min term
// varies per sub-block, so a dropped dmin or swapped scale/min nibble shows).
std::vector<uint8_t> random_kq(bool q5, int N, int K, uint32_t seed) {
    const int bytes = q5 ? 176 : 144;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dd(1e-3f, 4e-3f);
    const size_t blocks = static_cast<size_t>(N) * (K / 256);
    std::vector<uint8_t> w(blocks * bytes);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = w.data() + (b * bytes);
        for (int i = 4; i < bytes; ++i)
            p[i] = static_cast<uint8_t>(rng());
        const half d = __float2half(dd(rng));
        const half m = __float2half(dd(rng));
        std::memcpy(p, &d, 2);
        std::memcpy(p + 2, &m, 2);
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

struct KqBufs {
    int M = 0, N = 0, K = 0;
    bool q5 = false;
    uint8_t* wq = nullptr;
    half *a = nullptr, *w16 = nullptr, *ref = nullptr, *own = nullptr;
    void alloc(bool q5k, int m, int n, int k, cudaStream_t s) {
        q5 = q5k;
        M = m;
        N = n;
        K = k;
        const auto host = random_kq(q5, N, K, 0x45u + N + K + (q5 ? 1 : 0));
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
    QType qtype() const { return q5 ? QType::Q5_K : QType::Q4_K; }
    void release() const {
        cudaFree(wq);
        cudaFree(a);
        cudaFree(w16);
        cudaFree(ref);
        cudaFree(own);
    }
};

}  // namespace
