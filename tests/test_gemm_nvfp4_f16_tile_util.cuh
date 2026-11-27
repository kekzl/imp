#pragma once
// Shared by tests/test_gemm_nvfp4_f16_tile.cu and tests/bench/test_gemm_nvfp4_f16_tile_bench.cu.
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_grouped_nvfp4_f16_tile.h"
#include "compute/gemm_nvfp4_f16_tile.h"
#include "core/tensor.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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

float e4m3_host(uint8_t b) {
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    const float v = e == 0 ? (m / 8.0f) * std::ldexp(1.0f, -6)
                           : (1.0f + (m / 8.0f)) * std::ldexp(1.0f, e - 7);
    return (b & 0x80) != 0 ? -v : v;
}

float e2m1_host(uint8_t c) {
    static const float kV[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 0x8) != 0 ? -kV[c & 0x7] : kV[c & 0x7];
}

struct Nv4Bufs {
    int M = 0, N = 0, K = 0;
    uint8_t *w = nullptr, *sc = nullptr;
    half *a = nullptr, *w16 = nullptr, *ref = nullptr, *own = nullptr;
    // Any E2M1 bytes; positive E4M3 scales lo .. hi (0x01 subnormal .. 0x7E = 448), varied per 16-k block.
    void alloc(int m, int n, int k, int lo, int hi, cudaStream_t s) {
        M = m;
        N = n;
        K = k;
        std::mt19937 rng(0x7u + n + k);
        std::vector<uint8_t> hw(size_t(N) * (K / 2)), hs(size_t(N) * (K / 16));
        for (auto& b : hw)
            b = static_cast<uint8_t>(rng());
        for (auto& e : hs)
            e = static_cast<uint8_t>(lo + (rng() % (hi - lo + 1)));
        std::vector<half> h16(size_t(N) * K);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < K; ++c) {
                const uint8_t byte = hw[(size_t(r) * (K / 2)) + (c / 2)];
                const uint8_t code = (c & 1) != 0 ? byte >> 4 : byte & 0xF;
                const float sc16 = e4m3_host(hs[(size_t(r) * (K / 16)) + (c / 16)]);
                h16[(size_t(r) * K) + c] = __float2half(e2m1_host(code) * sc16 * (1.0f / 256.0f));
            }
        cudaMalloc(&w, hw.size());
        cudaMalloc(&sc, hs.size());
        cudaMalloc(&w16, h16.size() * 2);
        cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(sc, hs.data(), hs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(w16, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc(&a, size_t(M) * K * 2);
        cudaMalloc(&ref, size_t(M) * N * 2);
        cudaMalloc(&own, size_t(M) * N * 2);
        fill_act_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u);
        fill_act_kernel<<<1024, 256, 0, s>>>(ref, size_t(M) * N, 0x99u);
        cudaMemcpyAsync(own, ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s);
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
