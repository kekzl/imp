#pragma once
// Shared by tests/test_gemm_f16_tile.cu and tests/bench/test_gemm_f16_tile_bench.cu.
#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

__global__ void fill_f16_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

struct Bufs {
    half *a = nullptr, *w = nullptr, *ref = nullptr, *own = nullptr;
    int M = 0, N = 0, K = 0;
    void alloc(int m, int n, int k, cudaStream_t s) {
        M = m;
        N = n;
        K = k;
        cudaMalloc(&a, size_t(M) * K * 2);
        cudaMalloc(&w, size_t(N) * K * 2);
        cudaMalloc(&ref, size_t(M) * N * 2);
        cudaMalloc(&own, size_t(M) * N * 2);
        fill_f16_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u, 1.0f);
        fill_f16_kernel<<<1024, 256, 0, s>>>(w, size_t(N) * K, 0x77u, 0.05f);
        fill_f16_kernel<<<1024, 256, 0, s>>>(ref, size_t(M) * N, 0x99u, 1.0f);
        cudaMemcpyAsync(own, ref, size_t(M) * N * 2, cudaMemcpyDeviceToDevice, s);
    }
    void release() {
        cudaFree(a);
        cudaFree(w);
        cudaFree(ref);
        cudaFree(own);
    }
    void lib(float beta, cudaStream_t s) const {
        int64_t as[2] = {M, K}, ws[2] = {N, K}, ds[2] = {M, N};
        Tensor ta(a, QType::F16, 2, as, true), tw(w, QType::F16, 2, ws, true),
            td(ref, QType::F16, 2, ds, true);
        gemm(ta, tw, td, 1.0f, beta, s);
    }
};

}  // namespace
