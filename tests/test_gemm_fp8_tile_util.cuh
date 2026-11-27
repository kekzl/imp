#pragma once
// Shared by tests/test_gemm_fp8_tile.cu and tests/bench/test_gemm_fp8_tile_bench.cu.
#include "compute/gemm.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr float kAScale = 0.37f;
constexpr float kWScale = 0.051f;

// Finite E4M3 bytes from uniform floats in [-2, 2]; a NaN encoding (0x7F / 0xFF) cannot come out of the cast.
std::vector<uint8_t> random_e4m3(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h = h * 1664525u + 1013904223u;
        const float f = (static_cast<float>(h >> 8) / 8388608.0f - 1.0f) * 2.0f;
        v[i] = static_cast<uint8_t>(__nv_fp8_e4m3(f).__x);
    }
    return v;
}

struct Operands {
    int M = 0, N = 0, K = 0;
    void* a = nullptr;
    void* w = nullptr;
    float* a_scale = nullptr;
    float* w_scale = nullptr;
    half* y_ref = nullptr;
    half* y_own = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
};

void make_operands(Operands& o, int M, int N, int K) {
    o.M = M;
    o.N = N;
    o.K = K;
    const auto a = random_e4m3(size_t(M) * K, 0x1234u);
    const auto w = random_e4m3(size_t(N) * K, 0x9876u);
    ASSERT_EQ(cudaMalloc(&o.a, a.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.w, w.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(o.a, a.data(), a.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(o.w, w.data(), w.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.a_scale, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.w_scale, sizeof(float)), cudaSuccess);
    const float as = kAScale, ws = kWScale;
    ASSERT_EQ(cudaMemcpy(o.a_scale, &as, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(o.w_scale, &ws, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_ref, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_own, size_t(M) * N * sizeof(half)), cudaSuccess);
    o.ws_bytes = gemm_tile_streamk_workspace_bytes();
    ASSERT_EQ(cudaMalloc(&o.ws, o.ws_bytes), cudaSuccess);
}

void free_operands(Operands& o) {
    for (void* p : {o.a, o.w, static_cast<void*>(o.a_scale), static_cast<void*>(o.w_scale),
                    static_cast<void*>(o.y_ref), static_cast<void*>(o.y_own), o.ws})
        cudaFree(p);
    o = Operands{};
}

// cuBLASLt reference: D = a_scale * w_scale * A W^T + beta * D.
void run_ref(const Operands& o, float beta, cudaStream_t s) {
    const int64_t ash[2] = {o.M, o.K}, wsh[2] = {o.N, o.K}, csh[2] = {o.M, o.N};
    Tensor A(o.a, QType::FP8_E4M3, 2, ash, true);
    Tensor W(o.w, QType::FP8_E4M3, 2, wsh, true);
    Tensor C(o.y_ref, QType::F16, 2, csh, true);
    gemm_cublaslt(A, W, C, 1.0f, beta, o.a_scale, o.w_scale, s);
}

bool run_own(const Operands& o, float beta, int variant, cudaStream_t s) {
    const bool sk = variant == 3 || variant == 4 || variant == 6 || variant < 0;
    return gemm_fp8_tile({.a_data = o.a,
                          .w_data = o.w,
                          .a_scale = o.a_scale,
                          .w_scale = o.w_scale,
                          .beta = beta,
                          .d_fp16 = o.y_own,
                          .M = o.M,
                          .N = o.N,
                          .K = o.K,
                          .stream = s,
                          .variant = variant,
                          .workspace = sk ? o.ws : nullptr,
                          .workspace_bytes = sk ? o.ws_bytes : 0});
}

// 0 128x128, 1 128x256, 2 sync, 3 stream-K, 4 hybrid / split, 5 128x64, 6 128x64 split, -1 (auto)
constexpr int kVariants = 8;
int variant_of(int i) { return i < 7 ? i : -1; }

}  // namespace
