#pragma once
// Shared by tests/test_fp8_gemm_smallm.cu and tests/bench/test_fp8_gemm_smallm_bench.cu.
#include "compute/gemm.h"
#include "core/tensor.h"
#include "quant/fp8_gemm_smallm.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr float kAScale = 0.37f;
constexpr float kWScale = 0.051f;
constexpr int kMaxM = kFp8SmallMMaxM;

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

float e4m3_to_float(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

// Operands for one (N, K): kMaxM activation rows, the weight, scales, outputs, double reference.
struct Shape {
    int N = 0, K = 0;
    std::vector<uint8_t> a, w;
    std::vector<double> ref;  // [kMaxM][N], scaled, beta 0
    uint8_t* d_a = nullptr;
    uint8_t* d_w = nullptr;
    float* d_as = nullptr;
    float* d_ws = nullptr;
    half* d_y = nullptr;
    half* d_y2 = nullptr;

    void init(int n, int k, bool want_ref) {
        N = n;
        K = k;
        a = random_e4m3(size_t(kMaxM) * K, 0x1234u + n + k);
        w = random_e4m3(size_t(N) * K, 0x9876u + n);
        ASSERT_EQ(cudaMalloc(&d_a, a.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_w, w.size()), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_as, sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_ws, sizeof(float)), cudaSuccess);
        const float as = kAScale, ws = kWScale;
        ASSERT_EQ(cudaMemcpy(d_as, &as, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_ws, &ws, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y2, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        if (!want_ref)
            return;
        float lut[256];
        for (int i = 0; i < 256; ++i)
            lut[i] = e4m3_to_float(static_cast<uint8_t>(i));
        std::vector<double> wf(size_t(N) * K), af(size_t(kMaxM) * K);
        for (size_t i = 0; i < wf.size(); ++i)
            wf[i] = lut[w[i]];
        for (size_t i = 0; i < af.size(); ++i)
            af[i] = lut[a[i]];
        ref.assign(size_t(kMaxM) * N, 0.0);
        const double sc = double(kAScale) * double(kWScale);
        for (int m = 0; m < kMaxM; ++m)
            for (int n2 = 0; n2 < N; ++n2) {
                const double* ar = &af[size_t(m) * K];
                const double* wr = &wf[size_t(n2) * K];
                double s = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    s += ar[kk] * wr[kk];
                ref[size_t(m) * N + n2] = s * sc;
            }
    }
    void free_all() {
        for (void* p : {static_cast<void*>(d_a), static_cast<void*>(d_w), static_cast<void*>(d_as),
                        static_cast<void*>(d_ws), static_cast<void*>(d_y), static_cast<void*>(d_y2)})
            cudaFree(p);
        *this = Shape{};
    }
};

// Qwen3-8B q/o, k/v, gate/up, down, plus a 3-CTA shape with K not a multiple of a full round.
const int kShapes[][2] = {{4096, 4096}, {1024, 4096}, {12288, 4096}, {4096, 12288}, {48, 192}};

}  // namespace
