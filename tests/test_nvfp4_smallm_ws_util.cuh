#pragma once
// Shared by tests/test_nvfp4_smallm_ws.cu and tests/bench/test_nvfp4_smallm_ws_bench.cu.
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kMaxM = kNvfp4SmallMWsMaxM;
constexpr float kWTs = 0.0625f;  // weight tensor scale (the activation side is 1, as the dispatch quantizes)

uint32_t lcg(uint32_t& h) {
    h = (h * 1664525u) + 1013904223u;
    return h >> 8;
}

// Random nibbles and UE4M3 scales 2^-2 .. 2^2 (exponent 5..9), never NaN.
void random_nvfp4(std::vector<uint8_t>& packed, std::vector<uint8_t>& sf, int rows, int K, uint32_t seed) {
    packed.resize(size_t(rows) * K / 2);
    sf.resize(size_t(rows) * K / 16);
    uint32_t h = seed;
    for (auto& b : packed)
        b = static_cast<uint8_t>(lcg(h));
    for (auto& s : sf)
        s = static_cast<uint8_t>(((5 + (lcg(h) % 5)) << 3) | (lcg(h) & 7));
}

double fp4(uint8_t nib) {
    static const double lut[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
    return lut[nib & 15];
}

double ue4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + (m / 8.0), e - 7);
}

std::vector<double> decode(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& sf, int rows,
                           int K) {
    std::vector<double> v(size_t(rows) * K);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) {
            const uint8_t byte = packed[(size_t(r) * K / 2) + (k / 2)];
            v[(size_t(r) * K) + k] = fp4((k & 1) ? (byte >> 4) : byte) *
                                     ue4m3(sf[(size_t(r) * K / 16) + (k / 16)]);
        }
    return v;
}

struct Dev {
    void* p = nullptr;
    void* s = nullptr;
    void up(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& sf) {
        ASSERT_EQ(cudaMalloc(&p, packed.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&s, sf.size()), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(p, packed.data(), packed.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(s, sf.data(), sf.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }
    NvFP4QuantResult q(int rows, int K, float ts) const {
        NvFP4QuantResult r;
        r.packed_data = p;
        r.micro_scales = s;
        r.tensor_scale = ts;
        r.N = rows;
        r.K = K;
        return r;
    }
    void free_all() {
        cudaFree(p);
        cudaFree(s);
        p = s = nullptr;
    }
};

// Operands for one (N, K): kMaxM activation rows, the weight, outputs, double reference (beta 0).
struct Shape {
    int N = 0, K = 0;
    std::vector<uint8_t> xp, xs, wp, wsf;
    std::vector<double> ref;
    Dev x, w;
    half* d_y = nullptr;
    half* d_y2 = nullptr;

    void init(int n, int k, bool want_ref, uint32_t seed = 1) {
        N = n;
        K = k;
        random_nvfp4(xp, xs, kMaxM, K, 0x1234u + n + k + seed);
        random_nvfp4(wp, wsf, N, K, 0x9876u + n + seed);
        x.up(xp, xs);
        w.up(wp, wsf);
        ASSERT_EQ(cudaMalloc(&d_y, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y2, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        if (!want_ref)
            return;
        const auto xf = decode(xp, xs, kMaxM, K);
        const auto wf = decode(wp, wsf, N, K);
        ref.assign(size_t(kMaxM) * N, 0.0);
        for (int m = 0; m < kMaxM; ++m)
            for (int r = 0; r < N; ++r) {
                double s = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    s += xf[(size_t(m) * K) + kk] * wf[(size_t(r) * K) + kk];
                ref[(size_t(m) * N) + r] = s * kWTs;
            }
    }
    NvFP4QuantResult W() const { return w.q(N, K, kWTs); }
    NvFP4QuantResult X() const { return x.q(kMaxM, K, 1.0f); }
    void upload_x(const std::vector<uint8_t>& p, const std::vector<uint8_t>& s) {
        ASSERT_EQ(cudaMemcpy(x.p, p.data(), p.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(x.s, s.data(), s.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }
    void free_all() {
        x.free_all();
        w.free_all();
        cudaFree(d_y);
        cudaFree(d_y2);
        *this = Shape{};
    }
};

}  // namespace
