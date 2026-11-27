#pragma once
// Shared by tests/test_inkling_attn_paged.cu and tests/bench/test_inkling_attn_paged_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "compute/inkling_ops.h"

namespace {

constexpr int kNkv = 8, kHd = 128, kDr = 16, kExt = 1024, kKd = kNkv * kHd;

template <typename T>
T* upload(const std::vector<T>& h, std::vector<void*>& owned) {
    T* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess)
        return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    owned.push_back(d);
    return d;
}

// A shuffled table of ceil(ctx / block) entries over ctx / block + 5 pool blocks, and the pool with key k
// of kv ([2][ctx, kKd]: K then V) at row tbl[k / block] * block + k % block; unused rows hold other noise.
struct Paged {
    std::vector<int> tbl;
    std::vector<half> pool;  // [2][pool rows, kKd]
    size_t rows = 0;
};

Paged make_paged(const std::vector<half>& kv, int ctx, int block, uint32_t seed) {
    Paged p;
    const int n_log = (ctx + block - 1) / block, n_phys = n_log + 5;
    std::vector<int> ids(static_cast<size_t>(n_phys));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    p.tbl.assign(ids.begin(), ids.begin() + n_log);
    p.rows = static_cast<size_t>(n_phys) * block;
    p.pool.resize(2 * p.rows * kKd);
    std::uniform_real_distribution<float> noise(-3.0f, 3.0f);
    for (auto& v : p.pool)
        v = __float2half_rn(noise(rng));
    for (int side = 0; side < 2; ++side)
        for (int k = 0; k < ctx; ++k) {
            const size_t at = static_cast<size_t>(p.tbl[static_cast<size_t>(k / block)]) * block + k % block;
            std::memcpy(&p.pool[(side * p.rows + at) * kKd], &kv[(static_cast<size_t>(side) * ctx + k) * kKd],
                        kKd * sizeof(half));
        }
    return p;
}

// Inputs of one attention call: rows of q | rel | sink | gate, scales, distance table.
struct Inputs {
    std::vector<void*> owned;
    half* x = nullptr;
    float *qs = nullptr, *rs = nullptr, *proj = nullptr;
    int stride = 0;
};

Inputs make_inputs(int nh, int rows, uint32_t seed) {
    Inputs in;
    const int qd = nh * kHd, rd = nh * kDr;
    in.stride = 2 * qd + rd + nh;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    std::vector<half> x(static_cast<size_t>(rows) * in.stride);
    std::vector<float> qs(static_cast<size_t>(nh) * kHd), rs(static_cast<size_t>(nh) * kDr),
        proj(static_cast<size_t>(kExt) * kDr);
    for (auto& v : x)
        v = __float2half_rn(4.0f * u(rng));
    for (auto& v : qs)
        v = 1.0f + u(rng);
    for (auto& v : rs)
        v = 1.0f + u(rng);
    for (auto& v : proj)
        v = u(rng);
    in.x = upload(x, in.owned), in.qs = upload(qs, in.owned), in.rs = upload(rs, in.owned);
    in.proj = upload(proj, in.owned);
    return in;
}

inkling::AttnArgs base_args(const Inputs& in, int nh, int ctx, half* out, const int* d_pos) {
    inkling::AttnArgs a;
    const int qd = nh * kHd, rd = nh * kDr;
    a.q = in.x, a.rel = in.x + qd, a.sink = in.x + qd + rd, a.gate = in.x + qd + rd + nh;
    a.q_scale = in.qs, a.rel_scale = in.rs, a.proj = in.proj;
    a.out = out, a.d_pos = d_pos, a.max_ctx = ctx;
    a.stride = in.stride, a.n_heads = nh, a.n_kv = kNkv, a.hd = kHd, a.d_rel = kDr, a.rel_extent = kExt;
    a.scale = 1.0f / std::sqrt(static_cast<float>(kHd)), a.q_thr = 22.6f, a.rel_thr = 8.0f;
    a.log_alpha = 0.1f, a.log_floor = 1024.0f;
    return a;
}

// Output of one call; out is filled with NaN first so a path that writes nothing cannot match.
std::vector<half> run(const inkling::AttnArgs& a, bool gemm) {
    const size_t n = static_cast<size_t>(a.T) * a.n_heads * a.hd;
    cudaMemset(a.out, 0xff, n * sizeof(half));
    if (gemm)
        EXPECT_TRUE(inkling::attention_rows(a, nullptr));
    else
        inkling::attention(a, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> h(n);
    cudaMemcpy(h.data(), a.out, n * sizeof(half), cudaMemcpyDeviceToHost);
    return h;
}

size_t differing(const std::vector<half>& x, const std::vector<half>& y) {
    size_t d = x.size() == y.size() ? 0 : x.size() + y.size();
    for (size_t i = 0; i < std::min(x.size(), y.size()); ++i)
        d += std::memcmp(&x[i], &y[i], sizeof(half)) != 0;
    return d;
}

}  // namespace
