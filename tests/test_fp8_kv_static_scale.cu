// Static power-of-two FP8 KV scale (kv_cache.fp8_scale, src/core/kv_fp8_scale.h) through the real
// write (write_kv_cache_fp8_fused_kernel) and read (paged_kv_gather_fp8_to_fp16) kernels against a
// double reference. Mutant: drop the inv_scale multiply in write_kv_cache_fp8_fused_kernel
// (executor_kv_write_kernels.cu) -> StaticScaleWithinE4M3Bound fails.

#include "compute/attention_paged.h"
#include "compute/kv_gather.h"
#include "core/kv_fp8_scale.h"
#include "exec/executor_kernels.h"
#include "exec/executor_kernels.cuh"
#include "test_cuda_skip.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

namespace imp {
namespace {

constexpr int kBlockSize = 16;
constexpr int kNkv = 2;
constexpr int kHd = 128;
constexpr int kRow = kNkv * kHd;
constexpr int kTokens = 64;                   // 4 blocks
constexpr double kE4M3NormalMin = 0.015625;   // 2^-6
constexpr double kE4M3SubStep = 0.001953125;  // 2^-9

// V-like layer: amax 3.0 with most mass below the E4M3 normal floor at scale 1 (Qwen3-1.7B V amax
// from 1.2, docs/DETERMINISM.md). Deterministic seed; one exact amax element.
std::vector<half> make_rows(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> small(
        0.0f, 0.003f);  // unit-scale MSE / static MSE ~56 (step 2^-9 vs 2.5 % rms)
    std::normal_distribution<float> wide(0.0f, 0.6f);
    std::vector<half> x(static_cast<size_t>(kTokens) * kRow);
    for (size_t i = 0; i < x.size(); ++i) {
        float v = (i % 4 == 0) ? wide(rng) : small(rng);
        v = std::fmax(-3.0f, std::fmin(3.0f, v));
        x[i] = __float2half(v);
    }
    x[17] = __float2half(3.0f);
    return x;
}

// K = xk, V = xv written at scales sk / sv into a paged FP8 cache [n_blocks, kBlockSize, kNkv, kHd]
// (non-identity block table); gather() reads one side back to FP16 [kTokens, kRow].
struct Fp8PagedCache {
    static constexpr int kBlocks = kTokens / kBlockSize;
    static constexpr size_t kCacheElems = static_cast<size_t>(kBlocks) * kBlockSize * kRow;
    __nv_fp8_e4m3* kc = nullptr;
    __nv_fp8_e4m3* vc = nullptr;
    int* bt = nullptr;

    Fp8PagedCache(const std::vector<half>& xk, const std::vector<half>& xv, float sk, float sv) {
        half* d_k = nullptr;
        half* d_v = nullptr;
        int* d_pos = nullptr;
        std::vector<int> pos(kTokens);
        for (int i = 0; i < kTokens; ++i)
            pos[i] = i;
        std::vector<int> h_bt(kBlocks);
        for (int b = 0; b < kBlocks; ++b)
            h_bt[b] = kBlocks - 1 - b;
        cudaMalloc(&d_k, xk.size() * sizeof(half));
        cudaMalloc(&d_v, xv.size() * sizeof(half));
        cudaMalloc(&d_pos, kTokens * sizeof(int));
        cudaMalloc(&bt, kBlocks * sizeof(int));
        cudaMalloc(&kc, kCacheElems);
        cudaMalloc(&vc, kCacheElems);
        cudaMemcpy(d_k, xk.data(), xk.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, xv.data(), xv.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, pos.data(), kTokens * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(bt, h_bt.data(), kBlocks * sizeof(int), cudaMemcpyHostToDevice);
        write_kv_cache_fp8_fused_kernel<<<dim3(kTokens, 2), 256>>>(d_k, d_v, d_pos, bt, kc, vc, 1.0f / sk,
                                                                   1.0f / sv, kBlockSize * kRow, kRow,
                                                                   kBlockSize, kTokens, 0, 1);
        cudaDeviceSynchronize();
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_pos);
    }
    ~Fp8PagedCache() {
        cudaFree(kc);
        cudaFree(vc);
        cudaFree(bt);
    }
    Fp8PagedCache(const Fp8PagedCache&) = delete;
    Fp8PagedCache& operator=(const Fp8PagedCache&) = delete;

    [[nodiscard]] std::vector<half> gather(bool v_side, float scale) const {
        half* d_out = nullptr;
        cudaMalloc(&d_out, static_cast<size_t>(kTokens) * kRow * sizeof(half));
        paged_kv_gather_fp8_to_fp16(d_out, v_side ? vc : kc, bt, scale, kTokens, kBlockSize, kNkv, kHd,
                                    nullptr);
        std::vector<half> out(static_cast<size_t>(kTokens) * kRow);
        cudaMemcpy(out.data(), d_out, out.size() * sizeof(half), cudaMemcpyDeviceToHost);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        cudaFree(d_out);
        return out;
    }
};

// K and V both = x; returns the gathered FP16 K rows [kTokens, kRow].
std::vector<half> write_then_gather(const std::vector<half>& x, float scale) {
    return Fp8PagedCache(x, x, scale, scale).gather(false, scale);
}

// #3002 invariant at a non-unit scale: the in-place prefill round trip equals write + gather.
// Returns {K, V} rounded at sk / sv.
std::pair<std::vector<half>, std::vector<half>> round_inplace(const std::vector<half>& xk,
                                                              const std::vector<half>& xv, float sk,
                                                              float sv) {
    half* d_k = nullptr;
    half* d_v = nullptr;
    cudaMalloc(&d_k, xk.size() * sizeof(half));
    cudaMalloc(&d_v, xv.size() * sizeof(half));
    cudaMemcpy(d_k, xk.data(), xk.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, xv.data(), xv.size() * sizeof(half), cudaMemcpyHostToDevice);
    round_kv_fp8_inplace_kernel<<<dim3(kTokens, 2), 256>>>(d_k, d_v, kRow, kRow, kRow, kTokens, 1.0f / sk, sk,
                                                           1.0f / sv, sv);
    std::vector<half> k(xk.size());
    std::vector<half> v(xv.size());
    cudaMemcpy(k.data(), d_k, k.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), d_v, v.size() * sizeof(half), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    cudaFree(d_k);
    cudaFree(d_v);
    return {k, v};
}

float static_scale_for(const std::vector<half>& x) {
    float amax = 0.0f;
    for (half h : x)
        amax = std::fmax(amax, std::fabs(__half2float(h)));
    return fp8_kv_scale_from_amax(amax, amax, 2.0f);  // amax 3 -> 3*2/448 -> 2^-6
}

uint16_t bits(half h) {
    uint16_t b = 0;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

TEST(Fp8KvStaticScale, StaticScaleWithinE4M3Bound) {
    SKIP_IF_NO_CUDA();
    const auto x = make_rows(1234);
    const float s = static_scale_for(x);
    ASSERT_EQ(s, 0.015625f);
    const auto y = write_then_gather(x, s);
    for (size_t i = 0; i < x.size(); ++i) {
        const double ref = __half2float(x[i]);
        const double got = __half2float(y[i]);
        const double a = std::fabs(ref);
        // Normal E4M3 range: half an ULP of a 3-bit mantissa (2^-4 relative) plus FP16 output (2^-11).
        // Below it: half a subnormal step, both scaled by s.
        const double bound = a >= kE4M3NormalMin * s ? a * (0.0625 + 0.00049) : 0.5 * kE4M3SubStep * s + 1e-7;
        ASSERT_LE(std::fabs(got - ref), bound) << "i=" << i << " ref=" << ref << " got=" << got;
    }
}

TEST(Fp8KvStaticScale, PowerOfTwoKeepsTheUnitGridOnNormals) {
    SKIP_IF_NO_CUDA();
    const auto x = make_rows(99);
    const float s = static_scale_for(x);
    const auto y_static = write_then_gather(x, s);
    const auto y_unit = write_then_gather(x, 1.0f);
    int compared = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double a = std::fabs(__half2float(x[i]));
        if (a < kE4M3NormalMin)
            continue;  // subnormal at scale 1: the only place a static scale may differ
        ASSERT_EQ(bits(y_static[i]), bits(y_unit[i])) << "i=" << i;
        ++compared;
    }
    EXPECT_GT(compared, 1000);
}

TEST(Fp8KvStaticScale, StaticScaleCutsSubnormalError) {
    SKIP_IF_NO_CUDA();
    const auto x = make_rows(7);
    const float s = static_scale_for(x);
    const auto y_static = write_then_gather(x, s);
    const auto y_unit = write_then_gather(x, 1.0f);
    double se_static = 0.0;
    double se_unit = 0.0;
    int small = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double ref = __half2float(x[i]);
        if (std::fabs(ref) >= kE4M3NormalMin)
            continue;
        se_static += std::pow(__half2float(y_static[i]) - ref, 2);
        se_unit += std::pow(__half2float(y_unit[i]) - ref, 2);
        ++small;
    }
    ASSERT_GT(small, 5000);
    // Unit scale: absolute step 2^-9 on |x| < 2^-6; scale 2^-6 makes them normal (2^-4 relative).
    EXPECT_LT(se_static * 16.0, se_unit) << "static " << se_static << " unit " << se_unit;
}

TEST(Fp8KvStaticScale, InplaceRoundTripMatchesWriteGather) {
    SKIP_IF_NO_CUDA();
    const auto x = make_rows(5);
    const float s = static_scale_for(x);
    const auto a = write_then_gather(x, s);
    const auto b = round_inplace(x, x, s, s).first;
    for (size_t i = 0; i < x.size(); ++i)
        ASSERT_EQ(bits(a[i]), bits(b[i])) << "i=" << i;
}

// Gemma-4-like layer (docs/DETERMINISM.md): K amax ~1.5, V amax ~15, so the split scales differ by 2^4.
std::vector<half> scaled_rows(uint32_t seed, float f) {
    auto x = make_rows(seed);
    for (half& h : x)
        h = __float2half(__half2float(h) * f);
    return x;
}

float amax_of(const std::vector<half>& x) {
    float a = 0.0f;
    for (half h : x)
        a = std::fmax(a, std::fabs(__half2float(h)));
    return a;
}

// Mutant: write_kv_cache_fp8_fused_kernel / round_kv_fp8_inplace_kernel use the K scale for V -> fails.
TEST(Fp8KvStaticScale, SplitScalesRoundTripPerSide) {
    SKIP_IF_NO_CUDA();
    const auto xk = scaled_rows(11, 0.5f);
    const auto xv = scaled_rows(12, 5.0f);
    const Fp8KvScalePair s = fp8_kv_scales_from_amax(amax_of(xk), amax_of(xv), 2.0f, true);
    ASSERT_EQ(s.k, 0.0078125f);  // 1.5 * 2 / 448 -> 2^-7
    ASSERT_EQ(s.v, 0.125f);      // 15 * 2 / 448 -> 2^-3
    const Fp8PagedCache cache(xk, xv, s.k, s.v);
    const auto gk = cache.gather(false, s.k);
    const auto gv = cache.gather(true, s.v);
    const auto [rk, rv] = round_inplace(xk, xv, s.k, s.v);
    for (size_t i = 0; i < xk.size(); ++i) {
        ASSERT_EQ(bits(gk[i]), bits(rk[i])) << "K i=" << i;
        ASSERT_EQ(bits(gv[i]), bits(rv[i])) << "V i=" << i;
        // Normal range: 2^-4 relative (3-bit mantissa) plus the FP16 output.
        for (const auto& [ref_h, got_h, sc] :
             {std::tuple{xk[i], gk[i], s.k}, std::tuple{xv[i], gv[i], s.v}}) {
            const double ref = __half2float(ref_h);
            const double got = __half2float(got_h);
            const double a = std::fabs(ref);
            const double bound = a >= kE4M3NormalMin * sc ? a * (0.0625 + 0.00049)
                                                          : 0.5 * kE4M3SubStep * sc + 1e-7;
            ASSERT_LE(std::fabs(got - ref), bound) << "i=" << i << " ref=" << ref << " got=" << got;
        }
    }
}

// The FP8 decode kernels take one kv_scale; fp8_kv_decode_scales folds k / v into the softmax scale.
// Reference: double attention over the gathered K (s.k) and V (s.v). Mutant: pass (scale, s.v) unfolded
// in executor_attention_decode.cpp's shape -> logits 16x off -> fails.
TEST(Fp8KvStaticScale, SplitScalesDecodeFoldMatchesReference) {
    SKIP_IF_NO_CUDA();
    constexpr int kHeads = 4;  // GQA 2
    const auto xk = scaled_rows(21, 0.5f);
    const auto xv = scaled_rows(22, 5.0f);
    const Fp8KvScalePair s = fp8_kv_scales_from_amax(amax_of(xk), amax_of(xv), 2.0f, true);
    ASSERT_NE(s.k, s.v);
    const Fp8PagedCache cache(xk, xv, s.k, s.v);
    const auto gk = cache.gather(false, s.k);
    const auto gv = cache.gather(true, s.v);

    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<half> q(static_cast<size_t>(kHeads) * kHd);
    for (half& h : q)
        h = __float2half(u(rng) * 4.0f);
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHd));

    half* d_q = nullptr;
    half* d_o = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_q, q.size() * sizeof(half));
    cudaMalloc(&d_o, q.size() * sizeof(half));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx, &kTokens, sizeof(int), cudaMemcpyHostToDevice);
    const int64_t q_shape[4] = {1, 1, kHeads, kHd};
    const int64_t kv_shape[4] = {Fp8PagedCache::kBlocks, kBlockSize, kNkv, kHd};
    Tensor tq(d_q, QType::F16, 4, q_shape, true);
    Tensor to(d_o, QType::F16, 4, q_shape, true);
    Tensor tk(cache.kc, QType::FP8_E4M3, 4, kv_shape, true);
    Tensor tv(cache.vc, QType::FP8_E4M3, 4, kv_shape, true);
    const Fp8KvDecodeScales ds = fp8_kv_decode_scales(softmax_scale, s);
    EXPECT_EQ(ds.softmax_scale * ds.kv_scale, softmax_scale * s.k);  // exact: power-of-two ratio
    paged_attention_set_splitk_scratch(nullptr, 0);
    paged_attention_decode_fp8(tq, tk, tv, to, cache.bt, d_ctx, kBlockSize, ds.softmax_scale, ds.kv_scale,
                               kTokens);
    std::vector<half> o(q.size());
    cudaMemcpy(o.data(), d_o, o.size() * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaFree(d_q);
    cudaFree(d_o);
    cudaFree(d_ctx);

    double max_logit_spread = 0.0;
    for (int h = 0; h < kHeads; ++h) {
        const int kvh = h / (kHeads / kNkv);
        std::vector<double> logit(kTokens);
        double mx = -1e300;
        double mn = 1e300;
        for (int t = 0; t < kTokens; ++t) {
            double dot = 0.0;
            for (int d = 0; d < kHd; ++d)
                dot += static_cast<double>(__half2float(q[(h * kHd) + d])) *
                       __half2float(gk[(static_cast<size_t>(t) * kRow) + (kvh * kHd) + d]);
            logit[t] = dot * softmax_scale;
            mx = std::max(mx, logit[t]);
            mn = std::min(mn, logit[t]);
        }
        max_logit_spread = std::max(max_logit_spread, mx - mn);
        double z = 0.0;
        for (double& l : logit)
            z += (l = std::exp(l - mx));
        for (int d = 0; d < kHd; ++d) {
            double ref = 0.0;
            for (int t = 0; t < kTokens; ++t)
                ref += logit[t] / z * __half2float(gv[(static_cast<size_t>(t) * kRow) + (kvh * kHd) + d]);
            const double got = __half2float(o[(h * kHd) + d]);
            ASSERT_NEAR(got, ref, 2e-3 + (4e-3 * std::fabs(ref))) << "head " << h << " d " << d;
        }
    }
    EXPECT_GT(max_logit_spread, 0.5) << "flat softmax: a wrong K scale would pass unnoticed";
}

}  // namespace
}  // namespace imp
