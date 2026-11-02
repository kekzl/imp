// Static power-of-two FP8 KV scale (kv_cache.fp8_scale, src/core/kv_fp8_scale.h) through the real
// write (write_kv_cache_fp8_fused_kernel) and read (paged_kv_gather_fp8_to_fp16) kernels against a
// double reference. Mutant: drop the inv_scale multiply in write_kv_cache_fp8_fused_kernel
// (executor_kv_write_kernels.cu) -> StaticScaleWithinE4M3Bound fails.

#include "compute/kv_gather.h"
#include "core/kv_fp8_scale.h"
#include "exec/executor_kernels.h"
#include "exec/executor_kernels.cuh"
#include "test_cuda_skip.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
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

// K and V both = x; returns the gathered FP16 K rows [kTokens, kRow].
std::vector<half> write_then_gather(const std::vector<half>& x, float scale) {
    half* d_k = nullptr;
    half* d_v = nullptr;
    int* d_pos = nullptr;
    int* d_bt = nullptr;
    __nv_fp8_e4m3* d_kc = nullptr;
    __nv_fp8_e4m3* d_vc = nullptr;
    half* d_out = nullptr;
    const int n_blocks = kTokens / kBlockSize;
    const size_t cache_elems = static_cast<size_t>(n_blocks) * kBlockSize * kRow;
    std::vector<int> pos(kTokens);
    for (int i = 0; i < kTokens; ++i)
        pos[i] = i;
    std::vector<int> bt(n_blocks);
    for (int b = 0; b < n_blocks; ++b)
        bt[b] = n_blocks - 1 - b;  // non-identity physical order
    cudaMalloc(&d_k, x.size() * sizeof(half));
    cudaMalloc(&d_v, x.size() * sizeof(half));
    cudaMalloc(&d_pos, kTokens * sizeof(int));
    cudaMalloc(&d_bt, n_blocks * sizeof(int));
    cudaMalloc(&d_kc, cache_elems);
    cudaMalloc(&d_vc, cache_elems);
    cudaMalloc(&d_out, x.size() * sizeof(half));
    cudaMemcpy(d_k, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos.data(), kTokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, bt.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice);
    write_kv_cache_fp8_fused_kernel<<<dim3(kTokens, 2), 256>>>(d_k, d_v, d_pos, d_bt, d_kc, d_vc,
                                                               1.0f / scale, kBlockSize * kRow, kRow,
                                                               kBlockSize, kTokens, 0, 1);
    paged_kv_gather_fp8_to_fp16(d_out, d_kc, d_bt, scale, kTokens, kBlockSize, kNkv, kHd, nullptr);
    std::vector<half> out(x.size());
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_pos);
    cudaFree(d_bt);
    cudaFree(d_kc);
    cudaFree(d_vc);
    cudaFree(d_out);
    return out;
}

// #3002 invariant at a non-unit scale: the in-place prefill round trip equals write + gather.
std::vector<half> round_inplace(const std::vector<half>& x, float scale) {
    half* d_k = nullptr;
    half* d_v = nullptr;
    cudaMalloc(&d_k, x.size() * sizeof(half));
    cudaMalloc(&d_v, x.size() * sizeof(half));
    cudaMemcpy(d_k, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
    round_kv_fp8_inplace_kernel<<<dim3(kTokens, 2), 256>>>(d_k, d_v, kRow, kRow, kRow, kTokens, 1.0f / scale,
                                                           scale);
    std::vector<half> out(x.size());
    cudaMemcpy(out.data(), d_k, out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    cudaFree(d_k);
    cudaFree(d_v);
    return out;
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
    const auto b = round_inplace(x, s);
    for (size_t i = 0; i < x.size(); ++i)
        ASSERT_EQ(bits(a[i]), bits(b[i])) << "i=" << i;
}

}  // namespace
}  // namespace imp
