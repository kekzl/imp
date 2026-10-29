// gemm_mxfp4_tile (own MXFP4 prefill GEMM, ROADMAP row 123) against a decode-then-fp32 reference on the
// same operand bytes (E2M1 nibbles, UE8M0 per 32 K in SfAtom layout).

#include "quant/mxfp4_sfatom.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace imp;

namespace {

// Amplitude x 2^-3..2^3 per (row, 32-K group): every UE8M0 of a row differs from its neighbours, so a
// scale read from the wrong group shows in the output.
__global__ void fill_kernel(half* p, size_t n, int K, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        const int e = static_cast<int>(((i % K) / 32 + (i / K)) % 7) - 3;
        p[i] = __float2half(ldexpf(amp, e) * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

__device__ float mx4_elem(const uint8_t* data, const uint8_t* sf, int row, int k, int K) {
    constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint8_t byte = data[(static_cast<size_t>(row) * K + k) / 2];
    const int nib = (k & 1) ? byte >> 4 : byte & 0xf;
    const float v = (nib & 8) ? -kE2M1[nib & 7] : kE2M1[nib & 7];
    const int g = k / 32;
    const int rl = row % 128;
    const size_t off = (static_cast<size_t>((row / 128) * ((K + 127) / 128) + g / 4) * 512) +
                       ((rl % 32) * 16) + ((rl / 32) * 4) + (g % 4);
    return v * exp2f(static_cast<float>(sf[off]) - 127.0f);
}

// One thread per output: D[m,n] = alpha * sum_k A[m,k] W[n,k], fp32.
__global__ void ref_kernel(const uint8_t* a, const uint8_t* a_sf, const uint8_t* w, const uint8_t* w_sf,
                           float alpha, half* d, int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N)
        return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc = fmaf(mx4_elem(a, a_sf, m, k, K), mx4_elem(w, w_sf, n, k, K), acc);
    d[static_cast<size_t>(m) * N + n] = __float2half(alpha * acc);
}

struct Operands {
    int M = 0, N = 0, K = 0;
    void *a = nullptr, *a_sf = nullptr, *w = nullptr, *w_sf = nullptr, *ws = nullptr;
    size_t ws_bytes = 0;
    half* y_ref = nullptr;
    half* y_own = nullptr;
};

void quantize(const half* src, void** data, void** sf, int rows, int K, cudaStream_t s) {
    ASSERT_EQ(cudaMalloc(data, size_t(rows) * K / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(sf, cutlass_mxfp4_sf_size(rows, K)), cudaSuccess);
    cudaMemsetAsync(*sf, 0, cutlass_mxfp4_sf_size(rows, K), s);
    quantize_fp16_to_mxfp4_cutlass(src, *data, *sf, rows, K, s);
}

void make_operands(Operands& o, int M, int N, int K, cudaStream_t s) {
    o.M = M;
    o.N = N;
    o.K = K;
    half* w = nullptr;
    half* x = nullptr;
    ASSERT_EQ(cudaMalloc(&w, size_t(N) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&x, size_t(M) * K * sizeof(half)), cudaSuccess);
    fill_kernel<<<1024, 256, 0, s>>>(w, size_t(N) * K, K, 0x1234u, 0.08f);
    fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M) * K, K, 0x9876u, 2.0f);
    quantize(w, &o.w, &o.w_sf, N, K, s);
    quantize(x, &o.a, &o.a_sf, M, K, s);
    o.ws_bytes = gemm_tile_streamk_workspace_bytes();
    ASSERT_EQ(cudaMalloc(&o.ws, o.ws_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_ref, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_own, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    cudaFree(w);
    cudaFree(x);
}

void free_operands(Operands& o) {
    for (void* p : {o.a, o.a_sf, o.w, o.w_sf, o.ws, static_cast<void*>(o.y_ref), static_cast<void*>(o.y_own)})
        cudaFree(p);
}

constexpr int kVariants = 5;  // 0 128x128 PW, 1 128x256, 2 sync, 3 stream-K, then -1 (auto)
constexpr int kVariantBk[kVariants] = {128, 128, 256, 128, 128};
constexpr float kAlpha = 0.75f;

class Mxfp4GemmTileTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

// Same operand bytes, fp32 accumulate in both: outputs agree to fp16 rounding plus accumulation order.
TEST_P(Mxfp4GemmTileTest, MatchesDecodedReference) {
    const auto [M, N, K] = GetParam();
    Operands o;
    make_operands(o, M, N, K, s_);
    ref_kernel<<<dim3((N + 127) / 128, M), 128, 0, s_>>>(static_cast<const uint8_t*>(o.a),
                                                         static_cast<const uint8_t*>(o.a_sf),
                                                         static_cast<const uint8_t*>(o.w),
                                                         static_cast<const uint8_t*>(o.w_sf), kAlpha, o.y_ref,
                                                         M, N, K);
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<half> ref(size_t(M) * N), own(size_t(M) * N);
    ASSERT_EQ(cudaMemcpy(ref.data(), o.y_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int v = 0; v < kVariants; ++v) {
        if (K % kVariantBk[v] != 0)
            continue;
        cudaMemsetAsync(o.y_own, 0xff, ref.size() * sizeof(half), s_);
        ASSERT_TRUE(gemm_mxfp4_tile(o.a, o.a_sf, o.w, o.w_sf, kAlpha, o.y_own, M, N, K, s_, v < 4 ? v : -1,
                                    o.ws, o.ws_bytes));
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_diff = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double r = __half2float(ref[i]);
            const double x = __half2float(own[i]);
            if (!std::isfinite(x))
                ++bad;
            max_ref = std::max(max_ref, std::fabs(r));
            max_diff = std::max(max_diff, std::fabs(r - x));
        }
        std::fprintf(stderr, "[mx4 v%d] M=%d N=%d K=%d max|ref|=%.4g max|diff|=%.4g\n", v, M, N, K, max_ref,
                     max_diff);
        EXPECT_EQ(bad, 0u) << "variant " << v;
        EXPECT_GT(max_ref, 0.0);
        EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << "variant " << v;
    }
    free_operands(o);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Mxfp4GemmTileTest,
                         ::testing::Values(std::make_tuple(128, 128, 128), std::make_tuple(200, 384, 512),
                                           std::make_tuple(33, 136, 1024), std::make_tuple(512, 1024, 2048),
                                           std::make_tuple(1000, 2056, 256),
                                           std::make_tuple(1536, 2304, 512)));  // 216 tiles: stream-K

// K % 128 != 0 (no whole SfAtom per stage): refused, never a partial result.
TEST(Mxfp4GemmTile, RefusesPartialAtomK) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    Operands o;
    make_operands(o, 128, 128, 192, s);
    EXPECT_FALSE(gemm_mxfp4_tile(o.a, o.a_sf, o.w, o.w_sf, 1.0f, o.y_own, 128, 128, 192, s));
    free_operands(o);
    cudaStreamDestroy(s);
}
