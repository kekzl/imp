// gemm_nvfp4_f16_tile (fused NVFP4 W4A16 prompt GEMM, ROADMAP B-85): (1) bit-equal to gemm_f16_tile on the
// exact fp16 weight e2m1 * e4m3 * 2^-8 built on the host; (2) against imp's own quantize_fp16_to_nvfp4 +
// dequantize_nvfp4_to_fp16 + gemm_f16_tile (nibble order, scale mapping, alpha = tensor_scale * 256);
// (3) a timing table against the NVFP4 tile GEMM shapes of Flint attention (informational).

#include "compute/gemm_f16_tile.h"
#include "compute/gemm_nvfp4_f16_tile.h"
#include "core/tensor.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace imp;

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

size_t count_diff(const half* x, const half* y, size_t n, size_t* nonzero) {
    std::vector<uint16_t> a(n), b(n);
    cudaMemcpy(a.data(), x, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), y, n * 2, cudaMemcpyDeviceToHost);
    size_t d = 0, nz = 0;
    for (size_t i = 0; i < n; ++i) {
        d += a[i] != b[i] ? 1 : 0;
        nz += (a[i] & 0x7fff) != 0 ? 1 : 0;
    }
    if (nonzero != nullptr)
        *nonzero = nz;
    return d;
}

// M, N, K, acc16, beta, variant, scale byte lo, hi
using Shape = std::tuple<int, int, int, bool, float, int, int, int>;
class GemmNvfp4TileTest : public ::testing::TestWithParam<Shape> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

TEST_P(GemmNvfp4TileTest, BitEqualToExactF16WeightTile) {
    const auto [M, N, K, acc16, beta, variant, lo, hi] = GetParam();
    Nv4Bufs b;
    b.alloc(M, N, K, lo, hi, s_);
    const float alpha = 0.37f * 256.0f;
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, M, N, K, alpha, beta, acc16, s_, variant));
    ASSERT_TRUE(gemm_nvfp4_f16_tile(b.a, b.w, b.sc, b.own, M, N, K, alpha, beta, acc16, s_, variant));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    size_t nonzero = 0;
    const size_t diff = count_diff(b.ref, b.own, size_t(M) * N, &nonzero);
    std::fprintf(stderr,
                 "[nvfp4 tile] M=%d N=%d K=%d acc16=%d beta=%.0f v%d sc %#x..%#x: %zu differ, %zu nonzero\n",
                 M, N, K, acc16 ? 1 : 0, beta, variant, lo, hi, diff, nonzero);
    EXPECT_EQ(diff, 0u);
    EXPECT_GT(nonzero, size_t(M) * N / 2);
    b.release();
}

// Scales 0x01..0x7E cover E4M3 subnormals up to 448 (fp32 accumulate: fp16 sums could overflow at 448);
// 0x30..0x48 keeps fp16-accumulate sums finite.
INSTANTIATE_TEST_SUITE_P(Shapes, GemmNvfp4TileTest,
                         ::testing::Values(Shape{128, 256, 256, true, 0.0f, 2, 0x30, 0x48},
                                           Shape{200, 392, 768, true, 0.0f, 1, 0x30, 0x48},
                                           Shape{300, 1032, 1024, true, 1.0f, 1, 0x30, 0x48},
                                           Shape{512, 2048, 2048, true, 0.0f, 1, 0x30, 0x48},
                                           Shape{257, 520, 2048, false, 0.0f, -1, 0x01, 0x7E},
                                           Shape{130, 264, 512, false, 1.0f, -1, 0x01, 0x7E},
                                           Shape{2048, 4096, 2048, false, 0.0f, -1, 0x01, 0x7E}));

// Semantics against imp's own NVFP4 quantizer + dequantizer: same weight up to fp16 rounding of the
// dequantized value (one rounding there, none here), so a small relative error, not bit equality.
TEST(GemmNvfp4TileQuant, MatchesDequantizeNvfp4) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int M = 384, N = 1024, K = 2048;
    half *wsrc = nullptr, *wdq = nullptr, *a = nullptr, *ref = nullptr, *own = nullptr;
    cudaMalloc(&wsrc, size_t(N) * K * 2);
    cudaMalloc(&wdq, size_t(N) * K * 2);
    cudaMalloc(&a, size_t(M) * K * 2);
    cudaMalloc(&ref, size_t(M) * N * 2);
    cudaMalloc(&own, size_t(M) * N * 2);
    fill_act_kernel<<<1024, 256, 0, s>>>(wsrc, size_t(N) * K, 0x13u);
    fill_act_kernel<<<1024, 256, 0, s>>>(a, size_t(M) * K, 0x51u);
    int64_t ws[2] = {N, K};
    Tensor tw(wsrc, QType::F16, 2, ws, true);
    NvFP4QuantResult q;
    quantize_fp16_to_nvfp4(tw, q, s);
    dequantize_nvfp4_to_fp16(q, wdq, s);
    ASSERT_TRUE(gemm_f16_tile(a, wdq, ref, M, N, K, 1.0f, 0.0f, false, s, -1));
    ASSERT_TRUE(gemm_nvfp4_f16_tile(a, q.packed_data, q.micro_scales, own, M, N, K, q.tensor_scale * 256.0f,
                                    0.0f, false, s, -1));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    std::vector<half> hr(size_t(M) * N), ho(size_t(M) * N);
    cudaMemcpy(hr.data(), ref, hr.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ho.data(), own, ho.size() * 2, cudaMemcpyDeviceToHost);
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < hr.size(); ++i) {
        const double r = __half2float(hr[i]);
        const double d = __half2float(ho[i]) - r;
        num += d * d;
        den += r * r;
    }
    const double rel = std::sqrt(num / den);
    std::fprintf(stderr, "[nvfp4 tile] vs quantize+dequantize M=%d N=%d K=%d: relative L2 error %.3e\n", M, N,
                 K, rel);
    EXPECT_GT(den, 0.0);
    EXPECT_LT(rel, 2e-3);
    free_nvfp4_result(q);
    cudaFree(wsrc);
    cudaFree(wdq);
    cudaFree(a);
    cudaFree(ref);
    cudaFree(own);
    cudaStreamDestroy(s);
}

// Timing on Flint attention shapes (hidden 2048, q 4096, kv 512): fused W4A16 vs gemm_f16_tile on an FP16 W.
TEST(GemmNvfp4TileBench, FlintAttentionShapes) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][2] = {{4096, 2048}, {512, 2048}, {2048, 4096}};
    std::fprintf(stderr, "%6s %6s %6s %10s %10s\n", "M", "N", "K", "f16_us", "nvfp4_us");
    for (const auto& nk : shapes)
        for (int M : {512, 2048}) {
            Nv4Bufs b;
            b.alloc(M, nk[0], nk[1], 0x30, 0x48, s);
            auto time_us = [&](auto&& fn) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                for (int i = 0; i < 10; ++i)
                    fn();
                cudaEventRecord(e0, s);
                for (int i = 0; i < 20; ++i)
                    fn();
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms / 20;
            };
            const double t16 = time_us(
                [&] { (void)gemm_f16_tile(b.a, b.w16, b.ref, M, b.N, b.K, 1.0f, 0.0f, false, s, -1); });
            const double t4 = time_us([&] {
                (void)gemm_nvfp4_f16_tile(b.a, b.w, b.sc, b.own, M, b.N, b.K, 1.0f, 0.0f, false, s, -1);
            });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            std::fprintf(stderr, "%6d %6d %6d %10.1f %10.1f\n", M, b.N, b.K, t16, t4);
            b.release();
        }
    cudaStreamDestroy(s);
}
