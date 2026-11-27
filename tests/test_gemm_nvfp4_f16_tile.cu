// gemm_nvfp4_f16_tile (fused NVFP4 W4A16 prompt GEMM, ROADMAP B-85): (1) bit-equal to gemm_f16_tile on the
// exact fp16 weight e2m1 * e4m3 * 2^-8 built on the host; (2) against imp's own quantize_fp16_to_nvfp4 +
// dequantize_nvfp4_to_fp16 + gemm_f16_tile (nibble order, scale mapping, alpha = tensor_scale * 256);
// (3) timing cases: tests/bench/test_gemm_nvfp4_f16_tile_bench.cu.

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

using namespace imp;
#include "test_gemm_nvfp4_f16_tile_util.cuh"

namespace {

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
    NvFP4Owned q;
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
    cudaFree(wsrc);
    cudaFree(wdq);
    cudaFree(a);
    cudaFree(ref);
    cudaFree(own);
    cudaStreamDestroy(s);
}

// Grouped over expert-sorted rows: each expert's rows bit-equal to gemm_nvfp4_f16_tile on that expert alone
// (counts 0, 1, 127, 128, 129 and multi-tile; Flint gate/up 512x2048 and down 2048x512 shapes).
TEST(GemmGroupedNvfp4Tile, BitEqualPerExpert) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int counts[] = {0, 1, 127, 128, 129, 300, 0, 517, 64, 2};
    const int ne = static_cast<int>(sizeof(counts) / sizeof(counts[0]));
    std::vector<int32_t> off(ne + 1, 0);
    for (int e = 0; e < ne; ++e)
        off[e + 1] = off[e] + counts[e];
    const int rows = off[ne];
    for (const auto& nk : {std::pair<int, int>{512, 2048}, std::pair<int, int>{2048, 512}}) {
        const int N = nk.first, K = nk.second;
        std::mt19937 rng(0x5u + N);
        const size_t wst = size_t(N) * (K / 2), sst = size_t(N) * (K / 16);
        std::vector<uint8_t> hw(wst * ne), hs(sst * ne);
        for (auto& b : hw)
            b = static_cast<uint8_t>(rng());
        for (auto& b : hs)
            b = static_cast<uint8_t>(0x01 + (rng() % 0x7E));
        std::vector<float> ts(ne);
        for (auto& t : ts)
            t = 1e-4f * static_cast<float>(1 + (rng() % 50));
        uint8_t *w = nullptr, *sc = nullptr;
        float* dts = nullptr;
        int32_t* doff = nullptr;
        half *a = nullptr, *ref = nullptr, *own = nullptr;
        cudaMalloc(&w, hw.size());
        cudaMalloc(&sc, hs.size());
        cudaMalloc(&dts, ts.size() * 4);
        cudaMalloc(&doff, off.size() * 4);
        cudaMalloc(&a, size_t(rows) * K * 2);
        cudaMalloc(&ref, size_t(rows) * N * 2);
        cudaMalloc(&own, size_t(rows) * N * 2);
        cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(sc, hs.data(), hs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dts, ts.data(), ts.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(doff, off.data(), off.size() * 4, cudaMemcpyHostToDevice);
        fill_act_kernel<<<1024, 256, 0, s>>>(a, size_t(rows) * K, 0x77u);
        cudaMemsetAsync(ref, 0, size_t(rows) * N * 2, s);
        cudaMemsetAsync(own, 0xFF, size_t(rows) * N * 2, s);
        for (int e = 0; e < ne; ++e)
            if (counts[e] > 0)
                ASSERT_TRUE(gemm_nvfp4_f16_tile(a + size_t(off[e]) * K, w + e * wst, sc + e * sst,
                                                ref + size_t(off[e]) * N, counts[e], N, K, ts[e] * 256.0f,
                                                0.0f, false, s, -1));
        GroupedNvfp4F16Args g;
        g.a = a;
        g.rows = rows;
        g.N = N;
        g.K = K;
        g.ne = ne;
        g.d_offsets = doff;
        g.w = w;
        g.w_stride = wst;
        g.sc = sc;
        g.sc_stride = sst;
        g.d_tensor_scales = dts;
        g.d = own;
        ASSERT_TRUE(gemm_grouped_nvfp4_f16_tile(g, s));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        size_t nonzero = 0;
        const size_t diff = count_diff(ref, own, size_t(rows) * N, &nonzero);
        std::fprintf(stderr, "[grouped nvfp4 tile] ne=%d rows=%d N=%d K=%d: %zu differ, %zu nonzero\n", ne,
                     rows, N, K, diff, nonzero);
        EXPECT_EQ(diff, 0u);
        EXPECT_GT(nonzero, size_t(rows) * N / 2);
        cudaFree(w);
        cudaFree(sc);
        cudaFree(dts);
        cudaFree(doff);
        cudaFree(a);
        cudaFree(ref);
        cudaFree(own);
    }
    cudaStreamDestroy(s);
}
