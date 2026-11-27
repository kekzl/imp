// gemm_mxfp4_f16_tile (fused MXFP4 W4A16 prompt GEMM, ROADMAP row 123) against dequant_mxfp4_linear_to_fp16
// + gemm_f16_tile: same fp16 W bits, same per-element k order, so the outputs must match bit for bit.
// Repeat test here. Timing cases: tests/bench/test_gemm_mxfp4_f16_tile_bench.cu.

#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_mxfp4_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "quant/mxfp4_sfatom.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

using namespace imp;
#include "test_gemm_mxfp4_f16_tile_util.cuh"

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

class GemmMxfp4TileTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool, float, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

class GemmMxfp4RepeatTest : public ::testing::TestWithParam<int> {};

}  // namespace

TEST_P(GemmMxfp4TileTest, BitEqualToDequantPlusF16Tile) {
    const auto [M, N, K, acc16, beta, variant] = GetParam();
    Mx4Bufs b;
    b.alloc(M, N, K, s_);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_TRUE(gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    size_t nonzero = 0;
    const size_t diff = count_diff(b.ref, b.own, size_t(M) * N, &nonzero);
    std::fprintf(stderr, "[mxfp4 tile] M=%d N=%d K=%d acc16=%d beta=%.0f v%d: %zu differ, %zu nonzero\n", M,
                 N, K, acc16 ? 1 : 0, beta, variant, diff, nonzero);
    EXPECT_EQ(diff, 0u);
    EXPECT_GT(nonzero, size_t(M) * N / 2);
    b.release();
}

// variant: 1 128x128, 2 256x128 (fp16 accumulate; every tile sums k in the same order); fp32: one config.
INSTANTIATE_TEST_SUITE_P(Shapes, GemmMxfp4TileTest,
                         ::testing::Values(std::make_tuple(128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 1),
                                           std::make_tuple(512, 1024, 2560, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 2),
                                           std::make_tuple(512, 2048, 2560, true, 0.0f, 2),
                                           std::make_tuple(257, 520, 2048, false, 0.0f, -1),
                                           std::make_tuple(130, 264, 512, false, 1.0f, -1)));

// Repeat runs on a multi-wave shape stay bit-identical to the reference (the slot-release race of #2913).
TEST_P(GemmMxfp4RepeatTest, BitIdenticalAcrossRuns) {
    const int variant = GetParam();
    cudaStream_t s;
    cudaStreamCreate(&s);
    Mx4Bufs b;
    b.alloc(1536, 9216, 2560, s);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
    int bad = 0;
    for (int it = 0; it < 10; ++it) {
        ASSERT_TRUE(gemm_mxfp4_f16_tile(b.a, b.w, b.sc, b.own, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        bad += count_diff(b.ref, b.own, size_t(b.M) * b.N, nullptr) != 0 ? 1 : 0;
    }
    std::fprintf(stderr, "[mxfp4 tile] repeat 1536x9216x2560 v%d: %d / 10 runs differ\n", variant, bad);
    EXPECT_EQ(bad, 0);
    b.release();
    cudaStreamDestroy(s);
}

INSTANTIATE_TEST_SUITE_P(Variants, GemmMxfp4RepeatTest, ::testing::Values(1, 2));
