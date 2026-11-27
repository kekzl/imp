// gemm_q6k_f16_tile (fused Q6_K prompt GEMM, ROADMAP row 123) against dequant_gpu + gemm_f16_tile on the
// same tile config: the decode is the dequant_q6k_v2 math, so the outputs must match bit for bit.
// Timing cases: tests/bench/test_gemm_q6k_f16_tile_bench.cu.

#include "compute/gemm.h"
#include "compute/gemm_f16_tile.h"
#include "compute/gemm_q6k_f16_tile.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;
#include "test_gemm_q6k_f16_tile_util.cuh"

namespace {

class GemmQ6kRepeatTest : public ::testing::TestWithParam<int> {};

class GemmQ6kTileTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool, float, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

TEST_P(GemmQ6kTileTest, BitEqualToDequantPlusF16Tile) {
    const auto [M, N, K, acc16, beta, variant] = GetParam();
    Q6Bufs b;
    b.alloc(M, N, K, s_);
    dequant_gpu(b.wq, b.w16, QType::Q6_K, N, K, s_);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_TRUE(gemm_q6k_f16_tile(b.a, b.wq, b.own, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<uint16_t> r(size_t(M) * N), o(size_t(M) * N);
    cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
    size_t diff = 0, nonzero = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        diff += r[i] != o[i] ? 1 : 0;
        nonzero += (r[i] & 0x7fff) != 0 ? 1 : 0;
    }
    std::fprintf(stderr, "[q6k tile] M=%d N=%d K=%d acc16=%d beta=%.0f v%d: %zu / %zu differ, %zu nonzero\n",
                 M, N, K, acc16 ? 1 : 0, beta, variant, diff, r.size(), nonzero);
    EXPECT_EQ(diff, 0u);
    EXPECT_GT(nonzero, r.size() / 2);
    b.release();
}

// variant: 1 128x128, 2 256x128 (fp16 accumulate; every tile sums k in the same order, so the 128x256
// f16 tile is the reference for both); the fp32 accumulator has one config (variant ignored).
INSTANTIATE_TEST_SUITE_P(Shapes, GemmQ6kTileTest,
                         ::testing::Values(std::make_tuple(128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 1),
                                           std::make_tuple(512, 1024, 5120, true, 0.0f, 1),
                                           std::make_tuple(300, 1032, 1024, true, 1.0f, 2),
                                           std::make_tuple(512, 1024, 5120, true, 0.0f, 2),
                                           std::make_tuple(257, 520, 2048, false, 0.0f, -1),
                                           std::make_tuple(130, 264, 512, false, 1.0f, -1)));

// Repeat runs on a multi-wave shape stay bit-identical to the reference: without the release fence in the
// MMA loop the decoders overwrote a slot behind its last LDSM (19/20 runs differed, max |d| 106).
TEST_P(GemmQ6kRepeatTest, BitIdenticalAcrossRuns) {
    const int variant = GetParam();
    cudaStream_t s;
    cudaStreamCreate(&s);
    Q6Bufs b;
    b.alloc(1536, 5120, 5120, s);
    dequant_gpu(b.wq, b.w16, QType::Q6_K, b.N, b.K, s);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    std::vector<uint16_t> r(size_t(b.M) * b.N), o(r.size());
    cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int it = 0; it < 10; ++it) {
        ASSERT_TRUE(gemm_q6k_f16_tile(b.a, b.wq, b.own, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
        bad += r != o ? 1 : 0;
    }
    std::fprintf(stderr, "[q6k tile] repeat 1536x5120x5120 v%d: %d / 10 runs differ\n", variant, bad);
    EXPECT_EQ(bad, 0);
    b.release();
    cudaStreamDestroy(s);
}

INSTANTIATE_TEST_SUITE_P(Variants, GemmQ6kRepeatTest, ::testing::Values(1, 2));
