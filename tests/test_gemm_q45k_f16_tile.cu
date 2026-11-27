// gemm_q45k_f16_tile (fused Q4_K / Q5_K prompt GEMM, ROADMAP row 123) against dequant_gpu + gemm_f16_tile on
// the same tile config: the decode is the dequant_q45k_v2 math, so the outputs must match bit for bit.
// Timing cases: tests/bench/test_gemm_q45k_f16_tile_bench.cu.

#include "compute/gemm_f16_tile.h"
#include "compute/gemm_q45k_f16_tile.h"
#include "compute/mmq_q8_imma.h"
#include "core/process_diag.h"
#include "core/qtype.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;
#include "test_gemm_q45k_f16_tile_util.cuh"

namespace {

size_t count_diff(const KqBufs& b, size_t* nonzero) {
    std::vector<uint16_t> r(size_t(b.M) * b.N), o(r.size());
    cudaMemcpy(r.data(), b.ref, r.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.data(), b.own, o.size() * 2, cudaMemcpyDeviceToHost);
    size_t diff = 0;
    *nonzero = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        diff += r[i] != o[i] ? 1 : 0;
        *nonzero += (r[i] & 0x7fff) != 0 ? 1 : 0;
    }
    return diff;
}

class GemmQ45kTileTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int, bool, float, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

TEST_P(GemmQ45kTileTest, BitEqualToDequantPlusF16Tile) {
    const auto [q5, M, N, K, acc16, beta, variant] = GetParam();
    KqBufs b;
    b.alloc(q5, M, N, K, s_);
    dequant_gpu(b.wq, b.w16, b.qtype(), N, K, s_);
    ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_TRUE(gemm_q45k_f16_tile(q5, b.a, b.wq, b.own, M, N, K, 1.0f, beta, acc16, s_, variant));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    size_t nonzero = 0;
    const size_t diff = count_diff(b, &nonzero);
    std::fprintf(stderr,
                 "[q45k tile] %s M=%d N=%d K=%d acc16=%d beta=%.0f v%d: %zu / %d differ, %zu nonzero\n",
                 q5 ? "Q5_K" : "Q4_K", M, N, K, acc16 ? 1 : 0, beta, variant, diff, M * N, nonzero);
    EXPECT_EQ(diff, 0u);
    EXPECT_GT(nonzero, size_t(M) * N / 2);
    b.release();
}

// variant: 1 128x128, 2 256x128 (fp16 accumulate, same k order in every tile); fp32 accumulate has one
// config.
INSTANTIATE_TEST_SUITE_P(Shapes, GemmQ45kTileTest,
                         ::testing::Values(std::make_tuple(true, 128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(true, 200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(true, 300, 1032, 1024, true, 1.0f, 1),
                                           std::make_tuple(true, 512, 1024, 5120, true, 0.0f, 2),
                                           std::make_tuple(true, 257, 520, 2048, false, 0.0f, -1),
                                           std::make_tuple(false, 128, 256, 256, true, 0.0f, 2),
                                           std::make_tuple(false, 200, 392, 768, true, 0.0f, 1),
                                           std::make_tuple(false, 300, 1032, 1024, true, 1.0f, 2),
                                           std::make_tuple(false, 512, 1024, 5120, true, 0.0f, 1),
                                           std::make_tuple(false, 130, 264, 512, false, 1.0f, -1)));

// Repeat runs on a multi-wave shape stay bit-identical (the skeleton's release fence, #2913).
TEST(GemmQ45kRepeat, BitIdenticalAcrossRuns) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (int variant : {1, 2}) {
        KqBufs b;
        b.alloc(true, 1536, 5120, 5120, s);
        dequant_gpu(b.wq, b.w16, QType::Q5_K, b.N, b.K, s);
        ASSERT_TRUE(gemm_f16_tile(b.a, b.w16, b.ref, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
        int bad = 0;
        for (int it = 0; it < 10; ++it) {
            ASSERT_TRUE(
                gemm_q45k_f16_tile(true, b.a, b.wq, b.own, b.M, b.N, b.K, 1.0f, 0.0f, true, s, variant));
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            size_t nz = 0;
            bad += count_diff(b, &nz) != 0 ? 1 : 0;
        }
        std::fprintf(stderr, "[q45k tile] repeat 1536x5120x5120 v%d: %d / 10 runs differ\n", variant, bad);
        EXPECT_EQ(bad, 0);
        b.release();
    }
    cudaStreamDestroy(s);
}
