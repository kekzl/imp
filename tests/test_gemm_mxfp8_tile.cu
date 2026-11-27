// gemm_mxfp8_tile (own MXFP8 prefill GEMM, ROADMAP row 123) against a host double reference decoded from the
// same operand bytes, plain SfAtom weights and the FP8 row-scale view
// Timing cases: tests/bench/test_gemm_mxfp8_tile_bench.cu.

#include "quant/fp8_quant.h"
#include "quant/mxfp8_sfatom.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace imp;
#include "test_gemm_mxfp8_tile_util.cuh"

namespace {

float e4m3_to_float(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    const float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}

// E4M3 [rows, K] + SfAtom UE8M0 (128 rows x 4 K-groups per 512-byte atom) -> float rows on the host.
std::vector<float> decode_mx8(const void* data, const void* sf, int rows, int K) {
    std::vector<uint8_t> q(size_t(rows) * K), e(cutlass_mxfp8_sf_size(rows, K));
    cudaMemcpy(q.data(), data, q.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(e.data(), sf, e.size(), cudaMemcpyDeviceToHost);
    const int n_k_tiles = (K + 127) / 128;
    std::vector<float> out(q.size());
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) {
            const int g = k / 32, rl = r % 128;
            const int off = ((r / 128) * n_k_tiles + g / 4) * 512 + (rl % 32) * 16 + (rl / 32) * 4 + g % 4;
            out[size_t(r) * K + k] = e4m3_to_float(q[size_t(r) * K + k]) * std::ldexp(1.0f, e[off] - 127);
        }
    return out;
}

// D = A W^T (times col_scale) in double from the decoded operand bytes.
std::vector<double> host_reference(const Operands& o) {
    const std::vector<float> a = decode_mx8(o.act, o.act_sf, o.M, o.K);
    const std::vector<float> w = decode_mx8(o.cw.data, o.cw.scale_factors, o.N, o.K);
    std::vector<float> col(o.N, 1.0f);
    if (o.cw.col_scale)
        cudaMemcpy(col.data(), o.cw.col_scale, col.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<double> ref(size_t(o.M) * o.N);
    for (int m = 0; m < o.M; ++m)
        for (int n = 0; n < o.N; ++n) {
            const float* ar = &a[size_t(m) * o.K];
            const float* wr = &w[size_t(n) * o.K];
            double acc = 0.0;
            for (int k = 0; k < o.K; ++k)
                acc += double(ar[k]) * wr[k];
            ref[size_t(m) * o.N + n] = acc * col[n];
        }
    return ref;
}

constexpr int kVariantBk[kVariants] = {64, 64, 128, 64, 64};

class Mxfp8GemmTileTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

// Exact products of the operand bytes, fp32 accumulate in the kernel: fp16 rounding plus accumulation order.
TEST_P(Mxfp8GemmTileTest, MatchesHostReference) {
    const auto [M, N, K, rows] = GetParam();
    Operands o;
    make_operands(o, M, N, K, rows, s_);
    const std::vector<double> ref = host_reference(o);
    std::vector<half> own(size_t(M) * N);
    for (int v = 0; v < kVariants; ++v) {
        if (K % kVariantBk[v] != 0)
            continue;
        cudaMemsetAsync(o.y_own, 0xff, ref.size() * sizeof(half), s_);
        ASSERT_TRUE(gemm_mxfp8_tile(o.act, o.act_sf, o.cw.data, o.cw.scale_factors, o.cw.col_scale, o.y_own,
                                    M, N, K, s_, v < 4 ? v : -1, o.ws, o.ws_bytes));
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_diff = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double r = ref[i];
            const double x = __half2float(own[i]);
            if (!std::isfinite(x))
                ++bad;
            max_ref = std::max(max_ref, std::fabs(r));
            max_diff = std::max(max_diff, std::fabs(r - x));
        }
        std::fprintf(stderr, "[mx8 v%d] M=%d N=%d K=%d rows=%d max|ref|=%.4g max|diff|=%.4g\n", v, M, N, K,
                     int(rows), max_ref, max_diff);
        EXPECT_EQ(bad, 0u) << "variant " << v;
        EXPECT_GT(max_ref, 0.0);
        EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << "variant " << v;
    }
    free_operands(o);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, Mxfp8GemmTileTest,
    ::testing::Values(std::make_tuple(128, 128, 128, false), std::make_tuple(200, 384, 576, false),
                      std::make_tuple(33, 136, 1024, false), std::make_tuple(512, 1024, 2048, false),
                      std::make_tuple(1000, 2056, 256, false), std::make_tuple(96, 384, 1024, true),
                      std::make_tuple(300, 2056, 576, true),
                      std::make_tuple(1536, 2304, 512, true)));  // 216 tiles: hybrid stream-K

