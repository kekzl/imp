// gemm_nvfp4_tile (own NVFP4 prefill GEMM, ROADMAP row 123) against the library-free reference on the same
// operand bytes, data-parallel rows invariant to M
// (informational). Timing cases: tests/bench/test_nvfp4_gemm_tile_bench.cu.

#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "nvfp4_ref_gemm.cuh"
#include "quant/nvfp4_gemm_tile.h"
#include "compute/cutlass_nvfp4_owned.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace imp;
#include "test_nvfp4_gemm_tile_util.cuh"

namespace {

// Signed values with magnitudes over 2^-6..2^6 (wide dynamic range per 16-element block).
__global__ void fill_wide_kernel(half* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        const float u = (static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f;
        const float e = (static_cast<float>(h >> 16) / 32768.0f) - 1.0f;
        p[i] = __float2half(u * exp2f(6.0f * e));
    }
}

bool run_reference(const Operands& o, cudaStream_t s) {
    return imp::test::ref_nvfp4_gemm(o.act, o.act_sf, o.cw.data, o.cw.scale_factors, o.cw.tensor_scale,
                                     o.y_ref, o.M, o.N, o.K, s);
}

// Variants: 0 TMA 128x128 x4 + producer warp, 1 TMA 128x256 x3, 2 sync 128x128 BK256x2, -1 auto.
constexpr int kCheckedVariants = 7;  // 3 = stream-K 128x128, 4 = hybrid / split, 5 = 128x64, 6 = 128x64 split
// TMA variants take K % 64 (partial last stage).
constexpr int kVariantBk[kCheckedVariants] = {64, 64, 256, 64, 64, 64, 64};

class Nvfp4GemmTileTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

// Same operand bytes, fp32 accumulate in both: outputs agree to fp16 rounding plus
// accumulation order (rel. to the output scale).
TEST_P(Nvfp4GemmTileTest, MatchesReference) {
    const auto [M, N, K] = GetParam();
    Operands o;
    make_operands(o, M, N, K, s_);
    ASSERT_TRUE(run_reference(o, s_));
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<half> ref(size_t(M) * N), own(size_t(M) * N);
    cudaMemcpy(ref.data(), o.y_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (int v = 0; v < kCheckedVariants; ++v) {
        if (K % kVariantBk[v] != 0)
            continue;
        cudaMemsetAsync(o.y_own, 0xff, ref.size() * sizeof(half), s_);
        ASSERT_TRUE(gemm_nvfp4_tile(o.act, o.act_sf, o.cw.data, o.cw.scale_factors, o.cw.tensor_scale,
                                    o.y_own, M, N, K, s_, v, o.ws, o.ws_bytes));
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
        std::fprintf(stderr, "[tile v%d] M=%d N=%d K=%d max|ref|=%.4g max|diff|=%.4g\n", v, M, N, K, max_ref,
                     max_diff);
        EXPECT_EQ(bad, 0u) << "variant " << v;
        EXPECT_GT(max_ref, 0.0);
        EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << "variant " << v;
    }
    free_operands(o);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Nvfp4GemmTileTest,
                         ::testing::Values(std::make_tuple(128, 128, 128), std::make_tuple(200, 384, 512),
                                           std::make_tuple(33, 136, 1024), std::make_tuple(512, 1024, 5120),
                                           std::make_tuple(1000, 2056, 256),
                                           std::make_tuple(1536, 2304, 512),     // 216 tiles: hybrid stream-K
                                           std::make_tuple(200, 384, 192),       // K % 128 = 64
                                           std::make_tuple(2048, 2816, 2112)));  // Gemma-4 down_proj

// fp32 D (LM head logits) against the fp32 reference: decode-batch and PPL-chunk rows, a 128x64 grid,
// stream-K shapes; ragged M and N.
TEST(Nvfp4GemmTileF32, MatchesReference) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][3] = {{4, 18432, 2048}, {33, 1024, 512}, {512, 16384, 2048}, {200, 384, 192}};
    for (const auto& sh : shapes) {
        Operands o;
        make_operands(o, sh[0], sh[1], sh[2], s);
        float* ref = nullptr;
        float* own = nullptr;
        const size_t n = size_t(o.M) * o.N;
        ASSERT_EQ(cudaMalloc(&ref, n * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&own, n * sizeof(float)), cudaSuccess);
        cudaMemsetAsync(own, 0xff, n * sizeof(float), s);
        ASSERT_TRUE(imp::test::ref_nvfp4_gemm(o.act, o.act_sf, o.cw.data, o.cw.scale_factors,
                                              o.cw.tensor_scale, ref, o.M, o.N, o.K, s));
        ASSERT_TRUE(gemm_nvfp4_tile_f32(o.act, o.act_sf, o.cw.data, o.cw.scale_factors, o.cw.tensor_scale,
                                        own, o.M, o.N, o.K, s, o.ws, o.ws_bytes));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        std::vector<float> r(n), x(n);
        cudaMemcpy(r.data(), ref, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(x.data(), own, n * sizeof(float), cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_diff = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) {
            bad += std::isfinite(x[i]) ? 0 : 1;
            max_ref = std::max(max_ref, std::fabs(double(r[i])));
            max_diff = std::max(max_diff, std::fabs(double(r[i]) - double(x[i])));
        }
        std::fprintf(stderr, "[tile f32] M=%d N=%d K=%d max|ref|=%.4g max|diff|=%.4g\n", o.M, o.N, o.K,
                     max_ref, max_diff);
        EXPECT_EQ(bad, 0u);
        EXPECT_GT(max_ref, 0.0);
        EXPECT_LE(max_diff, 1e-4 * max_ref) << o.M << "x" << o.N << "x" << o.K;
        cudaFree(ref);
        cudaFree(own);
        free_operands(o);
    }
    cudaStreamDestroy(s);
}

// runtime.deterministic pins data-parallel tiles: a row's bits must not depend on M, the prefill chunk length
// (#2556). Variants 0 and 5 (data-parallel) must match the M=317 tail rows of M=2048/2112 bitwise; stream-K
// (variant 3) sums K partials in another order and must differ at q_gate M=2048, else this checks nothing.
TEST(Nvfp4GemmTile, DataParallelRowsInvariantToM) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][2] = {{12288, 5120}, {16384, 5120}, {5120, 17408}};  // q_gate, gdn_qkvz, down
    const int small_m = 317;
    for (const auto& nk : shapes) {
        const int N = nk[0], K = nk[1];
        for (int big_m : {2048, 2112}) {
            Operands big;
            make_operands(big, big_m, N, K, s);
            // Activations over 2^-6..2^6: a narrow range sums exactly in fp32, so any K order gives the same
            // bits.
            half* x = nullptr;
            ASSERT_EQ(cudaMalloc(&x, size_t(big_m) * K * sizeof(half)), cudaSuccess);
            fill_wide_kernel<<<1024, 256, 0, s>>>(x, size_t(big_m) * K, 0x5151u);
            quantize_fp16_to_nvfp4_sfatom(x, big.act, big.act_sf, big_m, K, s);
            const size_t tail = size_t(big_m - small_m);
            void* small_act = nullptr;
            void* small_sf = nullptr;
            half* y_small = nullptr;
            ASSERT_EQ(cudaMalloc(&small_act, size_t(small_m) * K / 2), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&small_sf, nvfp4_sfatom_size(small_m, K)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&y_small, size_t(small_m) * N * sizeof(half)), cudaSuccess);
            cudaMemsetAsync(small_sf, 0, nvfp4_sfatom_size(small_m, K), s);
            quantize_fp16_to_nvfp4_sfatom(x + tail * K, small_act, small_sf, small_m, K, s);
            std::vector<half> h_small(size_t(small_m) * N), h_big(size_t(big_m) * N), h_dp0(h_big.size());
            auto run = [&](int v, half* y, const void* a, const void* a_sf, int m, std::vector<half>& out) {
                EXPECT_TRUE(gemm_nvfp4_tile(a, a_sf, big.cw.data, big.cw.scale_factors, big.cw.tensor_scale,
                                            y, m, N, K, s, v, big.ws, big.ws_bytes))
                    << "variant " << v;
                EXPECT_EQ(cudaStreamSynchronize(s), cudaSuccess);
                cudaMemcpy(out.data(), y, out.size() * sizeof(half), cudaMemcpyDeviceToHost);
            };
            for (int v : {0, 5}) {
                run(v, big.y_own, big.act, big.act_sf, big_m, h_big);
                run(v, y_small, small_act, small_sf, small_m, h_small);
                EXPECT_EQ(memcmp(h_small.data(), h_big.data() + tail * N, h_small.size() * sizeof(half)), 0)
                    << "N=" << N << " K=" << K << " M=" << big_m << " variant " << v
                    << ": data-parallel rows depend on M";
                if (v == 0)
                    h_dp0 = h_big;
                else
                    EXPECT_EQ(memcmp(h_dp0.data(), h_big.data(), h_big.size() * sizeof(half)), 0)
                        << "variant 5 differs from variant 0 at N=" << N << " M=" << big_m;
            }
            // q_gate M=2048: 1536 tiles over 170 SMs, uniform stream-K splits tiles across units (#2556
            // engine case).
            if (N == 12288 && big_m == 2048) {
                run(3, big.y_own, big.act, big.act_sf, big_m, h_big);
                EXPECT_NE(memcmp(h_dp0.data(), h_big.data(), h_big.size() * sizeof(half)), 0)
                    << "stream-K matched data-parallel bitwise at q_gate M=2048";
            }
            cudaFree(x);
            cudaFree(small_act);
            cudaFree(small_sf);
            cudaFree(y_small);
            free_operands(big);
        }
    }
    cudaStreamDestroy(s);
}
