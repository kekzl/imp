// gemm_fp8_tile (own FP8 E4M3 x E4M3 prefill GEMM, per-tensor scales) against cuBLASLt FP8 on the same
// operand bytes and scales: beta = 0 and beta = 1, all tile variants, refusal contract
// Timing cases: tests/bench/test_gemm_fp8_tile_bench.cu.

#include "compute/gemm.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace imp;

#include "test_gemm_fp8_tile_util.cuh"
namespace {

std::vector<half> random_half(size_t n, uint32_t seed, float amp) {
    std::vector<half> v(n);
    uint32_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h = h * 1664525u + 1013904223u;
        v[i] = __float2half(amp * (static_cast<float>(h >> 8) / 8388608.0f - 1.0f));
    }
    return v;
}

class Fp8GemmTileTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
        cudaStreamCreate(&s_);
    }
    void TearDown() override {
        if (s_)
            cudaStreamDestroy(s_);
    }
    cudaStream_t s_ = nullptr;
};

// Compares own vs ref on the host; returns false on a non-finite own value or a tolerance miss.
void compare(const std::vector<half>& ref, const std::vector<half>& own, const char* tag, int v, int M, int N,
             int K) {
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
    std::fprintf(stderr, "[fp8 %s v%d] M=%d N=%d K=%d max|ref|=%.4g max|diff|=%.4g\n", tag, v, M, N, K,
                 max_ref, max_diff);
    EXPECT_EQ(bad, 0u) << tag << " variant " << v;
    EXPECT_GT(max_ref, 0.0);
    // Both arms accumulate fp32 in different orders: fp16 rounding plus order, relative to the output peak.
    EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << tag << " variant " << v;
}

}  // namespace

// beta = 0, output pre-poisoned with 0xff (NaN) so an unwritten tile fails.
TEST_P(Fp8GemmTileTest, MatchesCublasLt) {
    const auto [M, N, K] = GetParam();
    Operands o;
    make_operands(o, M, N, K);
    run_ref(o, 0.0f, s_);
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<half> ref(size_t(M) * N), own(size_t(M) * N);
    ASSERT_EQ(cudaMemcpy(ref.data(), o.y_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int i = 0; i < kVariants; ++i) {
        const int v = variant_of(i);
        cudaMemsetAsync(o.y_own, 0xff, ref.size() * sizeof(half), s_);
        if (v == 2 && K % 128 != 0) {  // sync mainloop stages BK 128: refuses K % 128 by design
            EXPECT_FALSE(run_own(o, 0.0f, v, s_));
            continue;
        }
        ASSERT_TRUE(run_own(o, 0.0f, v, s_)) << "variant " << v << " refused a legal shape";
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        compare(ref, own, "beta0", v, M, N, K);
    }
    free_operands(o);
}

// beta = 1: both arms start from the same fp16 residual and add the scaled product to it.
TEST_P(Fp8GemmTileTest, BetaOneAccumulates) {
    const auto [M, N, K] = GetParam();
    Operands o;
    make_operands(o, M, N, K);
    const auto resid = random_half(size_t(M) * N, 0x5555u, 3.0f);
    ASSERT_EQ(cudaMemcpy(o.y_ref, resid.data(), resid.size() * sizeof(half), cudaMemcpyHostToDevice),
              cudaSuccess);
    run_ref(o, 1.0f, s_);
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<half> ref(size_t(M) * N), own(size_t(M) * N);
    ASSERT_EQ(cudaMemcpy(ref.data(), o.y_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int i = 0; i < kVariants; ++i) {
        const int v = variant_of(i);
        ASSERT_EQ(cudaMemcpy(o.y_own, resid.data(), resid.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        if (v == 2 && K % 128 != 0) {
            EXPECT_FALSE(run_own(o, 1.0f, v, s_));
            continue;
        }
        ASSERT_TRUE(run_own(o, 1.0f, v, s_)) << "variant " << v << " refused a legal shape";
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        compare(ref, own, "beta1", v, M, N, K);
    }
    free_operands(o);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Fp8GemmTileTest,
                         ::testing::Values(std::make_tuple(512, 4096, 4096), std::make_tuple(512, 1024, 4096),
                                           std::make_tuple(77, 4096, 4096),
                                           std::make_tuple(2048, 12288, 4096),
                                           std::make_tuple(512, 4096, 12288), std::make_tuple(1, 1024, 4096),
                                           std::make_tuple(130, 136, 192)));

// K % 64 != 0 and N % 8 != 0 are refused and D stays byte-identical.
TEST(Fp8GemmTileRefusal, IllegalShapesReturnFalseAndLeaveOutput) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
        GTEST_SKIP() << "no CUDA device";
    cudaStream_t s;
    cudaStreamCreate(&s);
    struct Shape {
        int M, N, K;
    };
    for (const Shape sh : {Shape{64, 64, 96}, Shape{64, 60, 128}}) {
        Operands o;
        make_operands(o, sh.M, sh.N, sh.K);
        const size_t bytes = size_t(sh.M) * sh.N * sizeof(half);
        cudaMemset(o.y_own, 0xab, bytes);
        for (int i = 0; i < kVariants; ++i)
            EXPECT_FALSE(run_own(o, 0.0f, variant_of(i), s))
                << "M=" << sh.M << " N=" << sh.N << " K=" << sh.K;
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        std::vector<uint8_t> host(bytes);
        cudaMemcpy(host.data(), o.y_own, bytes, cudaMemcpyDeviceToHost);
        EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](uint8_t b) { return b == 0xab; }))
            << "refused call wrote D (N=" << sh.N << " K=" << sh.K << ")";
        free_operands(o);
    }
    cudaStreamDestroy(s);
}
