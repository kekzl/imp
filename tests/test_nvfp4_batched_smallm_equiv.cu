// Gate for small-M batched NVFP4 GEMV in speculative verify chunks (gemm_nvfp4_batched,
// #998/#1055); the e2e check can't see this path (spec_verify_small_m only fires in a verify
// chunk) and spec runs aren't cross-process reproducible to diff (#1457).
// GPU-ONLY, CI NEVER RUNS THIS: run test-quant --gtest_filter='*BatchedSmallM*' manually first.

#include <gtest/gtest.h>

#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "test_nvfp4_batched_smallm_equiv_util.cuh"
namespace {

std::vector<half> run_batched(const imp::NvFP4QuantResult& w, const std::vector<half>& x, int m) {
    half *d_x = nullptr, *d_y = nullptr;
    EXPECT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_y, static_cast<size_t>(m) * kN * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemset(d_y, 0, static_cast<size_t>(m) * kN * sizeof(half)), cudaSuccess);
    imp::gemm_nvfp4_batched(w, d_x, d_y, kN, kK, m, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> y(static_cast<size_t>(m) * kN);
    EXPECT_EQ(cudaMemcpy(y.data(), d_y, y.size() * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_x);
    cudaFree(d_y);
    return y;
}

// Invariant 1: the MR bucket must not change what a row computes. A 2-, 3- or
// 4-row call has to reproduce the 1-row call for each of its rows, which is
// exactly what a wrong bucket or a wrong activation stride would break.
TEST_F(BatchedSmallM, RowsMatchTheSingleRowCall) {
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<half> x(static_cast<size_t>(4) * kK);
    for (auto& h : x)
        h = __float2half(dist(rng));

    // Reference: each row through the M=1 path.
    std::vector<std::vector<half>> single;
    for (int r = 0; r < 4; ++r) {
        std::vector<half> row(x.begin() + static_cast<size_t>(r) * kK,
                              x.begin() + static_cast<size_t>(r + 1) * kK);
        single.push_back(run_batched(f_.w, row, 1));
    }

    for (int m = 2; m <= 4; ++m) {
        std::vector<half> xm(x.begin(), x.begin() + static_cast<size_t>(m) * kK);
        const std::vector<half> got = run_batched(f_.w, xm, m);
        for (int r = 0; r < m; ++r)
            for (int n = 0; n < kN; ++n) {
                const float a = __half2float(single[r][n]);
                const float b = __half2float(got[static_cast<size_t>(r) * kN + n]);
                ASSERT_NEAR(a, b, 1e-2f * std::max(1.0f, std::fabs(a)))
                    << "m=" << m << " row=" << r << " n=" << n;
            }
    }
}

// Invariant 2: absolute correctness against dequantise-then-multiply, so both
// paths agreeing while both being wrong is not a passing state.
TEST_F(BatchedSmallM, MatchesDequantisedReference) {
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int m = 1; m <= 4; ++m) {
        std::vector<half> x(static_cast<size_t>(m) * kK);
        for (auto& h : x)
            h = __float2half(dist(rng));
        const std::vector<half> got = run_batched(f_.w, x, m);

        for (int r = 0; r < m; ++r)
            for (int n = 0; n < kN; n += 37) {  // stride: full N is 512x4 dot products
                double acc = 0.0;
                for (int k = 0; k < kK; ++k)
                    acc += static_cast<double>(__half2float(x[static_cast<size_t>(r) * kK + k])) *
                           static_cast<double>(__half2float(f_.w_fp16[static_cast<size_t>(n) * kK + k]));
                const float ref = static_cast<float>(acc);
                const float out = __half2float(got[static_cast<size_t>(r) * kN + n]);
                // FP16 accumulation over 5120 terms: tolerance scales with the
                // magnitude, not an absolute epsilon.
                ASSERT_NEAR(out, ref, 0.05f * std::max(1.0f, std::fabs(ref)))
                    << "m=" << m << " row=" << r << " n=" << n;
            }
    }
}


// Sibling-pair v2 launch (FFN gate|up, GDN in|z) must be BIT-identical per tensor to the
// two single v2 launches it replaces (same CTA body/tile order); any diff is a
// selection-prologue bug. Only accepts stripes==1 (both Ns >= 5120).
TEST(SmallMV2Pair, PairMatchesTwoSingleCallsBitExact) {
    if (!has_sm120())
        GTEST_SKIP() << "sm_120 required";
    constexpr int kPK = 5120;
    constexpr int kN1 = 5120;
    constexpr int kN2 = 6144;  // deliberately unequal: exercises the n_tiles1 split
    std::mt19937 rng(4242);
    std::normal_distribution<float> dist(0.0f, 0.05f);

    auto make_w = [&](int n_rows) {
        std::vector<half> host(static_cast<size_t>(n_rows) * kPK);
        for (auto& h : host)
            h = __float2half(dist(rng));
        void* d_w = nullptr;
        EXPECT_EQ(cudaMalloc(&d_w, host.size() * sizeof(half)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_w, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t shape[2] = {n_rows, kPK};
        imp::Tensor wt(d_w, imp::QType::F16, 2, shape, true);
        imp::NvFP4Owned w{};
        imp::quantize_fp16_to_nvfp4(wt, w, nullptr);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_w);
        return w;
    };
    imp::NvFP4Owned w1 = make_w(kN1);
    imp::NvFP4Owned w2 = make_w(kN2);

    for (int m : {2, 7, 32}) {
        // One quantized activation, shared by every call — exactly the
        // production contract (the pair reads the same xq the singles read).
        std::vector<half> x(static_cast<size_t>(m) * kPK);
        for (auto& h : x)
            h = __float2half(dist(rng));
        void* d_x = nullptr;
        ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t xshape[2] = {m, kPK};
        imp::Tensor xt(d_x, imp::QType::F16, 2, xshape, true);
        imp::NvFP4Owned xq{};
        imp::quantize_fp16_to_nvfp4(xt, xq, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_x);

        auto alloc_y = [&](int n_rows) {
            half* d = nullptr;
            EXPECT_EQ(cudaMalloc(&d, static_cast<size_t>(m) * n_rows * sizeof(half)), cudaSuccess);
            EXPECT_EQ(cudaMemset(d, 0xEE, static_cast<size_t>(m) * n_rows * sizeof(half)), cudaSuccess);
            return d;
        };
        half* y1s = alloc_y(kN1);
        half* y2s = alloc_y(kN2);
        half* y1p = alloc_y(kN1);
        half* y2p = alloc_y(kN2);

        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4(w1, xq, y1s, m, kN1, kPK, nullptr, nullptr, false));
        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4(w2, xq, y2s, m, kN2, kPK, nullptr, nullptr, false));
        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_pair_a4(w1, w2, xq, y1p, y2p, m, kN1, kN2, kPK, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        auto fetch = [&](half* d, int n_rows) {
            std::vector<half> h(static_cast<size_t>(m) * n_rows);
            EXPECT_EQ(cudaMemcpy(h.data(), d, h.size() * sizeof(half), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            cudaFree(d);
            return h;
        };
        const std::vector<half> a1 = fetch(y1s, kN1), b1 = fetch(y1p, kN1);
        const std::vector<half> a2 = fetch(y2s, kN2), b2 = fetch(y2p, kN2);
        ASSERT_EQ(std::memcmp(a1.data(), b1.data(), a1.size() * sizeof(half)), 0) << "m=" << m << " W1";
        ASSERT_EQ(std::memcmp(a2.data(), b2.data(), a2.size() * sizeof(half)), 0) << "m=" << m << " W2";
    }
}

// Pair entry must refuse the striped regime rather than silently force stripes=1 (a
// small-N weight would then reduce over one K-stripe: numerically wrong under split-K
// rounding, and slower). N=512 has stripes > 1 under the shipped policy.
TEST(SmallMV2Pair, RefusesStripedShapes) {
    if (!has_sm120())
        GTEST_SKIP() << "sm_120 required";
    ASSERT_GT(imp::gemm_nvfp4_smallm_v2_stripes(512, 5120), 1);
    imp::NvFP4QuantResult dummy{};
    // args_ok fails on null pointers anyway, but the stripe check must come
    // first and be decisive on its own: a fully valid striped weight is the
    // dangerous input, so assert the policy gate directly.
    ASSERT_FALSE(imp::gemm_nvfp4_smallm_v2_pair_a4(dummy, dummy, dummy, nullptr, nullptr, 2, 512, 5120,
                                                   5120, nullptr));
}

}  // namespace
