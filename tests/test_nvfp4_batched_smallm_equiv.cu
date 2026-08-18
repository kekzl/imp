// tests/test_nvfp4_batched_smallm_equiv.cu
//
// Gate for the small-M batched NVFP4 GEMV that speculative verify chunks route
// into (`gemm_nvfp4_batched`, #998/#1055).
//
// It exists because the end-to-end equivalence check for that routing cannot
// see it. `ctx.spec_verify_small_m` is only true inside a verify chunk, so a
// no-speculation arm exercises the OLD path by construction and comes back
// byte-identical whatever the new path computes, while the speculative arms are
// not reproducible across processes (#1457) and cannot be diffed either. A
// change that reroutes 48 of 64 layers onto a different kernel family therefore
// had no gate at all until this file.
//
// Two invariants, both deterministic and both at the shapes a GDN hybrid
// actually uses:
//   1. row m of a batched M-row call equals the single-row call for that row,
//      which is what the MR bucket selection (<1>/<2>/<3>/<4>) must preserve;
//   2. the result matches a dequantise-then-multiply reference, so "both paths
//      agree" cannot be satisfied by both being wrong the same way.

#include <gtest/gtest.h>

#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

bool has_sm120() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess)
        return false;
    return p.major == 12;
}

// Shapes from Qwen3.8-27B's GDN layers, which are the ones the verify chunk
// routes: d_model 5120 into the in/out projections. K stays a multiple of 16.
constexpr int kK = 5120;
constexpr int kN = 512;  // one N tile; the kernel walks N in blocks anyway

struct Fixture {
    imp::NvFP4QuantResult w{};
    std::vector<half> w_fp16;  // dequantised reference weight, [kN, kK]
    ~Fixture() { imp::free_nvfp4_result(w); }
};

// Quantise a random weight and keep its dequantised form as the reference.
void build(Fixture& f) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::vector<half> host(static_cast<size_t>(kN) * kK);
    for (auto& h : host)
        h = __float2half(dist(rng));

    void* d_w = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, host.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice),
              cudaSuccess);
    int64_t shape[2] = {kN, kK};
    imp::Tensor wt(d_w, imp::QType::F16, 2, shape, /*on_device=*/true);
    imp::quantize_fp16_to_nvfp4(wt, f.w, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // The reference multiplies the weight the kernel actually holds, not the
    // pre-quantisation one: this test gates the GEMV, not the quantiser.
    void* d_deq = nullptr;
    ASSERT_EQ(cudaMalloc(&d_deq, host.size() * sizeof(half)), cudaSuccess);
    imp::dequantize_nvfp4_to_fp16(f.w, d_deq, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    f.w_fp16.resize(host.size());
    ASSERT_EQ(cudaMemcpy(f.w_fp16.data(), d_deq, host.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_deq);
    cudaFree(d_w);
}

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

class BatchedSmallM : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!has_sm120())
            GTEST_SKIP() << "sm_120 required";
        build(f_);
    }
    Fixture f_;
};

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

}  // namespace
