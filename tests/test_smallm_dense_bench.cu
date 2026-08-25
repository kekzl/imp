// =============================================================================
// test_smallm_dense_bench.cu — gemm_grouped_nvfp4_smallM as a DENSE small-M GEMM
// =============================================================================
//
// The hand-rolled mxf4nvf4 grouped GEMM (M-tiles 16/32/64/128, plain row-major
// UE4M3 scales) ships as an MoE-prefill opt-in. Batched decode at n_seq<=32
// runs the same shapes through the CUTLASS 128x128 cooperative tile, measured
// at 41.4 us on M=32 N=5120 K=5120 — 19% of the weight floor (grid 40 CTAs).
// With n_experts=1 the grouped kernel IS a dense small-M GEMM; this test
// checks correctness against a host dequant walk and benches that shape —
// the bench documents WHY it is not wired (92.3 us vs CUTLASS 41.4: the
// persistent grouped design has the same N-limited parallelism).
//
// GPU required — skips cleanly without one.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <vector>

#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct QuantBuf {
    imp::NvFP4QuantResult q{};
    void quantize(const std::vector<__half>& src, int rows, int K) {
        void* d = nullptr;
        ASSERT_EQ(cudaMalloc(&d, src.size() * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d, src.data(), src.size() * sizeof(__half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t shp[2] = {rows, K};
        imp::Tensor t(d, imp::QType::F16, 2, shp, /*on_device=*/true);
        imp::quantize_fp16_to_nvfp4(t, q, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_NE(q.packed_data, nullptr);
        cudaFree(d);
    }
};

float host_dequant(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
                   float tensor_scale, int n, int k, int K) {
    static const float lut[16] = {0.0f, 0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    uint8_t byte = packed[(size_t)n * (K / 2) + k / 2];
    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
    uint8_t se = scales[(size_t)n * (K / 16) + k / 16];
    int exp = (se >> 3) & 0xF;
    int mant = se & 0x7;  // UE4M3: unsigned
    float sf = (exp == 0) ? (mant / 8.0f) * std::pow(2.0f, -6.0f)
                          : (1.0f + mant / 8.0f) * std::pow(2.0f, exp - 7);
    return lut[nib] * sf * tensor_scale;
}

class SmallMDenseTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!gpu_available()) GTEST_SKIP() << "no CUDA device";
        if (!imp::gemm_grouped_nvfp4_smallM_available())
            GTEST_SKIP() << "smallM grouped GEMM unavailable on this device";
    }
};

// One "expert", dense shape: D = A @ B^T with both sides NVFP4 + row-major
// scales, alpha = tsA * tsB.
static bool run_dense(const imp::NvFP4QuantResult& a, const imp::NvFP4QuantResult& b, void* d_out,
                      int M, int N, int K, float* d_alpha, cudaStream_t stream) {
    int hM[1] = {M};
    const void* pa[1] = {a.packed_data};
    const void* psa[1] = {a.micro_scales};
    const void* pb[1] = {b.packed_data};
    const void* psb[1] = {b.micro_scales};
    void* pd[1] = {d_out};
    return imp::gemm_grouped_nvfp4_smallM(1, hM, N, K, pa, psa, pb, psb, pd, d_alpha, stream);
}

TEST_F(SmallMDenseTest, MatchesHostReference) {
    const int M = 24, N = 128, K = 512;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<__half> a_h((size_t)M * K), b_h((size_t)N * K);
    for (auto& v : a_h) v = __float2half(dist(rng));
    for (auto& v : b_h) v = __float2half(dist(rng));
    QuantBuf A, B;
    A.quantize(a_h, M, K);
    B.quantize(b_h, N, K);

    void* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
    float* d_alpha = nullptr;
    float alpha = A.q.tensor_scale * B.q.tensor_scale;
    ASSERT_EQ(cudaMalloc(&d_alpha, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_TRUE(run_dense(A.q, B.q, d_y, M, N, K, d_alpha, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__half> y_h((size_t)M * N);
    ASSERT_EQ(cudaMemcpy(y_h.data(), d_y, y_h.size() * sizeof(__half), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<uint8_t> ap((size_t)M * K / 2), as((size_t)M * K / 16);
    std::vector<uint8_t> bp((size_t)N * K / 2), bs((size_t)N * K / 16);
    ASSERT_EQ(cudaMemcpy(ap.data(), A.q.packed_data, ap.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(as.data(), A.q.micro_scales, as.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(bp.data(), B.q.packed_data, bp.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(bs.data(), B.q.micro_scales, bs.size(), cudaMemcpyDeviceToHost), cudaSuccess);

    double max_rel = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += host_dequant(ap, as, A.q.tensor_scale, m, k, K) *
                       host_dequant(bp, bs, B.q.tensor_scale, n, k, K);
            double got = __half2float(y_h[(size_t)m * N + n]);
            double rel = std::abs(got - ref) / std::max(1.0, std::abs(ref));
            max_rel = std::max(max_rel, rel);
        }
    }
    EXPECT_LT(max_rel, 2e-2) << "max relative error " << max_rel;

    cudaFree(d_y); cudaFree(d_alpha);
}

TEST_F(SmallMDenseTest, BenchDecodeShape) {
    const int M = 32, N = 5120, K = 5120;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<__half> a_h((size_t)M * K), b_h((size_t)N * K);
    for (auto& v : a_h) v = __float2half(dist(rng));
    for (auto& v : b_h) v = __float2half(dist(rng));
    QuantBuf A, B;
    A.quantize(a_h, M, K);
    B.quantize(b_h, N, K);

    void* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
    float* d_alpha = nullptr;
    float alpha = A.q.tensor_scale * B.q.tensor_scale;
    ASSERT_EQ(cudaMalloc(&d_alpha, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    for (int i = 0; i < 300; ++i)
        run_dense(A.q, B.q, d_y, M, N, K, d_alpha, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int iters = 300;
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        run_dense(A.q, B.q, d_y, M, N, K, d_alpha, nullptr);
    cudaEventRecord(t1);
    ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    const double us = ms * 1000.0 / iters;
    const double bytes = (double)N * K / 2 + (double)N * K / 16;
    printf("grouped-smallM dense M=%d N=%d K=%d: %.2f us/GEMM, %.0f GB/s weight read "
           "(floor 8.23 us; CUTLASS 128x128 cooperative measured 41.4 us)\n",
           M, N, K, us, bytes / (us * 1e-6) / 1e9);
    // MEASURED 2026-08-25: 92.3 us — the persistent grouped design also
    // bottoms out on this dense shape (its parallelism is N/N-tile work
    // items, the same ~40 units that starve CUTLASS), so it is NOT wired
    // into the dense batch path. Fifth refutation for the M=32 lever; see
    // docs/plans/2026-08-24-qwen38-port.md. The bar below is a regression
    // anchor on the measurement, not a target.
    EXPECT_LT(us, 140.0);

    cudaFree(d_y); cudaFree(d_alpha);
}

}  // namespace
