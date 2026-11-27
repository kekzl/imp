// Small-M NVFP4 GEMM: correctness vs per-element dequant reference (W4A16 family, same as
// M=1 decode GEMVs); bandwidth must beat CUTLASS's measured 41.4us/19% of weight floor at
// M=32,N=5120,K=5120 (docs/archive/plans/2026-08-24-qwen38-port.md).
// Bandwidth cases (above 40% of floor): tests/bench/test_nvfp4_smallm_bench.cu.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"
#include "test_nvfp4_smallm_util.cuh"

namespace {

// Host dequant of one weight element from the packed buffers.
float host_dequant(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
                   float tensor_scale, int n, int k, int K) {
    static const float lut[16] = {0.0f, 0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    uint8_t byte = packed[(size_t)n * (K / 2) + k / 2];
    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
    uint8_t se = scales[(size_t)n * (K / 16) + k / 16];
    // FP8 E4M3 decode
    int sign = (se >> 7) ? -1 : 1;
    int exp = (se >> 3) & 0xF;
    int mant = se & 0x7;
    float sf = (exp == 0) ? sign * (mant / 8.0f) * std::pow(2.0f, -6.0f)
                          : sign * (1.0f + mant / 8.0f) * std::pow(2.0f, exp - 7);
    return lut[nib] * sf * tensor_scale;
}

TEST_F(NvFP4SmallMTest, MatchesHostReference) {
    const int M = 19, N = 48, K = 512;  // deliberately not tile-aligned in N/M
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Random FP16 source weight, quantized on device by the shipping quantizer.
    std::vector<__half> w_h((size_t)N * K);
    for (auto& v : w_h) v = __float2half(dist(rng));
    void* d_w = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, w_h.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, w_h.data(), w_h.size() * sizeof(__half), cudaMemcpyHostToDevice),
              cudaSuccess);
    imp::NvFP4Owned q{};
    {
        int64_t shp[2] = {N, K};
        imp::Tensor wt(d_w, imp::QType::F16, 2, shp, /*on_device=*/true);
        imp::quantize_fp16_to_nvfp4(wt, q, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_NE(q.packed_data, nullptr);
    }

    std::vector<__half> x_h((size_t)M * K);
    for (auto& v : x_h) v = __float2half(dist(rng));
    void *d_x = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, x_h.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, x_h.data(), x_h.size() * sizeof(__half), cudaMemcpyHostToDevice),
              cudaSuccess);

    void* d_ws = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ws, imp::gemm_nvfp4_smallm_workspace_bytes(N)), cudaSuccess);
    ASSERT_TRUE(imp::gemm_nvfp4_smallm(q, static_cast<const half*>(d_x), static_cast<half*>(d_y), M, N,
                                       K, d_ws, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__half> y_h((size_t)M * N);
    ASSERT_EQ(cudaMemcpy(y_h.data(), d_y, y_h.size() * sizeof(__half), cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Host reference from the ACTUAL quantized buffers.
    std::vector<uint8_t> packed((size_t)N * K / 2), scales((size_t)N * K / 16);
    ASSERT_EQ(cudaMemcpy(packed.data(), q.packed_data, packed.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(scales.data(), q.micro_scales, scales.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);

    double max_rel = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += host_dequant(packed, scales, q.tensor_scale, n, k, K) *
                       __half2float(x_h[(size_t)m * K + k]);
            double got = __half2float(y_h[(size_t)m * N + n]);
            double rel = std::abs(got - ref) / std::max(1.0, std::abs(ref));
            max_rel = std::max(max_rel, rel);
        }
    }
    // FP16 accumulation inside a 256-element tile + FP32 across tiles: a few
    // e-3 relative against the FP64 host walk is the expected envelope.
    EXPECT_LT(max_rel, 2e-2) << "max relative error " << max_rel;

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
}

TEST_F(NvFP4SmallMTest, A4MatchesHostReference) {
    const int M = 19, N = 48, K = 512;
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<__half> w_h((size_t)N * K), x_h((size_t)M * K);
    for (auto& v : w_h) v = __float2half(dist(rng));
    for (auto& v : x_h) v = __float2half(dist(rng));
    QuantBuf W, X;
    W.quantize(w_h, N, K);
    X.quantize(x_h, M, K);

    void* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
    void* d_ws = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ws, imp::gemm_nvfp4_smallm_workspace_bytes(N)), cudaSuccess);
    ASSERT_TRUE(imp::gemm_nvfp4_smallm_a4(W.q, X.q, static_cast<half*>(d_y), M, N, K, d_ws, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__half> y_h((size_t)M * N);
    ASSERT_EQ(cudaMemcpy(y_h.data(), d_y, y_h.size() * sizeof(__half), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<uint8_t> wp((size_t)N * K / 2), wsc((size_t)N * K / 16);
    std::vector<uint8_t> xp((size_t)M * K / 2), xsc((size_t)M * K / 16);
    ASSERT_EQ(cudaMemcpy(wp.data(), W.q.packed_data, wp.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(wsc.data(), W.q.micro_scales, wsc.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(xp.data(), X.q.packed_data, xp.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(xsc.data(), X.q.micro_scales, xsc.size(), cudaMemcpyDeviceToHost), cudaSuccess);

    double max_rel = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += host_dequant(xp, xsc, X.q.tensor_scale, m, k, K) *
                       host_dequant(wp, wsc, W.q.tensor_scale, n, k, K);
            double got = __half2float(y_h[(size_t)m * N + n]);
            double rel = std::abs(got - ref) / std::max(1.0, std::abs(ref));
            max_rel = std::max(max_rel, rel);
        }
    }
    EXPECT_LT(max_rel, 2e-2) << "max relative error " << max_rel;
    cudaFree(d_y); cudaFree(d_ws);
}

}  // namespace
