// CUTLASS sm_120 MXFP8 x MXFP8 GEMM (gemm.mxfp8_gdn_proj_prefill): the quantizer's SfAtom
// scales and E4M3 bytes must reproduce an FP32 reference within the E4M3 budget, at the GDN
// projection shapes, and stay finite on the case that broke the FP8 SSM_IN attempt (coherent
// activation against 0.01-magnitude weight rows, docs/plans/2026-08-31-fp8-ssm-prefill.md).

#include "compute/gemm_cutlass_mxfp8_sm120.h"

#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

using namespace imp;

namespace {

struct Problem {
    int M, N, K;
    std::vector<float> a, w;  // row-major [M,K], [N,K]
};

Problem make_problem(int M, int N, int K, float w_mag, bool coherent_a, unsigned seed) {
    Problem p{M, N, K, {}, {}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    p.a.resize(static_cast<size_t>(M) * K);
    p.w.resize(static_cast<size_t>(N) * K);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            p.a[static_cast<size_t>(m) * K + k] = coherent_a ? 1.0f + 0.05f * nd(rng) : nd(rng);
    for (auto& v : p.w)
        v = w_mag * nd(rng);
    return p;
}

// Runs quantize(A), quantize(W), GEMM; returns D as float and the reference A W^T.
void run(const Problem& p, std::vector<float>& d, std::vector<float>& ref) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    std::vector<half> a_h(p.a.size()), w_h(p.w.size());
    for (size_t i = 0; i < p.a.size(); i++)
        a_h[i] = __float2half(p.a[i]);
    for (size_t i = 0; i < p.w.size(); i++)
        w_h[i] = __float2half(p.w[i]);
    half *d_a, *d_w, *d_out;
    ASSERT_EQ(cudaMalloc(&d_a, a_h.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, w_h.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(p.M) * p.N * sizeof(half)), cudaSuccess);
    cudaMemcpy(d_a, a_h.data(), a_h.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w_h.data(), w_h.size() * sizeof(half), cudaMemcpyHostToDevice);

    CutlassMxFP8Weight w;
    w.N = p.N;
    w.K = p.K;
    w.data_bytes = static_cast<size_t>(p.N) * p.K;
    w.sf_bytes = cutlass_mxfp8_sf_size(p.N, p.K);
    ASSERT_EQ(cudaMalloc(&w.data, w.data_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&w.scale_factors, w.sf_bytes), cudaSuccess);
    quantize_fp16_to_mxfp8_cutlass(d_w, w.data, w.scale_factors, p.N, p.K, stream);
    void *act, *act_sf, *ws = nullptr;
    const size_t sf_bytes = cutlass_mxfp8_sf_size(p.M, p.K);
    const size_t ws_bytes = gemm_mxfp8_cutlass_sm120_workspace(p.M, p.N, p.K);
    ASSERT_EQ(cudaMalloc(&act, static_cast<size_t>(p.M) * p.K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&act_sf, sf_bytes), cudaSuccess);
    if (ws_bytes)
        ASSERT_EQ(cudaMalloc(&ws, ws_bytes), cudaSuccess);
    quantize_fp16_to_mxfp8_cutlass(d_a, act, act_sf, p.M, p.K, stream);
    ASSERT_TRUE(gemm_mxfp8_cutlass_sm120(act, act_sf, w, d_out, p.M, p.N, p.K, ws, ws_bytes, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<half> out_h(static_cast<size_t>(p.M) * p.N);
    cudaMemcpy(out_h.data(), d_out, out_h.size() * sizeof(half), cudaMemcpyDeviceToHost);
    d.resize(out_h.size());
    for (size_t i = 0; i < out_h.size(); i++)
        d[i] = __half2float(out_h[i]);
    ref.assign(d.size(), 0.0f);
    for (int m = 0; m < p.M; m++)
        for (int n = 0; n < p.N; n++) {
            double acc = 0.0;
            for (int k = 0; k < p.K; k++)
                acc += static_cast<double>(__half2float(a_h[static_cast<size_t>(m) * p.K + k])) *
                       __half2float(w_h[static_cast<size_t>(n) * p.K + k]);
            ref[static_cast<size_t>(m) * p.N + n] = static_cast<float>(acc);
        }
    cudaFree(w.data);
    cudaFree(w.scale_factors);
    cudaFree(act);
    cudaFree(act_sf);
    if (ws)
        cudaFree(ws);
    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
}

double rel_rms(const std::vector<float>& d, const std::vector<float>& ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < d.size(); i++) {
        const double e = static_cast<double>(d[i]) - ref[i];
        num += e * e;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1.0));
}

}  // namespace

// Gaussian operands at a GDN in_proj-like shape (M not a tile multiple): E4M3 keeps 3
// mantissa bits, so a K=2048 dot product reads ~3.7 % relative RMS with both operands E4M3 (E2M1 would read
// ~15 %).
TEST(CutlassMxFP8Gemm, MatchesFp32ReferenceWithinE4M3Budget) {
    const Problem p = make_problem(/*M=*/200, /*N=*/512, /*K=*/2048, 1.0f, false, 7u);
    std::vector<float> d, ref;
    run(p, d, ref);
    if (::testing::Test::HasFatalFailure())
        return;
    const double err = rel_rms(d, ref);
    EXPECT_LT(err, 0.05) << "relative RMS " << err;
    EXPECT_GT(err, 1e-5) << "an exact match means the GEMM read the FP16 source, not the E4M3 copy";
}

// Coherent activation (mean 1.0) against 0.01-magnitude weight rows: the per-block power-of-two
// scale keeps every intermediate in range, where a per-tensor act scale plus an FP16 rescale
// of out / row_scale went non-finite (FP8 SSM_IN, 2026-09-01).
TEST(CutlassMxFP8Gemm, TinyWeightRowsAgainstCoherentActivationStayFinite) {
    const Problem p = make_problem(/*M=*/64, /*N=*/256, /*K=*/1024, 0.01f, true, 11u);
    std::vector<float> d, ref;
    run(p, d, ref);
    if (::testing::Test::HasFatalFailure())
        return;
    for (float v : d)
        ASSERT_TRUE(std::isfinite(v));
    EXPECT_LT(rel_rms(d, ref), 0.05);
}

// K that is not a multiple of the 128-element SfAtom K tile (K = 5120 = 40 tiles, but the
// 32-group count per row is not a multiple of 4 for K = 1120): the atom padding must not leak
// into the product.
TEST(CutlassMxFP8Gemm, PartialSfAtomKTile) {
    const Problem p = make_problem(/*M=*/48, /*N=*/128, /*K=*/1120, 1.0f, false, 3u);
    std::vector<float> d, ref;
    run(p, d, ref);
    if (::testing::Test::HasFatalFailure())
        return;
    EXPECT_LT(rel_rms(d, ref), 0.05);
}
