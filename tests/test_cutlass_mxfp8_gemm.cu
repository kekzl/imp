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

static float e4m3_to_float(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    const float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}

// Same addressing as mx8_sfatom_offset (128 rows x 4 K-groups per 512-byte atom).
static int sfatom_offset(int row, int k_group, int n_k_tiles) {
    const int row_local = row % 128, k_local = k_group % 4;
    return ((row / 128) * n_k_tiles + k_group / 4) * 512 + (row_local % 32) * 16 + (row_local / 32) * 4 + k_local;
}

// The rebuild of a freed F16 GDN weight from its MXFP8 copy (released_source_gemm_): every element
// equals the host decode E4M3(byte) * 2^(scale - 127) bit for bit, and the rebuilt weight stays
// within E4M3 rounding of the source. Partial atoms in both dimensions (300 x 1120).
TEST(CutlassMxFP8Gemm, DequantizeMatchesTheHostDecode) {
    constexpr int N = 300, K = 1120;
    std::mt19937 rng(5u);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::vector<half> src(static_cast<size_t>(N) * K);
    for (auto& v : src)
        v = __float2half(dist(rng));
    const size_t sf_bytes = cutlass_mxfp8_sf_size(N, K);
    void *d_src = nullptr, *d_back = nullptr, *q1 = nullptr, *sf1 = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_back, src.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&q1, src.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&sf1, sf_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src, src.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    quantize_fp16_to_mxfp8_cutlass(d_src, q1, sf1, N, K, nullptr);
    CutlassMxFP8Weight w;
    w.data = q1;
    w.scale_factors = sf1;
    w.N = N;
    w.K = K;
    dequantize_mxfp8_cutlass_to_fp16(w, d_back, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> hq1(src.size()), hs1(sf_bytes);
    std::vector<half> back(src.size());
    cudaMemcpy(hq1.data(), q1, hq1.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs1.data(), sf1, sf_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(back.data(), d_back, back.size() * sizeof(half), cudaMemcpyDeviceToHost);
    const int n_k_tiles = (K + 127) / 128;
    int mismatches = 0;
    for (int r = 0; r < N; ++r) {
        for (int k = 0; k < K; ++k) {
            const size_t i = static_cast<size_t>(r) * K + k;
            const float scale = std::ldexp(1.0f, hs1[sfatom_offset(r, k / 32, n_k_tiles)] - 127);
            const half want = __float2half(e4m3_to_float(hq1[i]) * scale);
            if (__half_as_ushort(want) != __half_as_ushort(back[i]))
                ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "rebuilt elements differ from the host decode";
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        const double a = __half2float(src[i]), b = __half2float(back[i]);
        err += (a - b) * (a - b);
        ref += a * a;
    }
    EXPECT_LT(std::sqrt(err / ref), 0.05) << "rebuild outside E4M3 rounding of the source";
    for (void* p : {d_src, d_back, q1, sf1})
        cudaFree(p);
}
