// Q4_K IMMA (qkind 1) vs the ggml dequant formula in double: MoE grouped and dense small-M. Own file:
// test_mmq_q8_imma.cu sits at the function-size gate. Outlier channels (every 37th K at 20x) price the
// per-128 activation scale (#2441).
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the IMMA activation scratch
namespace {

// 144-B super-block: d, dmin, 12-B 6-bit scales/mins, qs[128]; element l of 32-byte group g is the
// low (sub-block 2g) or high (2g + 1) nibble of qs[32g + l].
double q4k_weight(const uint8_t* bp, int i) {
    __half dh, mh;
    std::memcpy(&dh, bp, 2);
    std::memcpy(&mh, bp + 2, 2);
    const uint8_t* sc = bp + 4;
    const uint8_t* qs = bp + 16;
    const int g = i / 64, hi = (i % 64) / 32, l = i % 32, j = 2 * g + hi;
    const int s = j < 4 ? (sc[j] & 63) : ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
    const int m = j < 4 ? (sc[j + 4] & 63) : ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    const int q = hi ? (qs[g * 32 + l] >> 4) : (qs[g * 32 + l] & 0xF);
    return __half2float(dh) * s * q - __half2float(mh) * m;
}

void gen_q4k_blocks(std::vector<uint8_t>& W, std::mt19937& rng) {
    for (size_t b = 0; b < W.size() / 144; ++b) {
        uint8_t* bp = W.data() + b * 144;
        const __half dh = __float2half(0.0005f + 0.00002f * (rng() % 100));
        const __half mh = __float2half(0.0002f + 0.00001f * (rng() % 100));
        std::memcpy(bp, &dh, 2);
        std::memcpy(bp + 2, &mh, 2);
        for (int i = 4; i < 144; ++i)
            bp[i] = static_cast<uint8_t>(rng() & 0xFF);
    }
}

double q4k_moe_nrmse(const std::vector<int>& rows_per, int N, int K, unsigned seed, float outlier) {
    const int ne = static_cast<int>(rows_per.size());
    std::vector<int32_t> h_off(ne + 1, 0);
    int max_rows = 0;
    for (int e = 0; e < ne; ++e) {
        h_off[e + 1] = h_off[e] + rows_per[e];
        max_rows = std::max(max_rows, rows_per[e]);
    }
    const int expanded = h_off[ne];
    const int sbpr = K / 256;
    std::vector<uint8_t> W(static_cast<size_t>(ne) * N * sbpr * 144);
    std::mt19937 rng(seed);
    gen_q4k_blocks(W, rng);
    std::vector<__half> x(static_cast<size_t>(expanded) * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2half(nd(rng) * ((i % K) % 37 == 0 ? outlier : 1.0f));

    uint8_t* d_w;
    __half *d_x, *d_out;
    int32_t* d_off;
    EXPECT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_out, static_cast<size_t>(expanded) * N * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_off, h_off.size() * sizeof(int32_t)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    EXPECT_TRUE(
        mmq_imma_moe_gemm(d_w, /*qkind=*/1, d_x, d_out, d_off, max_rows, expanded, ne, N, K, nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(expanded) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    double err2 = 0, ref2 = 0;
    for (int e = 0; e < ne; ++e)
        for (int r = h_off[e]; r < h_off[e + 1]; ++r)
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int sb = 0; sb < sbpr; ++sb) {
                    const uint8_t* bp = W.data() + ((static_cast<size_t>(e) * N + n) * sbpr + sb) * 144;
                    for (int i = 0; i < 256; ++i)
                        ref += static_cast<double>(__half2float(x[(size_t)r * K + sb * 256 + i])) *
                               q4k_weight(bp, i);
                }
                const double got = __half2float(out[(size_t)r * N + n]);
                err2 += (got - ref) * (got - ref);
                ref2 += ref * ref;
            }
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_off);
    mmq_q8_imma_release_all();
    return std::sqrt(err2 / std::max(ref2, 1e-30));
}

TEST(MmqQ8Imma, MoeGroupedQ4K) {
    // BM=32 (max rows < 96) and BM=128 variants; N=192 leaves a partial 128-column tile.
    const double small_m = q4k_moe_nrmse({70, 41}, 192, 512, 401, 1.0f);
    const double large_m = q4k_moe_nrmse({70, 41, 130}, 192, 512, 402, 1.0f);
    const double outl = q4k_moe_nrmse({70, 41}, 192, 512, 403, 20.0f);
    std::fprintf(stderr, "[q4k moe] NRMSE BM=32 %.3e  BM=128 %.3e  outlier-20x %.3e\n", small_m, large_m,
                 outl);
    EXPECT_LT(small_m, 2e-2) << "Q4_K MoE NRMSE, BM=32";
    EXPECT_LT(large_m, 2e-2) << "Q4_K MoE NRMSE, BM=128";
    EXPECT_LT(outl, 2e-2) << "Q4_K MoE NRMSE, outlier channels";
}

// Dense Q4_K fixture: W[N][K] and 128 activation rows; run() feeds the first M rows into a residual
// (beta 1) or a zeroed output (beta 0) and returns out - residual.
struct Q4kDense {
    int N, K;
    std::vector<uint8_t> W;
    std::vector<__half> x;
    Q4kDense(int n, int k, unsigned seed) : N(n), K(k), W(static_cast<size_t>(n) * (k / 256) * 144) {
        std::mt19937 rng(seed);
        gen_q4k_blocks(W, rng);
        x.resize(static_cast<size_t>(128) * K);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : x)
            v = __float2half(nd(rng));
    }
    std::vector<float> run(int M, bool beta1, bool split) const {
        std::vector<__half> resid(static_cast<size_t>(M) * N, __float2half(0.0f));
        std::mt19937 rng(7);
        if (beta1)
            for (auto& v : resid)
                v = __float2half(static_cast<float>(rng() % 64) * 0.25f - 8.0f);
        uint8_t* d_w;
        __half *d_x, *d_out;
        EXPECT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_x, static_cast<size_t>(M) * K * 2), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out, resid.size() * 2), cudaSuccess);
        cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), static_cast<size_t>(M) * K * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, resid.data(), resid.size() * 2, cudaMemcpyHostToDevice);
        EXPECT_TRUE(mmq_q4k_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr, beta1 ? 1.0f : 0.0f, split));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<__half> out(resid.size());
        cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_out);
        mmq_q8_imma_release_all();
        std::vector<float> got(out.size());
        for (size_t i = 0; i < out.size(); ++i)
            got[i] = __half2float(out[i]) - __half2float(resid[i]);
        return got;
    }
    double nrmse(const std::vector<float>& got, int M) const {
        const int sbpr = K / 256;
        double err2 = 0, ref2 = 0;
        for (int r = 0; r < M; ++r)
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int sb = 0; sb < sbpr; ++sb) {
                    const uint8_t* bp = W.data() + (static_cast<size_t>(n) * sbpr + sb) * 144;
                    for (int i = 0; i < 256; ++i)
                        ref += static_cast<double>(__half2float(x[(size_t)r * K + sb * 256 + i])) *
                               q4k_weight(bp, i);
                }
                const double d = got[(size_t)r * N + n] - ref;
                err2 += d * d;
                ref2 += ref * ref;
            }
        return std::sqrt(err2 / std::max(ref2, 1e-30));
    }
};

// M <= 32: BM=32 tile, rows 16..31 skipped at M <= 16; K=2816 = 22 steps split 3 x 7 + 1 over 8 slices.
TEST(MmqQ8Imma, Q4KDenseSmallMSplitNRMSE) {
    const Q4kDense f(384, 2816, 501);
    const std::pair<int, bool> cases[] = {{2, false}, {9, true}, {16, false}, {17, true}, {32, false}};
    for (const auto& [M, beta1] : cases)
        EXPECT_LT(f.nrmse(f.run(M, beta1, true), M), 2e-2) << "Q4_K dense M=" << M << " beta1=" << beta1;
}

// Above 32 rows with the split allowed (not runtime.deterministic): the 64-K-step GGUF kernel, M=100 / 128
// one partial and one full row block.
TEST(MmqQ8Imma, Q4KDenseLargeMNRMSE) {
    const Q4kDense f(384, 2816, 503);
    for (const auto& [M, beta1] : {std::pair{100, false}, {128, true}})
        EXPECT_LT(f.nrmse(f.run(M, beta1, true), M), 2e-2) << "Q4_K dense M=" << M << " beta1=" << beta1;
}

// Without the split a short row is the same bits as inside a 100-row BM=128 call (#2152); the split
// path really runs (some values move).
TEST(MmqQ8Imma, Q4KDenseSmallMNoSplitMatchesBM128) {
    const Q4kDense f(384, 2816, 502);
    const std::vector<float> big = f.run(100, false, false);
    const std::vector<float> plain = f.run(20, false, false);
    const std::vector<float> split = f.run(20, false, true);
    size_t plain_diff = 0, split_diff = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        plain_diff += plain[i] != big[i];
        split_diff += split[i] != big[i];
    }
    EXPECT_EQ(plain_diff, 0u);
    EXPECT_GT(split_diff, 0u);
}

// A BM=128 grid with a short last wave (128 rows, n_sms + n_sms / 8 column tiles) runs its trailing column
// tiles K-split: the leading columns keep their bits, the tail columns move and stay accurate.
TEST(MmqQ8Imma, Q4KDenseWaveTailSplit) {
    int n_sms = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, 0), cudaSuccess);
    const int n0 = n_sms * 128;
    const Q4kDense f(n0 + ((n_sms / 8) * 128), 256, 504);
    for (const bool beta1 : {false, true}) {
        const std::vector<float> plain = f.run(128, beta1, false);
        const std::vector<float> tail = f.run(128, beta1, true);
        size_t head_diff = 0, tail_diff = 0;
        for (int r = 0; r < 128; ++r)
            for (int n = 0; n < f.N; ++n) {
                const size_t i = (static_cast<size_t>(r) * f.N) + n;
                (n < n0 ? head_diff : tail_diff) += plain[i] != tail[i];
            }
        EXPECT_EQ(head_diff, 0u) << "beta1=" << beta1;
        EXPECT_GT(tail_diff, 0u) << "beta1=" << beta1;
        EXPECT_LT(f.nrmse(tail, 128), 2e-2) << "beta1=" << beta1;
    }
}

}  // namespace
}  // namespace imp
