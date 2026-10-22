// Q6_K MoE grouped IMMA (qkind 2, 210-B blocks read in place) vs the ggml dequant formula.
// Own file: test_mmq_q8_imma.cu sits at the function-size gate. Also the dense Q6_K tests (k256 kernel).
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the IMMA activation scratch
namespace {

// 210-B super-block: ql[128], qh[64], int8 scales[16], half d at 208.
double q6k_weight(const uint8_t* bp, int i) {
    const int group = i >> 7, within = i & 127, quad = within >> 5, l = within & 31;
    const uint8_t qlb = bp[(group << 6) + ((quad & 1) << 5) + l];
    const int low4 = quad >= 2 ? (qlb >> 4) : (qlb & 0xF);
    const int high2 = (bp[128 + (group << 5) + l] >> (quad * 2)) & 3;
    __half dh;
    std::memcpy(&dh, bp + 208, 2);
    return __half2float(dh) * static_cast<int8_t>(bp[192 + (i >> 4)]) * (((high2 << 4) | low4) - 32);
}

double q6k_moe_nrmse(const std::vector<int>& rows_per, int N, int K, unsigned seed) {
    const int ne = static_cast<int>(rows_per.size());
    std::vector<int32_t> h_off(ne + 1, 0);
    int max_rows = 0;
    for (int e = 0; e < ne; ++e) {
        h_off[e + 1] = h_off[e] + rows_per[e];
        max_rows = std::max(max_rows, rows_per[e]);
    }
    const int expanded = h_off[ne];
    const int sbpr = K / 256;
    std::vector<uint8_t> W(static_cast<size_t>(ne) * N * sbpr * 210);
    std::mt19937 rng(seed);
    for (size_t b = 0; b < W.size() / 210; ++b) {
        uint8_t* bp = W.data() + b * 210;
        for (int i = 0; i < 192; ++i)
            bp[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 16; ++i)
            bp[192 + i] = static_cast<uint8_t>(static_cast<int8_t>((rng() % 65) - 32));
        const __half dh = __float2half(0.0005f + 0.0003f * (rng() % 100));
        std::memcpy(bp + 208, &dh, 2);
    }
    std::vector<__half> x(static_cast<size_t>(expanded) * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x)
        v = __float2half(nd(rng));

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
        mmq_imma_moe_gemm(d_w, /*qkind=*/2, d_x, d_out, d_off, max_rows, expanded, ne, N, K, nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(expanded) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    double err2 = 0, ref2 = 0;
    for (int e = 0; e < ne; ++e)
        for (int row = h_off[e]; row < h_off[e + 1]; row += 3)
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int sb = 0; sb < sbpr; ++sb) {
                    const uint8_t* bp = W.data() + ((static_cast<size_t>(e) * N + n) * sbpr + sb) * 210;
                    for (int i = 0; i < 256; ++i)
                        ref += static_cast<double>(__half2float(x[(size_t)row * K + sb * 256 + i])) *
                               q6k_weight(bp, i);
                }
                const double got = __half2float(out[(size_t)row * N + n]);
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

// BM=32 (max rows < 96) and BM=128; an empty expert; K=768 = 3 superblocks, so the 210-B block
// parity alternates row to row; N=192 leaves a partial 128-column tile.
TEST(MmqQ8Imma, MoeGroupedQ6K) {
    for (const auto& rows : {std::vector<int>{70, 0, 41}, std::vector<int>{70, 41, 130}})
        EXPECT_LT(q6k_moe_nrmse(rows, 192, 768, 601), 2e-2) << "Q6_K MoE NRMSE, max rows " << rows[2];
}

// ---- Q6_K dense — NRMSE vs full-dequant reference (per-16 scales) ----
namespace q6k_helpers {
constexpr int kSuper = 256;
constexpr int kBytes = 210;
inline void gen_q6k(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    const int bpr = K / kSuper;
    W.resize(static_cast<size_t>(N) * bpr * kBytes);
    std::mt19937 rng(seed);
    for (size_t b = 0; b < W.size() / kBytes; ++b) {
        uint8_t* bp = W.data() + b * kBytes;
        for (int i = 0; i < 192; ++i)
            bp[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 16; ++i)
            bp[192 + i] = static_cast<uint8_t>(static_cast<int8_t>((rng() % 65) - 32));
        __half dh = __float2half(0.0005f + 0.0003f * (rng() % 100));
        std::memcpy(bp + 208, &dh, 2);
    }
}
inline int q6_element(const uint8_t* bp, int i) {
    const int group = i >> 7, within = i & 127, quad = within >> 5, l = within & 31;
    const int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    const int qh_idx = (group << 5) + l;
    const uint8_t qlb = bp[ql_idx];
    const uint8_t low4 = (quad >= 2) ? ((qlb >> 4) & 0xF) : (qlb & 0xF);
    const uint8_t high2 = (bp[128 + qh_idx] >> (quad * 2)) & 0x3;
    return static_cast<int>((high2 << 4) | low4) - 32;
}
}  // namespace q6k_helpers

// GGUF 210-B blocks read in place; odd K/256 alternates the block 4-B parity row to row.
// beta1: out holds a residual before the call and must come back as residual + x @ W^T.
// w_off: weight base offset in bytes (2 = not 16-B aligned: the register-staged fallback).
void run_q6k_dense(int M, int N, int K, bool beta1 = false, int w_off = 0) {
    using namespace q6k_helpers;
    std::vector<uint8_t> W;
    gen_q6k(W, N, K, 311);
    std::vector<__half> x(static_cast<size_t>(M) * K);
    std::mt19937 rng(312);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x)
        v = __float2half(nd(rng));

    uint8_t* d_w;
    __half *d_x, *d_out;
    ASSERT_EQ(cudaMalloc(&d_w, W.size() + 16), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(M) * N * 2), cudaSuccess);
    cudaMemcpy(d_w + w_off, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    std::vector<__half> resid(static_cast<size_t>(M) * N, __float2half(0.0f));
    if (beta1)
        for (auto& v : resid)
            v = __float2half(nd(rng) * 4.0f);
    cudaMemcpy(d_out, resid.data(), resid.size() * 2, cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q6k_imma_gemm(d_w + w_off, d_x, d_out, M, N, K, nullptr, beta1 ? 1.0f : 0.0f));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(M) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    const int bpr = K / kSuper;
    double err2 = 0, ref2 = 0;
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; m += (M <= 32 ? 1 : 7)) {
            double ref = 0;
            for (int sblk = 0; sblk < bpr; ++sblk) {
                const uint8_t* bp = W.data() + (static_cast<size_t>(n) * bpr + sblk) * kBytes;
                __half dh;
                std::memcpy(&dh, bp + 208, 2);
                const double d = __half2float(dh);
                const int8_t* sc = reinterpret_cast<const int8_t*>(bp + 192);
                for (int i = 0; i < kSuper; ++i)
                    ref += static_cast<double>(__half2float(x[(size_t)m * K + sblk * kSuper + i])) *
                           (d * sc[i >> 4] * q6_element(bp, i));
            }
            const double got = static_cast<double>(__half2float(out[(size_t)m * N + n])) -
                               static_cast<double>(__half2float(resid[(size_t)m * N + n]));
            err2 += (got - ref) * (got - ref);
            ref2 += ref * ref;
        }
    }
    const double nrmse = std::sqrt(err2 / std::max(ref2, 1e-30));
    EXPECT_LT(nrmse, 2e-2) << "Q6K dense NRMSE K=" << K;
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    mmq_q8_imma_release_all();
}

TEST(MmqQ8Imma, Q6KDenseNRMSE) { run_q6k_dense(128, 128, 512); }
// K=768: 3 superblocks per row, so the block 4-B parity flips row to row; N=192 leaves a partial tile.
TEST(MmqQ8Imma, Q6KDenseOddSuperblocksNRMSE) { run_q6k_dense(200, 192, 768); }
// Short prompt chunks (2..32 rows): BM=32 tile, M-tail zero-filled, beta 1 = o/down residual add.
TEST(MmqQ8Imma, Q6KDenseSmallM2) { run_q6k_dense(2, 256, 512); }
TEST(MmqQ8Imma, Q6KDenseSmallM17OddSuperblocks) { run_q6k_dense(17, 192, 768); }
TEST(MmqQ8Imma, Q6KDenseSmallM32Beta1) { run_q6k_dense(32, 256, 1024, true); }
TEST(MmqQ8Imma, Q6KDenseSmallM9Beta1) { run_q6k_dense(9, 128, 512, true); }
// 11 superblocks: K-split 3+3+3+2, per-row block parity flips, N=384 = 3 tiles.
TEST(MmqQ8Imma, Q6KDenseSmallMSplitOddSuperblocks) { run_q6k_dense(5, 384, 2816); }
// Weight base 2 bytes past 16-B alignment: register-staged BM=32 fallback, beta 0 and 1.
TEST(MmqQ8Imma, Q6KDenseSmallMUnalignedFallback) {
    run_q6k_dense(7, 256, 768, false, 2);
    run_q6k_dense(7, 256, 768, true, 2);
}
TEST(MmqQ8Imma, Q6KDenseDeclinesBetween32And64) {
    // 33..63 rows: refused, the caller keeps dequant + cuBLAS. Refused before any pointer is read.
    EXPECT_FALSE(mmq_q6k_imma_gemm(nullptr, nullptr, nullptr, 33, 128, 512, nullptr));
    EXPECT_FALSE(mmq_q6k_imma_gemm(nullptr, nullptr, nullptr, 63, 128, 512, nullptr));
    EXPECT_FALSE(mmq_q6k_imma_gemm(nullptr, nullptr, nullptr, 1, 128, 512, nullptr));
}
}  // namespace
}  // namespace imp
