// Q6_K MoE grouped IMMA (qkind 2, 210-B blocks read in place) vs the ggml dequant formula.
// Own file: test_mmq_q8_imma.cu sits at the function-size gate.
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

}  // namespace
}  // namespace imp
