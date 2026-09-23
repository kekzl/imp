// Q5_K MoE grouped IMMA (qkind 4) vs the ggml dequant formula. Own file: test_mmq_q8_imma.cu sits at
// the function-size gate (the checker reads its arena macro onward as one body).
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

// 176-B super-block: d, dmin, 12-B 6-bit scales/mins, qh[32], qs[128]; element l of 32-byte
// group g takes bit 2g (low nibble) or 2g+1 (high nibble) of qh[l].
double q5k_weight(const uint8_t* bp, int i) {
    __half dh, mh;
    std::memcpy(&dh, bp, 2);
    std::memcpy(&mh, bp + 2, 2);
    const uint8_t* sc = bp + 4;
    const uint8_t* qh = bp + 16;
    const uint8_t* ql = bp + 48;
    const int g = i / 64, hi = (i % 64) / 32, l = i % 32, j = 2 * g + hi;
    const int s = j < 4 ? (sc[j] & 63) : ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
    const int m = j < 4 ? (sc[j + 4] & 63) : ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    int q = hi ? (ql[g * 32 + l] >> 4) : (ql[g * 32 + l] & 0xF);
    if (qh[l] & (1u << j))
        q += 16;
    return __half2float(dh) * s * q - __half2float(mh) * m;
}

double q5k_moe_nrmse(const std::vector<int>& rows_per, int N, int K, unsigned seed) {
    const int ne = static_cast<int>(rows_per.size());
    std::vector<int32_t> h_off(ne + 1, 0);
    int max_rows = 0;
    for (int e = 0; e < ne; ++e) {
        h_off[e + 1] = h_off[e] + rows_per[e];
        max_rows = std::max(max_rows, rows_per[e]);
    }
    const int expanded = h_off[ne];
    const int sbpr = K / 256;
    std::vector<uint8_t> W(static_cast<size_t>(ne) * N * sbpr * 176);
    std::mt19937 rng(seed);
    for (size_t b = 0; b < W.size() / 176; ++b) {
        uint8_t* bp = W.data() + b * 176;
        const __half dh = __float2half(0.0005f + 0.00002f * (rng() % 100));
        const __half mh = __float2half(0.0002f + 0.00001f * (rng() % 100));
        std::memcpy(bp, &dh, 2);
        std::memcpy(bp + 2, &mh, 2);
        for (int i = 4; i < 176; ++i)
            bp[i] = static_cast<uint8_t>(rng() & 0xFF);
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
        mmq_imma_moe_gemm(d_w, /*qkind=*/4, d_x, d_out, d_off, max_rows, expanded, ne, N, K, nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(expanded) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    double err2 = 0, ref2 = 0;
    for (int e = 0; e < ne; ++e)
        for (int r = h_off[e]; r < h_off[e + 1]; ++r)
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int sb = 0; sb < sbpr; ++sb) {
                    const uint8_t* bp = W.data() + ((static_cast<size_t>(e) * N + n) * sbpr + sb) * 176;
                    for (int i = 0; i < 256; ++i)
                        ref += static_cast<double>(__half2float(x[(size_t)r * K + sb * 256 + i])) *
                               q5k_weight(bp, i);
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

TEST(MmqQ8Imma, MoeGroupedQ5K) {
    // BM=32 (max rows < 96) and BM=128 variants; N=192 leaves a partial 128-column tile.
    const double small_m = q5k_moe_nrmse({70, 41}, 192, 512, 501);
    const double large_m = q5k_moe_nrmse({70, 41, 130}, 192, 512, 502);
    EXPECT_LT(small_m, 2e-2) << "Q5_K MoE NRMSE, BM=32";
    EXPECT_LT(large_m, 2e-2) << "Q5_K MoE NRMSE, BM=128";
}

}  // namespace
}  // namespace imp
