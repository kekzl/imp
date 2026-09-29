// #2267: Q8_0 IMMA tall tiles (BM=160 / 192, 10 / 12 warps) against BM=128, bit-exact.
// Same warp tile, same k order and scale expression per output: any differing half is a defect.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);

namespace {

void gen_q8_blocks(std::vector<uint8_t>& w, int N, int K, unsigned seed) {
    w.resize(static_cast<size_t>(N) * (K / 32) * 34);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> q(-128, 127);
    std::uniform_real_distribution<float> d(0.001f, 0.05f);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        const __half h = __float2half(d(rng));
        std::memcpy(&w[b * 34], &h, 2);
        for (int i = 0; i < 32; ++i)
            w[b * 34 + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q(rng)));
    }
}

// N=6208 = 48.5 tiles: N-tail. 49 CTAs x 4 > 170 SMs keeps M=33 off the BM=32 small-grid rule;
// the last_plane_bm assert fails if a larger GPU takes that rule instead.
TEST(MmqQ8Imma, TallTilesBitIdenticalToBm128) {
    const int N = 6208, K = 1024, max_m = 2048;
    std::vector<uint8_t> W;
    gen_q8_blocks(W, N, K, 2267);
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(max_m) * K * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(max_m) * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice), cudaSuccess);
    std::mt19937 rng(2268);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int M : {33, 512, 2048}) {
        std::vector<__half> x(static_cast<size_t>(M) * K), base(static_cast<size_t>(M) * N);
        for (auto& v : x)
            v = __float2half(nd(rng));
        for (auto& v : base)
            v = __float2half(nd(rng));
        ASSERT_EQ(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
        for (float beta : {0.0f, 1.0f}) {
            std::vector<std::vector<__half>> outs;
            for (int bm : {128, 160, 192}) {
                cudaMemcpy(d_out, base.data(), base.size() * sizeof(__half), cudaMemcpyHostToDevice);
                ASSERT_TRUE(
                    mmq_q8_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr, beta, /*allow_splitk=*/false, bm));
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                ASSERT_EQ(mmq_q8_imma_last_plane_bm(), bm) << "M=" << M << " beta=" << beta;
                outs.emplace_back(base.size());
                cudaMemcpy(outs.back().data(), d_out, base.size() * sizeof(__half), cudaMemcpyDeviceToHost);
            }
            size_t moved = 0;  // the GEMM must have written something, or equality proves nothing
            for (size_t i = 0; i < base.size(); ++i)
                moved += std::memcmp(&outs[0][i], &base[i], sizeof(__half)) != 0;
            EXPECT_GT(moved, base.size() / 2) << "M=" << M << " beta=" << beta;
            for (size_t v = 1; v < outs.size(); ++v) {
                size_t diff = 0;
                for (size_t i = 0; i < base.size(); ++i)
                    diff += std::memcmp(&outs[0][i], &outs[v][i], sizeof(__half)) != 0;
                EXPECT_EQ(diff, 0u) << "M=" << M << " beta=" << beta << " bm=" << (v == 1 ? 160 : 192);
            }
        }
    }
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    mmq_q8_imma_release_all();
}

}  // namespace
}  // namespace imp
