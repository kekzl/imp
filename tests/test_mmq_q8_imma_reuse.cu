// #2469: MoE grouped IMMA reuses the gate quantize for up (reuse_act).
// Own file: test_mmq_q8_imma.cu sits at the function-size gate.
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

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the IMMA activation scratch
namespace {

constexpr int kBlk = 32;
constexpr int kBlockBytes = 34;  // half d + 32 s8

void gen_q8_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    ASSERT_EQ(K % kBlk, 0);
    const int bpr = K / kBlk;
    W.resize(static_cast<size_t>(N) * bpr * kBlockBytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    for (size_t b = 0; b < W.size() / kBlockBytes; ++b) {
        uint8_t* bp = W.data() + b * kBlockBytes;
        __half d = __float2half(dd(rng));
        std::memcpy(bp, &d, 2);
        for (int i = 0; i < 32; ++i)
            bp[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(qd(rng)));
    }
}

TEST(MmqQ8Imma, MoeGroupedReuseActIsBitIdentical) {
    const int ne = 4, N = 128, K = 256;
    const int32_t h_off[ne + 1] = {0, 37, 37, 133, 152};
    const int expanded = h_off[ne], max_rows = 96;
    std::vector<uint8_t> Wg, Wu;
    gen_q8_weight(Wg, ne * N, K, 93);
    gen_q8_weight(Wu, ne * N, K, 94);
    std::vector<__half> x(static_cast<size_t>(expanded) * K), y(x.size());
    std::mt19937 rng(95);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x)
        v = __float2half(nd(rng));
    for (auto& v : y)
        v = __float2half(nd(rng));

    uint8_t *d_wg, *d_wu;
    __half *d_x, *d_y, *d_out;
    int32_t* d_off;
    const size_t out_bytes = static_cast<size_t>(expanded) * N * 2;
    ASSERT_EQ(cudaMalloc(&d_wg, Wg.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_wu, Wu.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, y.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, out_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, sizeof(h_off)), cudaSuccess);
    cudaMemcpy(d_wg, Wg.data(), Wg.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wu, Wu.data(), Wu.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_off, sizeof(h_off), cudaMemcpyHostToDevice);

    auto run = [&](const uint8_t* w, const __half* in, bool reuse) {
        EXPECT_TRUE(
            mmq_imma_moe_gemm(w, 0, in, d_out, d_off, max_rows, expanded, ne, N, K, nullptr, 0, reuse));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint16_t> h(out_bytes / 2);
        cudaMemcpy(h.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
        return h;
    };

    const auto up_fresh = run(d_wu, d_x, false);
    run(d_wg, d_x, false);
    EXPECT_EQ(run(d_wu, d_x, true), up_fresh) << "reused quantize differs from a fresh one";

    run(d_wg, d_x, false);
    run(d_wg, d_y, false);  // last quantize is y
    EXPECT_EQ(run(d_wu, d_x, true), up_fresh) << "reuse served another input's quantize";

    run(d_wg, d_x, false);
    cudaMemcpy(d_x, y.data(), y.size() * 2, cudaMemcpyHostToDevice);  // same pointer, new content
    EXPECT_EQ(run(d_wu, d_x, true), up_fresh) << "reuse_act did not skip the quantize";
    EXPECT_NE(run(d_wu, d_x, false), up_fresh);

    cudaFree(d_wg);
    cudaFree(d_wu);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    cudaFree(d_off);
    mmq_q8_imma_release_all();
}

}  // namespace
}  // namespace imp
