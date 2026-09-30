// mmq_imma_q8_mbar_kernel (#2267e, 2 CTAs/SM) against the legacy mmq_imma_kernel, both on the
// same s8 planes, selected via mmq_q8_imma_set_legacy_kernel.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the IMMA activation / split-K scratch
namespace {

// Q8_0 blocks: half d in [0.001, 0.05], 32 s8 in [-127, 127].
void gen_q8_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    W.resize(static_cast<size_t>(N) * (K / 32) * 34);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    for (size_t b = 0; b < W.size() / 34; ++b) {
        const __half d = __float2half(dd(rng));
        std::memcpy(&W[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) W[b * 34 + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(qd(rng)));
    }
}

// Per output |new - old| <= 1e-3 * max(|old|, rms(old)); new output bit-identical across two runs.
// moe = one expert through mmq_imma_moe_gemm (the only entry that takes M = 1).
void run_new_vs_legacy(int M, int N, int K, float beta, bool moe, unsigned seed) {
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, seed);
    std::vector<__half> x(static_cast<size_t>(M) * K), out0(static_cast<size_t>(M) * N);
    std::mt19937 rng(seed * 13 + 5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half(nd(rng));
    for (auto& v : out0) v = __float2half(nd(rng));  // beta = 1 residual

    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_out = nullptr;
    int32_t* d_off = nullptr;
    const int32_t h_off[2] = {0, M};
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, out0.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, sizeof(h_off)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_off, h_off, sizeof(h_off), cudaMemcpyHostToDevice), cudaSuccess);

    auto run = [&](bool legacy, std::vector<__half>& out) {
        mmq_q8_imma_set_legacy_kernel(legacy);
        ASSERT_EQ(cudaMemcpy(d_out, out0.data(), out0.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
        const bool ok = moe ? mmq_imma_moe_gemm(d_w, /*qkind=*/0, d_x, d_out, d_off, M, M, 1, N, K, nullptr)
                            : mmq_q8_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr, beta);
        ASSERT_TRUE(ok);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        out.resize(out0.size());
        ASSERT_EQ(cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    };
    std::vector<__half> o_old, o_new, o_new2;
    run(true, o_old);
    run(false, o_new);
    run(false, o_new2);
    mmq_q8_imma_set_legacy_kernel(false);
    if (::testing::Test::HasFatalFailure()) return;

    double sq = 0.0;
    for (const __half& h : o_old) sq += static_cast<double>(__half2float(h)) * __half2float(h);
    const double rms = std::sqrt(sq / static_cast<double>(o_old.size()));
    ASSERT_GT(rms, 0.0);
    double max_rel = 0.0;
    size_t bad = 0, nondet = 0;
    for (size_t i = 0; i < o_old.size(); ++i) {
        const double a = __half2float(o_old[i]), b = __half2float(o_new[i]);
        const double rel = std::fabs(b - a) / std::max(std::fabs(a), rms);
        max_rel = std::max(max_rel, rel);
        bad += rel > 1e-3 ? 1 : 0;
        nondet += std::memcmp(&o_new[i], &o_new2[i], sizeof(__half)) != 0 ? 1 : 0;
    }
    printf("new_vs_legacy M=%d N=%d K=%d beta=%.0f moe=%d rms=%.4f max_rel=%.3e bad=%zu nondet=%zu\n", M, N,
           K, beta, moe ? 1 : 0, rms, max_rel, bad, nondet);
    EXPECT_EQ(bad, 0u) << "max_rel " << max_rel;
    EXPECT_EQ(nondet, 0u);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_off);
    mmq_q8_imma_release_all();
}

TEST(MmqQ8Imma, NewVsLegacyM1Moe) { run_new_vs_legacy(1, 2048, 4096, 0.0f, true, 2301); }
TEST(MmqQ8Imma, NewVsLegacyM16SplitK) { run_new_vs_legacy(16, 2048, 4096, 0.0f, false, 2302); }
TEST(MmqQ8Imma, NewVsLegacyM33SmallGrid) { run_new_vs_legacy(33, 2048, 4096, 0.0f, false, 2303); }
TEST(MmqQ8Imma, NewVsLegacyM512) { run_new_vs_legacy(512, 2048, 4096, 0.0f, false, 2304); }
TEST(MmqQ8Imma, NewVsLegacyM512Beta1) { run_new_vs_legacy(512, 2048, 4096, 1.0f, false, 2305); }
TEST(MmqQ8Imma, NewVsLegacyM2048) { run_new_vs_legacy(2048, 4096, 4096, 0.0f, false, 2306); }
TEST(MmqQ8Imma, NewVsLegacyM2048NTail) { run_new_vs_legacy(2048, 320, 4096, 0.0f, false, 2307); }

}  // namespace
}  // namespace imp
