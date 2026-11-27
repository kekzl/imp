// gemm_f16_narrow_prefill: one-launch alpha/beta projection of GDN prefill rows. Checks vs a
// double CPU reference at the Qwen3.8-27B shape (K=5120, N=48/pair) for M=33..4096 (every
// split the launcher picks, partial row tiles), the single-pair form, bitwise determinism
// (fixed-order cluster split-K reduce), refusals. Wall time:
// tests/bench/test_gemm_f16_narrow_prefill_bench.cu.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gemm.h"
#include "compute/gemm_f16_narrow_prefill.h"
#include "core/process_diag.h"
#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;
#include "test_gemm_f16_narrow_prefill_util.cuh"

namespace {

void expect_close(const std::vector<__half>& got, const std::vector<double>& ref, const char* tag) {
    ASSERT_EQ(got.size(), ref.size()) << tag;
    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = __half2float(got[i]);
        const double tol = 4e-3 + 4e-3 * std::fabs(ref[i]);  // FP16 output rounding + FP32 accumulate
        max_err = std::max(max_err, std::fabs(g - ref[i]));
        ASSERT_NEAR(g, ref[i], tol) << tag << " at " << i;
    }
    printf("  %s: max |err| %.3e over %zu outputs\n", tag, max_err, got.size());
}

void check_case(Case& c, const char* what, size_t ws_limit = SIZE_MAX) {
    ASSERT_TRUE(c.run(nullptr, ws_limit)) << what;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << what;
    std::vector<double> r0, r1;
    ref_gemm(c.A, c.W0, r0, c.M, c.N0, c.K);
    char tag[64];
    snprintf(tag, sizeof tag, "%s alpha", what);
    expect_close(c.out(0), r0, tag);
    if (c.N1 > 0) {
        ref_gemm(c.A, c.W1, r1, c.M, c.N1, c.K);
        snprintf(tag, sizeof tag, "%s beta", what);
        expect_close(c.out(1), r1, tag);
    }
}

}  // namespace

TEST(GemmF16NarrowPrefill, MatchesReferenceAtQwen38Shape) {
    const int K = 5120, N = 48;
    for (int M : {33, 64, 100, 512, 1000, 4096}) {
        Case c(M, N, N, K, 1234u + static_cast<uint32_t>(M));
        char what[32];
        snprintf(what, sizeof what, "M=%d", M);
        check_case(c, what);
    }
}

TEST(GemmF16NarrowPrefill, SplitOneWithoutWorkspace) {
    Case c(512, 48, 48, 5120, 7u);
    check_case(c, "M=512 ws=0", 0);
}

TEST(GemmF16NarrowPrefill, SinglePairWiderNAndQwen36Shape) {
    Case one(300, 128, 0, 2048, 3u);
    check_case(one, "single pair N=128");
    Case q36(512, 32, 32, 2048, 5u);
    check_case(q36, "Qwen3.6 M=512");
}

TEST(GemmF16NarrowPrefill, DeterministicAcrossLaunches) {
    for (int M : {512, 4096}) {
        Case c(M, 48, 48, 5120, 99u);
        ASSERT_TRUE(c.run());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto a0 = c.out(0), b0 = c.out(1);
        for (int i = 0; i < 20; ++i)
            ASSERT_TRUE(c.run());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto a1 = c.out(0), b1 = c.out(1);
        ASSERT_EQ(memcmp(a0.data(), a1.data(), a0.size() * 2), 0) << "M=" << M;
        ASSERT_EQ(memcmp(b0.data(), b1.data(), b0.size() * 2), 0) << "M=" << M;
    }
}

// runtime.deterministic launches split 1 (no workspace): a row's bits must not depend on M, the
// prefill chunk length (#2556). The split-K default must differ, else this test checks nothing.
TEST(GemmF16NarrowPrefill, SplitOneRowsInvariantToM) {
    const int K = 5120, N = 48, big_m = 147, small_m = 33;
    Case big(big_m, N, N, K, 11u);
    Case small(small_m, N, N, K, 12u);
    const size_t row_off = static_cast<size_t>(big_m - small_m) * K * 2;
    ASSERT_EQ(cudaMemcpy(small.dA.p, static_cast<char*>(big.dA.p) + row_off,
                         static_cast<size_t>(small_m) * K * 2, cudaMemcpyDeviceToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(small.dW0.p, big.dW0.p, static_cast<size_t>(N) * K * 2, cudaMemcpyDeviceToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(small.dW1.p, big.dW1.p, static_cast<size_t>(N) * K * 2, cudaMemcpyDeviceToDevice),
              cudaSuccess);
    ASSERT_TRUE(big.run(nullptr, 0));
    ASSERT_TRUE(small.run(nullptr, 0));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const size_t tail = static_cast<size_t>(big_m - small_m) * N;
    for (int pair = 0; pair < 2; ++pair) {
        const auto b = big.out(pair), s = small.out(pair);
        ASSERT_EQ(memcmp(b.data() + tail, s.data(), s.size() * 2), 0) << "pair " << pair;
    }
    const auto split1 = big.out(0);
    ASSERT_TRUE(big.run());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_NE(memcmp(split1.data(), big.out(0).data(), split1.size() * 2), 0) << "split-K matched split 1";
}

TEST(GemmF16NarrowPrefill, RefusesUnsupportedShapes) {
    Case c(64, 48, 48, 5120, 1u);
    const half* A = static_cast<const half*>(c.dA.p);
    const half* W0 = static_cast<const half*>(c.dW0.p);
    half* C0 = static_cast<half*>(c.dC0.p);
    EXPECT_FALSE(gemm_f16_narrow_prefill(A, 0, 5120, W0, C0, 48, nullptr, nullptr, 0, c.ws.p, c.ws_bytes, 0));
    EXPECT_FALSE(
        gemm_f16_narrow_prefill(A, 64, 5128, W0, C0, 48, nullptr, nullptr, 0, c.ws.p, c.ws_bytes, 0));
    EXPECT_FALSE(
        gemm_f16_narrow_prefill(A, 64, 5120, W0, C0, 44, nullptr, nullptr, 0, c.ws.p, c.ws_bytes, 0));
    EXPECT_FALSE(gemm_f16_narrow_prefill(A, 64, 5120, W0, C0, 48, W0, C0, 96, c.ws.p, c.ws_bytes, 0));
    EXPECT_FALSE(gemm_f16_narrow_prefill(A, 64, 5120, W0, C0, 48, nullptr, C0, 48, c.ws.p, c.ws_bytes, 0));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
