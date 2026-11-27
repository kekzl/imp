// gemm_f16_narrow_smallm: one-launch alpha/beta projection of batched GDN decode. Checks
// vs a double CPU reference at the Qwen3.8-27B shape (K=5120, N=48/pair) for M=1,5,16,17,32,
// the single-pair form, bitwise determinism (fixed-order split-K reduce), and that
// unsupported shapes are refused rather than computed wrong.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gemm.h"
#include "compute/gemm_f16_narrow_smallm.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace imp;
#include "test_gemm_f16_narrow_smallm_util.cuh"

namespace {

// Reference over the FP16-rounded inputs, double accumulate.
void ref_gemm(const std::vector<__half>& A, const std::vector<__half>& W, std::vector<double>& C, int M,
              int N, int K) {
    C.assign(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += static_cast<double>(__half2float(A[static_cast<size_t>(m) * K + k])) *
                     static_cast<double>(__half2float(W[static_cast<size_t>(n) * K + k]));
            C[static_cast<size_t>(m) * N + n] = s;
        }
}

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

}  // namespace

TEST(GemmF16NarrowSmallM, MatchesReferenceAtQwen38Shape) {
    const int K = 5120, N = 48;
    for (int M : {1, 5, 16, 17, 32}) {
        Case c(M, N, N, K, 1234u + static_cast<uint32_t>(M));
        ASSERT_TRUE(c.run()) << "M=" << M;
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<double> r0, r1;
        ref_gemm(c.A, c.W0, r0, M, N, K);
        ref_gemm(c.A, c.W1, r1, M, N, K);
        char tag0[32], tag1[32];
        snprintf(tag0, sizeof tag0, "M=%d alpha", M);
        snprintf(tag1, sizeof tag1, "M=%d beta", M);
        expect_close(c.out(0), r0, tag0);
        expect_close(c.out(1), r1, tag1);
    }
}

TEST(GemmF16NarrowSmallM, SinglePairAndWiderN) {
    const int K = 4096;
    for (int N : {16, 64, 128}) {
        Case c(32, N, 0, K, 77u + static_cast<uint32_t>(N));
        ASSERT_TRUE(c.run());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<double> r0;
        ref_gemm(c.A, c.W0, r0, 32, N, K);
        char tag[32];
        snprintf(tag, sizeof tag, "N=%d single", N);
        expect_close(c.out(0), r0, tag);
    }
}

TEST(GemmF16NarrowSmallM, DeterministicAcrossLaunches) {
    Case c(32, 48, 48, 5120, 99u);
    ASSERT_TRUE(c.run());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto a0 = c.out(0), b0 = c.out(1);
    for (int i = 0; i < 50; ++i)
        ASSERT_TRUE(c.run());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto a1 = c.out(0), b1 = c.out(1);
    for (size_t i = 0; i < a0.size(); ++i) {
        ASSERT_EQ(__half_as_ushort(a0[i]), __half_as_ushort(a1[i])) << "alpha " << i;
        ASSERT_EQ(__half_as_ushort(b0[i]), __half_as_ushort(b1[i])) << "beta " << i;
    }
}

TEST(GemmF16NarrowSmallM, RefusesUnsupportedShapes) {
    Case c(32, 48, 48, 5120, 5u);
    const half* A = static_cast<const half*>(c.dA.p);
    const half* W = static_cast<const half*>(c.dW0.p);
    half* C = static_cast<half*>(c.dC0.p);
    EXPECT_FALSE(
        gemm_f16_narrow_smallm(A, 33, 5120, W, C, 48, nullptr, nullptr, 0, c.ws.p, c.ws_bytes, nullptr));
    EXPECT_FALSE(
        gemm_f16_narrow_smallm(A, 32, 5000, W, C, 48, nullptr, nullptr, 0, c.ws.p, c.ws_bytes, nullptr));
    EXPECT_FALSE(
        gemm_f16_narrow_smallm(A, 32, 5120, W, C, 40, nullptr, nullptr, 0, c.ws.p, c.ws_bytes, nullptr));
    EXPECT_FALSE(gemm_f16_narrow_smallm(A, 32, 5120, W, C, 48, nullptr, nullptr, 0, c.ws.p, 64, nullptr));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
