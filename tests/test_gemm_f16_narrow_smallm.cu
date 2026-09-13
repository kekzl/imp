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

namespace {

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        h[i] = __float2half(v[i]);
    return h;
}

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

struct Case {
    int M, N0, N1, K;
    std::vector<__half> A, W0, W1;
    DevBuf dA, dW0, dW1, dC0, dC1, ws;
    size_t ws_bytes;

    Case(int m, int n0, int n1, int k, uint32_t seed)
        : M(m),
          N0(n0),
          N1(n1),
          K(k),
          dA(static_cast<size_t>(m) * k * 2),
          dW0(static_cast<size_t>(n0) * k * 2),
          dW1(static_cast<size_t>(n1 > 0 ? n1 : 1) * k * 2),
          dC0(static_cast<size_t>(m) * n0 * 2),
          dC1(static_cast<size_t>(m) * (n1 > 0 ? n1 : 1) * 2),
          ws(gemm_f16_narrow_smallm_workspace_bytes(n0 + n1)),
          ws_bytes(gemm_f16_narrow_smallm_workspace_bytes(n0 + n1)) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> na(0.0f, 1.0f), nw(0.0f, 0.05f);
        std::vector<float> a(static_cast<size_t>(m) * k), w0(static_cast<size_t>(n0) * k),
            w1(static_cast<size_t>(n1) * k);
        for (auto& x : a)
            x = na(rng);
        for (auto& x : w0)
            x = nw(rng);
        for (auto& x : w1)
            x = nw(rng);
        A = to_half(a);
        W0 = to_half(w0);
        W1 = to_half(w1);
        cudaMemcpy(dA.p, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dW0.p, W0.data(), W0.size() * 2, cudaMemcpyHostToDevice);
        if (n1 > 0)
            cudaMemcpy(dW1.p, W1.data(), W1.size() * 2, cudaMemcpyHostToDevice);
        cudaMemset(ws.p, 0, ws_bytes);
    }

    bool run(cudaStream_t s = nullptr) {
        return gemm_f16_narrow_smallm(static_cast<const half*>(dA.p), M, K, static_cast<const half*>(dW0.p),
                                      static_cast<half*>(dC0.p), N0,
                                      N1 > 0 ? static_cast<const half*>(dW1.p) : nullptr,
                                      N1 > 0 ? static_cast<half*>(dC1.p) : nullptr, N1, ws.p, ws_bytes, s);
    }

    std::vector<__half> out(int pair) const {
        const int n = pair == 0 ? N0 : N1;
        std::vector<__half> h(static_cast<size_t>(M) * n);
        cudaMemcpy(h.data(), pair == 0 ? dC0.p : dC1.p, h.size() * 2, cudaMemcpyDeviceToHost);
        return h;
    }
};

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

// Informational: per-launch wall time inside a replayed CUDA graph (the kernel's actual
// regime) vs the two cuBLAS gemm() calls it replaces. No threshold (timing anchors flake in
// the full suite); WSL2 host launch cost would dominate an eager loop.
TEST(GemmF16NarrowSmallM, BenchQwen38Shape) {
    Case c(32, 48, 48, 5120, 11u);
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    const int64_t a_shape[2] = {32, 5120}, w_shape[2] = {48, 5120}, c_shape[2] = {32, 48};
    Tensor tA(c.dA.p, QType::F16, 2, a_shape, true), tW0(c.dW0.p, QType::F16, 2, w_shape, true),
        tW1(c.dW1.p, QType::F16, 2, w_shape, true), tC0(c.dC0.p, QType::F16, 2, c_shape, true),
        tC1(c.dC1.p, QType::F16, 2, c_shape, true);
    auto cublas_pair = [&]() {
        gemm(tA, tW0, tC0, 1.0f, 0.0f, s);
        gemm(tA, tW1, tC1, 1.0f, 0.0f, s);
    };
    // Warm both paths past the idle downclock (>1 s of work).
    cudaEvent_t w0, w1;
    cudaEventCreate(&w0);
    cudaEventCreate(&w1);
    cudaEventRecord(w0, s);
    for (int i = 0; i < 20000; ++i) {
        ASSERT_TRUE(c.run(s));
        if ((i % 10) == 0)
            cublas_pair();
    }
    cudaEventRecord(w1, s);
    ASSERT_EQ(cudaEventSynchronize(w1), cudaSuccess);
    float warm_ms = 0.0f;
    cudaEventElapsedTime(&warm_ms, w0, w1);
    printf("  warmup: %.0f ms\n", warm_ms);

    constexpr int kPerGraph = 64;
    auto time_graph = [&](const char* tag, auto&& body) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
        for (int i = 0; i < kPerGraph; ++i)
            body();
        ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
        for (int i = 0; i < 20; ++i)
            ASSERT_EQ(cudaGraphLaunch(ge, s), cudaSuccess);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int reps = 50;
        cudaEventRecord(e0, s);
        for (int i = 0; i < reps; ++i)
            cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        ASSERT_EQ(cudaEventSynchronize(e1), cudaSuccess);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        printf("  %s: %.2f us per launch in graph (%d launches x %d replays, L2-resident)\n", tag,
               ms * 1000.0f / (reps * kPerGraph), kPerGraph, reps);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    };
    time_graph("narrow kernel (alpha+beta)", [&]() { c.run(s); });
    gemm_set_lt_capture_allowed(true);
    time_graph("cuBLAS gemm() x2 (alpha, beta)", cublas_pair);
    gemm_set_lt_capture_allowed(false);

    // Numerics against the path it replaces: both accumulate in FP32 and round
    // once to FP16, in different summation orders. Report how many outputs
    // differ and by how much (1 FP16 ulp is the expected class).
    cublas_pair();
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    const auto ref0 = c.out(0), ref1 = c.out(1);
    ASSERT_TRUE(c.run(s));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    const auto got0 = c.out(0), got1 = c.out(1);
    size_t n_diff = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < ref0.size(); ++i) {
        for (int p = 0; p < 2; ++p) {
            const float r = __half2float(p ? ref1[i] : ref0[i]);
            const float g = __half2float(p ? got1[i] : got0[i]);
            if (r != g) {
                ++n_diff;
                max_abs = std::max(max_abs, static_cast<double>(std::fabs(r - g)));
                max_rel = std::max(max_rel,
                                   static_cast<double>(std::fabs(r - g) / std::max(std::fabs(r), 1e-3f)));
            }
            EXPECT_NEAR(g, r, 1e-2 + 4e-3 * std::fabs(r)) << "pair " << p << " at " << i;
        }
    }
    printf("  vs cuBLAS: %zu of %zu outputs differ, max |diff| %.3e, max rel %.3e\n", n_diff, 2 * ref0.size(),
           max_abs, max_rel);
    cudaStreamDestroy(s);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
