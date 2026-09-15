// gemm_f16_narrow_prefill: one-launch alpha/beta projection of GDN prefill rows. Checks vs a
// double CPU reference at the Qwen3.8-27B shape (K=5120, N=48/pair) for M=33..4096 (every
// split the launcher picks, partial row tiles), the single-pair form, bitwise determinism
// (fixed-order split-K reduce), refusals, and the wall time against the two cuBLAS GEMMs.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gemm.h"
#include "compute/gemm_f16_narrow_prefill.h"
#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    std::vector<float> a(A.size()), w(W.size());
    for (size_t i = 0; i < A.size(); ++i)
        a[i] = __half2float(A[i]);
    for (size_t i = 0; i < W.size(); ++i)
        w[i] = __half2float(W[i]);
    C.assign(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            const float* ar = &a[static_cast<size_t>(m) * K];
            const float* wr = &w[static_cast<size_t>(n) * K];
            for (int k = 0; k < K; ++k)
                s += static_cast<double>(ar[k]) * static_cast<double>(wr[k]);
            C[static_cast<size_t>(m) * N + n] = s;
        }
}

constexpr int kSplitMax = 32;

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
          ws(gemm_f16_narrow_prefill_workspace_bytes(m, n0 + n1, kSplitMax)),
          ws_bytes(gemm_f16_narrow_prefill_workspace_bytes(m, n0 + n1, kSplitMax)) {
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

    bool run(cudaStream_t s = nullptr, size_t ws_limit = SIZE_MAX) {
        const size_t b = std::min(ws_bytes, ws_limit);
        return gemm_f16_narrow_prefill(static_cast<const half*>(dA.p), M, K, static_cast<const half*>(dW0.p),
                                       static_cast<half*>(dC0.p), N0,
                                       N1 > 0 ? static_cast<const half*>(dW1.p) : nullptr,
                                       N1 > 0 ? static_cast<half*>(dC1.p) : nullptr, N1, b ? ws.p : nullptr,
                                       b, s);
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

TEST(GemmF16NarrowPrefill, BenchVsCublas) {
    struct Shape {
        const char* name;
        int M, N, K;
    };
    const Shape shapes[] = {{"Qwen3.8 pp512", 512, 48, 5120},
                            {"Qwen3.8 pp4096", 4096, 48, 5120},
                            {"Qwen3.6 pp512", 512, 32, 2048}};
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    for (const auto& sh : shapes) {
        Case c(sh.M, sh.N, sh.N, sh.K, 11u);
        const int64_t a_shape[2] = {sh.M, sh.K}, w_shape[2] = {sh.N, sh.K}, c_shape[2] = {sh.M, sh.N};
        Tensor tA(c.dA.p, QType::F16, 2, a_shape, true), tW0(c.dW0.p, QType::F16, 2, w_shape, true),
            tW1(c.dW1.p, QType::F16, 2, w_shape, true), tC0(c.dC0.p, QType::F16, 2, c_shape, true),
            tC1(c.dC1.p, QType::F16, 2, c_shape, true);
        auto cublas_pair = [&]() {
            gemm(tA, tW0, tC0, 1.0f, 0.0f, s);
            gemm(tA, tW1, tC1, 1.0f, 0.0f, s);
        };
        auto time_us = [&](auto&& body, int reps) {
            for (int i = 0; i < 50; ++i)
                body();
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, s);
            for (int i = 0; i < reps; ++i)
                body();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return ms * 1000.0f / reps;
        };
        // Warm past the idle downclock (>1 s of work), then time each path twice, alternating.
        cudaEvent_t w0, w1;
        cudaEventCreate(&w0);
        cudaEventCreate(&w1);
        cudaEventRecord(w0, s);
        float warm_ms = 0.0f;
        while (warm_ms < 1200.0f) {
            for (int i = 0; i < 500; ++i) {
                ASSERT_TRUE(c.run(s));
                if ((i % 10) == 0)
                    cublas_pair();
            }
            cudaEventRecord(w1, s);
            ASSERT_EQ(cudaEventSynchronize(w1), cudaSuccess);
            cudaEventElapsedTime(&warm_ms, w0, w1);
        }
        cudaEventDestroy(w0);
        cudaEventDestroy(w1);
        const int reps = 200;
        const float n0 = time_us([&]() { c.run(s); }, reps);
        const float b0 = time_us(cublas_pair, reps);
        const float n1 = time_us([&]() { c.run(s); }, reps);
        const float b1 = time_us(cublas_pair, reps);
        printf("  %s (M=%d N=%d x 2 K=%d): narrow %.2f / %.2f us, cuBLAS x2 %.2f / %.2f us (L2-resident)\n",
               sh.name, sh.M, sh.N, sh.K, n0, n1, b0, b1);
        struct Tune {
            int split, target;
        };
        const Tune tunes[] = {{32, 256}, {16, 256}, {16, 128}, {8, 128}, {8, 64}, {4, 64}};
        for (const auto& t : tunes) {
            gemm_f16_narrow_prefill_tune(t.split, t.target);
            const float ta = time_us([&]() { c.run(s); }, reps);
            const float tb = time_us([&]() { c.run(s); }, reps);
            printf("    split<=%2d target %3d: %.2f / %.2f us\n", t.split, t.target, ta, tb);
        }
        gemm_f16_narrow_prefill_tune(0, 0);

        // Numerics against the path it replaces (FP32 accumulate, one FP16 rounding, other order).
        cublas_pair();
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        const auto ref0 = c.out(0), ref1 = c.out(1);
        ASSERT_TRUE(c.run(s));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        const auto got0 = c.out(0), got1 = c.out(1);
        size_t n_diff = 0;
        double max_abs = 0.0;
        for (size_t i = 0; i < ref0.size(); ++i) {
            for (int p = 0; p < 2; ++p) {
                const float r = __half2float(p ? ref1[i] : ref0[i]);
                const float g = __half2float(p ? got1[i] : got0[i]);
                if (r != g) {
                    ++n_diff;
                    max_abs = std::max(max_abs, static_cast<double>(std::fabs(r - g)));
                }
                EXPECT_NEAR(g, r, 1e-2 + 4e-3 * std::fabs(r)) << sh.name << " pair " << p << " at " << i;
            }
        }
        printf("  %s vs cuBLAS: %zu of %zu outputs differ, max |diff| %.3e\n", sh.name, n_diff,
               2 * ref0.size(), max_abs);
        // Which of the two is closer to the double reference (cuBLAS picks hhh, FP16 accumulate).
        std::vector<double> r0;
        ref_gemm(c.A, c.W0, r0, sh.M, sh.N, sh.K);
        double e_narrow = 0.0, e_cublas = 0.0, s_narrow = 0.0, s_cublas = 0.0;
        for (size_t i = 0; i < r0.size(); ++i) {
            const double dn = std::fabs(__half2float(got0[i]) - r0[i]);
            const double dc = std::fabs(__half2float(ref0[i]) - r0[i]);
            e_narrow = std::max(e_narrow, dn);
            e_cublas = std::max(e_cublas, dc);
            s_narrow += dn * dn;
            s_cublas += dc * dc;
        }
        printf("  %s vs fp64 reference (alpha): narrow max %.3e rms %.3e, cuBLAS max %.3e rms %.3e\n",
               sh.name, e_narrow, std::sqrt(s_narrow / r0.size()), e_cublas, std::sqrt(s_cublas / r0.size()));
    }
    cudaStreamDestroy(s);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
