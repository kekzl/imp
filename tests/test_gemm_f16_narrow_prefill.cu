// gemm_f16_narrow_prefill: one-launch alpha/beta projection of GDN prefill rows. Checks vs a
// double CPU reference at the Qwen3.8-27B shape (K=5120, N=48/pair) for M=33..4096 (every
// split the launcher picks, partial row tiles), the single-pair form, bitwise determinism
// (fixed-order cluster split-K reduce), refusals, and the wall time against the two cuBLASLt GEMMs.

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

constexpr int kSplitMax = 8;

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

// Wall time vs the two cuBLASLt GEMMs in both accumulate modes. FP16 accumulate is what Qwen3.8 runs in
// situ (gemm.cublas_fp16_acc); FP32 is the old comparison. Median of 9 alternating 50-launch windows.
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
    const bool fp16_acc_was = process_diag_cublas_fp16_acc();
    for (const auto& sh : shapes) {
        Case c(sh.M, sh.N, sh.N, sh.K, 11u);
        const int64_t a_shape[2] = {sh.M, sh.K}, w_shape[2] = {sh.N, sh.K}, c_shape[2] = {sh.M, sh.N};
        Tensor tA(c.dA.p, QType::F16, 2, a_shape, true), tW0(c.dW0.p, QType::F16, 2, w_shape, true),
            tW1(c.dW1.p, QType::F16, 2, w_shape, true), tC0(c.dC0.p, QType::F16, 2, c_shape, true),
            tC1(c.dC1.p, QType::F16, 2, c_shape, true);
        auto cublas_pair = [&](bool acc16) {
            process_diag_set_cublas_fp16_acc(acc16);
            gemm(tA, tW0, tC0, 1.0f, 0.0f, s);
            gemm(tA, tW1, tC1, 1.0f, 0.0f, s);
        };
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        auto window_us = [&](auto&& body, int reps) {
            cudaEventRecord(e0, s);
            for (int i = 0; i < reps; ++i)
                body();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            return ms * 1000.0f / reps;
        };
        // Warm past the idle downclock (>1 s of work) with every arm in the mix.
        float warm_ms = 0.0f;
        cudaEventRecord(e0, s);
        while (warm_ms < 1200.0f) {
            for (int i = 0; i < 500; ++i) {
                ASSERT_TRUE(c.run(s));
                if ((i % 10) == 0) {
                    cublas_pair(true);
                    cublas_pair(false);
                }
            }
            cudaEventRecord(e1, s);
            ASSERT_EQ(cudaEventSynchronize(e1), cudaSuccess);
            cudaEventElapsedTime(&warm_ms, e0, e1);
        }
        constexpr int kWindows = 9;
        constexpr int kReps = 50;
        auto median = [](std::vector<float> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        std::vector<float> tn, t16, t32;
        for (int w = 0; w < kWindows; ++w) {
            tn.push_back(window_us([&]() { (void)c.run(s); }, kReps));
            t16.push_back(window_us([&]() { cublas_pair(true); }, kReps));
            t32.push_back(window_us([&]() { cublas_pair(false); }, kReps));
        }
        printf(
            "  %s (M=%d N=%d x 2 K=%d): narrow %.2f us, cuBLAS x2 fp16-acc %.2f us, fp32-acc %.2f us "
            "(median of %d, L2-resident)\n",
            sh.name, sh.M, sh.N, sh.K, median(tn), median(t16), median(t32), kWindows);
        struct Tune {
            int split, target, bm;
        };
        const Tune tunes[] = {{8, 256, 64}, {8, 256, 32}, {4, 256, 32},
                              {8, 512, 32}, {8, 512, 64}, {4, 128, 64}};
        for (const auto& t : tunes) {
            gemm_f16_narrow_prefill_tune(t.split, t.target, t.bm);
            std::vector<float> tt;
            for (int w = 0; w < kWindows; ++w)
                tt.push_back(window_us([&]() { (void)c.run(s); }, kReps));
            printf("    split<=%d target %3d bm %d: %.2f us\n", t.split, t.target, t.bm, median(tt));
        }
        gemm_f16_narrow_prefill_tune(0, 0, 0);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);

        // Numerics against the path it replaces in situ (cuBLAS FP16 accumulate) and FP32 cuBLAS.
        ASSERT_TRUE(c.run(s));
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        const auto got0 = c.out(0), got1 = c.out(1);
        std::vector<double> r0;
        ref_gemm(c.A, c.W0, r0, sh.M, sh.N, sh.K);
        auto err = [&](const std::vector<__half>& v, double& mx, double& rms) {
            mx = 0.0;
            double ss = 0.0;
            for (size_t i = 0; i < r0.size(); ++i) {
                const double d = std::fabs(__half2float(v[i]) - r0[i]);
                mx = std::max(mx, d);
                ss += d * d;
            }
            rms = std::sqrt(ss / r0.size());
        };
        double mn, rn;
        err(got0, mn, rn);
        for (const bool acc16 : {true, false}) {
            cublas_pair(acc16);
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const auto ref0 = c.out(0), ref1 = c.out(1);
            size_t n_diff = 0;
            for (size_t i = 0; i < ref0.size(); ++i)
                for (int p = 0; p < 2; ++p) {
                    const float r = __half2float(p ? ref1[i] : ref0[i]);
                    const float g = __half2float(p ? got1[i] : got0[i]);
                    n_diff += (r != g) ? 1 : 0;
                    if (!acc16)  // FP16 accumulate drifts 1e-2..5e-2 from fp64 itself: reported, not gated
                        EXPECT_NEAR(g, r, 1e-2 + 4e-3 * std::fabs(r))
                            << sh.name << " pair " << p << " at " << i;
                }
            double mc, rc;
            err(ref0, mc, rc);
            printf(
                "  %s vs fp64 reference (alpha): narrow max %.3e rms %.3e, cuBLAS %s max %.3e rms %.3e, "
                "%zu of %zu outputs differ\n",
                sh.name, mn, rn, acc16 ? "fp16-acc" : "fp32-acc", mc, rc, n_diff, 2 * ref0.size());
        }
    }
    process_diag_set_cublas_fp16_acc(fp16_acc_was);
    cudaStreamDestroy(s);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
