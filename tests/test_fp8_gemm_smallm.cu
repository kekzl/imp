// gemm_fp8_smallm (batched-decode FP8 x FP8, M 1..32) against a double reference on the decoded E4M3
// bytes and against cuBLASLt (the route it replaces), beta 0 / 1, row invariance over M and batch
// position, refusals, plus a cold-L2 timing table on Qwen3-8B shapes (disabled by default).

#include "compute/gemm.h"
#include "core/tensor.h"
#include "quant/fp8_gemm_smallm.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace imp;

namespace {

constexpr float kAScale = 0.37f;
constexpr float kWScale = 0.051f;
constexpr int kMaxM = kFp8SmallMMaxM;

std::vector<uint8_t> random_e4m3(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h = h * 1664525u + 1013904223u;
        const float f = (static_cast<float>(h >> 8) / 8388608.0f - 1.0f) * 2.0f;
        v[i] = static_cast<uint8_t>(__nv_fp8_e4m3(f).__x);
    }
    return v;
}

std::vector<half> random_half(size_t n, uint32_t seed, float amp) {
    std::vector<half> v(n);
    uint32_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h = h * 1664525u + 1013904223u;
        v[i] = __float2half(amp * (static_cast<float>(h >> 8) / 8388608.0f - 1.0f));
    }
    return v;
}

float e4m3_to_float(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

// Operands for one (N, K): kMaxM activation rows, the weight, scales, outputs, double reference.
struct Shape {
    int N = 0, K = 0;
    std::vector<uint8_t> a, w;
    std::vector<double> ref;  // [kMaxM][N], scaled, beta 0
    uint8_t* d_a = nullptr;
    uint8_t* d_w = nullptr;
    float* d_as = nullptr;
    float* d_ws = nullptr;
    half* d_y = nullptr;
    half* d_y2 = nullptr;

    void init(int n, int k, bool want_ref) {
        N = n;
        K = k;
        a = random_e4m3(size_t(kMaxM) * K, 0x1234u + n + k);
        w = random_e4m3(size_t(N) * K, 0x9876u + n);
        ASSERT_EQ(cudaMalloc(&d_a, a.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_w, w.size()), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_as, sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_ws, sizeof(float)), cudaSuccess);
        const float as = kAScale, ws = kWScale;
        ASSERT_EQ(cudaMemcpy(d_as, &as, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_ws, &ws, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y2, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        if (!want_ref)
            return;
        float lut[256];
        for (int i = 0; i < 256; ++i)
            lut[i] = e4m3_to_float(static_cast<uint8_t>(i));
        std::vector<double> wf(size_t(N) * K), af(size_t(kMaxM) * K);
        for (size_t i = 0; i < wf.size(); ++i)
            wf[i] = lut[w[i]];
        for (size_t i = 0; i < af.size(); ++i)
            af[i] = lut[a[i]];
        ref.assign(size_t(kMaxM) * N, 0.0);
        const double sc = double(kAScale) * double(kWScale);
        for (int m = 0; m < kMaxM; ++m)
            for (int n2 = 0; n2 < N; ++n2) {
                const double* ar = &af[size_t(m) * K];
                const double* wr = &wf[size_t(n2) * K];
                double s = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    s += ar[kk] * wr[kk];
                ref[size_t(m) * N + n2] = s * sc;
            }
    }
    void free_all() {
        for (void* p : {static_cast<void*>(d_a), static_cast<void*>(d_w), static_cast<void*>(d_as),
                        static_cast<void*>(d_ws), static_cast<void*>(d_y), static_cast<void*>(d_y2)})
            cudaFree(p);
        *this = Shape{};
    }
};

std::vector<half> download(const half* d, size_t n) {
    std::vector<half> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

bool has_device() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Qwen3-8B q/o, k/v, gate/up, down, plus a 3-CTA shape with K not a multiple of a full round.
const int kShapes[][2] = {{4096, 4096}, {1024, 4096}, {12288, 4096}, {4096, 12288}, {48, 192}};
const int kMs[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32};

}  // namespace

// Every M against the double reference: only fp32 accumulation order and the fp16 output rounding differ.
TEST(Fp8GemmSmallMTest, MatchesDoubleReferenceEveryM) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    for (const auto& nk : kShapes) {
        Shape s;
        s.init(nk[0], nk[1], true);
        double max_ref = 0.0;
        for (double v : s.ref)
            max_ref = std::max(max_ref, std::fabs(v));
        for (int M : kMs) {
            cudaMemset(s.d_y, 0xff,
                       size_t(kMaxM) * s.N * sizeof(half));  // NaN poison: rows < M must be written
            ASSERT_TRUE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, M, s.N, s.K, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto y = download(s.d_y, size_t(kMaxM) * s.N);
            double max_diff = 0.0;
            size_t bad = 0;
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < s.N; ++n) {
                    const double r = s.ref[size_t(m) * s.N + n];
                    const double x = __half2float(y[size_t(m) * s.N + n]);
                    if (!std::isfinite(x))
                        ++bad;
                    max_diff = std::max(max_diff, std::fabs(r - x) - std::ldexp(std::fabs(r), -11));
                }
            // Rows >= M stay untouched (poison bytes).
            for (size_t i = size_t(M) * s.N; i < y.size(); ++i) {
                uint16_t b;
                std::memcpy(&b, &y[i], 2);
                if (b != 0xffff) {
                    ++bad;
                    break;
                }
            }
            EXPECT_EQ(bad, 0u) << "N=" << s.N << " K=" << s.K << " M=" << M;
            EXPECT_LE(max_diff, 1e-4 * max_ref + 1e-6) << "N=" << s.N << " K=" << s.K << " M=" << M;
        }
        // Informational: own vs cuBLASLt distance to the double reference at M 16.
        double err_own = 0.0, err_lt = 0.0;
        {
            constexpr int M = 16;
            const int64_t ash[2] = {M, s.K}, wsh[2] = {s.N, s.K}, csh[2] = {M, s.N};
            Tensor A(s.d_a, QType::FP8_E4M3, 2, ash, true);
            Tensor W(s.d_w, QType::FP8_E4M3, 2, wsh, true);
            Tensor C(s.d_y2, QType::F16, 2, csh, true);
            gemm_cublaslt(A, W, C, 1.0f, 0.0f, s.d_as, s.d_ws, nullptr);
            ASSERT_TRUE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, M, s.N, s.K, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto own = download(s.d_y, size_t(M) * s.N);
            const auto lt = download(s.d_y2, size_t(M) * s.N);
            for (size_t i = 0; i < own.size(); ++i) {
                err_own = std::max(err_own, std::fabs(double(__half2float(own[i])) - s.ref[i]));
                err_lt = std::max(err_lt, std::fabs(double(__half2float(lt[i])) - s.ref[i]));
            }
        }
        std::fprintf(stderr, "[fp8 smallm] N=%d K=%d max|ref|=%.4g M16 max|err| own %.3e cuBLASLt %.3e\n",
                     s.N, s.K, max_ref, err_own, err_lt);
        s.free_all();
    }
}

// beta = 1 adds the scaled product to the fp16 residual, beta = 0.5 scales it (Lt semantics).
TEST(Fp8GemmSmallMTest, BetaAccumulates) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    Shape s;
    s.init(48, 192, true);
    const auto resid = random_half(size_t(kMaxM) * s.N, 0x5555u, 3.0f);
    for (float beta : {1.0f, 0.5f})
        for (int M : {1, 9, 16, 32}) {
            ASSERT_EQ(cudaMemcpy(s.d_y, resid.data(), resid.size() * sizeof(half), cudaMemcpyHostToDevice),
                      cudaSuccess);
            ASSERT_TRUE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, beta, s.d_y, M, s.N, s.K, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto y = download(s.d_y, resid.size());
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < s.N; ++n) {
                    const size_t i = size_t(m) * s.N + n;
                    const double want = s.ref[i] + beta * double(__half2float(resid[i]));
                    EXPECT_NEAR(__half2float(y[i]), want, std::ldexp(std::fabs(want), -10) + 1e-4)
                        << "beta " << beta << " M=" << M << " m=" << m << " n=" << n;
                }
        }
    s.free_all();
}

// #2152: a row's bits do not depend on M, nor on its position in the batch.
TEST(Fp8GemmSmallMTest, RowBitsIndependentOfMAndPosition) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    for (const auto& nk : kShapes) {
        Shape s;
        s.init(nk[0], nk[1], false);
        const size_t out = size_t(kMaxM) * s.N;
        ASSERT_TRUE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y2, kMaxM, s.N, s.K, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto full = download(s.d_y2, out);
        for (int M : kMs) {
            ASSERT_TRUE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, M, s.N, s.K, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto y = download(s.d_y, size_t(M) * s.N);
            EXPECT_EQ(std::memcmp(y.data(), full.data(), y.size() * sizeof(half)), 0)
                << "N=" << s.N << " K=" << s.K << " M=" << M;
        }
        // Reversed batch: row r lands at position kMaxM-1-r (another lane, another n-tile).
        std::vector<uint8_t> rev(s.a.size());
        for (int m = 0; m < kMaxM; ++m)
            std::memcpy(&rev[size_t(kMaxM - 1 - m) * s.K], &s.a[size_t(m) * s.K], s.K);
        ASSERT_EQ(cudaMemcpy(s.d_a, rev.data(), rev.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_TRUE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, kMaxM, s.N, s.K, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto y = download(s.d_y, out);
        for (int m = 0; m < kMaxM; ++m)
            EXPECT_EQ(std::memcmp(&y[size_t(kMaxM - 1 - m) * s.N], &full[size_t(m) * s.N],
                                  s.N * sizeof(half)),
                      0)
                << "N=" << s.N << " row " << m;
        s.free_all();
    }
}

// Against the cuBLASLt route it replaces (row-pinned FP8 x FP8), M 2..16.
TEST(Fp8GemmSmallMTest, MatchesCublasLt) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    for (const auto& nk : kShapes) {
        Shape s;
        s.init(nk[0], nk[1], false);
        for (int M : {2, 5, 8, 16}) {
            const int64_t ash[2] = {M, s.K}, wsh[2] = {s.N, s.K}, csh[2] = {M, s.N};
            Tensor A(s.d_a, QType::FP8_E4M3, 2, ash, true);
            Tensor W(s.d_w, QType::FP8_E4M3, 2, wsh, true);
            Tensor C(s.d_y2, QType::F16, 2, csh, true);
            gemm_cublaslt(A, W, C, 1.0f, 0.0f, s.d_as, s.d_ws, nullptr);
            ASSERT_TRUE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, M, s.N, s.K, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto ref = download(s.d_y2, size_t(M) * s.N);
            const auto own = download(s.d_y, size_t(M) * s.N);
            double max_ref = 0.0, max_diff = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double r = __half2float(ref[i]);
                max_ref = std::max(max_ref, std::fabs(r));
                max_diff = std::max(max_diff, std::fabs(r - double(__half2float(own[i]))));
            }
            EXPECT_GT(max_ref, 0.0);
            EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << "N=" << s.N << " K=" << s.K << " M=" << M;
        }
        s.free_all();
    }
}

// Sibling launch (q|k|v, gate|up): every tensor bit-equal to its own single launch, M 1..32.
TEST(Fp8GemmSmallMTest, MultiBitEqualToSingle) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    // Qwen3-8B q, k, v at K 4096 (distinct weights and scales), and gate|up as a pair.
    Shape q, k, v;
    q.init(4096, 4096, false);
    k.init(1024, 4096, false);
    v.init(1024, 4096, false);
    const float ws2 = 0.023f;
    ASSERT_EQ(cudaMemcpy(k.d_ws, &ws2, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    Shape* sh[3] = {&q, &k, &v};
    for (int count : {2, 3})
        for (int M : {1, 8, 9, 16, 17, 32}) {
            Fp8SmallMSibling t[3];
            for (int i = 0; i < count; ++i) {
                t[i] = {sh[i]->d_w, sh[i]->d_ws, sh[i]->d_y, sh[i]->N};
                ASSERT_TRUE(gemm_fp8_smallm(q.d_a, sh[i]->d_w, q.d_as, sh[i]->d_ws, 0.0f, sh[i]->d_y2, M,
                                            sh[i]->N, sh[i]->K, nullptr));
                cudaMemset(sh[i]->d_y, 0xff, size_t(kMaxM) * sh[i]->N * sizeof(half));
            }
            ASSERT_TRUE(gemm_fp8_smallm_multi(t, count, q.d_a, q.d_as, M, 4096, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            for (int i = 0; i < count; ++i) {
                const auto a = download(sh[i]->d_y, size_t(M) * sh[i]->N);
                const auto b = download(sh[i]->d_y2, size_t(M) * sh[i]->N);
                EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(half)), 0)
                    << "count " << count << " tensor " << i << " M=" << M;
            }
        }
    Fp8SmallMSibling bad[2] = {{q.d_w, q.d_ws, q.d_y, 4096}, {k.d_w, k.d_ws, k.d_y, 1000}};
    EXPECT_FALSE(gemm_fp8_smallm_multi(bad, 2, q.d_a, q.d_as, 4, 4096, nullptr));
    EXPECT_FALSE(gemm_fp8_smallm_multi(bad, 1, q.d_a, q.d_as, 4, 4096, nullptr));
    for (Shape* s : sh)
        s->free_all();
}

// Illegal shapes and pointers return false and leave y byte-identical.
TEST(Fp8GemmSmallMTest, RefusesIllegalShapes) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    Shape s;
    s.init(48, 192, false);
    const size_t bytes = size_t(kMaxM) * s.N * sizeof(half);
    cudaMemset(s.d_y, 0xab, bytes);
    struct Bad {
        int M, N, K;
    };
    for (const Bad b : {Bad{0, 48, 192}, Bad{33, 48, 192}, Bad{4, 40, 192}, Bad{4, 48, 160}, Bad{4, 0, 192}})
        EXPECT_FALSE(gemm_fp8_smallm(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, b.M, b.N, b.K, nullptr))
            << b.M << " " << b.N << " " << b.K;
    EXPECT_FALSE(gemm_fp8_smallm(s.d_a, s.d_w, nullptr, s.d_ws, 0.0f, s.d_y, 4, 48, 192, nullptr));
    EXPECT_FALSE(gemm_fp8_smallm(s.d_a + 1, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, 4, 48, 192, nullptr));
    EXPECT_FALSE(gemm_fp8_smallm_tuned(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, 4, 48, 192, nullptr, 3, 4));
    EXPECT_FALSE(
        gemm_fp8_smallm_tuned(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, 4, 48, 192, nullptr, 16, 2));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> host(bytes);
    cudaMemcpy(host.data(), s.d_y, bytes, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](uint8_t v) { return v == 0xab; }));
    s.free_all();
}

// Cold-L2 timing, Qwen3-8B decode shapes: each call follows a 256 MiB memset flush, median of 31 calls
// after >= 0.3 s warm-up. --gtest_also_run_disabled_tests --gtest_filter=*Fp8GemmSmallMBench*.
TEST(Fp8GemmSmallMBench, DISABLED_VsCublasLt) {
    cudaStream_t st;
    cudaStreamCreate(&st);
    constexpr size_t kFlushBytes = size_t(256) << 20;
    void* flush = nullptr;
    ASSERT_EQ(cudaMalloc(&flush, kFlushBytes), cudaSuccess);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto median_us = [&](auto&& fn) {
        auto one = [&]() {
            cudaMemsetAsync(flush, 0x5a, kFlushBytes, st);
            cudaEventRecord(e0, st);
            fn();
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1000.0 * ms;
        };
        double t = 0.0;
        while (t < 3e5)
            t += one();
        std::vector<double> v;
        for (int i = 0; i < 31; ++i)
            v.push_back(one());
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const int cfg[][2] = {{8, 1}, {8, 2}, {8, 4}};
    std::fprintf(stderr, "%4s %6s %6s %8s %8s %8s %8s %8s %7s\n", "M", "N", "K", "lt", "gemv16", "w8u1",
                 "w8u2", "w8u4", "GB/s");
    for (const auto& nk : kShapes) {
        if (nk[0] < 1024)
            continue;
        Shape s;
        s.init(nk[0], nk[1], false);
        half* x16 = nullptr;
        cudaMalloc(&x16, size_t(s.K) * sizeof(half));
        cudaMemset(x16, 0, size_t(s.K) * sizeof(half));
        for (int M : {1, 2, 4, 8, 12, 16, 32}) {
            const int64_t ash[2] = {M, s.K}, wsh[2] = {s.N, s.K}, csh[2] = {M, s.N};
            Tensor A(s.d_a, QType::FP8_E4M3, 2, ash, true);
            Tensor W(s.d_w, QType::FP8_E4M3, 2, wsh, true);
            Tensor C(s.d_y2, QType::F16, 2, csh, true);
            const double t_lt = median_us([&] { gemm_cublaslt(A, W, C, 1.0f, 0.0f, s.d_as, s.d_ws, st); });
            double t_gemv = 0.0;
            if (M == 1) {
                const int64_t xsh[1] = {s.K}, ysh[1] = {s.N};
                Tensor X(x16, QType::F16, 1, xsh, true);
                Tensor Y(s.d_y2, QType::F16, 1, ysh, true);
                t_gemv = median_us([&] { gemv_fp8(W, X, Y, kWScale, st); });
            }
            double t_c[3];
            for (int c = 0; c < 3; ++c)
                t_c[c] = median_us([&] {
                    (void)gemm_fp8_smallm_tuned(s.d_a, s.d_w, s.d_as, s.d_ws, 0.0f, s.d_y, M, s.N, s.K, st,
                                                cfg[c][0], cfg[c][1]);
                });
            const double best = *std::min_element(t_c, t_c + 3);
            std::fprintf(stderr, "%4d %6d %6d %8.2f %8.2f %8.2f %8.2f %8.2f %7.0f\n", M, s.N, s.K, t_lt,
                         t_gemv, t_c[0], t_c[1], t_c[2], double(s.N) * s.K / (best * 1e3));
        }
        cudaFree(x16);
        s.free_all();
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(flush);
    cudaStreamDestroy(st);
}
