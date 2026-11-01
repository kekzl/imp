// gemm_fp8_tile (own FP8 E4M3 x E4M3 prefill GEMM, per-tensor scales) against cuBLASLt FP8 on the same
// operand bytes and scales: beta = 0 and beta = 1, all tile variants, refusal contract, plus a cold-L2
// timing table on Qwen3-8B prefill shapes (informational, disabled by default).

#include "compute/gemm.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace imp;

namespace {

constexpr float kAScale = 0.37f;
constexpr float kWScale = 0.051f;

// Finite E4M3 bytes from uniform floats in [-2, 2]; a NaN encoding (0x7F / 0xFF) cannot come out of the cast.
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

struct Operands {
    int M = 0, N = 0, K = 0;
    void* a = nullptr;
    void* w = nullptr;
    float* a_scale = nullptr;
    float* w_scale = nullptr;
    half* y_ref = nullptr;
    half* y_own = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
};

void make_operands(Operands& o, int M, int N, int K) {
    o.M = M;
    o.N = N;
    o.K = K;
    const auto a = random_e4m3(size_t(M) * K, 0x1234u);
    const auto w = random_e4m3(size_t(N) * K, 0x9876u);
    ASSERT_EQ(cudaMalloc(&o.a, a.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.w, w.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(o.a, a.data(), a.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(o.w, w.data(), w.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.a_scale, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.w_scale, sizeof(float)), cudaSuccess);
    const float as = kAScale, ws = kWScale;
    ASSERT_EQ(cudaMemcpy(o.a_scale, &as, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(o.w_scale, &ws, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_ref, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_own, size_t(M) * N * sizeof(half)), cudaSuccess);
    o.ws_bytes = gemm_tile_streamk_workspace_bytes();
    ASSERT_EQ(cudaMalloc(&o.ws, o.ws_bytes), cudaSuccess);
}

void free_operands(Operands& o) {
    for (void* p : {o.a, o.w, static_cast<void*>(o.a_scale), static_cast<void*>(o.w_scale),
                    static_cast<void*>(o.y_ref), static_cast<void*>(o.y_own), o.ws})
        cudaFree(p);
    o = Operands{};
}

// cuBLASLt reference: D = a_scale * w_scale * A W^T + beta * D.
void run_ref(const Operands& o, float beta, cudaStream_t s) {
    const int64_t ash[2] = {o.M, o.K}, wsh[2] = {o.N, o.K}, csh[2] = {o.M, o.N};
    Tensor A(o.a, QType::FP8_E4M3, 2, ash, true);
    Tensor W(o.w, QType::FP8_E4M3, 2, wsh, true);
    Tensor C(o.y_ref, QType::F16, 2, csh, true);
    gemm_cublaslt(A, W, C, 1.0f, beta, o.a_scale, o.w_scale, s);
}

bool run_own(const Operands& o, float beta, int variant, cudaStream_t s) {
    const bool sk = variant == 3 || variant == 4 || variant == 6 || variant < 0;
    return gemm_fp8_tile(o.a, o.w, o.a_scale, o.w_scale, beta, o.y_own, o.M, o.N, o.K, s, variant,
                         sk ? o.ws : nullptr, sk ? o.ws_bytes : 0);
}

// 0 128x128, 1 128x256, 2 sync, 3 stream-K, 4 hybrid / split, 5 128x64, 6 128x64 split, -1 (auto)
constexpr int kVariants = 8;
int variant_of(int i) { return i < 7 ? i : -1; }

class Fp8GemmTileTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
        cudaStreamCreate(&s_);
    }
    void TearDown() override {
        if (s_)
            cudaStreamDestroy(s_);
    }
    cudaStream_t s_ = nullptr;
};

// Compares own vs ref on the host; returns false on a non-finite own value or a tolerance miss.
void compare(const std::vector<half>& ref, const std::vector<half>& own, const char* tag, int v, int M, int N,
             int K) {
    double max_ref = 0.0, max_diff = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = __half2float(ref[i]);
        const double x = __half2float(own[i]);
        if (!std::isfinite(x))
            ++bad;
        max_ref = std::max(max_ref, std::fabs(r));
        max_diff = std::max(max_diff, std::fabs(r - x));
    }
    std::fprintf(stderr, "[fp8 %s v%d] M=%d N=%d K=%d max|ref|=%.4g max|diff|=%.4g\n", tag, v, M, N, K,
                 max_ref, max_diff);
    EXPECT_EQ(bad, 0u) << tag << " variant " << v;
    EXPECT_GT(max_ref, 0.0);
    // Both arms accumulate fp32 in different orders: fp16 rounding plus order, relative to the output peak.
    EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << tag << " variant " << v;
}

}  // namespace

// beta = 0, output pre-poisoned with 0xff (NaN) so an unwritten tile fails.
TEST_P(Fp8GemmTileTest, MatchesCublasLt) {
    const auto [M, N, K] = GetParam();
    Operands o;
    make_operands(o, M, N, K);
    run_ref(o, 0.0f, s_);
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<half> ref(size_t(M) * N), own(size_t(M) * N);
    ASSERT_EQ(cudaMemcpy(ref.data(), o.y_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int i = 0; i < kVariants; ++i) {
        const int v = variant_of(i);
        cudaMemsetAsync(o.y_own, 0xff, ref.size() * sizeof(half), s_);
        if (v == 2 && K % 128 != 0) {  // sync mainloop stages BK 128: refuses K % 128 by design
            EXPECT_FALSE(run_own(o, 0.0f, v, s_));
            continue;
        }
        ASSERT_TRUE(run_own(o, 0.0f, v, s_)) << "variant " << v << " refused a legal shape";
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        compare(ref, own, "beta0", v, M, N, K);
    }
    free_operands(o);
}

// beta = 1: both arms start from the same fp16 residual and add the scaled product to it.
TEST_P(Fp8GemmTileTest, BetaOneAccumulates) {
    const auto [M, N, K] = GetParam();
    Operands o;
    make_operands(o, M, N, K);
    const auto resid = random_half(size_t(M) * N, 0x5555u, 3.0f);
    ASSERT_EQ(cudaMemcpy(o.y_ref, resid.data(), resid.size() * sizeof(half), cudaMemcpyHostToDevice),
              cudaSuccess);
    run_ref(o, 1.0f, s_);
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    std::vector<half> ref(size_t(M) * N), own(size_t(M) * N);
    ASSERT_EQ(cudaMemcpy(ref.data(), o.y_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int i = 0; i < kVariants; ++i) {
        const int v = variant_of(i);
        ASSERT_EQ(cudaMemcpy(o.y_own, resid.data(), resid.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        if (v == 2 && K % 128 != 0) {
            EXPECT_FALSE(run_own(o, 1.0f, v, s_));
            continue;
        }
        ASSERT_TRUE(run_own(o, 1.0f, v, s_)) << "variant " << v << " refused a legal shape";
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        compare(ref, own, "beta1", v, M, N, K);
    }
    free_operands(o);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Fp8GemmTileTest,
                         ::testing::Values(std::make_tuple(512, 4096, 4096), std::make_tuple(512, 1024, 4096),
                                           std::make_tuple(77, 4096, 4096),
                                           std::make_tuple(2048, 12288, 4096),
                                           std::make_tuple(512, 4096, 12288), std::make_tuple(1, 1024, 4096),
                                           std::make_tuple(130, 136, 192)));

// K % 64 != 0 and N % 8 != 0 are refused and D stays byte-identical.
TEST(Fp8GemmTileRefusal, IllegalShapesReturnFalseAndLeaveOutput) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
        GTEST_SKIP() << "no CUDA device";
    cudaStream_t s;
    cudaStreamCreate(&s);
    struct Shape {
        int M, N, K;
    };
    for (const Shape sh : {Shape{64, 64, 96}, Shape{64, 60, 128}}) {
        Operands o;
        make_operands(o, sh.M, sh.N, sh.K);
        const size_t bytes = size_t(sh.M) * sh.N * sizeof(half);
        cudaMemset(o.y_own, 0xab, bytes);
        for (int i = 0; i < kVariants; ++i)
            EXPECT_FALSE(run_own(o, 0.0f, variant_of(i), s))
                << "M=" << sh.M << " N=" << sh.N << " K=" << sh.K;
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        std::vector<uint8_t> host(bytes);
        cudaMemcpy(host.data(), o.y_own, bytes, cudaMemcpyDeviceToHost);
        EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](uint8_t b) { return b == 0xab; }))
            << "refused call wrote D (N=" << sh.N << " K=" << sh.K << ")";
        free_operands(o);
    }
    cudaStreamDestroy(s);
}

// Cold-L2 timing on Qwen3-8B prefill projections: each call follows a 256 MiB memset flush, median of 31
// timed calls after >= 1 s of warm-up. DISABLED_ so no lane runs it: --gtest_also_run_disabled_tests
// --gtest_filter=*Fp8GemmTileBench*.
TEST(Fp8GemmTileBench, DISABLED_VsCublasLt) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    constexpr size_t kFlushBytes = size_t(256) << 20;
    void* flush = nullptr;
    ASSERT_EQ(cudaMalloc(&flush, kFlushBytes), cudaSuccess);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    // Median device time of one call in us; the flush runs outside the event pair.
    auto median_us = [&](auto&& fn) {
        auto one = [&]() {
            cudaMemsetAsync(flush, 0x5a, kFlushBytes, s);
            cudaEventRecord(e0, s);
            fn();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1000.0 * ms;
        };
        double t = 0.0;
        while (t < 1e6)  // warm for >= 1 s
            t += one();
        std::vector<double> v;
        for (int i = 0; i < 31; ++i)
            v.push_back(one());
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    // Qwen3-8B: q/o 4096x4096, k/v 1024x4096, gate or up 12288x4096 (and the fused 24576), down 4096x12288.
    const int nk[][2] = {{4096, 4096}, {1024, 4096}, {12288, 4096}, {24576, 4096}, {4096, 12288}};
    const int ms[] = {512, 2048};
    std::fprintf(stderr, "%6s %6s %6s %9s %9s %9s %9s %9s %9s %9s %9s %9s %7s\n", "M", "N", "K", "cublasLt",
                 "t128", "t256", "sync", "sk", "hybrid", "t64", "t64sk", "auto", "auto_TF");
    for (const auto& s_nk : nk) {
        for (int M : ms) {
            Operands o;
            make_operands(o, M, s_nk[0], s_nk[1]);
            const double t_ref = median_us([&] { run_ref(o, 0.0f, s); });
            double t_v[kVariants];
            for (int i = 0; i < kVariants; ++i)
                t_v[i] = median_us([&] { (void)run_own(o, 0.0f, variant_of(i), s); });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const double tf = 2.0 * o.M * o.N * double(o.K) / (t_v[kVariants - 1] * 1e-6) / 1e12;
            std::fprintf(stderr, "%6d %6d %6d %9.1f", o.M, o.N, o.K, t_ref);
            for (double t : t_v)
                std::fprintf(stderr, " %9.1f", t);
            std::fprintf(stderr, " %7.0f\n", tf);
            free_operands(o);
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(flush);
    cudaStreamDestroy(s);
}
