// Timing cases of tests/test_gemm_f16_narrow_prefill.cu, built into test-bench (make test-bench), not into
// the default GPU run.
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
