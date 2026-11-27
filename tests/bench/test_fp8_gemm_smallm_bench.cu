// Timing cases of tests/test_fp8_gemm_smallm.cu, built into test-bench (make test-bench), not into the
// default GPU run.
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

#include "test_fp8_gemm_smallm_util.cuh"

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
