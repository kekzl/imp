// Timing cases of tests/test_attention_fmha_hd512.cu, built into test-bench (make test-bench), not into the
// default GPU run.

#include <gtest/gtest.h>
#include "compute/attention_cublas.h"
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "test_attention_fmha_hd512_util.cuh"

namespace imp {
namespace {

class FmhaHd512Test : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

// DISABLED isolated A/B: WMMA hd=512 vs cuBLAS FP32-S at Gemma-4 shapes (nh=16,nkv=8,hd=512);
// run with --gtest_also_run_disabled_tests. Clock warmup >1s first (RTX 5090 idle-downclocked).
TEST_F(FmhaHd512Test, DISABLED_BenchVsCublas) {
    const int HD = 512, NH = 16, NKV = 8;
    const float scale = 1.0f / std::sqrt((float)HD);
    for (int Sq : {512, 2048}) {
        const int Skv = Sq;
        const size_t q_elems = (size_t)Sq * NH * HD, kv_elems = (size_t)Skv * NKV * HD;
        std::vector<half> Qh(q_elems), Kh(kv_elems), Vh(kv_elems);
        lcg_fill(Qh, 0x1111u, 4.0f);
        lcg_fill(Kh, 0x2222u, 4.0f);
        lcg_fill(Vh, 0x3333u, 1.0f);
        void *d_q, *d_k, *d_v, *d_o, *d_s;
        const size_t s_elems = (size_t)4 * NH * Sq * Skv;
        cudaMalloc(&d_q, q_elems * 2);
        cudaMalloc(&d_k, kv_elems * 2);
        cudaMalloc(&d_v, kv_elems * 2);
        cudaMalloc(&d_o, q_elems * 2);
        cudaMalloc(&d_s, s_elems * 2);
        cudaMemcpy(d_q, Qh.data(), q_elems * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, Kh.data(), kv_elems * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, Vh.data(), kv_elems * 2, cudaMemcpyHostToDevice);
        int64_t q2[2] = {Sq, (int64_t)NH * HD}, kv2[2] = {Skv, (int64_t)NKV * HD};
        int64_t s3[3] = {NH, Sq, 4 * (int64_t)Skv};
        int64_t q4[4] = {1, Sq, NH, HD}, kv4[4] = {1, Skv, NKV, HD};
        Tensor Q2(d_q, QType::F16, 2, q2, true), K2(d_k, QType::F16, 2, kv2, true);
        Tensor V2(d_v, QType::F16, 2, kv2, true), O2(d_o, QType::F16, 2, q2, true);
        Tensor S3(d_s, QType::F16, 3, s3, true);
        Tensor Q4(d_q, QType::F16, 4, q4, true), K4(d_k, QType::F16, 4, kv4, true);
        Tensor V4(d_v, QType::F16, 4, kv4, true), O4(d_o, QType::F16, 4, q4, true);

        auto time_ms = [&](bool fmha, int reps) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a, stream_);
            for (int i = 0; i < reps; i++) {
                if (fmha)
                    // timing loop: the launch status is not under test
                    (void)fmha_sm120_prefill(Q4, K4, V4, O4, scale, true, 0, 0.0f, stream_, 0, nullptr);
                else
                    attention_cublas_prefill({.Q = &Q2,
                                              .K = &K2,
                                              .V = &V2,
                                              .O = &O2,
                                              .S = &S3,
                                              .n_heads = NH,
                                              .n_kv_heads = NKV,
                                              .head_dim = HD,
                                              .scale = scale,
                                              .causal = true,
                                              .softcap = 0.0f,
                                              .q_offset = 0,
                                              .stream = stream_});
            }
            cudaEventRecord(b, stream_);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return ms / reps;
        };
        // Warm the clocks (>1s of work) before timing.
        for (int w = 0; w < 40; w++) {
            // timing loop: the launch status is not under test
            (void)fmha_sm120_prefill(Q4, K4, V4, O4, scale, true, 0, 0.0f, stream_, 0, nullptr);
            attention_cublas_prefill({.Q = &Q2,
                                      .K = &K2,
                                      .V = &V2,
                                      .O = &O2,
                                      .S = &S3,
                                      .n_heads = NH,
                                      .n_kv_heads = NKV,
                                      .head_dim = HD,
                                      .scale = scale,
                                      .causal = true,
                                      .softcap = 0.0f,
                                      .q_offset = 0,
                                      .stream = stream_});
        }
        cudaStreamSynchronize(stream_);
        double best_c = 1e9, best_f = 1e9;
        for (int t = 0; t < 3; t++) {
            best_c = std::min(best_c, (double)time_ms(false, 20));
            best_f = std::min(best_f, (double)time_ms(true, 20));
        }
        printf("[hd512 bench pp%-4d] cublas_fp32s %.3f ms | fmha %.3f ms | speedup %.2fx\n", Sq, best_c,
               best_f, best_c / best_f);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_s);
    }
}

// DISABLED long-context fallback bench: Sq=2048 continuation vs long KV (8k/16k), the actual
// production shape for hd=512 (materialized S overflows the workspace). FMHA whole-chunk vs
// cuBLAS 256-row slices. Run with --gtest_also_run_disabled_tests.
TEST_F(FmhaHd512Test, DISABLED_BenchLongCtxFallback) {
    const int HD = 512, NH = 16, NKV = 8;
    const float scale = 1.0f / std::sqrt((float)HD);
    const int Sq = 2048, SLICE = 256;
    for (int Skv : {8192, 16384}) {
        const int q_offset = Skv - Sq;
        const size_t q_elems = (size_t)Sq * NH * HD, kv_elems = (size_t)Skv * NKV * HD;
        std::vector<half> Qh(q_elems), Kh(kv_elems), Vh(kv_elems);
        lcg_fill(Qh, 0x1111u, 4.0f);
        lcg_fill(Kh, 0x2222u, 4.0f);
        lcg_fill(Vh, 0x3333u, 1.0f);
        void *d_q, *d_k, *d_v, *d_o, *d_s;
        const size_t s_elems = (size_t)4 * NH * SLICE * Skv;  // per-slice score buffer (FP32-S)
        ASSERT_EQ(cudaMalloc(&d_q, q_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_k, kv_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_v, kv_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_o, q_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_s, s_elems * 2), cudaSuccess);
        cudaMemcpy(d_q, Qh.data(), q_elems * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, Kh.data(), kv_elems * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, Vh.data(), kv_elems * 2, cudaMemcpyHostToDevice);

        int64_t kv2[2] = {Skv, (int64_t)NKV * HD};
        int64_t s3[3] = {NH, SLICE, 4 * (int64_t)Skv};
        int64_t q4[4] = {1, Sq, NH, HD}, kv4[4] = {1, Skv, NKV, HD};
        Tensor K2(d_k, QType::F16, 2, kv2, true), V2(d_v, QType::F16, 2, kv2, true);
        Tensor S3(d_s, QType::F16, 3, s3, true);
        Tensor Q4(d_q, QType::F16, 4, q4, true), K4(d_k, QType::F16, 4, kv4, true);
        Tensor V4(d_v, QType::F16, 4, kv4, true), O4(d_o, QType::F16, 4, q4, true);

        auto run_fmha = [&]() {
            bool ran = fmha_sm120_prefill(Q4, K4, V4, O4, scale, true, 0, 0.0f, stream_, q_offset, nullptr);
            ASSERT_TRUE(ran);
        };
        auto run_cublas_sliced = [&](int slice) {
            for (int off = 0; off < Sq; off += slice) {
                int64_t q2s[2] = {slice, (int64_t)NH * HD};
                Tensor Q2s(static_cast<half*>(d_q) + (size_t)off * NH * HD, QType::F16, 2, q2s, true);
                Tensor O2s(static_cast<half*>(d_o) + (size_t)off * NH * HD, QType::F16, 2, q2s, true);
                attention_cublas_prefill({.Q = &Q2s,
                                          .K = &K2,
                                          .V = &V2,
                                          .O = &O2s,
                                          .S = &S3,
                                          .n_heads = NH,
                                          .n_kv_heads = NKV,
                                          .head_dim = HD,
                                          .scale = scale,
                                          .causal = true,
                                          .softcap = 0.0f,
                                          .q_offset = q_offset + off,
                                          .stream = stream_});
            }
        };

        // Warm the clocks: >1.2s of sustained work before timing (idle downclock).
        {
            auto t0 = std::chrono::steady_clock::now();
            do {
                run_fmha();
                run_cublas_sliced(SLICE);
                cudaStreamSynchronize(stream_);
            } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 1.2);
        }

        auto time_ms = [&](int slice /* 0 = fmha */, int reps) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a, stream_);
            for (int i = 0; i < reps; i++) {
                if (slice == 0)
                    run_fmha();
                else
                    run_cublas_sliced(slice);
            }
            cudaEventRecord(b, stream_);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return ms / reps;
        };
        double best_f = 1e9;
        for (int t = 0; t < 3; t++)
            best_f = std::min(best_f, (double)time_ms(0, 5));
        printf("[hd512 longctx Sq=%d Skv=%-5d] fmha %.3f ms\n", Sq, Skv, best_f);
        // Slice sweep: the production workspace cap (attn_scores_mib) limits the
        // feasible slice at a given ctx — smaller slices trade GEMM efficiency
        // for workspace. 256 needs the full 384 MiB at 16k ctx; 64 fits at 64k.
        for (int slice : {16, 32, 64, 128, 256}) {
            double best_c = 1e9;
            for (int t = 0; t < 3; t++)
                best_c = std::min(best_c, (double)time_ms(slice, 5));
            printf("[hd512 longctx Sq=%d Skv=%-5d] cublas_sliced(%3d) %.3f ms | fmha/cublas %.2fx\n", Sq, Skv,
                   slice, best_c, best_c / best_f);
        }
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_s);
    }
}

}  // namespace
}  // namespace imp
