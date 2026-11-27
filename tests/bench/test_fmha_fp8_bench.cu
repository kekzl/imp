// Timing cases of tests/test_fmha_fp8.cu, built into test-bench (make test-bench), not into the default GPU
// run.

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include "core/process_diag.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

namespace imp {
namespace {

class FmhaFA2Hd256Test : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
        process_diag_set_fa2_hd256(true);
    }
    void TearDown() override {
        process_diag_set_fa2_hd256(false);
        process_diag_set_fa2_f16acc(false);
        process_diag_set_fa2_pv_f16acc(false);
        cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_ = nullptr;
};

// Micro-benchmark: FA2 HD=256 (pv-f16) vs SMEM-tiled WMMA FMHA on the Qwen3.6-35B prefill
// shape (8Q/2KV heads, hd=256, 2048 tokens). Reports per-kernel ms + cross-path max rel
// error; stage-1 decision data, not a perf gate.
static void hd256_bench_vs_wmma(cudaStream_t stream_, int NH, int NKV, const char* tag, bool pv_f16 = true) {
    for (int sweep_sq : {512, 1024, 2048, 4096}) {
        const int B = 1, Sq = sweep_sq, Skv = sweep_sq, HD = 256;
        const bool causal = true;
        const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        process_diag_set_fa2_f16acc(true);
        process_diag_set_fa2_pv_f16acc(pv_f16);

        size_t q_elems = (size_t)B * Sq * NH * HD;
        size_t kv_elems = (size_t)B * Skv * NKV * HD;
        std::vector<half> Q_h(q_elems), K_h(kv_elems), V_h(kv_elems);
        for (size_t i = 0; i < q_elems; i++)
            Q_h[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6));
        for (size_t i = 0; i < kv_elems; i++) {
            K_h[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 11 + 5) % 13) - 6));
            V_h[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 13 + 7) % 13) - 6));
        }
        void *d_q, *d_k, *d_v, *d_o_fa2, *d_o_wmma;
        cudaMalloc(&d_q, q_elems * sizeof(half));
        cudaMalloc(&d_k, kv_elems * sizeof(half));
        cudaMalloc(&d_v, kv_elems * sizeof(half));
        cudaMalloc(&d_o_fa2, q_elems * sizeof(half));
        cudaMalloc(&d_o_wmma, q_elems * sizeof(half));
        cudaMemcpy(d_q, Q_h.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, K_h.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, V_h.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice);

        int64_t q_shape[] = {B, Sq, NH, HD};
        int64_t kv_shape[] = {B, Skv, NKV, HD};
        Tensor Qt(d_q, QType::F16, 4, q_shape, true);
        Tensor Kt(d_k, QType::F16, 4, kv_shape, true);
        Tensor Vt(d_v, QType::F16, 4, kv_shape, true);
        Tensor O_fa2(d_o_fa2, QType::F16, 4, q_shape, true);
        Tensor O_wmma(d_o_wmma, QType::F16, 4, q_shape, true);

        // correctness cross-check first
        ASSERT_TRUE(fmha_sm120_fa2_prefill(Qt, Kt, Vt, O_fa2, scale, causal, 0, 0.0f, stream_, 0));
        ASSERT_TRUE(fmha_sm120_prefill(Qt, Kt, Vt, O_wmma, scale, causal, 0, 0.0f, stream_, 0));
        cudaStreamSynchronize(stream_);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        {
            std::vector<half> a(q_elems), b(q_elems);
            cudaMemcpy(a.data(), d_o_fa2, q_elems * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), d_o_wmma, q_elems * sizeof(half), cudaMemcpyDeviceToHost);
            float max_rel = 0.0f;
            int nans = 0;
            for (size_t i = 0; i < q_elems; i++) {
                float x = __half2float(a[i]), y = __half2float(b[i]);
                if (std::isnan(x) || std::isnan(y)) {
                    nans++;
                    continue;
                }
                max_rel = std::max(max_rel, std::abs(x - y) / std::max(1.0f, std::abs(y)));
            }
            EXPECT_EQ(nans, 0);
            EXPECT_LT(max_rel, 0.03f) << "FA2-hd256 vs WMMA cross-path divergence";
            printf("[hd256-bench] cross-path max_rel=%.5f\n", max_rel);
        }

        auto time_kernel = [&](auto&& launch) {
            for (int w = 0; w < 3; w++)
                launch();
            cudaStreamSynchronize(stream_);
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
            constexpr int kIters = 20;
            cudaEventRecord(t0, stream_);
            for (int i = 0; i < kIters; i++)
                launch();
            cudaEventRecord(t1, stream_);
            cudaEventSynchronize(t1);
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
            return ms / kIters;
        };
        float fa2_ms = time_kernel([&] {
            // timing loop: the launch status is not under test
            (void)fmha_sm120_fa2_prefill(Qt, Kt, Vt, O_fa2, scale, causal, 0, 0.0f, stream_, 0);
        });
        float wmma_ms = time_kernel([&] {
            // timing loop: the launch status is not under test
            (void)fmha_sm120_prefill(Qt, Kt, Vt, O_wmma, scale, causal, 0, 0.0f, stream_, 0);
        });
        printf(
            "[hd256-bench] %s (Sq=%d NH=%d NKV=%d): FA2=%.3f ms  WMMA=%.3f ms  "
            "(FA2/WMMA = %.2fx)\n",
            tag, Sq, NH, NKV, fa2_ms, wmma_ms, fa2_ms / wmma_ms);

        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o_fa2);
        cudaFree(d_o_wmma);
    }  // sweep_sq
}

TEST_F(FmhaFA2Hd256Test, BenchVsWmma_Qwen36Shape) { hd256_bench_vs_wmma(stream_, 8, 2, "bkv64 Qwen3.6"); }
TEST_F(FmhaFA2Hd256Test, BenchVsWmma_Qwen38Shape) { hd256_bench_vs_wmma(stream_, 24, 4, "bkv64 Qwen3.8"); }

}  // namespace
}  // namespace imp
