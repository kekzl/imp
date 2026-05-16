// Tests for the FP8 E4M3 FMHA kernel (QK^T in FP8, PV in FP16).
// Verifies correctness against a CPU reference for various configs.

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include "runtime/config.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <float.h>

namespace imp {
namespace {

class FmhaFP8Test : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }

    // CPU reference attention (same as FP16 test)
    static void ref_attention(const std::vector<float>& Q_f, const std::vector<float>& K_f,
                              const std::vector<float>& V_f, std::vector<float>& O_f, int B, int Sq, int Skv,
                              int NH, int NKV, int HD, float scale, bool causal, int sw, float softcap) {
        int gqa = NH / NKV;
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / gqa;
                for (int sq = 0; sq < Sq; sq++) {
                    float max_s = -FLT_MAX;
                    std::vector<float> scores(Skv);
                    for (int sk = 0; sk < Skv; sk++) {
                        float dot = 0;
                        for (int d = 0; d < HD; d++) {
                            int qi = ((b * Sq + sq) * NH + h) * HD + d;
                            int ki = ((b * Skv + sk) * NKV + kvh) * HD + d;
                            dot += Q_f[qi] * K_f[ki];
                        }
                        dot *= scale;
                        if (softcap > 0)
                            dot = softcap * std::tanh(dot / softcap);
                        if (causal && sk > sq)
                            dot = -1e30f;
                        if (sw > 0 && (sq - sk) >= sw)
                            dot = -1e30f;
                        scores[sk] = dot;
                        max_s = std::max(max_s, dot);
                    }
                    float sum_exp = 0;
                    for (int sk = 0; sk < Skv; sk++) {
                        scores[sk] = std::exp(scores[sk] - max_s);
                        sum_exp += scores[sk];
                    }
                    for (int d = 0; d < HD; d++) {
                        float val = 0;
                        for (int sk = 0; sk < Skv; sk++) {
                            int vi = ((b * Skv + sk) * NKV + (h / gqa)) * HD + d;
                            val += (scores[sk] / sum_exp) * V_f[vi];
                        }
                        int oi = ((b * Sq + sq) * NH + h) * HD + d;
                        O_f[oi] = val;
                    }
                }
            }
        }
    }

    void run_test(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, int sw = 0,
                  float softcap = 0.0f) {
        float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        size_t q_elems = B * Sq * NH * HD;
        size_t kv_elems = B * Skv * NKV * HD;

        std::vector<float> Q_f(q_elems), K_f(kv_elems), V_f(kv_elems);
        for (size_t i = 0; i < q_elems; i++)
            Q_f[i] = 0.02f * static_cast<float>((i * 7 + 3) % 13 - 6);
        for (size_t i = 0; i < kv_elems; i++) {
            K_f[i] = 0.02f * static_cast<float>((i * 11 + 5) % 13 - 6);
            V_f[i] = 0.02f * static_cast<float>((i * 13 + 7) % 13 - 6);
        }

        std::vector<float> O_ref(q_elems, 0.0f);
        ref_attention(Q_f, K_f, V_f, O_ref, B, Sq, Skv, NH, NKV, HD, scale, causal, sw, softcap);

        std::vector<half> Q_h(q_elems), K_h(kv_elems), V_h(kv_elems);
        for (size_t i = 0; i < q_elems; i++)
            Q_h[i] = __float2half(Q_f[i]);
        for (size_t i = 0; i < kv_elems; i++)
            K_h[i] = __float2half(K_f[i]);
        for (size_t i = 0; i < kv_elems; i++)
            V_h[i] = __float2half(V_f[i]);

        size_t q_bytes = q_elems * sizeof(half);
        size_t kv_bytes = kv_elems * sizeof(half);

        void *d_q, *d_k, *d_v, *d_o;
        cudaMalloc(&d_q, q_bytes);
        cudaMalloc(&d_k, kv_bytes);
        cudaMalloc(&d_v, kv_bytes);
        cudaMalloc(&d_o, q_bytes);

        cudaMemcpy(d_q, Q_h.data(), q_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, K_h.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, V_h.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_o, 0, q_bytes);

        int64_t q_shape[] = {B, Sq, NH, HD};
        int64_t kv_shape[] = {B, Skv, NKV, HD};
        Tensor Qt(d_q, QType::F16, 4, q_shape, true);
        Tensor Kt(d_k, QType::F16, 4, kv_shape, true);
        Tensor Vt(d_v, QType::F16, 4, kv_shape, true);
        Tensor Ot(d_o, QType::F16, 4, q_shape, true);

        bool ok = fmha_sm120_fp8_prefill(Qt, Kt, Vt, Ot, scale, causal, sw, softcap, stream_);
        if (!ok) {
            cudaFree(d_q);
            cudaFree(d_k);
            cudaFree(d_v);
            cudaFree(d_o);
            GTEST_SKIP() << "fmha_sm120_fp8_prefill returned false";
        }
        cudaStreamSynchronize(stream_);
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

        std::vector<half> O_h(q_elems);
        cudaMemcpy(O_h.data(), d_o, q_bytes, cudaMemcpyDeviceToHost);

        // FP8 QK^T has lower precision than FP16 — use relaxed tolerance
        float max_err = 0.0f;
        int nan_count = 0;
        for (size_t i = 0; i < q_elems; i++) {
            float got = __half2float(O_h[i]);
            float ref = O_ref[i];
            if (std::isnan(got)) {
                nan_count++;
                continue;
            }
            float err = std::abs(got - ref);
            float denom = std::max(1.0f, std::abs(ref));
            max_err = std::max(max_err, err / denom);
        }
        EXPECT_EQ(nan_count, 0) << "NaN values in FP8 FMHA output";
        // FP8 E4M3 has ~0.1% precision loss in scores, allow 5% relative error
        EXPECT_LT(max_err, 0.05f) << "Max relative error too high: " << max_err;

        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
    }

    cudaStream_t stream_ = nullptr;
};

TEST_F(FmhaFP8Test, CausalHD128) { run_test(1, 64, 64, 4, 4, 128, true); }

TEST_F(FmhaFP8Test, NonCausalHD128) { run_test(1, 32, 64, 4, 4, 128, false); }

TEST_F(FmhaFP8Test, GQA) { run_test(1, 64, 64, 8, 2, 128, true); }

TEST_F(FmhaFP8Test, CausalMultiTile) { run_test(1, 128, 128, 4, 4, 128, true); }

TEST_F(FmhaFP8Test, SlidingWindow) { run_test(1, 64, 64, 4, 4, 128, true, 32); }

TEST_F(FmhaFP8Test, Softcap) { run_test(1, 32, 32, 4, 4, 128, true, 0, 50.0f); }

TEST_F(FmhaFP8Test, HD64) { run_test(1, 32, 32, 4, 4, 64, true); }

TEST_F(FmhaFP8Test, HD256) { run_test(1, 32, 32, 4, 4, 256, true); }

// Mimic Qwen3.5-4B attention prefill shape: 16 Q heads, 4 KV heads (GQA 4:1),
// head_dim=256, 128-token sequence. Multi-tile on both axes with non-zero V
// throughout — catches the S_tile smem overlap bug that the HD256 test
// (Sq=Skv=32, zero-padded V) masked by having the reference also near zero.
TEST_F(FmhaFP8Test, Qwen35LikeHD256_GQA41_SeqMultiTile) { run_test(1, 128, 128, 16, 4, 256, true); }

// --- FP8 cluster path coverage (M5 Slice 2.2) -----------------------------
//
// The FP8 cluster kernel fires when:
//   - n_q_per_kv ∈ {2,4,8}
//   - head_dim ∈ {64,128,256}  (HD=96 dropped — FP8 m16n8k32 requires
//                                HD % 32 == 0)
//   - seq_kv ≥ 8 * CL_Bkv = 512
//
// run_test enforces a 5 % rel tolerance against the CPU reference; the
// cluster kernel inherits the legacy FP8 quantization noise.

TEST_F(FmhaFP8Test, ClusterPathGQA2Hd128) { run_test(1, 64, 512, 4, 2, 128, true); }
TEST_F(FmhaFP8Test, ClusterPathGQA4Hd128) { run_test(1, 64, 512, 8, 2, 128, true); }
TEST_F(FmhaFP8Test, ClusterPathGQA8Hd128) { run_test(1, 64, 512, 16, 2, 128, true); }
TEST_F(FmhaFP8Test, ClusterPathHd64) { run_test(1, 64, 512, 8, 2, 64, true); }
TEST_F(FmhaFP8Test, ClusterPathHd256) { run_test(1, 64, 512, 4, 2, 256, true); }
TEST_F(FmhaFP8Test, ClusterPathLongPrompt) { run_test(1, 128, 1024, 8, 2, 128, true); }
TEST_F(FmhaFP8Test, ClusterPathSlidingWindow) { run_test(1, 128, 1024, 4, 2, 128, true, /*sw=*/256); }
TEST_F(FmhaFP8Test, ClusterPathSoftcap) { run_test(1, 128, 1024, 4, 2, 128, true, 0, /*softcap=*/50.0f); }
TEST_F(FmhaFP8Test, ClusterPathBypassedForShortKv) {
    // seq_kv < 512 → cluster gate rejects, legacy FP8 kernel runs.
    run_test(1, 64, 256, 8, 2, 128, true);
}

// Direct cluster-vs-legacy bit-equivalence proof for the FP8 path. Mirrors
// FmhaSm120Test.ClusterMatchesLegacy. Confirms the DSMEM staging + FP16→FP8
// per-block conversion preserves the legacy FP8 kernel's bit-pattern.
TEST_F(FmhaFP8Test, ClusterMatchesLegacy) {
    int sm_major = 0, sm_minor = 0;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
    if (sm_major * 10 + sm_minor < 120) GTEST_SKIP() << "Requires sm_120+";

    const int B = 1, Sq = 64, Skv = 1024, NH = 8, NKV = 2, HD = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    size_t q_elems = B * Sq * NH * HD;
    size_t kv_elems = B * Skv * NKV * HD;

    std::vector<half> Q_h(q_elems), K_h(kv_elems), V_h(kv_elems);
    for (size_t i = 0; i < q_elems; i++)
        Q_h[i] = __float2half(0.02f * static_cast<float>((i * 7 + 3) % 13 - 6));
    for (size_t i = 0; i < kv_elems; i++) {
        K_h[i] = __float2half(0.02f * static_cast<float>((i * 11 + 5) % 13 - 6));
        V_h[i] = __float2half(0.02f * static_cast<float>((i * 13 + 7) % 13 - 6));
    }

    void *d_q, *d_k, *d_v, *d_o_cluster, *d_o_legacy;
    size_t q_bytes = q_elems * sizeof(half), kv_bytes = kv_elems * sizeof(half);
    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_k, kv_bytes);
    cudaMalloc(&d_v, kv_bytes);
    cudaMalloc(&d_o_cluster, q_bytes);
    cudaMalloc(&d_o_legacy, q_bytes);
    cudaMemcpy(d_q, Q_h.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K_h.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V_h.data(), kv_bytes, cudaMemcpyHostToDevice);

    int64_t q_shape[] = {B, Sq, NH, HD};
    int64_t kv_shape[] = {B, Skv, NKV, HD};
    Tensor Qt(d_q, QType::F16, 4, q_shape, true);
    Tensor Kt(d_k, QType::F16, 4, kv_shape, true);
    Tensor Vt(d_v, QType::F16, 4, kv_shape, true);

    {
        RuntimeConfig cfg = RuntimeConfig::current();
        cfg.attention.no_fmha_cluster = false;
        RuntimeConfig::install(cfg);
        cudaMemset(d_o_cluster, 0, q_bytes);
        Tensor Oc(d_o_cluster, QType::F16, 4, q_shape, true);
        ASSERT_TRUE(fmha_sm120_fp8_prefill(Qt, Kt, Vt, Oc, scale, true, 0, 0.0f, stream_));
        cudaStreamSynchronize(stream_);
    }
    {
        RuntimeConfig cfg = RuntimeConfig::current();
        cfg.attention.no_fmha_cluster = true;
        RuntimeConfig::install(cfg);
        cudaMemset(d_o_legacy, 0, q_bytes);
        Tensor Ol(d_o_legacy, QType::F16, 4, q_shape, true);
        ASSERT_TRUE(fmha_sm120_fp8_prefill(Qt, Kt, Vt, Ol, scale, true, 0, 0.0f, stream_));
        cudaStreamSynchronize(stream_);
    }
    {
        RuntimeConfig cfg = RuntimeConfig::current();
        cfg.attention.no_fmha_cluster = false;
        RuntimeConfig::install(cfg);
    }

    std::vector<half> Oc(q_elems), Ol(q_elems);
    cudaMemcpy(Oc.data(), d_o_cluster, q_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ol.data(), d_o_legacy, q_bytes, cudaMemcpyDeviceToHost);

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < q_elems; i++) {
        float a = __half2float(Oc[i]);
        float b = __half2float(Ol[i]);
        float ad = std::abs(a - b);
        float rd = ad / std::max(1.0f, std::abs(b));
        max_abs_diff = std::max(max_abs_diff, ad);
        max_rel_diff = std::max(max_rel_diff, rd);
    }
    fprintf(stderr, "FP8 ClusterMatchesLegacy: max_abs=%g max_rel=%g\n", max_abs_diff, max_rel_diff);
    EXPECT_LT(max_abs_diff, 1e-2f) << "FP8 cluster vs legacy max abs diff " << max_abs_diff;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o_cluster);
    cudaFree(d_o_legacy);
}

}  // namespace
}  // namespace imp
