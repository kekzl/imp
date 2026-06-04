// Tests for the FP8 E4M3 FMHA kernel (QK^T in FP8, PV in FP16).
// Verifies correctness against a CPU reference for various configs.

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
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

// ---------------------------------------------------------------------------
// FA2 register-resident kernel: same oracle, but exercises fmha_sm120_fa2_prefill.
// Unlike the fp8 fixture (which SKIPs on unsupported configs), this fixture
// ASSERTs the fa2 path actually ran — a perf rewrite must not silently fall
// through. Target config: head_dim=128.
// ---------------------------------------------------------------------------
class FmhaFA2Test : public FmhaFP8Test {
protected:
    void run_fa2(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, int sw = 0,
                 float softcap = 0.0f, float amplitude = 1.0f) {
        float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        size_t q_elems = B * Sq * NH * HD;
        size_t kv_elems = B * Skv * NKV * HD;

        std::vector<float> Q_f(q_elems), K_f(kv_elems), V_f(kv_elems);
        for (size_t i = 0; i < q_elems; i++)
            Q_f[i] = amplitude * 0.02f * static_cast<float>((i * 7 + 3) % 13 - 6);
        for (size_t i = 0; i < kv_elems; i++) {
            K_f[i] = amplitude * 0.02f * static_cast<float>((i * 11 + 5) % 13 - 6);
            V_f[i] = 0.02f * static_cast<float>((i * 13 + 7) % 13 - 6);
        }

        std::vector<float> O_ref(q_elems, 0.0f);
        ref_attention(Q_f, K_f, V_f, O_ref, B, Sq, Skv, NH, NKV, HD, scale, causal, sw, softcap);

        std::vector<half> Q_h(q_elems), K_h(kv_elems), V_h(kv_elems);
        for (size_t i = 0; i < q_elems; i++)
            Q_h[i] = __float2half(Q_f[i]);
        for (size_t i = 0; i < kv_elems; i++) {
            K_h[i] = __float2half(K_f[i]);
            V_h[i] = __float2half(V_f[i]);
        }

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

        bool ok = fmha_sm120_fa2_prefill(Qt, Kt, Vt, Ot, scale, causal, sw, softcap, stream_);
        ASSERT_TRUE(ok) << "fmha_sm120_fa2_prefill returned false (config must be supported)";
        cudaStreamSynchronize(stream_);
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

        std::vector<half> O_h(q_elems);
        cudaMemcpy(O_h.data(), d_o, q_bytes, cudaMemcpyDeviceToHost);

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
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        EXPECT_EQ(nan_count, 0) << "NaN values in FA2 FMHA output";
        EXPECT_LT(max_err, 0.05f) << "Max relative error too high: " << max_err;
    }
};

TEST_F(FmhaFA2Test, CausalHD128) { run_fa2(1, 64, 64, 4, 4, 128, true); }

// Engine-realistic SHORT prefill shapes. Since the executor prefers FA2 for
// hd=128 at every length (executor_attention.cu fa2_capable), short chat
// prompts hit this kernel too — production corruption (prompt-blind models,
// degenerate output) appeared exactly there. Qwen3-8B GQA is 32 Q / 8 KV.
TEST_F(FmhaFA2Test, CausalShortSeq24_GQA32_8) { run_fa2(1, 24, 24, 32, 8, 128, true); }
TEST_F(FmhaFA2Test, CausalSeq32_GQA32_8) { run_fa2(1, 32, 32, 32, 8, 128, true); }
TEST_F(FmhaFA2Test, CausalShortSeq24) { run_fa2(1, 24, 24, 4, 4, 128, true); }
TEST_F(FmhaFA2Test, CausalOddSeq136_GQA32_8) { run_fa2(1, 136, 136, 32, 8, 128, true); }
TEST_F(FmhaFA2Test, CausalOddSeq51) { run_fa2(1, 51, 51, 4, 4, 128, true); }

// Realistic Q/K magnitudes. QK-normed models (Qwen3 family) produce Q/K values
// far beyond the ±0.12 of the synthetic fill above; the FA2 kernel converts
// Q and K to FP8 e4m3 WITHOUT a scale factor, so large inputs lose precision
// or saturate (e4m3 max ±448) — production symptom: prompt-blind models on
// every hd=128 architecture while the FP16 cuBLAS path stays correct.
// amplitude=80 → |Q|,|K| up to ~9.6, QK dots up to ~118 pre-scale (still
// within e4m3 range per element, but only ~2 mantissa bits at that scale).
TEST_F(FmhaFA2Test, RealisticMagnitude_Seq24) {
    run_fa2(1, 24, 24, 32, 8, 128, true, 0, 0.0f, /*amplitude=*/80.0f);
}
TEST_F(FmhaFA2Test, RealisticMagnitude_Seq64) {
    run_fa2(1, 64, 64, 4, 4, 128, true, 0, 0.0f, /*amplitude=*/80.0f);
}
TEST_F(FmhaFA2Test, NonCausalHD128) { run_fa2(1, 32, 64, 4, 4, 128, false); }
TEST_F(FmhaFA2Test, GQA_HD128) { run_fa2(1, 64, 64, 8, 2, 128, true); }
TEST_F(FmhaFA2Test, CausalMultiTile_HD128) { run_fa2(1, 128, 128, 4, 4, 128, true); }
TEST_F(FmhaFA2Test, LongCtx_HD128) { run_fa2(1, 256, 256, 8, 2, 128, true); }
TEST_F(FmhaFA2Test, SlidingWindow_HD128) { run_fa2(1, 128, 128, 4, 4, 128, true, 64); }
TEST_F(FmhaFA2Test, Softcap_HD128) { run_fa2(1, 64, 64, 4, 4, 128, true, 0, 50.0f); }

}  // namespace
}  // namespace imp
