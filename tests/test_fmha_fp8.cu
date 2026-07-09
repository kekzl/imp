// Tests for the FP8 E4M3 FMHA kernel (QK^T in FP8, PV in FP16).
// Verifies correctness against a CPU reference for various configs.
//
// NOTE (#511): these parity cases use small synthetic values (±0.12) where
// e4m3 quantization error is invisible. On real model activations the raw
// (unscaled) Q/K→e4m3 conversion compounds per-layer score error into
// garbage (teacher-forced PPL gemma-3-12b 16.6→549, Qwen3-8B 40.5→4506) —
// which is why the kernel is opt-in (attention.fp8_fmha = "on") and NOT in
// the default dispatch chain. These tests pin indexing/masking only, not
// production quality.

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include "runtime/process_diag.h"
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
                              int NH, int NKV, int HD, float scale, bool causal, int sw, float softcap,
                              int q_offset = 0) {
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
                        if (causal && sk > (q_offset + sq))
                            dot = -1e30f;
                        if (sw > 0 && ((q_offset + sq) - sk) >= sw)
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
        // NOTE: the int cast before the -6 is load-bearing. `(i*7+3)%13 - 6`
        // with size_t i underflows unsigned whenever %13 < 6, producing
        // ±3.7e17 (→ ±inf as half) at ~46% of positions. The e4m3 satfinite
        // convert masked that in the fp8 kernel, and the NaN-poisoned CPU
        // reference made std::max() drop every comparison → vacuous pass.
        for (size_t i = 0; i < q_elems; i++)
            Q_f[i] = 0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6);
        for (size_t i = 0; i < kv_elems; i++) {
            K_f[i] = 0.02f * static_cast<float>(static_cast<int>((i * 11 + 5) % 13) - 6);
            V_f[i] = 0.02f * static_cast<float>(static_cast<int>((i * 13 + 7) % 13) - 6);
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
        int nan_count = 0, ref_nan_count = 0;
        for (size_t i = 0; i < q_elems; i++) {
            float got = __half2float(O_h[i]);
            float ref = O_ref[i];
            if (!std::isfinite(ref)) {
                ref_nan_count++;  // poisoned reference would silently skip comparisons
                continue;
            }
            if (std::isnan(got)) {
                nan_count++;
                continue;
            }
            float err = std::abs(got - ref);
            float denom = std::max(1.0f, std::abs(ref));
            max_err = std::max(max_err, err / denom);
        }
        EXPECT_EQ(ref_nan_count, 0) << "CPU reference is NaN/inf — test data is broken";
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

// --- #566 residue: fp8 kernel at hd=256, production-like long sizes ---
// gemma-3-12b (hd=256) prefill routes through THIS kernel once n crosses the
// FMHA threshold (head_dim % 32 gate — not the FP16 WMMA as assumed). The
// pre-#569 catastrophic window readings and the no-window PPL 10.5 came
// through here; pin the kernel against the fp64-style oracle at those shapes.
TEST_F(FmhaFP8Test, HD256_LongSeq) { run_test(1, 1536, 1536, 16, 8, 256, true); }
TEST_F(FmhaFP8Test, HD256_LongSeq_SlidingWindow1024) {
    run_test(1, 1536, 1536, 16, 8, 256, true, /*sw=*/1024);
}
TEST_F(FmhaFP8Test, HD128_LongSeq_SlidingWindow1024) {
    run_test(1, 1536, 1536, 16, 8, 128, true, /*sw=*/1024);
}

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
                 float softcap = 0.0f, float amplitude = 1.0f, bool fp16_qk = false, int q_offset = 0,
                 float tol_override = 0.0f) {
        float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        size_t q_elems = B * Sq * NH * HD;
        size_t kv_elems = B * Skv * NKV * HD;

        std::vector<float> Q_f(q_elems), K_f(kv_elems), V_f(kv_elems);
        // int cast before -6 is load-bearing — see run_test above.
        for (size_t i = 0; i < q_elems; i++)
            Q_f[i] = amplitude * 0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6);
        for (size_t i = 0; i < kv_elems; i++) {
            K_f[i] = amplitude * 0.02f * static_cast<float>(static_cast<int>((i * 11 + 5) % 13) - 6);
            V_f[i] = 0.02f * static_cast<float>(static_cast<int>((i * 13 + 7) % 13) - 6);
        }

        std::vector<float> O_ref(q_elems, 0.0f);
        ref_attention(Q_f, K_f, V_f, O_ref, B, Sq, Skv, NH, NKV, HD, scale, causal, sw, softcap, q_offset);

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

        bool ok = fmha_sm120_fa2_prefill(Qt, Kt, Vt, Ot, scale, causal, sw, softcap, stream_,
                                         q_offset, fp16_qk);
        ASSERT_TRUE(ok) << "fmha_sm120_fa2_prefill returned false (config must be supported)";
        cudaStreamSynchronize(stream_);
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

        std::vector<half> O_h(q_elems);
        cudaMemcpy(O_h.data(), d_o, q_bytes, cudaMemcpyDeviceToHost);

        float max_err = 0.0f;
        int nan_count = 0, ref_nan_count = 0;
        for (size_t i = 0; i < q_elems; i++) {
            float got = __half2float(O_h[i]);
            float ref = O_ref[i];
            if (!std::isfinite(ref)) {
                ref_nan_count++;  // poisoned reference would silently skip comparisons
                continue;
            }
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
        EXPECT_EQ(ref_nan_count, 0) << "CPU reference is NaN/inf — test data is broken";
        EXPECT_EQ(nan_count, 0) << "NaN values in FA2 FMHA output";
        // fp16 QK has no e4m3 score quantization — hold it to a 1% bound
        // (inputs are half-rounded vs the float reference; ~2^-11 per element
        // over a 128-dot + f16 P/V rounding in PV). fp8 QK keeps the
        // historical 5%.
        const float tol = (tol_override > 0.0f) ? tol_override : (fp16_qk ? 0.01f : 0.05f);
        EXPECT_LT(max_err, tol) << "Max relative error too high: " << max_err;
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

// ---------------------------------------------------------------------------
// FP16-QK variant (mma.m16n8k16.f16): the short-prefill replacement for the
// materialized cuBLAS path. No e4m3 quantization anywhere in QK → verified
// at 1% tolerance (5x tighter than the fp8 tests). The realistic-magnitude
// cases are the exact regime where the fp8 QK loses mantissa bits (#511) —
// fp16 must hold the tight bound there too.
// ---------------------------------------------------------------------------
TEST_F(FmhaFA2Test, FP16QK_CausalShortSeq24_GQA32_8) {
    run_fa2(1, 24, 24, 32, 8, 128, true, 0, 0.0f, 1.0f, /*fp16_qk=*/true);
}
TEST_F(FmhaFA2Test, FP16QK_CausalSeq64) { run_fa2(1, 64, 64, 4, 4, 128, true, 0, 0.0f, 1.0f, true); }
TEST_F(FmhaFA2Test, FP16QK_CausalOddSeq51) { run_fa2(1, 51, 51, 4, 4, 128, true, 0, 0.0f, 1.0f, true); }
TEST_F(FmhaFA2Test, FP16QK_CausalOddSeq136_GQA32_8) {
    run_fa2(1, 136, 136, 32, 8, 128, true, 0, 0.0f, 1.0f, true);
}
TEST_F(FmhaFA2Test, FP16QK_RealisticMagnitude_Seq24) {
    run_fa2(1, 24, 24, 32, 8, 128, true, 0, 0.0f, /*amplitude=*/80.0f, true);
}
TEST_F(FmhaFA2Test, FP16QK_RealisticMagnitude_Seq64) {
    run_fa2(1, 64, 64, 4, 4, 128, true, 0, 0.0f, /*amplitude=*/80.0f, true);
}
TEST_F(FmhaFA2Test, FP16QK_NonCausal) { run_fa2(1, 32, 64, 4, 4, 128, false, 0, 0.0f, 1.0f, true); }
TEST_F(FmhaFA2Test, FP16QK_GQA) { run_fa2(1, 64, 64, 8, 2, 128, true, 0, 0.0f, 1.0f, true); }
TEST_F(FmhaFA2Test, FP16QK_MultiTile) { run_fa2(1, 128, 128, 4, 4, 128, true, 0, 0.0f, 1.0f, true); }
TEST_F(FmhaFA2Test, FP16QK_LongCtx) { run_fa2(1, 256, 256, 8, 2, 128, true, 0, 0.0f, 1.0f, true); }
TEST_F(FmhaFA2Test, FP16QK_SlidingWindow) { run_fa2(1, 128, 128, 4, 4, 128, true, 64, 0.0f, 1.0f, true); }
TEST_F(FmhaFA2Test, FP16QK_Softcap) { run_fa2(1, 64, 64, 4, 4, 128, true, 0, 50.0f, 1.0f, true); }

// --- Chunk continuation (q_offset > 0) — issue #548 ---
// PR #553 measured wrong attention on Llama-3.2-3B chunk continuations
// (teacher-forced NLL 0.29 → 7.13 at chunk=64) while Qwen3-4B was bit-exact
// through the same kernel, and declined the fast path as a mitigation. These
// cases reproduce the failing production shapes at the kernel level:
// Llama-3.2-3B is GQA 24Q/8KV (ratio 3) vs Qwen3's 32/8 (ratio 4), prompts
// end on arbitrary (non-tile-multiple) KV lengths, and offsets are not
// Bq/Bkv multiples. seq_kv = q_offset + Sq exactly as the chunked gather
// produces it.
TEST_F(FmhaFA2Test, FP16QK_Chunked_GQA4) {
    run_fa2(1, 64, 512, 32, 8, 128, true, 0, 0.0f, 1.0f, true, /*q_offset=*/448);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_GQA3_LlamaShape) {
    run_fa2(1, 64, 512, 24, 8, 128, true, 0, 0.0f, 1.0f, true, /*q_offset=*/448);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_GQA3_OddKv) {
    // partial last KV tile (seq_kv % 64 != 0) + odd chunk length
    run_fa2(1, 51, 371, 24, 8, 128, true, 0, 0.0f, 1.0f, true, /*q_offset=*/320);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_OffsetNotTileMultiple) {
    run_fa2(1, 64, 292, 24, 8, 128, true, 0, 0.0f, 1.0f, true, /*q_offset=*/228);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_LongCtx) {
    // continuation chunk far past the threshold-class context (3k)
    run_fa2(1, 128, 3200, 24, 8, 128, true, 0, 0.0f, 1.0f, true, /*q_offset=*/3072);
}
TEST_F(FmhaFA2Test, FP8_Chunked_GQA3) {
    // e4m3 path with the same continuation geometry (looser 5% tol)
    run_fa2(1, 64, 512, 24, 8, 128, true, 0, 0.0f, 1.0f, false, /*q_offset=*/448);
}

// --- Bq=64/Bkv=32 occupancy band (#597) ---
// blocks_128 = ceil(Sq/128) × NH must land in [sm_count/2, sm_count) to select
// the 2-CTA/SM Bkv=32 kernel — these shapes give 96 on the 170-SM RTX 5090
// (GPU tests are local-only on that chip). Covers the halved KV tile against
// the same CPU oracle: multi-tile causal, partial last KV tile at Bkv=32
// granularity, chunk continuation, and sliding-window tile bounds.
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_CausalMultiTile) {
    run_fa2(1, 384, 384, 32, 8, 128, true, 0, 0.0f, 1.0f, true);
}
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_OddSeq) {
    run_fa2(1, 333, 333, 32, 8, 128, true, 0, 0.0f, 1.0f, true);
}
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_Chunked) {
    run_fa2(1, 384, 1408, 32, 8, 128, true, 0, 0.0f, 1.0f, true, /*q_offset=*/1024);
}
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_SlidingWindow) {
    run_fa2(1, 384, 384, 32, 8, 128, true, /*sw=*/64, 0.0f, 1.0f, true);
}

// --- PV f16-accumulate (attention.fa2_pv_f16acc, #667 follow-up) ---
// Same oracle; the dispatch reads the process-diag knobs, so the fixture
// toggles them around each case (and restores the pristine defaults).
// Tolerance is widened to 2%: O accumulates in f16 across KV tiles (the
// rescale-and-add rounding is exactly what this knob trades for full-rate
// HMMA); the production gate is teacher-forced PPL, this pins math/layout.
class FmhaFA2PvF16Test : public FmhaFA2Test {
protected:
    void run_pv(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, int sw = 0,
                float softcap = 0.0f, float amplitude = 1.0f, int q_offset = 0) {
        process_diag_set_fa2_f16acc(true);
        process_diag_set_fa2_pv_f16acc(true);
        run_fa2(B, Sq, Skv, NH, NKV, HD, causal, sw, softcap, amplitude, /*fp16_qk=*/true, q_offset,
                /*tol_override=*/0.02f);
        process_diag_set_fa2_f16acc(false);
        process_diag_set_fa2_pv_f16acc(false);
    }
};

TEST_F(FmhaFA2PvF16Test, CausalSeq64) { run_pv(1, 64, 64, 4, 4, 128, true); }
TEST_F(FmhaFA2PvF16Test, CausalShortSeq24_GQA32_8) { run_pv(1, 24, 24, 32, 8, 128, true); }
TEST_F(FmhaFA2PvF16Test, CausalOddSeq51) { run_pv(1, 51, 51, 4, 4, 128, true); }
TEST_F(FmhaFA2PvF16Test, RealisticMagnitude_Seq64) {
    run_pv(1, 64, 64, 4, 4, 128, true, 0, 0.0f, /*amplitude=*/80.0f);
}
TEST_F(FmhaFA2PvF16Test, NonCausal) { run_pv(1, 32, 64, 4, 4, 128, false); }
TEST_F(FmhaFA2PvF16Test, GQA) { run_pv(1, 64, 64, 8, 2, 128, true); }
TEST_F(FmhaFA2PvF16Test, MultiTile) { run_pv(1, 128, 128, 4, 4, 128, true); }
TEST_F(FmhaFA2PvF16Test, LongCtx) { run_pv(1, 256, 256, 8, 2, 128, true); }
TEST_F(FmhaFA2PvF16Test, SlidingWindow) { run_pv(1, 128, 128, 4, 4, 128, true, 64); }
TEST_F(FmhaFA2PvF16Test, Softcap) { run_pv(1, 64, 64, 4, 4, 128, true, 0, 50.0f); }
// chunk continuation + the two occupancy bands (Bq=64 twoslot / Bq=128)
TEST_F(FmhaFA2PvF16Test, Chunked_GQA3_OddKv) {
    run_pv(1, 51, 371, 24, 8, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/320);
}
TEST_F(FmhaFA2PvF16Test, Bkv32Band_CausalMultiTile) { run_pv(1, 384, 384, 32, 8, 128, true); }
TEST_F(FmhaFA2PvF16Test, Bq128Band_LongCtx) { run_pv(1, 768, 768, 32, 8, 128, true); }

// --- amax-scaled fp8-QK (attention.fp8_qk_scaled, #680) ---
// The raw e4m3 conversion loses mantissa bits at realistic Q/K magnitudes
// (#511); the scaled variant must hold a much tighter bound exactly there.
class FmhaFA2Fp8ScaledTest : public FmhaFA2Test {
protected:
    void run_scaled(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, float amp,
                    int q_offset = 0) {
        process_diag_set_fp8_qk_scaled(true);
        run_fa2(B, Sq, Skv, NH, NKV, HD, causal, 0, 0.0f, amp, /*fp16_qk=*/false, q_offset,
                /*tol_override=*/0.02f);
        process_diag_set_fp8_qk_scaled(false);
    }
};

TEST_F(FmhaFA2Fp8ScaledTest, Causal) { run_scaled(1, 64, 64, 4, 4, 128, true, 1.0f); }
TEST_F(FmhaFA2Fp8ScaledTest, RealisticMagnitude) { run_scaled(1, 64, 64, 4, 4, 128, true, 80.0f); }
TEST_F(FmhaFA2Fp8ScaledTest, RealisticMagnitude_GQA) {
    run_scaled(1, 136, 136, 32, 8, 128, true, 80.0f);
}
TEST_F(FmhaFA2Fp8ScaledTest, Chunked) { run_scaled(1, 64, 512, 24, 8, 128, true, 80.0f, 448); }

// --- Stage-1 HD=256 FA2 port (attention.fa2_hd256) ---
// The register-resident FA2 kernel instanced at HD=256 (fp16-qk only,
// Bq=64/Bkv=64/TWOSLOT). Shapes mirror the Qwen3.6 hybrid geometry
// (head_dim=256, kv_heads=2, GQA 4:1) plus the usual edge cases. The pv-f16
// variant is the production candidate; the f32/f16-acc variants are pinned
// for correctness even where they spill registers.
class FmhaFA2Hd256Test : public FmhaFA2Test {
protected:
    void SetUp() override {
        FmhaFA2Test::SetUp();
        process_diag_set_fa2_hd256(true);
    }
    void TearDown() override {
        process_diag_set_fa2_hd256(false);
        process_diag_set_fa2_f16acc(false);
        process_diag_set_fa2_pv_f16acc(false);
        FmhaFA2Test::TearDown();
    }
    // pv-f16 production variant (2% tol — f16 O accumulation, same bound as
    // the hd=128 PvF16 suite).
    void run_pv256(int B, int Sq, int Skv, int NH, int NKV, bool causal, int sw = 0,
                   float softcap = 0.0f, float amplitude = 1.0f, int q_offset = 0) {
        process_diag_set_fa2_f16acc(true);
        process_diag_set_fa2_pv_f16acc(true);
        run_fa2(B, Sq, Skv, NH, NKV, 256, causal, sw, softcap, amplitude, /*fp16_qk=*/true, q_offset,
                /*tol_override=*/0.02f);
    }
};

TEST_F(FmhaFA2Hd256Test, PvF16_CausalSeq64_GQA8_2) { run_pv256(1, 64, 64, 8, 2, true); }
TEST_F(FmhaFA2Hd256Test, PvF16_CausalMultiTile) { run_pv256(1, 256, 256, 8, 2, true); }
TEST_F(FmhaFA2Hd256Test, PvF16_OddSeq51) { run_pv256(1, 51, 51, 8, 2, true); }
TEST_F(FmhaFA2Hd256Test, PvF16_OddSeq200_GQA16_4) { run_pv256(1, 200, 200, 16, 4, true); }
TEST_F(FmhaFA2Hd256Test, PvF16_NonCausal) { run_pv256(1, 32, 64, 8, 2, false); }
TEST_F(FmhaFA2Hd256Test, PvF16_RealisticMagnitude) {
    run_pv256(1, 64, 64, 8, 2, true, 0, 0.0f, /*amplitude=*/80.0f);
}
TEST_F(FmhaFA2Hd256Test, PvF16_SlidingWindow) { run_pv256(1, 128, 128, 8, 2, true, /*sw=*/64); }
TEST_F(FmhaFA2Hd256Test, PvF16_Softcap) { run_pv256(1, 64, 64, 8, 2, true, 0, /*softcap=*/50.0f); }
TEST_F(FmhaFA2Hd256Test, PvF16_Chunked) {
    // chunk continuation: seq_kv = q_offset + Sq, non-tile-multiple lengths
    run_pv256(1, 51, 371, 8, 2, true, 0, 0.0f, 1.0f, /*q_offset=*/320);
}
TEST_F(FmhaFA2Hd256Test, PvF16_LongCtx1536) { run_pv256(1, 1536, 1536, 8, 2, true); }

// f32-acc and f16-acc-QK variants: correctness must hold even where the
// register footprint spills (they are A/B references, not the fast path).
TEST_F(FmhaFA2Hd256Test, F32Acc_CausalSeq64) {
    run_fa2(1, 64, 64, 8, 2, 256, true, 0, 0.0f, 1.0f, /*fp16_qk=*/true);
}
TEST_F(FmhaFA2Hd256Test, F16Acc_CausalMultiTile) {
    process_diag_set_fa2_f16acc(true);
    run_fa2(1, 200, 200, 8, 2, 256, true, 0, 0.0f, 1.0f, /*fp16_qk=*/true);
}

// fp8-qk mode must keep declining hd=256 (no e4m3 instance in stage 1).
TEST_F(FmhaFA2Hd256Test, Fp8QkStillDeclines) {
    float scale = 1.0f / 16.0f;
    const int elems = 1 * 32 * 8 * 256;
    void* d;
    cudaMalloc(&d, elems * sizeof(half) * 4);
    half* p = static_cast<half*>(d);
    int64_t q_shape[] = {1, 32, 8, 256};
    int64_t kv_shape[] = {1, 32, 2, 256};
    Tensor Qt(p, QType::F16, 4, q_shape, true);
    Tensor Kt(p + elems, QType::F16, 4, kv_shape, true);
    Tensor Vt(p + 2 * elems, QType::F16, 4, kv_shape, true);
    Tensor Ot(p + 3 * elems, QType::F16, 4, q_shape, true);
    EXPECT_FALSE(fmha_sm120_fa2_prefill(Qt, Kt, Vt, Ot, scale, true, 0, 0.0f, stream_, 0,
                                        /*fp16_qk=*/false));
    cudaFree(d);
}

// Micro-benchmark: FA2 HD=256 (pv-f16) vs the SMEM-tiled WMMA FMHA on the
// Qwen3.6-35B prefill shape (8 Q heads / 2 KV heads, hd=256, 2048 tokens).
// Reports per-kernel ms + cross-path max relative error. Not a perf gate —
// the stage-1 decision data (see PR body).
TEST_F(FmhaFA2Hd256Test, BenchVsWmma_Qwen36Shape) {
    for (int sweep_sq : {512, 1024, 2048, 4096}) {
    const int B = 1, Sq = sweep_sq, Skv = sweep_sq, NH = 8, NKV = 2, HD = 256;
    const bool causal = true;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    process_diag_set_fa2_f16acc(true);
    process_diag_set_fa2_pv_f16acc(true);

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
    ASSERT_TRUE(fmha_sm120_fa2_prefill(Qt, Kt, Vt, O_fa2, scale, causal, 0, 0.0f, stream_, 0, true));
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
        fmha_sm120_fa2_prefill(Qt, Kt, Vt, O_fa2, scale, causal, 0, 0.0f, stream_, 0, true);
    });
    float wmma_ms = time_kernel([&] {
        fmha_sm120_prefill(Qt, Kt, Vt, O_wmma, scale, causal, 0, 0.0f, stream_, 0);
    });
    printf("[hd256-bench] Qwen3.6 shape (Sq=%d NH=%d NKV=%d): FA2=%.3f ms  WMMA=%.3f ms  "
           "(FA2/WMMA = %.2fx)\n",
           Sq, NH, NKV, fa2_ms, wmma_ms, fa2_ms / wmma_ms);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o_fa2);
    cudaFree(d_o_wmma);
    }  // sweep_sq
}

}  // namespace
}  // namespace imp
