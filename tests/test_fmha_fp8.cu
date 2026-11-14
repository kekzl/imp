// FA2 register-resident FMHA (fmha_sm120_fa2_prefill, f16 QK^T) vs a CPU reference: indexing,
// masking, chunk continuation, accumulate variants and CTA order. The fp8-QK kernels these
// tests also covered were deleted with attention.fp8_fmha (teacher-forced PPL gemma-3-12b
// 16.6->549, #511).

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include "core/process_diag.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <float.h>

namespace imp {
namespace {

class FmhaTestBase : public ::testing::Test {
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

    cudaStream_t stream_ = nullptr;
};

// FA2 register-resident kernel (fmha_sm120_fa2_prefill): this fixture ASSERTs the FA2 path
// actually ran - a perf rewrite must not silently fall through.
class FmhaFA2Test : public FmhaTestBase {
protected:
    void run_fa2(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, int sw = 0,
                 float softcap = 0.0f, float amplitude = 1.0f, int q_offset = 0, float tol_override = 0.0f) {
        float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        size_t q_elems = B * Sq * NH * HD;
        size_t kv_elems = B * Skv * NKV * HD;

        std::vector<float> Q_f(q_elems), K_f(kv_elems), V_f(kv_elems);
        // int cast before -6 is load-bearing: unsigned wrap-around otherwise.
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

        bool ok = fmha_sm120_fa2_prefill(Qt, Kt, Vt, Ot, scale, causal, sw, softcap, stream_, q_offset);
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
        // f16 QK has no e4m3 score quantization: held to 1% (half-rounded inputs vs float
        // reference, ~2^-11/elem over a 128-dot + f16 P/V rounding).
        const float tol = (tol_override > 0.0f) ? tol_override : 0.01f;
        EXPECT_LT(max_err, tol) << "Max relative error too high: " << max_err;
    }
};

// FP16-QK variant (mma.m16n8k16.f16): the short-prefill replacement for materialized cuBLAS.
// No e4m3 quantization in QK, so verified at 1% (5x tighter than fp8); the
// realistic-magnitude cases are exactly where fp8 QK loses mantissa bits (#511) - fp16 must
// hold the tight bound there too.
TEST_F(FmhaFA2Test, FP16QK_CausalShortSeq24_GQA32_8) { run_fa2(1, 24, 24, 32, 8, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_CausalSeq64) { run_fa2(1, 64, 64, 4, 4, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_CausalOddSeq51) { run_fa2(1, 51, 51, 4, 4, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_CausalOddSeq136_GQA32_8) { run_fa2(1, 136, 136, 32, 8, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_RealisticMagnitude_Seq24) {
    run_fa2(1, 24, 24, 32, 8, 128, true, 0, 0.0f, /*amplitude=*/80.0f);
}
TEST_F(FmhaFA2Test, FP16QK_RealisticMagnitude_Seq64) {
    run_fa2(1, 64, 64, 4, 4, 128, true, 0, 0.0f, /*amplitude=*/80.0f);
}
TEST_F(FmhaFA2Test, FP16QK_NonCausal) { run_fa2(1, 32, 64, 4, 4, 128, false, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_GQA) { run_fa2(1, 64, 64, 8, 2, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_MultiTile) { run_fa2(1, 128, 128, 4, 4, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_LongCtx) { run_fa2(1, 256, 256, 8, 2, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_SlidingWindow) { run_fa2(1, 128, 128, 4, 4, 128, true, 64, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_Softcap) { run_fa2(1, 64, 64, 4, 4, 128, true, 0, 50.0f, 1.0f); }

// Chunk continuation (q_offset>0), #548: PR #553 measured wrong attention on Llama-3.2-3B
// chunk continuations (teacher-forced NLL 0.29->7.13 at chunk=64) while Qwen3-4B was
// bit-exact through the same kernel; the fast path was declined as a mitigation. Reproduces
// the failing shapes at kernel level: Llama's GQA ratio 3 (24Q/8KV) vs Qwen3's 4 (32/8),
// non-tile-multiple KV lengths, offsets not Bq/Bkv multiples.
TEST_F(FmhaFA2Test, FP16QK_Chunked_GQA4) {
    run_fa2(1, 64, 512, 32, 8, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/448);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_GQA3_LlamaShape) {
    run_fa2(1, 64, 512, 24, 8, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/448);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_GQA3_OddKv) {
    // partial last KV tile (seq_kv % 64 != 0) + odd chunk length
    run_fa2(1, 51, 371, 24, 8, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/320);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_OffsetNotTileMultiple) {
    run_fa2(1, 64, 292, 24, 8, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/228);
}
TEST_F(FmhaFA2Test, FP16QK_Chunked_LongCtx) {
    // continuation chunk far past the threshold-class context (3k)
    run_fa2(1, 128, 3200, 24, 8, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/3072);
}

// Bq=64/Bkv=32 occupancy band (#597): blocks_128 = ceil(Sq/128)*NH must land in
// [sm_count/2, sm_count) to select the 2-CTA/SM Bkv=32 kernel - 96 on the 170-SM RTX 5090.
// Covers multi-tile causal, partial last KV tile, chunk continuation, sliding-window bounds.
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_CausalMultiTile) {
    run_fa2(1, 384, 384, 32, 8, 128, true, 0, 0.0f, 1.0f);
}
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_OddSeq) { run_fa2(1, 333, 333, 32, 8, 128, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_Chunked) {
    run_fa2(1, 384, 1408, 32, 8, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/1024);
}
TEST_F(FmhaFA2Test, FP16QK_Bkv32Band_SlidingWindow) {
    run_fa2(1, 384, 384, 32, 8, 128, true, /*sw=*/64, 0.0f, 1.0f);
}

// PV f16-accumulate (process_diag_fa2_pv_f16acc, #667 follow-up): tolerance widened to 2% since
// O accumulates in f16 across KV tiles (the rescale-and-add rounding this knob trades for
// full-rate HMMA); production gate is teacher-forced PPL, this pins math/layout only.
class FmhaFA2PvF16Test : public FmhaFA2Test {
protected:
    void run_pv(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, int sw = 0,
                float softcap = 0.0f, float amplitude = 1.0f, int q_offset = 0) {
        process_diag_set_fa2_f16acc(true);
        process_diag_set_fa2_pv_f16acc(true);
        run_fa2(B, Sq, Skv, NH, NKV, HD, causal, sw, softcap, amplitude, q_offset,
                /*tol_override=*/0.02f);
        process_diag_set_fa2_f16acc(false);
        process_diag_set_fa2_pv_f16acc(false);
    }
};

// process_diag_fa2_dense_2cta: the Bq=128 TWOSLOT instance pinned to 128
// registers, two CTAs per SM. Shapes must reach the Bq=128 band
// (q-tiles x heads >= SM count): 1024 x 24 heads = 192 CTAs.
class FmhaFA2Dense2CtaTest : public FmhaFA2PvF16Test {
protected:
    void SetUp() override {
        FmhaFA2PvF16Test::SetUp();
        process_diag_set_fa2_dense_2cta(true);
    }
    void TearDown() override {
        process_diag_set_fa2_dense_2cta(false);
        FmhaFA2PvF16Test::TearDown();
    }
};
TEST_F(FmhaFA2Dense2CtaTest, Causal1024_24Q4KV) { run_pv(1, 1024, 1024, 24, 4, 128, true); }
TEST_F(FmhaFA2Dense2CtaTest, OddSeq1000) { run_pv(1, 1000, 1000, 24, 4, 128, true); }
TEST_F(FmhaFA2Dense2CtaTest, SlidingWindow) { run_pv(1, 1024, 1024, 24, 4, 128, true, /*sw=*/300); }
TEST_F(FmhaFA2Dense2CtaTest, Chunked) {
    run_pv(1, 1024, 2048, 24, 4, 128, true, 0, 0.0f, 1.0f, /*q_offset=*/1024);
}
TEST_F(FmhaFA2Dense2CtaTest, RealisticMagnitude) { run_pv(1, 1024, 1024, 24, 4, 128, true, 0, 0.0f, 80.0f); }
// process_diag_fa2_heavy_first: the causal CTA order is only a scheduling choice, every
// q-tile's rows are computed by exactly one CTA either way, so both orders must produce
// byte-identical output. 2048 rows x 24 heads (Bq=128, 2-CTA band); a reversal off-by-one
// leaves a tile's rows unwritten.
class FmhaFA2HeavyFirstTest : public FmhaFA2Test {
protected:
    void run_identity(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, int q_offset = 0) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        const size_t q_elems = static_cast<size_t>(B) * Sq * NH * HD;
        const size_t kv_elems = static_cast<size_t>(B) * Skv * NKV * HD;
        std::vector<half> Q_h(q_elems), K_h(kv_elems), V_h(kv_elems);
        for (size_t i = 0; i < q_elems; i++)
            Q_h[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6));
        for (size_t i = 0; i < kv_elems; i++) {
            K_h[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 11 + 5) % 13) - 6));
            V_h[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 13 + 7) % 13) - 6));
        }
        const size_t q_bytes = q_elems * sizeof(half), kv_bytes = kv_elems * sizeof(half);
        void *d_q, *d_k, *d_v, *d_o[2];
        cudaMalloc(&d_q, q_bytes);
        cudaMalloc(&d_k, kv_bytes);
        cudaMalloc(&d_v, kv_bytes);
        cudaMalloc(&d_o[0], q_bytes);
        cudaMalloc(&d_o[1], q_bytes);
        cudaMemcpy(d_q, Q_h.data(), q_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, K_h.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, V_h.data(), kv_bytes, cudaMemcpyHostToDevice);
        int64_t q_shape[] = {B, Sq, NH, HD};
        int64_t kv_shape[] = {B, Skv, NKV, HD};
        Tensor Qt(d_q, QType::F16, 4, q_shape, true);
        Tensor Kt(d_k, QType::F16, 4, kv_shape, true);
        Tensor Vt(d_v, QType::F16, 4, kv_shape, true);
        for (int order = 0; order < 2; order++) {
            cudaMemset(d_o[order], 0, q_bytes);
            process_diag_set_fa2_heavy_first(order == 1);
            Tensor Ot(d_o[order], QType::F16, 4, q_shape, true);
            bool ok = fmha_sm120_fa2_prefill(Qt, Kt, Vt, Ot, scale, causal, 0, 0.0f, stream_, q_offset);
            EXPECT_TRUE(ok) << "fa2 path must run in both orders";
        }
        process_diag_set_fa2_heavy_first(true);
        cudaStreamSynchronize(stream_);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        std::vector<half> O0(q_elems), O1(q_elems);
        cudaMemcpy(O0.data(), d_o[0], q_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(O1.data(), d_o[1], q_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o[0]);
        cudaFree(d_o[1]);
        size_t nonzero = 0;
        for (size_t i = 0; i < q_elems; i++)
            nonzero += (__half2float(O0[i]) != 0.0f);
        EXPECT_GT(nonzero, q_elems / 2) << "output is mostly zero: the kernel did not run the rows";
        EXPECT_EQ(std::memcmp(O0.data(), O1.data(), q_bytes), 0)
            << "natural and heavy-first CTA orders must be byte-identical";
    }
};
TEST_F(FmhaFA2HeavyFirstTest, Causal2048_24Q4KV_BitIdentical) { run_identity(1, 2048, 2048, 24, 4, 128, true); }
TEST_F(FmhaFA2HeavyFirstTest, Chunked1024of2048_BitIdentical) {
    run_identity(1, 1024, 2048, 24, 4, 128, true, /*q_offset=*/1024);
}
TEST_F(FmhaFA2HeavyFirstTest, OddSeq1000_BitIdentical) { run_identity(1, 1000, 1000, 24, 4, 128, true); }

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

// Stage-1 HD=256 FA2 port (process_diag_fa2_hd256): register-resident kernel at HD=256
// (fp16-qk only, Bq=64/Bkv=64/TWOSLOT), shapes mirroring Qwen3.6 hybrid geometry (hd=256,
// kv_heads=2, GQA 4:1). pv-f16 is the production candidate; f32/f16-acc variants are pinned
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
        run_fa2(B, Sq, Skv, NH, NKV, 256, causal, sw, softcap, amplitude, q_offset,
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
TEST_F(FmhaFA2Hd256Test, F32Acc_CausalSeq64) { run_fa2(1, 64, 64, 8, 2, 256, true, 0, 0.0f, 1.0f); }
TEST_F(FmhaFA2Hd256Test, F16Acc_CausalMultiTile) {
    process_diag_set_fa2_f16acc(true);
    run_fa2(1, 200, 200, 8, 2, 256, true, 0, 0.0f, 1.0f);
}

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
    printf("[hd256-bench] %s (Sq=%d NH=%d NKV=%d): FA2=%.3f ms  WMMA=%.3f ms  "
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
