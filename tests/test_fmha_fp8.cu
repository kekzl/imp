// FA2 register-resident FMHA (fmha_sm120_fa2_prefill, f16 QK^T) vs a CPU reference: indexing,
// masking, chunk continuation, accumulate variants and CTA order. The fp8-QK kernels these
// tests also covered were deleted with attention.fp8_fmha (teacher-forced PPL gemma-3-12b
// 16.6->549, #511). Timing cases: tests/bench/test_fmha_fp8_bench.cu.

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include "core/process_diag.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cmath>
#include <float.h>

#include "host_parallel.h"

namespace imp {
namespace {

// CPU reference attention, one (batch, head) per thread; each output's loop order is the serial one.
void ref_attention(const std::vector<float>& Q_f, const std::vector<float>& K_f,
                   const std::vector<float>& V_f, std::vector<float>& O_f, int B, int Sq, int Skv, int NH,
                   int NKV, int HD, float scale, bool causal, int sw, float softcap, int q_offset = 0) {
    int gqa = NH / NKV;
    imp_test::parallel_for(B * NH, [&](int bh) {
        const int b = bh / NH, h = bh % NH;
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
    });
}

// Kernel variants. The process_diag flags each one sets are the state its former suite ran under
// in a full test-attention run (FmhaFA2Test, FmhaFA2Dense2CtaTest, FmhaFA2HeavyFirstTest,
// FmhaFA2PvF16Test, FmhaFA2Hd256Test, in that order: dense_2cta and the accumulate flags leaked
// from one suite into the next). Now set per case, independent of order and filter.
enum class Variant {
    kDefault,     // production defaults; f16 QK held to 1% (half inputs vs float reference)
    kDense2Cta,   // Bq=128 TWOSLOT instance at 2 CTAs/SM; 2% (f16 O accumulation)
    kHeavyFirst,  // causal CTA order: natural vs heavy-first output byte-identical, no reference
    kPvF16,       // PV f16-accumulate (#667 follow-up), dense_2cta off; 2%
    kHd256PvF16,  // HD=256 port (#932), pv-f16 production variant; 2%
    kHd256F32,    // HD=256, f32 accumulate (A/B reference, may spill); 1%
    kHd256F16Qk,  // HD=256, f16-accumulate QK only; 1%
};

struct Fa2Case {
    const char* name;  // former Suite_Case
    Variant v;
    int B, Sq, Skv, NH, NKV, HD;
    bool causal;
    int sw = 0;
    float softcap = 0.0f;
    float amplitude = 1.0f;
    int q_offset = 0;
};

void set_flags(bool f16acc, bool pv_f16acc, bool dense_2cta) {
    process_diag_set_fa2_f16acc(f16acc);
    process_diag_set_fa2_pv_f16acc(pv_f16acc);
    process_diag_set_fa2_dense_2cta(dense_2cta);
    process_diag_set_fa2_hd256(true);
    process_diag_set_fa2_heavy_first(true);
}

// Returns the tolerance of the variant; kHeavyFirst has none.
float apply_variant(Variant v) {
    switch (v) {
        case Variant::kDefault:
            set_flags(true, true, true);
            return 0.01f;
        case Variant::kDense2Cta:
            set_flags(true, true, true);
            return 0.02f;
        case Variant::kHeavyFirst:
            set_flags(false, false, false);
            return 0.0f;
        case Variant::kPvF16:
            set_flags(true, true, false);
            return 0.02f;
        case Variant::kHd256PvF16:
            set_flags(true, true, false);
            return 0.02f;
        case Variant::kHd256F32:
            set_flags(false, false, false);
            return 0.01f;
        case Variant::kHd256F16Qk:
            set_flags(true, false, false);
            return 0.01f;
    }
    return 0.01f;
}

class FmhaFA2Prefill : public ::testing::TestWithParam<Fa2Case> {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override {
        set_flags(true, true, true);  // production defaults
        cudaStreamDestroy(stream_);
    }

    // FA2 register-resident kernel (fmha_sm120_fa2_prefill): ASSERTs the FA2 path actually
    // ran - a perf rewrite must not silently fall through.
    void run_fa2(const Fa2Case& c, float tol) {
        const int B = c.B, Sq = c.Sq, Skv = c.Skv, NH = c.NH, NKV = c.NKV, HD = c.HD;
        float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        size_t q_elems = B * Sq * NH * HD;
        size_t kv_elems = B * Skv * NKV * HD;

        std::vector<float> Q_f(q_elems), K_f(kv_elems), V_f(kv_elems);
        // int cast before -6 is load-bearing: unsigned wrap-around otherwise.
        for (size_t i = 0; i < q_elems; i++)
            Q_f[i] = c.amplitude * 0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6);
        for (size_t i = 0; i < kv_elems; i++) {
            K_f[i] = c.amplitude * 0.02f * static_cast<float>(static_cast<int>((i * 11 + 5) % 13) - 6);
            V_f[i] = 0.02f * static_cast<float>(static_cast<int>((i * 13 + 7) % 13) - 6);
        }

        std::vector<float> O_ref(q_elems, 0.0f);
        ref_attention(Q_f, K_f, V_f, O_ref, B, Sq, Skv, NH, NKV, HD, scale, c.causal, c.sw, c.softcap,
                      c.q_offset);

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

        bool ok = fmha_sm120_fa2_prefill(Qt, Kt, Vt, Ot, scale, c.causal, c.sw, c.softcap, stream_,
                                         c.q_offset);
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
            float e = std::abs(got - ref);
            float denom = std::max(1.0f, std::abs(ref));
            max_err = std::max(max_err, e / denom);
        }
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        EXPECT_EQ(ref_nan_count, 0) << "CPU reference is NaN/inf — test data is broken";
        EXPECT_EQ(nan_count, 0) << "NaN values in FA2 FMHA output";
        EXPECT_LT(max_err, tol) << "Max relative error too high: " << max_err;
    }

    // process_diag_fa2_heavy_first: the causal CTA order is only a scheduling choice, every
    // q-tile's rows are computed by exactly one CTA either way, so both orders must produce
    // byte-identical output. 2048 rows x 24 heads (Bq=128 band); a reversal off-by-one
    // leaves a tile's rows unwritten.
    void run_identity(const Fa2Case& c) {
        const int B = c.B, Sq = c.Sq, Skv = c.Skv, NH = c.NH, NKV = c.NKV, HD = c.HD;
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
            bool ok = fmha_sm120_fa2_prefill(Qt, Kt, Vt, Ot, scale, c.causal, 0, 0.0f, stream_, c.q_offset);
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

    cudaStream_t stream_ = nullptr;
};

TEST_P(FmhaFA2Prefill, MatchesReference) {
    const Fa2Case& c = GetParam();
    const float tol = apply_variant(c.v);
    if (c.v == Variant::kHeavyFirst)
        run_identity(c);
    else
        run_fa2(c, tol);
}

using V = Variant;
// clang-format off
const Fa2Case kCases[] = {
    // FP16-QK (mma.m16n8k16.f16): the short-prefill replacement for materialized cuBLAS. The
    // realistic-magnitude cases are exactly where fp8 QK lost mantissa bits (#511).
    {"FmhaFA2Test_FP16QK_CausalShortSeq24_GQA32_8", V::kDefault, 1, 24, 24, 32, 8, 128, true},
    {"FmhaFA2Test_FP16QK_CausalSeq64", V::kDefault, 1, 64, 64, 4, 4, 128, true},
    {"FmhaFA2Test_FP16QK_CausalOddSeq51", V::kDefault, 1, 51, 51, 4, 4, 128, true},
    {"FmhaFA2Test_FP16QK_CausalOddSeq136_GQA32_8", V::kDefault, 1, 136, 136, 32, 8, 128, true},
    {"FmhaFA2Test_FP16QK_RealisticMagnitude_Seq24", V::kDefault, 1, 24, 24, 32, 8, 128, true, 0, 0.0f, 80.0f},
    {"FmhaFA2Test_FP16QK_RealisticMagnitude_Seq64", V::kDefault, 1, 64, 64, 4, 4, 128, true, 0, 0.0f, 80.0f},
    {"FmhaFA2Test_FP16QK_NonCausal", V::kDefault, 1, 32, 64, 4, 4, 128, false},
    {"FmhaFA2Test_FP16QK_GQA", V::kDefault, 1, 64, 64, 8, 2, 128, true},
    {"FmhaFA2Test_FP16QK_MultiTile", V::kDefault, 1, 128, 128, 4, 4, 128, true},
    {"FmhaFA2Test_FP16QK_LongCtx", V::kDefault, 1, 256, 256, 8, 2, 128, true},
    {"FmhaFA2Test_FP16QK_SlidingWindow", V::kDefault, 1, 128, 128, 4, 4, 128, true, 64},
    {"FmhaFA2Test_FP16QK_Softcap", V::kDefault, 1, 64, 64, 4, 4, 128, true, 0, 50.0f},
    // Chunk continuation (q_offset>0), #548: PR #553 measured wrong attention on Llama-3.2-3B
    // chunk continuations (teacher-forced NLL 0.29->7.13 at chunk=64) while Qwen3-4B was
    // bit-exact through the same kernel. Llama's GQA ratio 3 (24Q/8KV) vs Qwen3's 4 (32/8),
    // non-tile-multiple KV lengths, offsets not Bq/Bkv multiples.
    {"FmhaFA2Test_FP16QK_Chunked_GQA4", V::kDefault, 1, 64, 512, 32, 8, 128, true, 0, 0.0f, 1.0f, 448},
    {"FmhaFA2Test_FP16QK_Chunked_GQA3_LlamaShape", V::kDefault, 1, 64, 512, 24, 8, 128, true, 0, 0.0f, 1.0f, 448},
    {"FmhaFA2Test_FP16QK_Chunked_GQA3_OddKv", V::kDefault, 1, 51, 371, 24, 8, 128, true, 0, 0.0f, 1.0f, 320},
    {"FmhaFA2Test_FP16QK_Chunked_OffsetNotTileMultiple", V::kDefault, 1, 64, 292, 24, 8, 128, true, 0, 0.0f, 1.0f, 228},
    {"FmhaFA2Test_FP16QK_Chunked_LongCtx", V::kDefault, 1, 128, 3200, 24, 8, 128, true, 0, 0.0f, 1.0f, 3072},
    // Bq=64/Bkv=32 occupancy band (#597): blocks_128 = ceil(Sq/128)*NH in [sm_count/2, sm_count)
    // selects the 2-CTA/SM Bkv=32 kernel - 96 on the 170-SM RTX 5090.
    {"FmhaFA2Test_FP16QK_Bkv32Band_CausalMultiTile", V::kDefault, 1, 384, 384, 32, 8, 128, true},
    {"FmhaFA2Test_FP16QK_Bkv32Band_OddSeq", V::kDefault, 1, 333, 333, 32, 8, 128, true},
    {"FmhaFA2Test_FP16QK_Bkv32Band_Chunked", V::kDefault, 1, 384, 1408, 32, 8, 128, true, 0, 0.0f, 1.0f, 1024},
    {"FmhaFA2Test_FP16QK_Bkv32Band_SlidingWindow", V::kDefault, 1, 384, 384, 32, 8, 128, true, 64},
    // process_diag_fa2_dense_2cta: Bq=128 TWOSLOT pinned to 128 registers, two CTAs per SM;
    // 1024 x 24 heads = 192 CTAs reach the Bq=128 band.
    {"FmhaFA2Dense2CtaTest_Causal1024_24Q4KV", V::kDense2Cta, 1, 1024, 1024, 24, 4, 128, true},
    {"FmhaFA2Dense2CtaTest_OddSeq1000", V::kDense2Cta, 1, 1000, 1000, 24, 4, 128, true},
    {"FmhaFA2Dense2CtaTest_SlidingWindow", V::kDense2Cta, 1, 1024, 1024, 24, 4, 128, true, 300},
    {"FmhaFA2Dense2CtaTest_Chunked", V::kDense2Cta, 1, 1024, 2048, 24, 4, 128, true, 0, 0.0f, 1.0f, 1024},
    {"FmhaFA2Dense2CtaTest_RealisticMagnitude", V::kDense2Cta, 1, 1024, 1024, 24, 4, 128, true, 0, 0.0f, 80.0f},
    {"FmhaFA2HeavyFirstTest_Causal2048_24Q4KV_BitIdentical", V::kHeavyFirst, 1, 2048, 2048, 24, 4, 128, true},
    {"FmhaFA2HeavyFirstTest_Chunked1024of2048_BitIdentical", V::kHeavyFirst, 1, 1024, 2048, 24, 4, 128, true, 0, 0.0f, 1.0f, 1024},
    {"FmhaFA2HeavyFirstTest_OddSeq1000_BitIdentical", V::kHeavyFirst, 1, 1000, 1000, 24, 4, 128, true},
    // PV f16-accumulate: O accumulates in f16 across KV tiles (the rescale-and-add rounding this
    // knob trades for full-rate HMMA); production gate is teacher-forced PPL, this pins math/layout.
    {"FmhaFA2PvF16Test_CausalSeq64", V::kPvF16, 1, 64, 64, 4, 4, 128, true},
    {"FmhaFA2PvF16Test_CausalShortSeq24_GQA32_8", V::kPvF16, 1, 24, 24, 32, 8, 128, true},
    {"FmhaFA2PvF16Test_CausalOddSeq51", V::kPvF16, 1, 51, 51, 4, 4, 128, true},
    {"FmhaFA2PvF16Test_RealisticMagnitude_Seq64", V::kPvF16, 1, 64, 64, 4, 4, 128, true, 0, 0.0f, 80.0f},
    {"FmhaFA2PvF16Test_NonCausal", V::kPvF16, 1, 32, 64, 4, 4, 128, false},
    {"FmhaFA2PvF16Test_GQA", V::kPvF16, 1, 64, 64, 8, 2, 128, true},
    {"FmhaFA2PvF16Test_MultiTile", V::kPvF16, 1, 128, 128, 4, 4, 128, true},
    {"FmhaFA2PvF16Test_LongCtx", V::kPvF16, 1, 256, 256, 8, 2, 128, true},
    {"FmhaFA2PvF16Test_SlidingWindow", V::kPvF16, 1, 128, 128, 4, 4, 128, true, 64},
    {"FmhaFA2PvF16Test_Softcap", V::kPvF16, 1, 64, 64, 4, 4, 128, true, 0, 50.0f},
    // chunk continuation + the two occupancy bands (Bq=64 twoslot / Bq=128)
    {"FmhaFA2PvF16Test_Chunked_GQA3_OddKv", V::kPvF16, 1, 51, 371, 24, 8, 128, true, 0, 0.0f, 1.0f, 320},
    {"FmhaFA2PvF16Test_Bkv32Band_CausalMultiTile", V::kPvF16, 1, 384, 384, 32, 8, 128, true},
    {"FmhaFA2PvF16Test_Bq128Band_LongCtx", V::kPvF16, 1, 768, 768, 32, 8, 128, true},
    // HD=256 FA2 port (process_diag_fa2_hd256): fp16-qk only, Bq=64/Bkv=64/TWOSLOT, Qwen3.6
    // hybrid geometry (hd=256, kv_heads=2, GQA 4:1).
    {"FmhaFA2Hd256Test_PvF16_CausalSeq64_GQA8_2", V::kHd256PvF16, 1, 64, 64, 8, 2, 256, true},
    {"FmhaFA2Hd256Test_PvF16_CausalMultiTile", V::kHd256PvF16, 1, 256, 256, 8, 2, 256, true},
    {"FmhaFA2Hd256Test_PvF16_OddSeq51", V::kHd256PvF16, 1, 51, 51, 8, 2, 256, true},
    {"FmhaFA2Hd256Test_PvF16_OddSeq200_GQA16_4", V::kHd256PvF16, 1, 200, 200, 16, 4, 256, true},
    {"FmhaFA2Hd256Test_PvF16_NonCausal", V::kHd256PvF16, 1, 32, 64, 8, 2, 256, false},
    {"FmhaFA2Hd256Test_PvF16_RealisticMagnitude", V::kHd256PvF16, 1, 64, 64, 8, 2, 256, true, 0, 0.0f, 80.0f},
    {"FmhaFA2Hd256Test_PvF16_SlidingWindow", V::kHd256PvF16, 1, 128, 128, 8, 2, 256, true, 64},
    {"FmhaFA2Hd256Test_PvF16_Softcap", V::kHd256PvF16, 1, 64, 64, 8, 2, 256, true, 0, 50.0f},
    {"FmhaFA2Hd256Test_PvF16_Chunked", V::kHd256PvF16, 1, 51, 371, 8, 2, 256, true, 0, 0.0f, 1.0f, 320},
    {"FmhaFA2Hd256Test_PvF16_LongCtx1536", V::kHd256PvF16, 1, 1536, 1536, 8, 2, 256, true},
    {"FmhaFA2Hd256Test_F32Acc_CausalSeq64", V::kHd256F32, 1, 64, 64, 8, 2, 256, true},
    {"FmhaFA2Hd256Test_F16Acc_CausalMultiTile", V::kHd256F16Qk, 1, 200, 200, 8, 2, 256, true},
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(Shapes, FmhaFA2Prefill, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<Fa2Case>& info) {
                             return std::string(info.param.name);
                         });

}  // namespace
}  // namespace imp
