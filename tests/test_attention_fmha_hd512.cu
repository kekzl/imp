// head_dim=512 prefill parity — WMMA FMHA (fmha_sm120) vs the cuBLAS materialized
// reference, both anchored to an fp64 CPU reference.
//
// hd=512 is the Gemma-4 global-attention layer geometry (and Qwen3.5-27B, see
// attention_cublas.cu:423). Before this test the tiled WMMA FMHA declined
// hd=512 and every hd=512 layer fell to the materialized cuBLAS path. The
// dispatch closes that gap with a Bq=16/Bkv=16 instantiation; this test is the
// validator's parity gate: the new fused kernel must (a) actually run at hd=512
// (not silently decline) and (b) match the trusted cuBLAS reference and an fp64
// reference within the f16 numerical class.
//
// Math mirrors attention_cublas_prefill exactly:
//   S = scale * (Q·K^T)  →  (softcap>0: S = softcap*tanh(S/softcap))
//   → mask (causal: j>abs_row; SWA: abs_row-j>=window; abs_row=q_offset+row)
//   → softmax → O = P·V.
//
// Inputs are heavy-tailed realistic magnitudes and rounded to f16 on the host so
// both device paths consume bit-identical inputs; the fp64 reference is computed
// from the same f16-rounded values.

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

namespace imp {
namespace {

// Integer LCG → f32 with multiply-only transforms (no libm), cubed to make a
// heavy tail, then rounded to f16 — the QK-normed-activation regime that broke
// benign-fill parity tests historically (#493/#512).
void lcg_fill(std::vector<half>& out, uint32_t seed, float amp) {
    uint32_t s = seed * 2654435761u + 1013904223u;
    for (auto& h : out) {
        s = s * 1664525u + 1013904223u;
        float u = ((s >> 8) & 0xFFFFFF) * (1.0f / 16777216.0f);  // [0,1)
        float c = (2.0f * u - 1.0f);
        c = c * c * c;  // cubed: heavy tail
        float v = c * amp;
        if (((s >> 4) & 0xFF) == 0)
            v *= 4.0f;  // ~1/256 outliers
        h = __float2half(v);
    }
}

// fp64 reference from f16-rounded inputs. Layout: Q[Sq,NH,HD], K/V[Skv,NKV,HD].
void ref_attention_f64(const std::vector<half>& Q, const std::vector<half>& K, const std::vector<half>& V,
                       std::vector<double>& O, int Sq, int Skv, int NH, int NKV, int HD, bool causal,
                       int sw, float softcap, float scale, int q_offset) {
    O.assign((size_t)Sq * NH * HD, 0.0);
    const int group = NH / NKV;
    for (int h = 0; h < NH; h++) {
        const int kvh = h / group;
        for (int i = 0; i < Sq; i++) {
            const int abs_row = q_offset + i;
            std::vector<double> s(Skv, 0.0);
            double m = -1e300;
            for (int j = 0; j < Skv; j++) {
                double dot = 0.0;
                for (int d = 0; d < HD; d++) {
                    dot += (double)__half2float(Q[((size_t)i * NH + h) * HD + d]) *
                           (double)__half2float(K[((size_t)j * NKV + kvh) * HD + d]);
                }
                double sc = dot * (double)scale;
                if (softcap > 0.0f)
                    sc = (double)softcap * std::tanh(sc / (double)softcap);
                bool masked = (causal && j > abs_row) || (sw > 0 && (abs_row - j) >= sw);
                s[j] = masked ? -1e300 : sc;
                if (s[j] > m)
                    m = s[j];
            }
            double denom = 0.0;
            for (int j = 0; j < Skv; j++) {
                s[j] = (s[j] <= -1e299) ? 0.0 : std::exp(s[j] - m);
                denom += s[j];
            }
            double inv = denom > 0.0 ? 1.0 / denom : 0.0;
            for (int d = 0; d < HD; d++) {
                double acc = 0.0;
                for (int j = 0; j < Skv; j++)
                    acc += s[j] * inv * (double)__half2float(V[((size_t)j * NKV + kvh) * HD + d]);
                O[((size_t)i * NH + h) * HD + d] = acc;
            }
        }
    }
}

struct Metrics {
    double max_rel = 0.0;
    double mean_rel = 0.0;
};

Metrics compare(const std::vector<float>& got, const std::vector<double>& ref) {
    Metrics m;
    double sum = 0.0;
    size_t n = ref.size();
    for (size_t i = 0; i < n; i++) {
        double denom = std::max(1e-2, std::fabs(ref[i]));  // abs floor: tiny-value stability
        double r = std::fabs((double)got[i] - ref[i]) / denom;
        m.max_rel = std::max(m.max_rel, r);
        sum += r;
    }
    m.mean_rel = sum / (double)n;
    return m;
}

class FmhaHd512Test : public ::testing::Test {
  protected:
    void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;

    // Runs one config through cuBLAS + WMMA FMHA at hd=512 and checks parity.
    void run(const std::string& name, int Sq, int Skv, int NH, int NKV, bool causal, int sw, float softcap,
             int q_offset) {
        const int HD = 512;
        const float scale = 1.0f / std::sqrt((float)HD);
        const size_t q_elems = (size_t)Sq * NH * HD;
        const size_t kv_elems = (size_t)Skv * NKV * HD;

        std::vector<half> Qh(q_elems), Kh(kv_elems), Vh(kv_elems);
        lcg_fill(Qh, 0x1111u, 4.0f);
        lcg_fill(Kh, 0x2222u, 4.0f);
        lcg_fill(Vh, 0x3333u, 1.0f);

        std::vector<double> ref;
        ref_attention_f64(Qh, Kh, Vh, ref, Sq, Skv, NH, NKV, HD, causal, sw, softcap, scale, q_offset);

        void *d_q, *d_k, *d_v, *d_o, *d_s;
        // 4x so cuBLAS takes the production FP32-S path: use_fp32_s needs the
        // score buffer >= 3x the (NH*Sq*Skv) element count (#677 — FP32 scores
        // in the front 2x, non-overlapping FP16 probs after). At hd=512 the
        // FP16-S path truncates the large QK^T scores and is materially less
        // accurate; Gemma-4 / Qwen3.5-27B run FP32-S in production.
        const size_t s_elems = (size_t)4 * NH * Sq * Skv;
        ASSERT_EQ(cudaMalloc(&d_q, q_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_k, kv_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_v, kv_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_o, q_elems * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_s, s_elems * 2), cudaSuccess);
        cudaMemcpy(d_q, Qh.data(), q_elems * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, Kh.data(), kv_elems * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, Vh.data(), kv_elems * 2, cudaMemcpyHostToDevice);

        int64_t q2[2] = {Sq, (int64_t)NH * HD};
        int64_t kv2[2] = {Skv, (int64_t)NKV * HD};
        int64_t s3[3] = {NH, Sq, 4 * (int64_t)Skv};
        int64_t q4[4] = {1, Sq, NH, HD};
        int64_t kv4[4] = {1, Skv, NKV, HD};
        Tensor Q2(d_q, QType::F16, 2, q2, true), K2(d_k, QType::F16, 2, kv2, true);
        Tensor V2(d_v, QType::F16, 2, kv2, true), O2(d_o, QType::F16, 2, q2, true);
        Tensor S3(d_s, QType::F16, 3, s3, true);
        Tensor Q4(d_q, QType::F16, 4, q4, true), K4(d_k, QType::F16, 4, kv4, true);
        Tensor V4(d_v, QType::F16, 4, kv4, true), O4(d_o, QType::F16, 4, q4, true);

        auto collect = [&]() {
            cudaStreamSynchronize(stream_);
            EXPECT_EQ(cudaGetLastError(), cudaSuccess) << name;
            std::vector<half> h(q_elems);
            cudaMemcpy(h.data(), d_o, q_elems * 2, cudaMemcpyDeviceToHost);
            std::vector<float> out(q_elems);
            for (size_t i = 0; i < q_elems; i++)
                out[i] = __half2float(h[i]);
            return out;
        };

        // cuBLAS materialized reference (the legacy pre-change path).
        cudaMemset(d_o, 0, q_elems * 2);
        attention_cublas_prefill(Q2, K2, V2, O2, S3, NH, NKV, HD, scale, causal, softcap, q_offset, stream_,
                                 sw);
        std::vector<float> o_cublas = collect();

        // WMMA FMHA — must actually run at hd=512 (the coverage gate).
        cudaMemset(d_o, 0, q_elems * 2);
        bool ran =
            fmha_sm120_prefill(Q4, K4, V4, O4, scale, causal, sw, softcap, stream_, q_offset, nullptr);
        ASSERT_TRUE(ran) << name << ": fmha_sm120_prefill declined hd=512 — the kernel must serve it";
        std::vector<float> o_fmha = collect();

        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_s);

        // The fp64 eager computation is the trusted reference. cuBLAS FP32-S is
        // the pre-change legacy path (an independent f16-class approximation).
        Metrics mf = compare(o_fmha, ref);      // FMHA vs trusted fp64 — correctness
        Metrics mc = compare(o_cublas, ref);    // legacy cuBLAS vs trusted fp64
        std::vector<double> cublas_d(o_cublas.begin(), o_cublas.end());
        Metrics mfc = compare(o_fmha, cublas_d);  // FMHA vs the legacy output

        printf("[hd512 %-18s] fmha_vs_ref max=%.2e mean=%.2e | cublas_vs_ref max=%.2e mean=%.2e | "
               "fmha_vs_cublas max=%.2e\n",
               name.c_str(), mf.max_rel, mf.mean_rel, mc.max_rel, mc.mean_rel, mfc.max_rel);

        // Correctness bar = the f16 numerical class vs the trusted fp64 reference
        // (a 512-long reduction rounds harder than HD<=256). The fused hd=512
        // kernel is the O(n) fallback for long-context S-matrix overflow, where
        // cuBLAS cannot run at all — so "no worse than cuBLAS" is not a meaningful
        // production bar (there is no cuBLAS to compare against there). Both paths
        // must simply land in the f16 class. The Bkv=32 tile trades ~0.9e-2 of
        // accuracy on rect+offset shapes for +40% throughput (both < 2.5e-2).
        // Thresholds frozen, not to be widened.
        EXPECT_LT(mf.max_rel, 2.5e-2) << name << ": FMHA hd=512 off vs the fp64 reference";
        EXPECT_LT(mf.mean_rel, 1e-3) << name << ": FMHA hd=512 mean error too high vs fp64";
        EXPECT_LT(mc.max_rel, 2.5e-2) << name << ": cuBLAS FP32-S reference itself off vs fp64";
    }
};

// Gemma-4 global-layer geometry: full causal attention, GQA, softcap.
TEST_F(FmhaHd512Test, Causal_GQA) { run("causal_gqa", 96, 96, 8, 4, true, 0, 0.0f, 0); }
TEST_F(FmhaHd512Test, Causal_GQA_Softcap) { run("causal_gqa_softcap", 96, 96, 8, 4, true, 0, 50.0f, 0); }
TEST_F(FmhaHd512Test, Causal_MHA) { run("causal_mha", 64, 64, 4, 4, true, 0, 0.0f, 0); }
// Non-square (chunked-continuation shape) with q_offset.
TEST_F(FmhaHd512Test, Rect_Offset) { run("rect_offset", 32, 160, 8, 2, true, 0, 0.0f, 128); }
// Tile-boundary edge: Skv just over a Bkv=16 tile; Sq over a Bq=16 tile.
TEST_F(FmhaHd512Test, TileEdge) { run("tile_edge", 17, 33, 8, 4, true, 0, 0.0f, 0); }
// Sliding window (kernel generality — hd=512 production is full-attention).
TEST_F(FmhaHd512Test, SlidingWindow) { run("sliding_window", 96, 96, 8, 4, true, 48, 0.0f, 0); }

// Isolated kernel A/B: the WMMA FMHA hd=512 vs the materialized cuBLAS FP32-S
// path it replaces, at Gemma-4 global-layer shapes (nh=16, nkv=8, hd=512).
// This isolates the per-shape prefill-attention win (the whole-model Gemma-4
// prefill is MoE-dequant-dominated and hd=512 is only 1/6 of attention layers,
// so an end-to-end delta would sit in prefill restart noise). DISABLED so the
// normal suite stays fast; run with --gtest_also_run_disabled_tests.
// Clock warmup >1s first (RTX 5090 idles downclocked, ~1s ramp — CLAUDE.md).
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
                    fmha_sm120_prefill(Q4, K4, V4, O4, scale, true, 0, 0.0f, stream_, 0, nullptr);
                else
                    attention_cublas_prefill(Q2, K2, V2, O2, S3, NH, NKV, HD, scale, true, 0.0f, 0, stream_);
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
            fmha_sm120_prefill(Q4, K4, V4, O4, scale, true, 0, 0.0f, stream_, 0, nullptr);
            attention_cublas_prefill(Q2, K2, V2, O2, S3, NH, NKV, HD, scale, true, 0.0f, 0, stream_);
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

// Parity of the q-row-sliced cuBLAS path (attention_cublas_prefill_sliced) —
// the S-overflow production route for hd=512 layers. The S buffer is sized so
// the FP32-S 3× rule forces 32-row slices (3 slices over Sq=96); the result
// must match the fp64 reference within the f16 class and track the whole-call
// FP32-S path (independent slicing of the same math).
TEST_F(FmhaHd512Test, SlicedCublasParity) {
    const int HD = 512, NH = 8, NKV = 4, Sq = 96, Skv = 160, q_offset = 64;
    const float scale = 1.0f / std::sqrt((float)HD);
    const float softcap = 50.0f;
    const size_t q_elems = (size_t)Sq * NH * HD, kv_elems = (size_t)Skv * NKV * HD;
    std::vector<half> Qh(q_elems), Kh(kv_elems), Vh(kv_elems);
    lcg_fill(Qh, 0x1111u, 4.0f);
    lcg_fill(Kh, 0x2222u, 4.0f);
    lcg_fill(Vh, 0x3333u, 1.0f);
    std::vector<double> ref;
    ref_attention_f64(Qh, Kh, Vh, ref, Sq, Skv, NH, NKV, HD, /*causal=*/true, /*sw=*/0, softcap, scale,
                      q_offset);

    void *d_q, *d_k, *d_v, *d_o, *d_s;
    // Sized to force ns=32: ns = s_elems / (3*NH*Skv) = 122880 / 3840 = 32.
    const size_t s_elems = (size_t)NH * Sq * Skv;
    ASSERT_EQ(cudaMalloc(&d_q, q_elems * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_k, kv_elems * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_v, kv_elems * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_o, q_elems * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, s_elems * 2 * 4), cudaSuccess);  // 4x: also holds the whole-call FP32-S run
    cudaMemcpy(d_q, Qh.data(), q_elems * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, Kh.data(), kv_elems * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, Vh.data(), kv_elems * 2, cudaMemcpyHostToDevice);

    int64_t q2[2] = {Sq, (int64_t)NH * HD}, kv2[2] = {Skv, (int64_t)NKV * HD};
    int64_t s3_small[3] = {NH, Sq, Skv};      // forces 32-row slices via the 3x FP32-S rule
    int64_t s3_big[3] = {NH, Sq, 4 * (int64_t)Skv};  // whole-call FP32-S reference
    Tensor Q2(d_q, QType::F16, 2, q2, true), K2(d_k, QType::F16, 2, kv2, true);
    Tensor V2(d_v, QType::F16, 2, kv2, true), O2(d_o, QType::F16, 2, q2, true);
    Tensor S_small(d_s, QType::F16, 3, s3_small, true);
    Tensor S_big(d_s, QType::F16, 3, s3_big, true);

    auto collect = [&]() {
        cudaStreamSynchronize(stream_);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        std::vector<half> h(q_elems);
        cudaMemcpy(h.data(), d_o, q_elems * 2, cudaMemcpyDeviceToHost);
        std::vector<float> out(q_elems);
        for (size_t i = 0; i < q_elems; i++)
            out[i] = __half2float(h[i]);
        return out;
    };

    cudaMemset(d_o, 0, q_elems * 2);
    attention_cublas_prefill(Q2, K2, V2, O2, S_big, NH, NKV, HD, scale, /*causal=*/true, softcap,
                             q_offset, stream_);
    std::vector<float> o_whole = collect();

    cudaMemset(d_o, 0, q_elems * 2);
    bool ran = attention_cublas_prefill_sliced(Q2, K2, V2, O2, S_small, NH, NKV, HD, scale,
                                               /*causal=*/true, softcap, q_offset, stream_);
    ASSERT_TRUE(ran) << "sliced path declined although a 32-row slice fits";
    std::vector<float> o_sliced = collect();

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_s);

    Metrics ms = compare(o_sliced, ref);
    Metrics mw = compare(o_whole, ref);
    std::vector<double> whole_d(o_whole.begin(), o_whole.end());
    Metrics msw = compare(o_sliced, whole_d);
    printf("[hd512 sliced_parity   ] sliced_vs_ref max=%.2e mean=%.2e | whole_vs_ref max=%.2e mean=%.2e "
           "| sliced_vs_whole max=%.2e\n",
           ms.max_rel, ms.mean_rel, mw.max_rel, mw.mean_rel, msw.max_rel);
    EXPECT_LT(ms.max_rel, 2.5e-2) << "sliced cuBLAS off vs the fp64 reference";
    EXPECT_LT(ms.mean_rel, 1e-3) << "sliced cuBLAS mean error too high vs fp64";
    // Same math, same FP32-S class — but slicing changes the GEMM batch
    // geometry, so the runs differ in fp16-P rounding (measured 1.2e-2, the
    // same mutual-f16 band as fmha_vs_cublas above). Gate at the class bound.
    EXPECT_LT(msw.max_rel, 2.5e-2) << "sliced diverges from the whole-call FP32-S path";
}

// Long-context fallback-regime bench: continuation chunk Sq=2048 against a long
// KV (Skv 8k/16k, q_offset = Skv - Sq). This is the shape the fused hd=512
// kernel actually serves in production — the materialized S-matrix overflows
// the workspace there, so (pre-dispatch) the executor had to run cuBLAS in thin
// s_cap-respecting row slices. Arms: FMHA whole-chunk vs cuBLAS in 256-row
// slices (the realistic alternative at this regime). DISABLED so the normal
// suite stays fast; run with --gtest_also_run_disabled_tests.
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
            bool ran =
                fmha_sm120_prefill(Q4, K4, V4, O4, scale, true, 0, 0.0f, stream_, q_offset, nullptr);
            ASSERT_TRUE(ran);
        };
        auto run_cublas_sliced = [&](int slice) {
            for (int off = 0; off < Sq; off += slice) {
                int64_t q2s[2] = {slice, (int64_t)NH * HD};
                Tensor Q2s(static_cast<half*>(d_q) + (size_t)off * NH * HD, QType::F16, 2, q2s, true);
                Tensor O2s(static_cast<half*>(d_o) + (size_t)off * NH * HD, QType::F16, 2, q2s, true);
                attention_cublas_prefill(Q2s, K2, V2, O2s, S3, NH, NKV, HD, scale, /*causal=*/true,
                                         /*softcap=*/0.0f, q_offset + off, stream_);
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
            printf("[hd512 longctx Sq=%d Skv=%-5d] cublas_sliced(%3d) %.3f ms | fmha/cublas %.2fx\n", Sq,
                   Skv, slice, best_c, best_c / best_f);
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
