// Paged decode attention quant variants vs fp64 reference — TEST_AUDIT.md
// risk #6 (the decode twin of risk #1 / the cross-path test).
//
// The decode hot path has SIX paged-attention implementations, one per KV
// dtype (kv_cache.dtype in src/runtime/config.h): F16, FP8-E4M3, INT8, INT4,
// NVFP4 (scalar), NVFP4-TC (tensor-core Q.K). Each serves EVERY decoded token
// when its KV mode is active, yet the only prior numeric oracle was
// test_paged_attention.cu, whose quant tests (INT4) build their CPU reference
// FROM THE DEQUANTIZED values — i.e. imp-vs-imp on the quant grid (class B
// tautology: it cannot see a quant kernel that is wrong, only one that is
// inconsistent with its own dequant). And those used benign sin/cos fills.
//
// This test fixes both:
//   * Ground truth is an fp64 single-query attention computed FROM THE ORIGINAL
//     f16 K/V (the bits the F16 kernel sees), NOT from the quantized grid.
//   * For the quant paths the K/V are host-quantized into each kernel's EXACT
//     cache layout, then run; their deviation from the original-f16 fp64
//     reference IS the quantization error, which we CHARACTERIZE (measured
//     envelope + printed stats), never bless at an unmeetable tolerance. Only
//     the F16 path is held strict (no score/value quantization at all).
//   * Data is the realistic heavy-tailed LCG regime (tests/refs/README.md §3,
//     amp=2 K/Q "mild" class, amp=1 V), the same recipe as the cross-path
//     golden — multiply-only f32 transforms so it is reproducible and free of
//     the periodic-%13 / size_t-underflow vacuity (#525).
//   * kv_len sweep {16, 64, 333, 1024}: 333 is deliberately NOT block-aligned
//     (block_size=16) to exercise the partial-tail block; 16/64 are the short
//     rows where quant score noise has no averaging to hide behind (#512),
//     1024 is the long-context dilution case.
//   * Hard no-NaN/Inf guard on EVERY path (the actual decode-corruption assert;
//     the existing NVFP4-TC test documents that synthetic random-byte NVFP4
//     input drives the scalar kernel to all-NaN — so a CORRECT host quantize
//     is mandatory, and we assert it stays finite).
//
// Tolerances (tests/refs/README.md §2): F16 ≤ 1e-2 rel vs fp64 (f16-rounded
// inputs over a hd-term dot + f16 P/V). Quant paths: characterized envelopes,
// measured on first run and frozen below with ~50% margin (dated). NVFP4's
// 1e-1 single-op class is a floor expectation, not the envelope — paged KV
// quant on short uncorrelated rows is looser, exactly the risk-#6 finding.

#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace imp {
namespace {

static constexpr int BLOCK_SIZE = 16;  // kKVBlockSize

// ---------------------------------------------------------------------------
// LCG fill — bit-exact mirror of tests/refs/gen_attention_crosspath_golden.py
// ::lcg_fill (f32 multiply-only, no libm). Heavy-tailed cubed-uniform, amp
// scales the envelope, 1/256 outliers at 2x. Produces identical f16 bits in
// any language; here it only needs to be self-consistent (the fp64 ref is
// computed from the same f16 values), so no committed golden is required —
// the independence comes from the fp64-vs-quant-grid separation, not from a
// numpy cross-check.
// ---------------------------------------------------------------------------
void lcg_fill(std::vector<half>& out, uint32_t seed, float amp) {
    uint32_t x = seed;
    for (auto& h : out) {
        x = x * 1664525u + 1013904223u;
        int32_t v = (int32_t)((x >> 8) & 0x3FFFu) - 8192;
        float f = (float)v * (1.0f / 8192.0f);
        float val = f * f * f * amp;
        if ((x & 0xFFu) == 0)
            val *= 2.0f;
        h = __float2half(val);
    }
}

// ---------------------------------------------------------------------------
// fp64 single-query attention reference, computed from the ORIGINAL f16 K/V.
// Q: [n_heads, head_dim]; K/V flat per kv_head: [kv_len, n_kv_heads, head_dim].
// O: [n_heads, head_dim]. GQA mapping kvh = h / (n_heads/n_kv_heads).
// ---------------------------------------------------------------------------
void ref_decode_f64(const std::vector<half>& Qh, const std::vector<half>& Kh, const std::vector<half>& Vh,
                    std::vector<double>& O, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                    float scale) {
    const int gqa = n_heads / n_kv_heads;
    O.assign((size_t)n_heads * head_dim, 0.0);
    std::vector<double> S(kv_len);
    for (int h = 0; h < n_heads; h++) {
        int kvh = h / gqa;
        double m = -1e300;
        for (int j = 0; j < kv_len; j++) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; d++) {
                dot += (double)__half2float(Qh[(size_t)h * head_dim + d]) *
                       (double)__half2float(Kh[((size_t)j * n_kv_heads + kvh) * head_dim + d]);
            }
            dot *= (double)scale;
            S[j] = dot;
            m = std::max(m, dot);
        }
        double l = 0.0;
        for (int j = 0; j < kv_len; j++) {
            S[j] = std::exp(S[j] - m);
            l += S[j];
        }
        for (int d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (int j = 0; j < kv_len; j++)
                acc += S[j] * (double)__half2float(Vh[((size_t)j * n_kv_heads + kvh) * head_dim + d]);
            O[(size_t)h * head_dim + d] = acc / l;
        }
    }
}

// ---------------------------------------------------------------------------
// Error statistics vs the fp64 reference. denom floored at 1 (outputs are
// O(1)-bounded softmax-weighted V averages, so absolute≈relative on the bulk
// and tiny refs don't explode the metric).
// ---------------------------------------------------------------------------
struct ErrStats {
    float max_rel = 0.0f;
    float p999 = 0.0f;
    int nan_count = 0;
    std::string str() const {
        char buf[96];
        snprintf(buf, sizeof(buf), "max_rel=%.4g p99.9=%.4g nan=%d", max_rel, p999, nan_count);
        return buf;
    }
};

ErrStats err_stats(const std::vector<float>& got, const std::vector<double>& ref) {
    ErrStats s;
    std::vector<float> errs;
    errs.reserve(got.size());
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i])) {
            s.nan_count++;
            continue;
        }
        double denom = std::max(1.0, std::fabs(ref[i]));
        float e = (float)(std::fabs((double)got[i] - ref[i]) / denom);
        errs.push_back(e);
        s.max_rel = std::max(s.max_rel, e);
    }
    if (!errs.empty()) {
        size_t k = (size_t)((double)(errs.size() - 1) * 0.999);
        std::nth_element(errs.begin(), errs.begin() + k, errs.end());
        s.p999 = errs[k];
    }
    return s;
}

// ---------------------------------------------------------------------------
// Quantizer helpers — each mirrors its kernel's dequant exactly, so the only
// error injected is the quantization itself (the thing being characterized).
// ---------------------------------------------------------------------------

// E2M1 (FP4) magnitude LUT + sign bit, matching cvt.rn.f16x2.e2m1x2.
constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Round a scaled magnitude to the nearest E2M1 code (round-to-nearest-even on
// ties is irrelevant here — the LUT gaps are wide; nearest-magnitude suffices).
uint8_t f_to_e2m1(float v) {
    float a = std::fabs(v);
    int best = 0;
    float bestd = 1e30f;
    for (int c = 0; c < 8; c++) {
        float d = std::fabs(a - kE2M1[c]);
        if (d < bestd) {
            bestd = d;
            best = c;
        }
    }
    uint8_t code = (uint8_t)best;
    if (v < 0.0f)
        code |= 0x8;
    return code;
}

// UE4M3 (FP8 E4M3, sign always 0) encode of a positive scale.
uint8_t f_to_ue4m3(float s) {
    __nv_fp8_e4m3 q = __nv_fp8_e4m3(s);
    uint8_t bits;
    memcpy(&bits, &q, 1);
    return bits;
}
float ue4m3_to_f(uint8_t b) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &b, 1);
    return (float)v;
}

// ---------------------------------------------------------------------------
// Build the F16 paged cache: [num_blocks, block_size, n_kv_heads, head_dim].
// Identity block table. Returns flat half buffer.
// ---------------------------------------------------------------------------
std::vector<half> build_f16_cache(const std::vector<half>& kv, int kv_len, int n_kv_heads, int head_dim,
                                  int num_blocks) {
    std::vector<half> cache((size_t)num_blocks * BLOCK_SIZE * n_kv_heads * head_dim, __float2half(0.0f));
    for (int s = 0; s < kv_len; s++) {
        int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
        for (int kvh = 0; kvh < n_kv_heads; kvh++) {
            size_t dst = ((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads * head_dim + (size_t)kvh * head_dim;
            size_t src = ((size_t)s * n_kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; d++)
                cache[dst + d] = kv[src + d];
        }
    }
    return cache;
}

// ---------------------------------------------------------------------------
// Device tensor helpers
// ---------------------------------------------------------------------------
void* up(const void* host, size_t bytes) {
    void* d = nullptr;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice);
    return d;
}

Tensor f16_tensor(void* d, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim = (int)shape.size();
    int i = 0;
    for (auto s : shape)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    t.data = d;
    return t;
}

Tensor raw_tensor(void* d, QType qt, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = qt;
    t.ndim = (int)shape.size();
    int i = 0;
    for (auto s : shape)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    t.data = d;
    return t;
}

std::vector<float> read_o(void* d_o, size_t elems) {
    std::vector<half> h(elems);
    cudaMemcpy(h.data(), d_o, elems * sizeof(half), cudaMemcpyDeviceToHost);
    std::vector<float> o(elems);
    for (size_t i = 0; i < elems; i++)
        o[i] = __half2float(h[i]);
    return o;
}

// ===========================================================================
// The fixture
// ===========================================================================
class PagedOracleTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;

    // Run all KV-dtype paths for one (config, kv_len) and assert/characterize.
    void run(const char* cfg_name, int kv_len, int n_heads, int n_kv_heads, int head_dim,
             // characterization envelopes (max_rel ceilings, frozen with margin)
             float env_fp8, float env_int8, float env_int4, float env_nvfp4) {
        char trace[160];
        snprintf(trace, sizeof(trace), "%s kv_len=%d nh=%d nkv=%d hd=%d", cfg_name, kv_len, n_heads,
                 n_kv_heads, head_dim);
        SCOPED_TRACE(trace);

        const float scale = 1.0f / std::sqrt((float)head_dim);
        const int num_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const size_t q_elems = (size_t)n_heads * head_dim;
        const size_t kv_elems = (size_t)kv_len * n_kv_heads * head_dim;

        // ---- inputs (seed varies per config+len so paths don't alias) ----
        const uint32_t seed = 0xC0DEu + (uint32_t)kv_len * 131u + (uint32_t)n_kv_heads * 17u;
        std::vector<half> Qh(q_elems), Kh(kv_elems), Vh(kv_elems);
        lcg_fill(Qh, seed + 1, 2.0f);
        lcg_fill(Kh, seed + 2, 2.0f);
        lcg_fill(Vh, seed + 3, 1.0f);  // V not QK-normed

        // ---- fp64 reference from the ORIGINAL f16 values ----
        std::vector<double> ref;
        ref_decode_f64(Qh, Kh, Vh, ref, kv_len, n_heads, n_kv_heads, head_dim, scale);

        // ---- block table (identity) + ctx_len, on device ----
        std::vector<int> bt(num_blocks);
        for (int i = 0; i < num_blocks; i++)
            bt[i] = i;
        int* d_bt = (int*)up(bt.data(), num_blocks * sizeof(int));
        int ctx = kv_len;
        int* d_ctx = (int*)up(&ctx, sizeof(int));

        void* d_q = up(Qh.data(), q_elems * sizeof(half));
        void* d_o = nullptr;
        cudaMalloc(&d_o, q_elems * sizeof(half));
        Tensor Q = f16_tensor(d_q, {1, 1, n_heads, head_dim});

        auto clear_o = [&] { cudaMemset(d_o, 0, q_elems * sizeof(half)); };

        // -------------------------------------------------------------------
        // 1. F16 paged — strict 1e-2 vs fp64 (no score/value quantization).
        // -------------------------------------------------------------------
        {
            auto Kc = build_f16_cache(Kh, kv_len, n_kv_heads, head_dim, num_blocks);
            auto Vc = build_f16_cache(Vh, kv_len, n_kv_heads, head_dim, num_blocks);
            void* d_k = up(Kc.data(), Kc.size() * sizeof(half));
            void* d_v = up(Vc.data(), Vc.size() * sizeof(half));
            Tensor K = f16_tensor(d_k, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
            Tensor V = f16_tensor(d_v, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
            Tensor O = f16_tensor(d_o, {1, 1, n_heads, head_dim});
            clear_o();
            paged_attention_decode(Q, K, V, O, d_bt, d_ctx, BLOCK_SIZE, scale, kv_len, 0, 0.0f, stream_,
                                   num_blocks);
            cudaStreamSynchronize(stream_);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "F16 paged launch";
            auto o = read_o(d_o, q_elems);
            ErrStats e = err_stats(o, ref);
            EXPECT_EQ(e.nan_count, 0) << "F16 paged: non-finite output";
            EXPECT_LT(e.max_rel, 1e-2f) << "F16 paged vs fp64 (must be strict — no quant): " << e.str();
            printf("[paged-oracle] %s F16:    %s\n", trace, e.str().c_str());
            cudaFree(d_k);
            cudaFree(d_v);
        }

        // -------------------------------------------------------------------
        // 2. FP8-E4M3 paged — per-tensor kv_scale, layout == F16 (1 byte).
        //    Dequant: val = e4m3(byte) * kv_scale. Quantize with a single
        //    tensor-wide scale (the kernel's contract).
        // -------------------------------------------------------------------
        {
            // per-tensor scale = absmax / 448 (e4m3 max), applied to both K and V
            float amax = 0.0f;
            for (auto& h : Kh)
                amax = std::max(amax, std::fabs(__half2float(h)));
            for (auto& h : Vh)
                amax = std::max(amax, std::fabs(__half2float(h)));
            float kv_scale = amax > 0 ? amax / 448.0f : 1.0f;
            float inv = 1.0f / kv_scale;
            auto quant = [&](const std::vector<half>& kv) {
                std::vector<uint8_t> out((size_t)num_blocks * BLOCK_SIZE * n_kv_heads * head_dim, 0);
                for (int s = 0; s < kv_len; s++) {
                    int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
                    for (int kvh = 0; kvh < n_kv_heads; kvh++) {
                        size_t dst =
                            ((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads * head_dim + (size_t)kvh * head_dim;
                        size_t src = ((size_t)s * n_kv_heads + kvh) * head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            __nv_fp8_e4m3 q = __nv_fp8_e4m3(__half2float(kv[src + d]) * inv);
                            memcpy(&out[dst + d], &q, 1);
                        }
                    }
                }
                return out;
            };
            auto Kq = quant(Kh), Vq = quant(Vh);
            void* d_k = up(Kq.data(), Kq.size());
            void* d_v = up(Vq.data(), Vq.size());
            Tensor K = raw_tensor(d_k, QType::FP8_E4M3, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
            Tensor V = raw_tensor(d_v, QType::FP8_E4M3, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
            Tensor O = f16_tensor(d_o, {1, 1, n_heads, head_dim});
            clear_o();
            paged_attention_decode_fp8(Q, K, V, O, d_bt, d_ctx, BLOCK_SIZE, scale, kv_scale, kv_len, 0, 0.0f,
                                       stream_, num_blocks);
            cudaStreamSynchronize(stream_);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "FP8 paged launch";
            auto o = read_o(d_o, q_elems);
            ErrStats e = err_stats(o, ref);
            EXPECT_EQ(e.nan_count, 0) << "FP8 paged: non-finite output (decode-corruption guard)";
            EXPECT_LT(e.max_rel, env_fp8) << "FP8 paged envelope exceeded: " << e.str();
            printf("[paged-oracle] %s FP8:    %s (env %.3g)\n", trace, e.str().c_str(), env_fp8);
            cudaFree(d_k);
            cudaFree(d_v);
        }

        // -------------------------------------------------------------------
        // 3. INT8 paged — per-(token,kv_head) FP16 scale; layout == F16
        //    (1 byte). Dequant: val = int8 * scale. (The kernel also quantizes
        //    Q internally — its q_scale tax is part of the characterized error.)
        // -------------------------------------------------------------------
        {
            const int scale_elems = num_blocks * BLOCK_SIZE * n_kv_heads;
            auto quant = [&](const std::vector<half>& kv, std::vector<half>& scales) {
                std::vector<int8_t> out((size_t)num_blocks * BLOCK_SIZE * n_kv_heads * head_dim, 0);
                scales.assign(scale_elems, __float2half(0.0f));
                for (int s = 0; s < kv_len; s++) {
                    int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
                    for (int kvh = 0; kvh < n_kv_heads; kvh++) {
                        size_t src = ((size_t)s * n_kv_heads + kvh) * head_dim;
                        float amax = 0.0f;
                        for (int d = 0; d < head_dim; d++)
                            amax = std::max(amax, std::fabs(__half2float(kv[src + d])));
                        float sc = amax > 0 ? amax / 127.0f : 1.0f;
                        float inv = 1.0f / sc;
                        scales[(blk * BLOCK_SIZE + slot) * n_kv_heads + kvh] = __float2half(sc);
                        size_t dst =
                            ((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads * head_dim + (size_t)kvh * head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            int q = (int)std::lround(__half2float(kv[src + d]) * inv);
                            q = std::max(-127, std::min(127, q));
                            out[dst + d] = (int8_t)q;
                        }
                    }
                }
                return out;
            };
            std::vector<half> ks, vs;
            auto Kq = quant(Kh, ks), Vq = quant(Vh, vs);
            void* d_k = up(Kq.data(), Kq.size());
            void* d_v = up(Vq.data(), Vq.size());
            void* d_ks = up(ks.data(), ks.size() * sizeof(half));
            void* d_vs = up(vs.data(), vs.size() * sizeof(half));
            Tensor K = raw_tensor(d_k, QType::INT8, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
            Tensor V = raw_tensor(d_v, QType::INT8, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
            Tensor O = f16_tensor(d_o, {1, 1, n_heads, head_dim});
            clear_o();
            paged_attention_decode_int8(Q, K, V, O, (const half*)d_ks, (const half*)d_vs, d_bt, d_ctx,
                                        BLOCK_SIZE, scale, kv_len, 0, 0.0f, stream_, num_blocks);
            cudaStreamSynchronize(stream_);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "INT8 paged launch";
            auto o = read_o(d_o, q_elems);
            ErrStats e = err_stats(o, ref);
            EXPECT_EQ(e.nan_count, 0) << "INT8 paged: non-finite output";
            EXPECT_LT(e.max_rel, env_int8) << "INT8 paged envelope exceeded: " << e.str();
            printf("[paged-oracle] %s INT8:   %s (env %.3g)\n", trace, e.str().c_str(), env_int8);
            cudaFree(d_k);
            cudaFree(d_v);
            cudaFree(d_ks);
            cudaFree(d_vs);
        }

        // -------------------------------------------------------------------
        // 4. INT4 paged — per-(token,kv_head) FP16 scale; packed nibbles
        //    [head_dim/2] (lo=even, hi=odd). Dequant: val = int4 * scale.
        // -------------------------------------------------------------------
        {
            const int half_hd = head_dim / 2;
            const int scale_elems = num_blocks * BLOCK_SIZE * n_kv_heads;
            auto quant = [&](const std::vector<half>& kv, std::vector<half>& scales) {
                std::vector<uint8_t> out((size_t)num_blocks * BLOCK_SIZE * n_kv_heads * half_hd, 0);
                scales.assign(scale_elems, __float2half(0.0f));
                for (int s = 0; s < kv_len; s++) {
                    int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
                    for (int kvh = 0; kvh < n_kv_heads; kvh++) {
                        size_t src = ((size_t)s * n_kv_heads + kvh) * head_dim;
                        float amax = 0.0f;
                        for (int d = 0; d < head_dim; d++)
                            amax = std::max(amax, std::fabs(__half2float(kv[src + d])));
                        float sc = amax > 0 ? amax / 7.0f : 1.0f;
                        float inv = 1.0f / sc;
                        scales[(blk * BLOCK_SIZE + slot) * n_kv_heads + kvh] = __float2half(sc);
                        size_t dst = ((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads * half_hd +
                                     (size_t)kvh * half_hd;
                        for (int d = 0; d < head_dim; d += 2) {
                            int q0 = (int)std::lround(__half2float(kv[src + d]) * inv);
                            int q1 = (int)std::lround(__half2float(kv[src + d + 1]) * inv);
                            q0 = std::max(-8, std::min(7, q0));
                            q1 = std::max(-8, std::min(7, q1));
                            out[dst + d / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
                        }
                    }
                }
                return out;
            };
            std::vector<half> ks, vs;
            auto Kq = quant(Kh, ks), Vq = quant(Vh, vs);
            void* d_k = up(Kq.data(), Kq.size());
            void* d_v = up(Vq.data(), Vq.size());
            void* d_ks = up(ks.data(), ks.size() * sizeof(half));
            void* d_vs = up(vs.data(), vs.size() * sizeof(half));
            Tensor K = raw_tensor(d_k, QType::INT8, {num_blocks, BLOCK_SIZE, n_kv_heads, half_hd});
            Tensor V = raw_tensor(d_v, QType::INT8, {num_blocks, BLOCK_SIZE, n_kv_heads, half_hd});
            Tensor O = f16_tensor(d_o, {1, 1, n_heads, head_dim});
            clear_o();
            paged_attention_decode_int4(Q, K, V, O, (const half*)d_ks, (const half*)d_vs, d_bt, d_ctx,
                                        BLOCK_SIZE, scale, kv_len, 0, 0.0f, stream_, num_blocks);
            cudaStreamSynchronize(stream_);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "INT4 paged launch";
            auto o = read_o(d_o, q_elems);
            ErrStats e = err_stats(o, ref);
            EXPECT_EQ(e.nan_count, 0) << "INT4 paged: non-finite output";
            EXPECT_LT(e.max_rel, env_int4) << "INT4 paged envelope exceeded: " << e.str();
            printf("[paged-oracle] %s INT4:   %s (env %.3g)\n", trace, e.str().c_str(), env_int4);
            cudaFree(d_k);
            cudaFree(d_v);
            cudaFree(d_ks);
            cudaFree(d_vs);
        }

        // -------------------------------------------------------------------
        // 5. + 6. NVFP4 paged (scalar) and NVFP4-TC. Shared host quantize:
        //    per-(token, kv_head, group-of-16) UE4M3 micro-scale + E2M1 nibble
        //    [head_dim/2]. Dequant: e2m1(nibble) * ue4m3(scale). A CORRECT host
        //    quantize is mandatory — the existing TC test documents that
        //    random-byte NVFP4 drives the scalar kernel to all-NaN.
        // -------------------------------------------------------------------
        {
            const int half_hd = head_dim / 2;
            const int sc_groups = head_dim / 16;
            const int sc_elems = num_blocks * BLOCK_SIZE * n_kv_heads * sc_groups;
            auto quant = [&](const std::vector<half>& kv, std::vector<uint8_t>& scales) {
                std::vector<uint8_t> out((size_t)num_blocks * BLOCK_SIZE * n_kv_heads * half_hd, 0);
                scales.assign(sc_elems, 0);
                for (int s = 0; s < kv_len; s++) {
                    int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
                    for (int kvh = 0; kvh < n_kv_heads; kvh++) {
                        size_t src = ((size_t)s * n_kv_heads + kvh) * head_dim;
                        size_t pdst = ((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads * half_hd +
                                      (size_t)kvh * half_hd;
                        size_t sdst = ((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads * sc_groups +
                                      (size_t)kvh * sc_groups;
                        for (int g = 0; g < sc_groups; g++) {
                            // group of 16 elements -> one E2M1-grid scale.
                            float amax = 0.0f;
                            for (int d = 0; d < 16; d++)
                                amax = std::max(amax, std::fabs(__half2float(kv[src + g * 16 + d])));
                            // E2M1 max magnitude = 6; scale = amax/6, then round
                            // through UE4M3 (the on-disk micro-scale).
                            float sc = amax > 0 ? amax / 6.0f : 1.0f;
                            uint8_t sc_byte = f_to_ue4m3(sc);
                            float sc_q = ue4m3_to_f(sc_byte);
                            if (sc_q <= 0.0f)
                                sc_q = 1.0f;
                            scales[sdst + g] = sc_byte;
                            float inv = 1.0f / sc_q;
                            for (int d = 0; d < 16; d += 2) {
                                int dd = g * 16 + d;
                                uint8_t lo = f_to_e2m1(__half2float(kv[src + dd]) * inv);
                                uint8_t hi = f_to_e2m1(__half2float(kv[src + dd + 1]) * inv);
                                out[pdst + dd / 2] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
                            }
                        }
                    }
                }
                return out;
            };
            std::vector<uint8_t> ks, vs;
            auto Kq = quant(Kh, ks), Vq = quant(Vh, vs);
            void* d_k = up(Kq.data(), Kq.size());
            void* d_v = up(Vq.data(), Vq.size());
            void* d_ks = up(ks.data(), ks.size());
            void* d_vs = up(vs.data(), vs.size());
            Tensor K = raw_tensor(d_k, QType::FP4_E2M1, {num_blocks, BLOCK_SIZE, n_kv_heads, half_hd});
            Tensor V = raw_tensor(d_v, QType::FP4_E2M1, {num_blocks, BLOCK_SIZE, n_kv_heads, half_hd});

            // 5. scalar
            {
                Tensor O = f16_tensor(d_o, {1, 1, n_heads, head_dim});
                clear_o();
                paged_attention_decode_nvfp4(Q, K, V, O, (const uint8_t*)d_ks, (const uint8_t*)d_vs, d_bt,
                                             d_ctx, BLOCK_SIZE, scale, kv_len, 0, 0.0f, stream_, num_blocks);
                cudaStreamSynchronize(stream_);
                ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "NVFP4 scalar paged launch";
                auto o = read_o(d_o, q_elems);
                ErrStats e = err_stats(o, ref);
                EXPECT_EQ(e.nan_count, 0)
                    << "NVFP4 scalar paged: non-finite output (the all-NaN trap — correct quant must avoid)";
                EXPECT_LT(e.max_rel, env_nvfp4) << "NVFP4 scalar paged envelope exceeded: " << e.str();
                printf("[paged-oracle] %s NVFP4:  %s (env %.3g)\n", trace, e.str().c_str(), env_nvfp4);
            }
            // 6. tensor-core Q.K variant (only hd=128 is the production TC shape;
            //    the kernel internally requires HEAD_DIM%16==0 — assert it runs).
            {
                Tensor O = f16_tensor(d_o, {1, 1, n_heads, head_dim});
                clear_o();
                paged_attention_decode_nvfp4_tc(Q, K, V, O, (const uint8_t*)d_ks, (const uint8_t*)d_vs, d_bt,
                                                d_ctx, BLOCK_SIZE, scale, kv_len, 0, 0.0f, stream_,
                                                num_blocks);
                cudaStreamSynchronize(stream_);
                ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "NVFP4-TC paged launch";
                auto o = read_o(d_o, q_elems);
                ErrStats e = err_stats(o, ref);
                EXPECT_EQ(e.nan_count, 0) << "NVFP4-TC paged: non-finite output";
                EXPECT_LT(e.max_rel, env_nvfp4) << "NVFP4-TC paged envelope exceeded: " << e.str();
                printf("[paged-oracle] %s NVFP4-TC: %s (env %.3g)\n", trace, e.str().c_str(), env_nvfp4);
            }
            cudaFree(d_k);
            cudaFree(d_v);
            cudaFree(d_ks);
            cudaFree(d_vs);
        }

        cudaFree(d_q);
        cudaFree(d_o);
        cudaFree(d_bt);
        cudaFree(d_ctx);
    }
};

// ===========================================================================
// Measured characterization envelopes — MEASURED 2026-06-04 on RTX 5090
// (sm_120a), first run of this suite (its birth certificate). All numbers are
// max_rel vs the original-f16 fp64 reference, denom max(1,|ref|), worst across
// both GQA32x8 and MHA8x8 configs at each kv_len. Ceilings below = worst
// observed + ~50% margin.
//
//   kv_len:        16        64       333      1024     ceiling (frozen)
//   F16        2.43e-4   2.29e-4   1.70e-4   6.08e-5   1e-2 (STRICT, no quant)
//   FP8        0.0216    0.0073    0.0051    0.0028    0.035
//   INT8       0.0039    0.0017    0.0007    0.0005    0.007
//   INT4       0.0609    0.0474    0.0193    0.0096    0.10
//   NVFP4      0.0670    0.0251    0.0134    0.0072    0.11
//   NVFP4-TC   0.0670    0.0251    0.0133    0.0072    0.11  (tracks scalar)
//
// Findings (real):
//   * Monotone improvement with kv_len — confirms the #512 mechanism in the
//     DECODE direction: per-key quant score-noise is worst when the softmax
//     averages over few keys (16) and dilutes as context grows (1024). The
//     short-row band is exactly where the cross-path prefill failure also bit.
//   * Quant ranking at hd=128: INT8 (per-token per-elem) << FP8 (per-tensor,
//     more mantissa/elem) < INT4 ≈ NVFP4 (4-bit nibbles). All stay well within
//     NVFP4's 1e-1 single-op class here — but only because the host quantize
//     is CORRECT; random-byte NVFP4 NaNs the scalar kernel (the TC test's
//     documented trap). The no-NaN guard is the load-bearing decode assert.
//   * NVFP4 scalar and NVFP4-TC agree to <2e-4 — the tensor-core Q.K dot is
//     numerically equivalent to the scalar dot on this data.
//   * F16 paged tracks fp64 at <2.5e-4, four orders under its 1e-2 budget —
//     the decode F16 path has no hidden score-precision tax.
// ===========================================================================

// GQA 32q/8kv, hd=128 — the production decode hot shape (Qwen3 / Llama family).
TEST_F(PagedOracleTest, GQA32x8_HD128_Sweep) {
    for (int kv_len : {16, 64, 333, 1024})
        run("gqa32x8", kv_len, 32, 8, 128, /*fp8*/ 0.035f, /*int8*/ 0.007f, /*int4*/ 0.10f, /*nvfp4*/ 0.11f);
}

// MHA 8q/8kv, hd=128 — the no-GQA path (each q-head its own kv-head).
TEST_F(PagedOracleTest, MHA8x8_HD128_Sweep) {
    for (int kv_len : {16, 64, 333, 1024})
        run("mha8x8", kv_len, 8, 8, 128, /*fp8*/ 0.035f, /*int8*/ 0.007f, /*int4*/ 0.10f, /*nvfp4*/ 0.11f);
}

}  // namespace
}  // namespace imp
