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
// Shared decode-path harness. Inputs, fp64 reference, block table, Q/O buffers
// are built ONCE per (shape, kv_len); each KV-dtype "policy" then quantizes K/V
// into its own cache layout, launches its kernel, and reports an ErrStats vs
// the shared reference. This is the parametrization seam (R8): the skeleton
// (reference + characterize) is shared, only the per-dtype quant+launch differs.
// ===========================================================================
struct PathCtx {
    cudaStream_t stream;
    int kv_len, n_heads, n_kv_heads, head_dim, num_blocks;
    float scale;
    size_t q_elems;
    const std::vector<half>* Kh;
    const std::vector<half>* Vh;
    const std::vector<double>* ref;
    Tensor Q;
    void* d_o;
    int* d_bt;
    int* d_ctx;
    void clear_o() const { cudaMemset(d_o, 0, q_elems * sizeof(half)); }
    Tensor O() const { return f16_tensor(d_o, {1, 1, n_heads, head_dim}); }
};

// ---------------------------------------------------------------------------
// KV-dtype policies. Each exposes name(), envelope(), strict(), and run(ctx):
// run() quantizes K/V into the kernel's exact layout, launches it, returns the
// max_rel vs the shared fp64 reference (and asserts launch success + finiteness
// inside, since those are dtype-agnostic guards). TYPED_TEST iterates them.
// NOTE: launch failures use EXPECT (not ASSERT) because run() returns ErrStats;
// a failed launch flags the EXPECT and then also blows the envelope check, so
// it cannot pass silently — it just doesn't abort the case early.
// ---------------------------------------------------------------------------
struct PathF16 {
    static const char* name() { return "F16"; }
    static bool strict() { return true; }
    static float envelope() { return 1e-2f; }  // STRICT — no quantization at all.
    static ErrStats run(const PathCtx& c) {
        auto Kc = build_f16_cache(*c.Kh, c.kv_len, c.n_kv_heads, c.head_dim, c.num_blocks);
        auto Vc = build_f16_cache(*c.Vh, c.kv_len, c.n_kv_heads, c.head_dim, c.num_blocks);
        void* d_k = up(Kc.data(), Kc.size() * sizeof(half));
        void* d_v = up(Vc.data(), Vc.size() * sizeof(half));
        Tensor K = f16_tensor(d_k, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, c.head_dim});
        Tensor V = f16_tensor(d_v, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, c.head_dim});
        Tensor O = c.O();
        c.clear_o();
        paged_attention_decode(c.Q, K, V, O, c.d_bt, c.d_ctx, BLOCK_SIZE, c.scale, c.kv_len, 0, 0.0f,
                               c.stream, c.num_blocks);
        cudaStreamSynchronize(c.stream);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "F16 paged launch";
        ErrStats e = err_stats(read_o(c.d_o, c.q_elems), *c.ref);
        cudaFree(d_k);
        cudaFree(d_v);
        return e;
    }
};

struct PathFP8 {
    static const char* name() { return "FP8"; }
    static bool strict() { return false; }
    static float envelope() { return 0.035f; }
    // per-tensor kv_scale, layout == F16 (1 byte). Dequant: val = e4m3 * kv_scale.
    static ErrStats run(const PathCtx& c) {
        float amax = 0.0f;
        for (auto& h : *c.Kh)
            amax = std::max(amax, std::fabs(__half2float(h)));
        for (auto& h : *c.Vh)
            amax = std::max(amax, std::fabs(__half2float(h)));
        float kv_scale = amax > 0 ? amax / 448.0f : 1.0f;
        float inv = 1.0f / kv_scale;
        auto quant = [&](const std::vector<half>& kv) {
            std::vector<uint8_t> out((size_t)c.num_blocks * BLOCK_SIZE * c.n_kv_heads * c.head_dim, 0);
            for (int s = 0; s < c.kv_len; s++) {
                int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
                for (int kvh = 0; kvh < c.n_kv_heads; kvh++) {
                    size_t dst = ((size_t)blk * BLOCK_SIZE + slot) * c.n_kv_heads * c.head_dim +
                                 (size_t)kvh * c.head_dim;
                    size_t src = ((size_t)s * c.n_kv_heads + kvh) * c.head_dim;
                    for (int d = 0; d < c.head_dim; d++) {
                        __nv_fp8_e4m3 q = __nv_fp8_e4m3(__half2float(kv[src + d]) * inv);
                        memcpy(&out[dst + d], &q, 1);
                    }
                }
            }
            return out;
        };
        auto Kq = quant(*c.Kh), Vq = quant(*c.Vh);
        void* d_k = up(Kq.data(), Kq.size());
        void* d_v = up(Vq.data(), Vq.size());
        Tensor K = raw_tensor(d_k, QType::FP8_E4M3, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, c.head_dim});
        Tensor V = raw_tensor(d_v, QType::FP8_E4M3, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, c.head_dim});
        Tensor O = c.O();
        c.clear_o();
        paged_attention_decode_fp8(c.Q, K, V, O, c.d_bt, c.d_ctx, BLOCK_SIZE, c.scale, kv_scale, c.kv_len, 0,
                                   0.0f, c.stream, c.num_blocks);
        cudaStreamSynchronize(c.stream);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "FP8 paged launch";
        ErrStats e = err_stats(read_o(c.d_o, c.q_elems), *c.ref);
        cudaFree(d_k);
        cudaFree(d_v);
        return e;
    }
};

struct PathINT8 {
    static const char* name() { return "INT8"; }
    static bool strict() { return false; }
    static float envelope() { return 0.007f; }
    // per-(token,kv_head) FP16 scale; layout == F16 (1 byte). val = int8 * scale.
    static ErrStats run(const PathCtx& c) {
        const int scale_elems = c.num_blocks * BLOCK_SIZE * c.n_kv_heads;
        auto quant = [&](const std::vector<half>& kv, std::vector<half>& scales) {
            std::vector<int8_t> out((size_t)c.num_blocks * BLOCK_SIZE * c.n_kv_heads * c.head_dim, 0);
            scales.assign(scale_elems, __float2half(0.0f));
            for (int s = 0; s < c.kv_len; s++) {
                int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
                for (int kvh = 0; kvh < c.n_kv_heads; kvh++) {
                    size_t src = ((size_t)s * c.n_kv_heads + kvh) * c.head_dim;
                    float amax = 0.0f;
                    for (int d = 0; d < c.head_dim; d++)
                        amax = std::max(amax, std::fabs(__half2float(kv[src + d])));
                    float sc = amax > 0 ? amax / 127.0f : 1.0f;
                    float inv = 1.0f / sc;
                    scales[(blk * BLOCK_SIZE + slot) * c.n_kv_heads + kvh] = __float2half(sc);
                    size_t dst = ((size_t)blk * BLOCK_SIZE + slot) * c.n_kv_heads * c.head_dim +
                                 (size_t)kvh * c.head_dim;
                    for (int d = 0; d < c.head_dim; d++) {
                        int q = (int)std::lround(__half2float(kv[src + d]) * inv);
                        q = std::max(-127, std::min(127, q));
                        out[dst + d] = (int8_t)q;
                    }
                }
            }
            return out;
        };
        std::vector<half> ks, vs;
        auto Kq = quant(*c.Kh, ks), Vq = quant(*c.Vh, vs);
        void* d_k = up(Kq.data(), Kq.size());
        void* d_v = up(Vq.data(), Vq.size());
        void* d_ks = up(ks.data(), ks.size() * sizeof(half));
        void* d_vs = up(vs.data(), vs.size() * sizeof(half));
        Tensor K = raw_tensor(d_k, QType::INT8, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, c.head_dim});
        Tensor V = raw_tensor(d_v, QType::INT8, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, c.head_dim});
        Tensor O = c.O();
        c.clear_o();
        paged_attention_decode_int8(c.Q, K, V, O, (const half*)d_ks, (const half*)d_vs, c.d_bt, c.d_ctx,
                                    BLOCK_SIZE, c.scale, c.kv_len, 0, 0.0f, c.stream, c.num_blocks);
        cudaStreamSynchronize(c.stream);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "INT8 paged launch";
        ErrStats e = err_stats(read_o(c.d_o, c.q_elems), *c.ref);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_ks);
        cudaFree(d_vs);
        return e;
    }
};

struct PathINT4 {
    static const char* name() { return "INT4"; }
    static bool strict() { return false; }
    static float envelope() { return 0.10f; }
    // per-(token,kv_head) FP16 scale; packed nibbles [head_dim/2] (lo=even,
    // hi=odd). val = int4 * scale.
    static ErrStats run(const PathCtx& c) {
        const int half_hd = c.head_dim / 2;
        const int scale_elems = c.num_blocks * BLOCK_SIZE * c.n_kv_heads;
        auto quant = [&](const std::vector<half>& kv, std::vector<half>& scales) {
            std::vector<uint8_t> out((size_t)c.num_blocks * BLOCK_SIZE * c.n_kv_heads * half_hd, 0);
            scales.assign(scale_elems, __float2half(0.0f));
            for (int s = 0; s < c.kv_len; s++) {
                int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
                for (int kvh = 0; kvh < c.n_kv_heads; kvh++) {
                    size_t src = ((size_t)s * c.n_kv_heads + kvh) * c.head_dim;
                    float amax = 0.0f;
                    for (int d = 0; d < c.head_dim; d++)
                        amax = std::max(amax, std::fabs(__half2float(kv[src + d])));
                    float sc = amax > 0 ? amax / 7.0f : 1.0f;
                    float inv = 1.0f / sc;
                    scales[(blk * BLOCK_SIZE + slot) * c.n_kv_heads + kvh] = __float2half(sc);
                    size_t dst =
                        ((size_t)blk * BLOCK_SIZE + slot) * c.n_kv_heads * half_hd + (size_t)kvh * half_hd;
                    for (int d = 0; d < c.head_dim; d += 2) {
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
        auto Kq = quant(*c.Kh, ks), Vq = quant(*c.Vh, vs);
        void* d_k = up(Kq.data(), Kq.size());
        void* d_v = up(Vq.data(), Vq.size());
        void* d_ks = up(ks.data(), ks.size() * sizeof(half));
        void* d_vs = up(vs.data(), vs.size() * sizeof(half));
        Tensor K = raw_tensor(d_k, QType::INT8, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, half_hd});
        Tensor V = raw_tensor(d_v, QType::INT8, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, half_hd});
        Tensor O = c.O();
        c.clear_o();
        paged_attention_decode_int4(c.Q, K, V, O, (const half*)d_ks, (const half*)d_vs, c.d_bt, c.d_ctx,
                                    BLOCK_SIZE, c.scale, c.kv_len, 0, 0.0f, c.stream, c.num_blocks);
        cudaStreamSynchronize(c.stream);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "INT4 paged launch";
        ErrStats e = err_stats(read_o(c.d_o, c.q_elems), *c.ref);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_ks);
        cudaFree(d_vs);
        return e;
    }
};

// Shared NVFP4 host quantize (scalar + TC use the same cache). per-(token,
// kv_head, group-of-16) UE4M3 micro-scale + E2M1 nibble [head_dim/2]. A CORRECT
// host quantize is mandatory — random-byte NVFP4 drives the scalar kernel NaN.
static std::vector<uint8_t> nvfp4_quant_kv(const PathCtx& c, const std::vector<half>& kv,
                                           std::vector<uint8_t>& scales) {
    const int half_hd = c.head_dim / 2;
    const int sc_groups = c.head_dim / 16;
    const int sc_elems = c.num_blocks * BLOCK_SIZE * c.n_kv_heads * sc_groups;
    std::vector<uint8_t> out((size_t)c.num_blocks * BLOCK_SIZE * c.n_kv_heads * half_hd, 0);
    scales.assign(sc_elems, 0);
    for (int s = 0; s < c.kv_len; s++) {
        int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
        for (int kvh = 0; kvh < c.n_kv_heads; kvh++) {
            size_t src = ((size_t)s * c.n_kv_heads + kvh) * c.head_dim;
            size_t pdst =
                ((size_t)blk * BLOCK_SIZE + slot) * c.n_kv_heads * half_hd + (size_t)kvh * half_hd;
            size_t sdst =
                ((size_t)blk * BLOCK_SIZE + slot) * c.n_kv_heads * sc_groups + (size_t)kvh * sc_groups;
            for (int g = 0; g < sc_groups; g++) {
                float amax = 0.0f;
                for (int d = 0; d < 16; d++)
                    amax = std::max(amax, std::fabs(__half2float(kv[src + g * 16 + d])));
                float sc = amax > 0 ? amax / 6.0f : 1.0f;  // E2M1 max magnitude = 6
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
}

struct PathNVFP4 {
    static const char* name() { return "NVFP4"; }
    static bool strict() { return false; }
    static float envelope() { return 0.11f; }
    static ErrStats run(const PathCtx& c) {
        std::vector<uint8_t> ks, vs;
        auto Kq = nvfp4_quant_kv(c, *c.Kh, ks), Vq = nvfp4_quant_kv(c, *c.Vh, vs);
        const int half_hd = c.head_dim / 2;
        void* d_k = up(Kq.data(), Kq.size());
        void* d_v = up(Vq.data(), Vq.size());
        void* d_ks = up(ks.data(), ks.size());
        void* d_vs = up(vs.data(), vs.size());
        Tensor K = raw_tensor(d_k, QType::FP4_E2M1, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, half_hd});
        Tensor V = raw_tensor(d_v, QType::FP4_E2M1, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, half_hd});
        Tensor O = c.O();
        c.clear_o();
        paged_attention_decode_nvfp4(c.Q, K, V, O, (const uint8_t*)d_ks, (const uint8_t*)d_vs, c.d_bt,
                                     c.d_ctx, BLOCK_SIZE, c.scale, c.kv_len, 0, 0.0f, c.stream, c.num_blocks);
        cudaStreamSynchronize(c.stream);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "NVFP4 scalar paged launch";
        ErrStats e = err_stats(read_o(c.d_o, c.q_elems), *c.ref);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_ks);
        cudaFree(d_vs);
        return e;
    }
};

struct PathNVFP4TC {
    static const char* name() { return "NVFP4-TC"; }
    static bool strict() { return false; }
    static float envelope() { return 0.11f; }
    static ErrStats run(const PathCtx& c) {
        std::vector<uint8_t> ks, vs;
        auto Kq = nvfp4_quant_kv(c, *c.Kh, ks), Vq = nvfp4_quant_kv(c, *c.Vh, vs);
        const int half_hd = c.head_dim / 2;
        void* d_k = up(Kq.data(), Kq.size());
        void* d_v = up(Vq.data(), Vq.size());
        void* d_ks = up(ks.data(), ks.size());
        void* d_vs = up(vs.data(), vs.size());
        Tensor K = raw_tensor(d_k, QType::FP4_E2M1, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, half_hd});
        Tensor V = raw_tensor(d_v, QType::FP4_E2M1, {c.num_blocks, BLOCK_SIZE, c.n_kv_heads, half_hd});
        Tensor O = c.O();
        c.clear_o();
        paged_attention_decode_nvfp4_tc(c.Q, K, V, O, (const uint8_t*)d_ks, (const uint8_t*)d_vs, c.d_bt,
                                        c.d_ctx, BLOCK_SIZE, c.scale, c.kv_len, 0, 0.0f, c.stream,
                                        c.num_blocks);
        cudaStreamSynchronize(c.stream);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "NVFP4-TC paged launch";
        ErrStats e = err_stats(read_o(c.d_o, c.q_elems), *c.ref);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_ks);
        cudaFree(d_vs);
        return e;
    }
};

// ===========================================================================
// TYPED_TEST over KV dtypes (R8 / audit Phase-2 R8: "TYPED_TEST über KV-Dtypes
// im paged-Oracle"). One typed fixture, one body; each KV dtype is a policy
// type. The two production decode shapes (GQA 32x8 and MHA 8x8 at hd=128) and
// the kv_len sweep {16, 64, 333, 1024} run inside the body. 333 is deliberately
// NOT block-aligned (partial-tail block); 16/64 are the short rows where quant
// score-noise has no averaging (#512); 1024 is the long-context dilution case.
// ===========================================================================
template <typename Path>
class PagedOracle : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;

    void run_shape(const char* cfg, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
        char trace[160];
        snprintf(trace, sizeof(trace), "%s %s kv_len=%d nh=%d nkv=%d hd=%d", Path::name(), cfg, kv_len,
                 n_heads, n_kv_heads, head_dim);
        SCOPED_TRACE(trace);

        const float scale = 1.0f / std::sqrt((float)head_dim);
        const int num_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const size_t q_elems = (size_t)n_heads * head_dim;
        const size_t kv_elems = (size_t)kv_len * n_kv_heads * head_dim;

        const uint32_t seed = 0xC0DEu + (uint32_t)kv_len * 131u + (uint32_t)n_kv_heads * 17u;
        std::vector<half> Qh(q_elems), Kh(kv_elems), Vh(kv_elems);
        lcg_fill(Qh, seed + 1, 2.0f);
        lcg_fill(Kh, seed + 2, 2.0f);
        lcg_fill(Vh, seed + 3, 1.0f);  // V not QK-normed

        std::vector<double> ref;
        ref_decode_f64(Qh, Kh, Vh, ref, kv_len, n_heads, n_kv_heads, head_dim, scale);

        std::vector<int> bt(num_blocks);
        for (int i = 0; i < num_blocks; i++)
            bt[i] = i;
        int* d_bt = (int*)up(bt.data(), num_blocks * sizeof(int));
        int ctx = kv_len;
        int* d_ctx = (int*)up(&ctx, sizeof(int));
        void* d_q = up(Qh.data(), q_elems * sizeof(half));
        void* d_o = nullptr;
        cudaMalloc(&d_o, q_elems * sizeof(half));

        PathCtx c{stream_, kv_len, n_heads,        n_kv_heads, head_dim, num_blocks,
                  scale,   q_elems, &Kh,           &Vh,        &ref,
                  f16_tensor(d_q, {1, 1, n_heads, head_dim}), d_o, d_bt, d_ctx};

        ErrStats e = Path::run(c);
        EXPECT_EQ(e.nan_count, 0) << Path::name() << " paged: non-finite output (decode-corruption guard)";
        if (Path::strict())
            EXPECT_LT(e.max_rel, Path::envelope())
                << Path::name() << " paged vs fp64 (STRICT — no quant): " << e.str();
        else
            EXPECT_LT(e.max_rel, Path::envelope())
                << Path::name() << " paged envelope exceeded: " << e.str();
        printf("[paged-oracle] %s: %s (env %.3g)\n", trace, e.str().c_str(), Path::envelope());

        cudaFree(d_q);
        cudaFree(d_o);
        cudaFree(d_bt);
        cudaFree(d_ctx);
    }
};

using KVDtypes = ::testing::Types<PathF16, PathFP8, PathINT8, PathINT4, PathNVFP4, PathNVFP4TC>;
TYPED_TEST_SUITE(PagedOracle, KVDtypes);

// ===========================================================================
// Measured characterization envelopes — MEASURED 2026-06-04 on RTX 5090
// (sm_120a), first run of this suite (its birth certificate), re-confirmed
// 2026-06-06 under the typed restructure (identical numbers — same quantizers
// and kernels, only the dispatch changed). All numbers are max_rel vs the
// original-f16 fp64 reference, denom max(1,|ref|), worst across both GQA32x8
// and MHA8x8 configs at each kv_len. Ceilings (Path::envelope()) = worst
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

TYPED_TEST(PagedOracle, HD128_Sweep) {
    for (int kv_len : {16, 64, 333, 1024}) {
        // GQA 32q/8kv — production decode hot shape (Qwen3 / Llama family).
        this->run_shape("gqa32x8", kv_len, 32, 8, 128);
        // MHA 8q/8kv — the no-GQA path (each q-head its own kv-head).
        this->run_shape("mha8x8", kv_len, 8, 8, 128);
    }
}

}  // namespace
}  // namespace imp
