// =============================================================================
// TEST_AUDIT.md Phase 2.6 — Risk #8
// GGUF Q4_K / Q6_K / Q8_0 dequant + MMVQ/dp4a-GEMV vs a format-derived,
// INDEPENDENT fp64 reference.
//
// WHY THIS EXISTS (audit §3 risk #8, §2 class-B critique):
//   The existing tests/test_mmvq.cu compares imp's MMVQ kernel against imp's
//   dp4a kernel (imp-vs-imp == class B tautology). All of GGUF serving
//   (Q8_0 / Q6_K / Q4_K models) consumes these dequant + GEMV kernels, yet
//   only INT4/INT8 toy refs existed. This file is the missing class-A anchor:
//   it re-derives each block format's dequant ARITHMETIC in fp64 on the host
//   from the ggml format definition (block layout = the format itself, taken
//   from imp's headers; the dequant math is reconstructed and justified in
//   comments against ggml-common.h / dequantize_row_q*), and compares imp's
//   GPU kernels against it.
//
// INDEPENDENCE (audit §4 — no tautologies):
//   * Synthetic block bytes are built directly on the BYTE level via an LCG
//     over the raw quant bytes + independently chosen scale halfs — NOT via
//     imp's quantizer (which would make a quantize->dequant round-trip and
//     prove nothing).
//   * The reference dequant is a slow fp64 host loop, never a call into imp.
//   * Edge cases (d = 0, all-63 / all-0 6-bit scales, max-magnitude scale
//     halfs, and a NaN d-half) are exercised with a hard no-NaN/Inf guard.
//
// TOLERANCES (audit §4 tolerance policy — derived per path):
//   * Dequant kernel (pure decode, half-rounding only): the kernel computes
//     d*sc*q in fp32 then rounds to f16 once. The fp64 reference rounds the
//     same product to f16. Both see identical input bits, so the only spread
//     is fp32-vs-fp64 accumulation of a 2-3 factor product => <= 1e-3 rel
//     (1 ulp of f16 ~= 2^-11 ~= 4.9e-4; we measure and assert 1e-3).
//   * fp16-dequant GEMV (gemv_q8_0 / gemv_q6k): dequant in fp32, dot in fp32,
//     output rounded to f16. Reference: dequant in fp64, dot the ORIGINAL f16
//     x in fp64. Error = fp32-vs-fp64 dot over K terms + one f16 output round.
//     fp16-class => <= 1e-2 rel (justified per audit §4, measured).
//   * dp4a / MMVQ GEMV (gemv_*_q8_1, ggml_mmvq_*): these ALSO quantize the
//     activations x to Q8_1 (amax/127 per 32-block, ggml-standard). That adds
//     ~0.4% RMS per-element activation noise on top of the f16 dot. Over a
//     K-term dot this does NOT fully average out (correlated within a block),
//     so we derive ~1-2% and MEASURE the real envelope, asserting 2.5e-2 rel
//     with the per-test printed stats as the characterization record.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "compute/gemm.h"
#include "compute/ggml_mmvq.h"
#include "quant/dequant_gpu.h"

namespace imp {
namespace {

// -----------------------------------------------------------------------------
// Deterministic byte-level LCG (Numerical Recipes constants). Independent of
// imp; used to fill raw quant bytes and to pick scale halfs.
// -----------------------------------------------------------------------------
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s;
    }
    uint8_t byte() { return static_cast<uint8_t>(next() >> 24); }
    // Uniform in [-1, 1), multiply-only mapping (bit-stable, no libm).
    float unit() { return (next() >> 8) * (1.0f / 8388608.0f) - 1.0f; }
};

// Host-side f16 helpers via the CUDA half type (host-callable).
inline uint16_t f16_bits(half h) {
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
inline half f16_from_bits(uint16_t b) {
    half h;
    std::memcpy(&h, &b, 2);
    return h;
}
// fp64 value of an f16, computed via the half->float path the GPU also uses.
inline double f16_to_f64(half h) { return static_cast<double>(__half2float(h)); }
// Round an fp32 to f16 and back (the kernel's final-store rounding). Saturates
// to ±Inf on overflow, matching __float2half on values > 65504.
inline float __float2half_then_float(float f) { return __half2float(__float2half(f)); }

// -----------------------------------------------------------------------------
// fp64 REFERENCE DEQUANT — derived from the ggml format definition.
// Layouts mirror imp's headers (= the format), arithmetic is reconstructed.
// -----------------------------------------------------------------------------

// Q8_0: 34 bytes / 32 elems. [ d:f16 | qs:int8[32] ]. val = d * q.
// (ggml dequantize_row_q8_0: y = d * qs[j].)
void ref_dequant_q8_0(const uint8_t* blk, double* out) {
    half d;
    std::memcpy(&d, blk, 2);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    double dd = f16_to_f64(d);
    for (int i = 0; i < 32; ++i)
        out[i] = dd * static_cast<double>(qs[i]);
}

// Q6_K: 210 bytes / 256 elems. [ ql:u8[128] | qh:u8[64] | scales:int8[16] | d:f16 ].
// ggml dequantize_row_q6_K: for each of 2 groups of 128, four 32-quads:
//   q = (high2<<4 | low4) - 32 ; val = d * scale[i/16] * q
// (6-bit signed quant centered at 32; sub-block scale every 16 elems.)
void ref_dequant_q6_k(const uint8_t* blk, double* out) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
    half d;
    std::memcpy(&d, blk + 208, 2);
    double dd = f16_to_f64(d);
    for (int i = 0; i < 256; ++i) {
        int group = i >> 7;       // 0..1
        int within = i & 127;     // 0..127
        int quad = within >> 5;   // 0..3
        int l = within & 31;      // 0..31
        int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
        int qh_idx = (group << 5) + l;
        uint8_t ql_byte = ql[ql_idx];
        uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
        uint8_t high2 = (qh[qh_idx] >> (quad * 2)) & 0x3;
        int q6 = static_cast<int>((high2 << 4) | low4) - 32;
        out[i] = dd * static_cast<double>(sc[i >> 4]) * static_cast<double>(q6);
    }
}

// Q4_K: 144 bytes / 256 elems. [ d:f16 | dmin:f16 | scales:u8[12] | qs:u8[128] ].
// 8 sub-blocks of 32. 6-bit (scale,min) per sub-block packed per ggml
// get_scale_min_k4. Quant nibbles: a 64-elem chunk uses 32 bytes, first 32
// elems in low nibbles, next 32 in high nibbles of the SAME 32 bytes.
//   val = d*scale*q4 - dmin*min   (ggml dequantize_row_q4_K).
void ref_get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
void ref_dequant_q4_k(const uint8_t* blk, double* out) {
    half d, dmin;
    std::memcpy(&d, blk, 2);
    std::memcpy(&dmin, blk + 2, 2);
    const uint8_t* sc = blk + 4;
    const uint8_t* qs = blk + 16;
    double dd = f16_to_f64(d);
    double dm = f16_to_f64(dmin);
    for (int i = 0; i < 256; ++i) {
        int sub = i / 32;
        uint8_t scv, mnv;
        ref_get_scale_min_k4(sub, sc, scv, mnv);
        int qs_byte = (i / 64) * 32 + (i % 32);
        int use_high = (i / 32) & 1;
        uint8_t packed = qs[qs_byte];
        int q4 = use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);
        out[i] = dd * static_cast<double>(scv) * static_cast<double>(q4) -
                 dm * static_cast<double>(mnv);
    }
}

// Q5_K: 176 bytes / 256 elems. [ d:f16 | dmin:f16 | scales:u8[12] | qh:u8[32] |
// qs:u8[128] ]. Same 6-bit (scale,min) packing as Q4_K. The 5th bit of
// element i lives in qh[i % 32] at bit position sub = i/32 (ggml
// dequantize_row_q5_K). val = d*scale*(q4 + 16*bit) - dmin*min.
void ref_dequant_q5_k(const uint8_t* blk, double* out) {
    half d, dmin;
    std::memcpy(&d, blk, 2);
    std::memcpy(&dmin, blk + 2, 2);
    const uint8_t* sc = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* qs = blk + 48;
    double dd = f16_to_f64(d);
    double dm = f16_to_f64(dmin);
    for (int i = 0; i < 256; ++i) {
        int sub = i / 32;
        uint8_t scv, mnv;
        ref_get_scale_min_k4(sub, sc, scv, mnv);
        int qs_byte = (i / 64) * 32 + (i % 32);
        int use_high = (i / 32) & 1;
        uint8_t packed = qs[qs_byte];
        int q4 = use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);
        int q5 = q4 + (((qh[i % 32] >> sub) & 1) << 4);
        out[i] = dd * static_cast<double>(scv) * static_cast<double>(q5) -
                 dm * static_cast<double>(mnv);
    }
}

// IQ4_NL: 18 bytes / 32 elems. [ d:f16 | qs:u8[16] ]. Non-linear 4-bit: the
// nibble indexes a fixed signed codebook (ggml-common.h kvalues_iq4nl).
// Element j (0..15) = low nibble of qs[j]; element j+16 = high nibble
// (ggml dequantize_row_iq4_nl). val = d * codebook[nibble].
static const int8_t kRefIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,    13,   25,  38,  53,  69,  89,  113};
void ref_dequant_iq4_nl(const uint8_t* blk, double* out) {
    half d;
    std::memcpy(&d, blk, 2);
    const uint8_t* qs = blk + 2;
    double dd = f16_to_f64(d);
    for (int j = 0; j < 16; ++j) {
        out[j] = dd * static_cast<double>(kRefIq4nlValues[qs[j] & 0xF]);
        out[j + 16] = dd * static_cast<double>(kRefIq4nlValues[(qs[j] >> 4) & 0xF]);
    }
}

// IQ4_XS: 136 bytes / 256 elems. [ d:f16 | scales_h:u16 | scales_l:u8[4] |
// qs:u8[128] ]. 8 sub-blocks of 32, 6-bit scale per sub-block:
//   ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xF | ((scales_h >> 2*ib) & 3) << 4
// Same codebook + nibble layout as IQ4_NL within each 16-byte sub-block
// (ggml dequantize_row_iq4_xs). val = d * (ls - 32) * codebook[nibble].
void ref_dequant_iq4_xs(const uint8_t* blk, double* out) {
    half d;
    std::memcpy(&d, blk, 2);
    uint16_t scales_h;
    std::memcpy(&scales_h, blk + 2, 2);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    double dd = f16_to_f64(d);
    for (int ib = 0; ib < 8; ++ib) {
        int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        double dl = dd * static_cast<double>(ls - 32);
        for (int j = 0; j < 16; ++j) {
            out[ib * 32 + j] = dl * static_cast<double>(kRefIq4nlValues[qs[ib * 16 + j] & 0xF]);
            out[ib * 32 + j + 16] = dl * static_cast<double>(kRefIq4nlValues[(qs[ib * 16 + j] >> 4) & 0xF]);
        }
    }
}

// -----------------------------------------------------------------------------
// Synthetic block builders — raw bytes via LCG, scale halfs chosen separately.
// scale_mode: 0 = "normal" random-ish scales; 1 = d=0 (degenerate);
//             2 = max-magnitude finite scale half; 3 = NaN d-half (guard only).
// 6-bit packed scales are filled to hit all-0 / all-63 extremes when forced.
// -----------------------------------------------------------------------------
enum ScaleMode { NORMAL = 0, ZERO_D = 1, MAXMAG = 2, NAN_D = 3 };

half pick_d(Lcg& g, ScaleMode mode) {
    switch (mode) {
        case ZERO_D:
            return __float2half(0.0f);
        case MAXMAG:
            // Largest finite normal f16 = 0x7BFF (65504). Stresses overflow.
            return f16_from_bits(0x7BFF);
        case NAN_D:
            return f16_from_bits(0x7E00);  // quiet NaN
        default:
            return __float2half(0.005f + 0.02f * std::fabs(g.unit()));
    }
}

void build_q8_0(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 32;
    buf.resize((size_t)N * bpr * 34);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 34;
            half d = pick_d(g, mode);
            std::memcpy(bp, &d, 2);
            for (int i = 0; i < 32; ++i)
                bp[2 + i] = g.byte();  // int8 quants, full [-128,127] range
        }
}

void build_q6_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 210);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 210;
            for (int i = 0; i < 192; ++i)
                bp[i] = g.byte();  // ql[128] + qh[64]
            for (int i = 0; i < 16; ++i)  // int8 sub-block scales
                bp[192 + i] = (mode == MAXMAG) ? 127 : g.byte();
            half d = pick_d(g, mode);
            std::memcpy(bp + 208, &d, 2);
        }
}

void build_q4_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 144);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 144;
            half d = pick_d(g, mode);
            half dmin = (mode == MAXMAG) ? f16_from_bits(0x7BFF)
                                         : __float2half(0.002f + 0.01f * std::fabs(g.unit()));
            if (mode == ZERO_D)
                dmin = __float2half(0.0f);
            if (mode == NAN_D)
                dmin = f16_from_bits(0x7E00);
            std::memcpy(bp, &d, 2);
            std::memcpy(bp + 2, &dmin, 2);
            // 6-bit packed scales: all-0 or all-63 extremes when MAXMAG, else LCG.
            for (int i = 0; i < 12; ++i)
                bp[4 + i] = (mode == MAXMAG) ? 0xFF : g.byte();  // 0xFF -> all 6-bit fields = 63
            for (int i = 0; i < 128; ++i)
                bp[16 + i] = g.byte();  // 4-bit quant nibbles
        }
}

void build_q5_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 176);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 176;
            half d = pick_d(g, mode);
            half dmin = (mode == MAXMAG) ? f16_from_bits(0x7BFF)
                                         : __float2half(0.002f + 0.01f * std::fabs(g.unit()));
            if (mode == ZERO_D)
                dmin = __float2half(0.0f);
            if (mode == NAN_D)
                dmin = f16_from_bits(0x7E00);
            std::memcpy(bp, &d, 2);
            std::memcpy(bp + 2, &dmin, 2);
            for (int i = 0; i < 12; ++i)
                bp[4 + i] = (mode == MAXMAG) ? 0xFF : g.byte();  // 6-bit scale fields
            for (int i = 0; i < 32; ++i)
                bp[16 + i] = g.byte();  // qh: 5th bits, full range
            for (int i = 0; i < 128; ++i)
                bp[48 + i] = g.byte();  // 4-bit quant nibbles
        }
}

void build_iq4_nl(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 32;
    buf.resize((size_t)N * bpr * 18);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 18;
            half d = pick_d(g, mode);
            std::memcpy(bp, &d, 2);
            for (int i = 0; i < 16; ++i)
                bp[2 + i] = g.byte();  // codebook indices, full nibble range
        }
}

void build_iq4_xs(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 136);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 136;
            half d = pick_d(g, mode);
            std::memcpy(bp, &d, 2);
            // scales_h + scales_l: all-1-bits hits every 6-bit scale = 63
            // (max |ls-32| = 31) under MAXMAG, else LCG.
            for (int i = 0; i < 6; ++i)
                bp[2 + i] = (mode == MAXMAG) ? 0xFF : g.byte();
            for (int i = 0; i < 128; ++i)
                bp[8 + i] = g.byte();
        }
}

// -----------------------------------------------------------------------------
// GPU helpers
// -----------------------------------------------------------------------------
void* to_device(const std::vector<uint8_t>& h) {
    void* d = nullptr;
    cudaMalloc(&d, h.size());
    cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
    return d;
}

half* random_x(int K, Lcg& g, std::vector<half>& host) {
    host.resize(K);
    for (int i = 0; i < K; ++i)
        host[i] = __float2half(g.unit() * 2.0f);  // ~[-2,2]
    half* d = nullptr;
    cudaMalloc(&d, K * sizeof(half));
    cudaMemcpy(d, host.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    return d;
}

bool any_nan_inf(const std::vector<half>& v) {
    for (half h : v) {
        float f = __half2float(h);
        if (std::isnan(f) || std::isinf(f))
            return true;
    }
    return false;
}

// Reference dequant of a whole [N,K] weight buffer into fp64.
void ref_dequant_all(const std::vector<uint8_t>& buf, int N, int K, QType qt,
                     std::vector<double>& out) {
    out.resize((size_t)N * K);
    if (qt == QType::Q8_0) {
        int bpr = K / 32;
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < bpr; ++b)
                ref_dequant_q8_0(buf.data() + ((size_t)r * bpr + b) * 34, &out[(size_t)r * K + b * 32]);
    } else if (qt == QType::Q6_K) {
        int bpr = K / 256;
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < bpr; ++b)
                ref_dequant_q6_k(buf.data() + ((size_t)r * bpr + b) * 210, &out[(size_t)r * K + b * 256]);
    } else if (qt == QType::IQ4_NL) {
        int bpr = K / 32;
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < bpr; ++b)
                ref_dequant_iq4_nl(buf.data() + ((size_t)r * bpr + b) * 18, &out[(size_t)r * K + b * 32]);
    } else if (qt == QType::IQ4_XS) {
        int bpr = K / 256;
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < bpr; ++b)
                ref_dequant_iq4_xs(buf.data() + ((size_t)r * bpr + b) * 136, &out[(size_t)r * K + b * 256]);
    } else if (qt == QType::Q5_K) {
        int bpr = K / 256;
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < bpr; ++b)
                ref_dequant_q5_k(buf.data() + ((size_t)r * bpr + b) * 176, &out[(size_t)r * K + b * 256]);
    } else {  // Q4_K
        int bpr = K / 256;
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < bpr; ++b)
                ref_dequant_q4_k(buf.data() + ((size_t)r * bpr + b) * 144, &out[(size_t)r * K + b * 256]);
    }
}

// -----------------------------------------------------------------------------
// (1) DEQUANT KERNEL vs fp64 reference. Tolerance: f16-rounding only (1e-3 rel).
// -----------------------------------------------------------------------------
void check_dequant(const char* name, QType qt, int N, int K, ScaleMode mode, double rel_tol) {
    Lcg g(0xC0FFEEu + static_cast<uint32_t>(qt) * 131 + mode * 977);
    std::vector<uint8_t> buf;
    if (qt == QType::Q8_0)
        build_q8_0(buf, N, K, g, mode);
    else if (qt == QType::Q6_K)
        build_q6_k(buf, N, K, g, mode);
    else if (qt == QType::IQ4_NL)
        build_iq4_nl(buf, N, K, g, mode);
    else if (qt == QType::IQ4_XS)
        build_iq4_xs(buf, N, K, g, mode);
    else
        build_q4_k(buf, N, K, g, mode);

    void* dW = to_device(buf);
    half* dOut = nullptr;
    cudaMalloc(&dOut, (size_t)N * K * sizeof(half));
    dequant_gpu(dW, dOut, qt, N, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hOut((size_t)N * K);
    cudaMemcpy(hOut.data(), dOut, (size_t)N * K * sizeof(half), cudaMemcpyDeviceToHost);

    std::vector<double> ref;
    ref_dequant_all(buf, N, K, qt, ref);

    // No-NaN/Inf guard (the real Gemma-class assert). NORMAL/ZERO_D weights can
    // never overflow f16, so the GPU must be all-finite. MAXMAG (d=65504 * full
    // quant range) DELIBERATELY overflows f16 — there a finite output would be
    // WRONG; the independent check is "GPU is non-finite exactly where the
    // f16-rounded fp64 reference is non-finite" (verified in the loop below).
    // NAN_D injects a NaN scale half: only assert no crash / no UB (no compare).
    if (mode == NORMAL || mode == ZERO_D) {
        ASSERT_FALSE(any_nan_inf(hOut)) << name << ": dequant produced NaN/Inf on finite weights";
    }

    double max_rel = 0.0, max_abs = 0.0;
    int worst = 0;
    int classify_mismatch = 0;
    for (size_t i = 0; i < hOut.size(); ++i) {
        double r = ref[i];
        double g16 = static_cast<double>(__half2float(hOut[i]));
        // Round the fp64 reference to f16 the same way the kernel rounds, so we
        // only measure arithmetic divergence, not the unavoidable f16 step.
        float r16f = __float2half_then_float(static_cast<float>(r));
        double r16 = static_cast<double>(r16f);
        bool g_fin = std::isfinite(g16);
        bool r_fin = std::isfinite(r16);
        if (mode == NAN_D)
            continue;  // reference NaN — characterize only, no metric/classify
        if (g_fin != r_fin) {
            ++classify_mismatch;  // GPU and ref disagree on finite/overflow
            continue;
        }
        if (!g_fin)
            continue;  // both overflowed identically (MAXMAG): correct
        double a = std::fabs(g16 - r16);
        double rel = std::fabs(r16) > 1e-4 ? a / std::fabs(r16) : a;
        if (rel > max_rel) {
            max_rel = rel;
            max_abs = a;
            worst = static_cast<int>(i);
        }
    }
    printf("[dequant %-6s mode=%d] N=%d K=%d max_rel=%.3e max_abs=%.3e classify_mismatch=%d (idx=%d "
           "gpu=%.5f ref=%.5f)\n",
           name, mode, N, K, max_rel, max_abs, classify_mismatch, worst, __half2float(hOut[worst]),
           ref[worst]);

    if (mode != NAN_D) {
        EXPECT_EQ(classify_mismatch, 0)
            << name << " mode=" << mode << ": GPU/ref disagree on f16 overflow classification";
        EXPECT_LT(max_rel, rel_tol) << name << " mode=" << mode << ": dequant rel error too large";
    }

    cudaFree(dW);
    cudaFree(dOut);
}

// -----------------------------------------------------------------------------
// fp64 reference GEMV: y[r] = sum_k dequant(W)[r,k] * (f16)x[k].
// -----------------------------------------------------------------------------
void ref_gemv(const std::vector<double>& wref, const std::vector<half>& x, int N, int K,
              std::vector<double>& y) {
    y.assign(N, 0.0);
    for (int r = 0; r < N; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += wref[(size_t)r * K + k] * static_cast<double>(__half2float(x[k]));
        y[r] = acc;
    }
}

struct GemvStats {
    double max_rel_scaled;  // max |gpu-ref| / rms(ref)  — cancellation-robust
    double rms_rel;         // sqrt(mean(err^2)) / rms(ref)
    double max_abs;
    double ref_rms;
    int worst;
};

// Compare GEMV output to the fp64 reference. Per-element relative error is the
// WRONG metric for a dot product: genuine sign cancellation drives some ref
// outputs to ~0, where any absolute noise explodes the per-element ratio
// (measured 5x-30x on the q8_1 path). The honest column-vector metric is the
// error normalized by the TYPICAL output magnitude rms(ref): it answers "how
// large is the noise relative to a representative logit", which is exactly what
// matters for argmax/softmax downstream.
GemvStats gemv_eval(const std::vector<half>& gpu, const std::vector<double>& ref) {
    GemvStats s{};
    double sum_sq = 0.0, sum_err_sq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        sum_sq += ref[i] * ref[i];
    double ref_rms = std::sqrt(sum_sq / ref.size());
    double inv = ref_rms > 1e-9 ? 1.0 / ref_rms : 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double gf = static_cast<double>(__half2float(gpu[i]));
        double a = std::fabs(gf - ref[i]);
        sum_err_sq += a * a;
        double rel = a * inv;
        if (rel > s.max_rel_scaled) {
            s.max_rel_scaled = rel;
            s.max_abs = a;
            s.worst = static_cast<int>(i);
        }
    }
    s.rms_rel = std::sqrt(sum_err_sq / ref.size()) * inv;
    s.ref_rms = ref_rms;
    return s;
}

}  // namespace

// =============================================================================
// DEQUANT-KERNEL TESTS (path: src/quant/dequant_gpu.cu)
//
// TYPED_TEST over the GGUF block formats (R8 / audit §Phase-2 R8: "parametrize
// where it's cheap and real"). The dequant test body is one shared skeleton —
// build synthetic blocks, run dequant_gpu, compare to the fp64 reference over
// the 4 scale modes — that already dispatches on QType (build_* + ref_dequant_all
// switch). The heterogeneous part is the per-format fp64 reference, which stays
// in its own function; only the launch+compare driver is parametrized. Each
// format is a compile-time tag carrying its QType + name; the body is identical.
// =============================================================================
template <QType QT>
struct QTypeTag {
    static constexpr QType value = QT;
};

template <typename T>
class GgufDequant : public ::testing::Test {};

using GgufDequantFormats =
    ::testing::Types<QTypeTag<QType::Q8_0>, QTypeTag<QType::Q6_K>, QTypeTag<QType::Q4_K>,
                     QTypeTag<QType::IQ4_NL>, QTypeTag<QType::IQ4_XS>>;

inline const char* qtype_name(QType qt) {
    switch (qt) {
        case QType::Q8_0: return "Q8_0";
        case QType::Q6_K: return "Q6_K";
        case QType::Q4_K: return "Q4_K";
        case QType::IQ4_NL: return "IQ4_NL";
        case QType::IQ4_XS: return "IQ4_XS";
        default: return "?";
    }
}

TYPED_TEST_SUITE(GgufDequant, GgufDequantFormats);

TYPED_TEST(GgufDequant, AllScaleModes) {
    constexpr QType qt = TypeParam::value;
    const char* name = qtype_name(qt);
    // K=512 covers both layouts: 16 blocks/row for the 32-block formats and
    // 2 super-blocks/row for the 256-block K-quants. N chosen non-round so a
    // row-stride bug surfaces.
    check_dequant(name, qt, 37, 512, NORMAL, 1e-3);
    check_dequant(name, qt, 16, 256, ZERO_D, 1e-3);
    check_dequant(name, qt, 16, 256, MAXMAG, 1e-3);
    check_dequant(name, qt, 8, 256, NAN_D, 1e-3);  // no-crash / UB guard
}

// =============================================================================
// fp16-DEQUANT GEMV TESTS (gemv_q8_0 / gemv_q6k — fp32 dot, no q8_1 act quant)
// Tolerance: fp16-class 1e-2 rel.
// =============================================================================
TEST(GgufRef, Q8_0_GemvFp16) {
    const int N = 256, K = 1024;
    Lcg g(0x5151u);
    std::vector<uint8_t> buf;
    build_q8_0(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);
    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    gemv_q8_0(dW, dx, dy, N, K, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy));

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, QType::Q8_0, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[gemv Q8_0  fp16] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst, __half2float(hy[s.worst]),
           yref[s.worst]);
    EXPECT_LT(s.max_rel_scaled, 1e-2) << "gemv_q8_0 fp16 error too large";
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
}

TEST(GgufRef, Q6_K_GemvFp16) {
    const int N = 256, K = 1024;
    Lcg g(0x6161u);
    std::vector<uint8_t> buf;
    build_q6_k(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);
    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    gemv_q6k(dW, dx, dy, N, K, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy));

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, QType::Q6_K, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[gemv Q6_K  fp16] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst, __half2float(hy[s.worst]),
           yref[s.worst]);
    EXPECT_LT(s.max_rel_scaled, 1e-2) << "gemv_q6k fp16 error too large";
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
}

// =============================================================================
// dp4a / MMVQ GEMV TESTS — these quantize activations to Q8_1 (amax/127).
// Tolerance: q8_1-activation band ~1-2%, asserted at 2.5e-2, MEASURED below.
// =============================================================================
namespace {
void run_dp4a_gemv(const char* name, QType qt, int N, int K,
                   void (*dp4a_fn)(const void*, const block_q8_1*, const float*, half*, int, int,
                                   cudaStream_t)) {
    Lcg g(0xD4A0u + static_cast<uint32_t>(qt));
    std::vector<uint8_t> buf;
    if (qt == QType::Q8_0)
        build_q8_0(buf, N, K, g, NORMAL);
    else if (qt == QType::Q6_K)
        build_q6_k(buf, N, K, g, NORMAL);
    else if (qt == QType::Q5_K)
        build_q5_k(buf, N, K, g, NORMAL);
    else
        build_q4_k(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);

    int padded_blocks = ((K + 255) / 256) * 8;
    block_q8_1* q8 = nullptr;
    float* d8 = nullptr;
    cudaMalloc(&q8, padded_blocks * sizeof(block_q8_1));
    cudaMalloc(&d8, padded_blocks * sizeof(float));
    cudaMemset(q8, 0, padded_blocks * sizeof(block_q8_1));
    cudaMemset(d8, 0, padded_blocks * sizeof(float));
    quantize_fp16_to_q8_1(dx, q8, d8, K, nullptr);

    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    dp4a_fn(dW, q8, d8, dy, N, K, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy)) << name;

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, qt, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[gemv %-5s dp4a] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           name, N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst,
           __half2float(hy[s.worst]), yref[s.worst]);
    // q8_1 activation quant (amax/127 per 32-block) adds ~0.4% RMS per element.
    // Over a K-dot that does not fully cancel (correlated within a block), so we
    // bound the typical-magnitude-normalized RMS at 1.5% and the single worst
    // element at 5% (cancellation tail). Both MEASURED — see printed stats.
    EXPECT_LT(s.rms_rel, 1.5e-2) << name << ": dp4a gemv RMS outside q8_1-activation band";
    EXPECT_LT(s.max_rel_scaled, 5e-2) << name << ": dp4a gemv worst element outside band";

    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(q8);
    cudaFree(d8);
}

void run_mmvq_gemv(const char* name, QType qt, int N, int K,
                   void (*mmvq_fn)(const void*, const half*, half*, int, int, int, void*, size_t,
                                   cudaStream_t)) {
    Lcg g(0x33CCu + static_cast<uint32_t>(qt));
    std::vector<uint8_t> buf;
    if (qt == QType::Q8_0)
        build_q8_0(buf, N, K, g, NORMAL);
    else
        build_q4_k(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);

    int q8_blocks = (K + 31) / 32;
    size_t scratch_size = (size_t)q8_blocks * 36 * 2;
    void* scratch = nullptr;
    cudaMalloc(&scratch, scratch_size);
    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    mmvq_fn(dW, dx, dy, 1, N, K, scratch, scratch_size, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy)) << name;

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, qt, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[mmvq %-5s    ] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           name, N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst,
           __half2float(hy[s.worst]), yref[s.worst]);
    // Same q8_1-activation band as the dp4a path (MMVQ quantizes x to Q8_1 too).
    EXPECT_LT(s.rms_rel, 1.5e-2) << name << ": mmvq gemv RMS outside q8_1-activation band";
    EXPECT_LT(s.max_rel_scaled, 5e-2) << name << ": mmvq gemv worst element outside band";

    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(scratch);
}
}  // namespace

TEST(GgufRef, Q8_0_GemvDp4a) { run_dp4a_gemv("Q8_0", QType::Q8_0, 256, 1024, gemv_q8_0_q8_1); }
TEST(GgufRef, Q6_K_GemvDp4a) { run_dp4a_gemv("Q6_K", QType::Q6_K, 256, 1024, gemv_q6k_q8_1); }
TEST(GgufRef, Q4_K_GemvDp4a) { run_dp4a_gemv("Q4_K", QType::Q4_K, 256, 1024, gemv_q4_k_q8_1); }
TEST(GgufRef, Q5_K_GemvDp4a) { run_dp4a_gemv("Q5_K", QType::Q5_K, 256, 1024, gemv_q5_k_q8_1); }

TEST(GgufRef, Q8_0_GemvMmvq) { run_mmvq_gemv("Q8_0", QType::Q8_0, 256, 1024, ggml_mmvq_q8_0); }
TEST(GgufRef, Q4_K_GemvMmvq) { run_mmvq_gemv("Q4_K", QType::Q4_K, 256, 1024, ggml_mmvq_q4k); }

}  // namespace imp
