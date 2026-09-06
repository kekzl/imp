#pragma once
// =============================================================================
// GGUF block formats, twice over, with no imp in either half.
//
// This header is the class-A anchor `tests/test_gguf_dequant_ref.cu` compares
// imp's dequant and GEMV kernels against. For every quant format the GGUF
// loader accepts it holds two independent pieces:
//
//   * a BYTE-LEVEL BUILDER that fills raw block bytes from a deterministic LCG
//     and picks the scale halfs separately — never imp's quantizer, so no
//     quantize->dequant round trip can make a wrong layout look right;
//   * an fp64 REFERENCE DEQUANT reconstructed from the ggml format definition
//     (`ggml-common.h` block structs for the layout, `ggml-quants.c`
//     `dequantize_row_*` for the arithmetic), justified per format in the
//     comments and never a call into imp.
//
// `kFormats` ties the two together per QType and is what makes the coverage
// gate two-way: a format `dequant_gpu_supported()` accepts with no row here is
// a numerics path nobody checks, and that is how Q3_K shipped with its high-bit
// plane read wrong in both of its kernels (AUDIT_arch_2026 D-5).
//
// It is a header because the builders and references are the reusable half; the
// GPU launches, tolerances and test bodies live in the .cu that includes it.
// =============================================================================

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "core/qtype.h"

namespace imp {
namespace gguf_ref {

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
inline void ref_dequant_q8_0(const uint8_t* blk, double* out) {
    half d;
    std::memcpy(&d, blk, 2);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    double dd = f16_to_f64(d);
    for (int i = 0; i < 32; ++i)
        out[i] = dd * static_cast<double>(qs[i]);
}

// Q4_0: 18 bytes / 32 elems. [ d:f16 | qs:u8[16] ]. 4-bit symmetric quant
// centered at 8. ggml dequantize_row_q4_0: for j in 0..15,
//   y[j]    = d * ((qs[j] & 0xF) - 8)   (low nibble)
//   y[j+16] = d * ((qs[j] >> 4)  - 8)   (high nibble)
inline void ref_dequant_q4_0(const uint8_t* blk, double* out) {
    half d;
    std::memcpy(&d, blk, 2);
    const uint8_t* qs = blk + 2;
    double dd = f16_to_f64(d);
    for (int j = 0; j < 16; ++j) {
        out[j] = dd * (static_cast<double>(qs[j] & 0xF) - 8.0);
        out[j + 16] = dd * (static_cast<double>((qs[j] >> 4) & 0xF) - 8.0);
    }
}

// Q6_K: 210 bytes / 256 elems. [ ql:u8[128] | qh:u8[64] | scales:int8[16] | d:f16 ].
// ggml dequantize_row_q6_K: for each of 2 groups of 128, four 32-quads:
//   q = (high2<<4 | low4) - 32 ; val = d * scale[i/16] * q
// (6-bit signed quant centered at 32; sub-block scale every 16 elems.)
inline void ref_dequant_q6_k(const uint8_t* blk, double* out) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
    half d;
    std::memcpy(&d, blk + 208, 2);
    double dd = f16_to_f64(d);
    for (int i = 0; i < 256; ++i) {
        int group = i >> 7;      // 0..1
        int within = i & 127;    // 0..127
        int quad = within >> 5;  // 0..3
        int l = within & 31;     // 0..31
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
inline void ref_get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
inline void ref_dequant_q4_k(const uint8_t* blk, double* out) {
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
        out[i] = dd * static_cast<double>(scv) * static_cast<double>(q4) - dm * static_cast<double>(mnv);
    }
}

// Q5_K: 176 bytes / 256 elems. [ d:f16 | dmin:f16 | scales:u8[12] | qh:u8[32] |
// qs:u8[128] ]. Same 6-bit (scale,min) packing as Q4_K. The 5th bit of
// element i lives in qh[i % 32] at bit position sub = i/32 (ggml
// dequantize_row_q5_K). val = d*scale*(q4 + 16*bit) - dmin*min.
inline void ref_dequant_q5_k(const uint8_t* blk, double* out) {
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
        out[i] = dd * static_cast<double>(scv) * static_cast<double>(q5) - dm * static_cast<double>(mnv);
    }
}

// IQ4_NL: 18 bytes / 32 elems. [ d:f16 | qs:u8[16] ]. Non-linear 4-bit: the
// nibble indexes a fixed signed codebook (ggml-common.h kvalues_iq4nl).
// Element j (0..15) = low nibble of qs[j]; element j+16 = high nibble
// (ggml dequantize_row_iq4_nl). val = d * codebook[nibble].
static const int8_t kRefIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,    13,   25,  38,  53,  69,  89,  113};
inline void ref_dequant_iq4_nl(const uint8_t* blk, double* out) {
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
inline void ref_dequant_iq4_xs(const uint8_t* blk, double* out) {
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

// =============================================================================
// AUDIT_arch_2026 D-5: the six GGUF formats imp dequantizes with no reference.
// Same independence rule as above — layout from the format definition
// (ggml-common.h block structs), arithmetic reconstructed from
// ggml-quants.c dequantize_row_*, never a call into imp.
// =============================================================================

// Q4_1: 20 bytes / 32 elems. [ d:f16 | m:f16 | qs:u8[16] ]. Asymmetric 4-bit:
// no offset on the quant, an additive per-block minimum instead.
// ggml dequantize_row_q4_1: y[j] = d*(qs[j]&0xF) + m, y[j+16] = d*(qs[j]>>4) + m.
inline void ref_dequant_q4_1(const uint8_t* blk, double* out) {
    half d, m;
    std::memcpy(&d, blk, 2);
    std::memcpy(&m, blk + 2, 2);
    const uint8_t* qs = blk + 4;
    double dd = f16_to_f64(d);
    double mm = f16_to_f64(m);
    for (int j = 0; j < 16; ++j) {
        out[j] = dd * static_cast<double>(qs[j] & 0xF) + mm;
        out[j + 16] = dd * static_cast<double>((qs[j] >> 4) & 0xF) + mm;
    }
}

// Q5_0: 22 bytes / 32 elems. [ d:f16 | qh:u8[4] | qs:u8[16] ]. The 5th bit of
// element j sits in bit j of the qh word, the 5th bit of element j+16 in bit
// j+12 — NOT j+16 (ggml packs the high halves 4 bits apart, the trap this
// reference exists to catch). Symmetric, centered at 16.
// ggml: xh_0 = ((qh >> j) << 4) & 0x10; xh_1 = ((qh >> (j+12))) & 0x10;
//       y[j] = d*(((qs[j]&0xF)|xh_0) - 16), y[j+16] = d*(((qs[j]>>4)|xh_1) - 16).
inline void ref_dequant_q5_0(const uint8_t* blk, double* out) {
    half d;
    std::memcpy(&d, blk, 2);
    uint32_t qh;
    std::memcpy(&qh, blk + 2, 4);
    const uint8_t* qs = blk + 6;
    double dd = f16_to_f64(d);
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = static_cast<uint8_t>((qh >> j) << 4) & 0x10;
        const uint8_t xh_1 = static_cast<uint8_t>(qh >> (j + 12)) & 0x10;
        const int x0 = static_cast<int>((qs[j] & 0x0F) | xh_0) - 16;
        const int x1 = static_cast<int>((qs[j] >> 4) | xh_1) - 16;
        out[j] = dd * static_cast<double>(x0);
        out[j + 16] = dd * static_cast<double>(x1);
    }
}

// Q5_1: 24 bytes / 32 elems. [ d:f16 | m:f16 | qh:u8[4] | qs:u8[16] ]. Q5_0's
// 5th-bit packing with Q4_1's additive minimum and no offset.
// ggml: y[j] = d*((qs[j]&0xF)|xh_0) + m, y[j+16] = d*((qs[j]>>4)|xh_1) + m.
inline void ref_dequant_q5_1(const uint8_t* blk, double* out) {
    half d, m;
    std::memcpy(&d, blk, 2);
    std::memcpy(&m, blk + 2, 2);
    uint32_t qh;
    std::memcpy(&qh, blk + 4, 4);
    const uint8_t* qs = blk + 8;
    double dd = f16_to_f64(d);
    double mm = f16_to_f64(m);
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = static_cast<uint8_t>((qh >> j) << 4) & 0x10;
        const uint8_t xh_1 = static_cast<uint8_t>(qh >> (j + 12)) & 0x10;
        const int x0 = static_cast<int>((qs[j] & 0x0F) | xh_0);
        const int x1 = static_cast<int>((qs[j] >> 4) | xh_1);
        out[j] = dd * static_cast<double>(x0) + mm;
        out[j + 16] = dd * static_cast<double>(x1) + mm;
    }
}

// Q2_K: 84 bytes / 256 elems. [ scales:u8[16] | qs:u8[64] | d:f16 | dmin:f16 ].
// 16 sub-blocks of 16; each scale byte holds a 4-bit scale (low) and a 4-bit
// min (high). 2-bit quants, four per byte: the 64-byte quant array is walked in
// two 128-element halves, and within a half the shift (0,2,4,6) selects the
// sub-block PAIR while l indexes the 16 elements — output order is
// (shift-major, then +0 / +16 group), which is what a naive "q[i/4] >> (2*(i%4))"
// reader gets wrong. ggml dequantize_row_q2_K: val = d*(sc&0xF)*q2 - dmin*(sc>>4).
inline void ref_dequant_q2_k(const uint8_t* blk, double* out) {
    const uint8_t* scales = blk;
    const uint8_t* qs = blk + 16;
    half d, dmin;
    std::memcpy(&d, blk + 80, 2);
    std::memcpy(&dmin, blk + 82, 2);
    const double dd = f16_to_f64(d);
    const double dm = f16_to_f64(dmin);
    int is = 0;
    double* y = out;
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* q = qs + (n / 128) * 32;
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = scales[is++];
            double dl = dd * static_cast<double>(sc & 0xF);
            double ml = dm * static_cast<double>(sc >> 4);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * static_cast<double>((q[l] >> shift) & 3) - ml;
            sc = scales[is++];
            dl = dd * static_cast<double>(sc & 0xF);
            ml = dm * static_cast<double>(sc >> 4);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * static_cast<double>((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
    }
}

// Q3_K: 110 bytes / 256 elems. [ hmask:u8[32] | qs:u8[64] | scales:u8[12] | d:f16 ].
// 16 sub-blocks of 16 with a 6-bit signed scale each, packed into 12 bytes by
// ggml's aux shuffle below (low 4 bits of the first 8 bytes, the missing 2 bits
// spread over the last 4). The quant is 2 low bits from qs plus ONE INVERTED
// high bit from hmask: a set hmask bit means "no -4", a clear bit subtracts 4.
// The hmask bit walks m = 1<<(j) across the 4 shifts, i.e. one bitplane per
// sub-block pair. ggml dequantize_row_q3_K: val = d*(scale-32)*(q2 - (hm?0:4)).
inline void ref_dequant_q3_k(const uint8_t* blk, double* out) {
    const uint8_t* hm = blk;
    const uint8_t* qs = blk + 32;
    half d;
    std::memcpy(&d, blk + 108, 2);
    const double d_all = f16_to_f64(d);

    // ggml's 12-byte -> 16 x 6-bit unpack, verbatim (kmask1/kmask2 shuffle).
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    std::memcpy(aux, blk + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    int is = 0;
    uint8_t m = 1;
    double* y = out;
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* q = qs + (n / 128) * 32;
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            double dl = d_all * static_cast<double>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (static_cast<double>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0.0 : 4.0));
            dl = d_all * static_cast<double>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (static_cast<double>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0.0 : 4.0));
            shift += 2;
            m <<= 1;
        }
    }
}

// Q8_K: 292 bytes / 256 elems. [ d:f32 | qs:int8[256] | bsums:int16[16] ].
// The only GGUF weight format here whose scale is a full float, and the bsums
// tail is dot-product bookkeeping that dequant must ignore.
// ggml dequantize_row_q8_K: y = d * qs[j].
inline void ref_dequant_q8_k(const uint8_t* blk, double* out) {
    float d;
    std::memcpy(&d, blk, 4);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
    const double dd = static_cast<double>(d);
    for (int i = 0; i < 256; ++i)
        out[i] = dd * static_cast<double>(qs[i]);
}
// -----------------------------------------------------------------------------
// Synthetic block builders — raw bytes via LCG, scale halfs chosen separately.
// scale_mode: 0 = "normal" random-ish scales; 1 = d=0 (degenerate);
//             2 = max-magnitude finite scale half; 3 = NaN d-half (guard only).
// 6-bit packed scales are filled to hit all-0 / all-63 extremes when forced.
// -----------------------------------------------------------------------------
enum ScaleMode { NORMAL = 0, ZERO_D = 1, MAXMAG = 2, NAN_D = 3 };

inline half pick_d(Lcg& g, ScaleMode mode) {
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

inline void build_q8_0(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
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

inline void build_q4_0(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 32;
    buf.resize((size_t)N * bpr * 18);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 18;
            half d = pick_d(g, mode);
            std::memcpy(bp, &d, 2);
            for (int i = 0; i < 16; ++i)
                bp[2 + i] = g.byte();  // packed 4-bit nibbles, full range
        }
}

inline void build_q6_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 210);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 210;
            for (int i = 0; i < 192; ++i)
                bp[i] = g.byte();         // ql[128] + qh[64]
            for (int i = 0; i < 16; ++i)  // int8 sub-block scales
                bp[192 + i] = (mode == MAXMAG) ? 127 : g.byte();
            half d = pick_d(g, mode);
            std::memcpy(bp + 208, &d, 2);
        }
}

inline void build_q4_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
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

inline void build_q5_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
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

inline void build_iq4_nl(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
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

inline void build_iq4_xs(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
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

// Q8_K's scale is a full float, not an f16 half — its own picker.
inline float pick_d_f32(Lcg& g, ScaleMode mode) {
    switch (mode) {
        case ZERO_D:
            return 0.0f;
        case MAXMAG:
            return 65504.0f;  // * 127 quants overflows f16 on store
        case NAN_D:
            return std::numeric_limits<float>::quiet_NaN();
        default:
            return 0.005f + 0.02f * std::fabs(g.unit());
    }
}

// The additive minimum of Q4_1 / Q5_1 and the dmin of Q2_K follow d's mode:
// MAXMAG must overflow through BOTH terms, ZERO_D must zero the whole block.
inline half pick_min(Lcg& g, ScaleMode mode) {
    switch (mode) {
        case ZERO_D:
            return __float2half(0.0f);
        case MAXMAG:
            return f16_from_bits(0x7BFF);
        case NAN_D:
            return f16_from_bits(0x7E00);
        default:
            return __float2half(0.002f + 0.01f * std::fabs(g.unit()));
    }
}

inline void build_q4_1(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 32;
    buf.resize((size_t)N * bpr * 20);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 20;
            half d = pick_d(g, mode);
            half m = pick_min(g, mode);
            std::memcpy(bp, &d, 2);
            std::memcpy(bp + 2, &m, 2);
            for (int i = 0; i < 16; ++i)
                bp[4 + i] = g.byte();  // packed 4-bit nibbles, full range
        }
}

inline void build_q5_0(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 32;
    buf.resize((size_t)N * bpr * 22);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 22;
            half d = pick_d(g, mode);
            std::memcpy(bp, &d, 2);
            for (int i = 0; i < 4; ++i)
                bp[2 + i] = g.byte();  // qh: every 5th bit exercised
            for (int i = 0; i < 16; ++i)
                bp[6 + i] = g.byte();
        }
}

inline void build_q5_1(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 32;
    buf.resize((size_t)N * bpr * 24);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 24;
            half d = pick_d(g, mode);
            half m = pick_min(g, mode);
            std::memcpy(bp, &d, 2);
            std::memcpy(bp + 2, &m, 2);
            for (int i = 0; i < 4; ++i)
                bp[4 + i] = g.byte();
            for (int i = 0; i < 16; ++i)
                bp[8 + i] = g.byte();
        }
}

inline void build_q2_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 84);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 84;
            // scales: 0xFF forces scale=15 AND min=15 (both 4-bit fields max).
            for (int i = 0; i < 16; ++i)
                bp[i] = (mode == MAXMAG) ? 0xFF : g.byte();
            for (int i = 0; i < 64; ++i)
                bp[16 + i] = g.byte();  // 2-bit quants, all four per byte
            half d = pick_d(g, mode);
            half dmin = pick_min(g, mode);
            std::memcpy(bp + 80, &d, 2);
            std::memcpy(bp + 82, &dmin, 2);
        }
}

inline void build_q3_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 110);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 110;
            for (int i = 0; i < 32; ++i)
                bp[i] = g.byte();  // hmask: both polarities of the inverted bit
            for (int i = 0; i < 64; ++i)
                bp[32 + i] = g.byte();
            // 6-bit packed scales: 0xFF puts every unpacked field at 63 (= +31
            // after the -32 bias), the max-magnitude end of the signed range.
            for (int i = 0; i < 12; ++i)
                bp[96 + i] = (mode == MAXMAG) ? 0xFF : g.byte();
            half d = pick_d(g, mode);
            std::memcpy(bp + 108, &d, 2);
        }
}

inline void build_q8_k(std::vector<uint8_t>& buf, int N, int K, Lcg& g, ScaleMode mode) {
    int bpr = K / 256;
    buf.resize((size_t)N * bpr * 292);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b) {
            uint8_t* bp = buf.data() + ((size_t)r * bpr + b) * 292;
            float d = pick_d_f32(g, mode);
            std::memcpy(bp, &d, 4);
            for (int i = 0; i < 256; ++i)
                bp[4 + i] = g.byte();  // int8 quants
            for (int i = 0; i < 32; ++i)
                bp[260 + i] = g.byte();  // bsums — dequant must ignore these
        }
}
// -----------------------------------------------------------------------------
// The format table: block geometry + the two independent halves (byte builder,
// fp64 reference) per QType. One row per format the GGUF loader accepts, so a
// format added to `dequant_gpu` with no row here is visible as a missing row
// rather than as a silently untested kernel (AUDIT_arch_2026 D-5).
// -----------------------------------------------------------------------------
struct FormatSpec {
    QType qt;
    const char* name;
    int block_bytes;
    int block_elems;
    void (*ref)(const uint8_t*, double*);
    void (*build)(std::vector<uint8_t>&, int, int, Lcg&, ScaleMode);
};

const FormatSpec kFormats[] = {
    {QType::Q8_0, "Q8_0", 34, 32, ref_dequant_q8_0, build_q8_0},
    {QType::Q4_0, "Q4_0", 18, 32, ref_dequant_q4_0, build_q4_0},
    {QType::Q4_1, "Q4_1", 20, 32, ref_dequant_q4_1, build_q4_1},
    {QType::Q5_0, "Q5_0", 22, 32, ref_dequant_q5_0, build_q5_0},
    {QType::Q5_1, "Q5_1", 24, 32, ref_dequant_q5_1, build_q5_1},
    {QType::Q2_K, "Q2_K", 84, 256, ref_dequant_q2_k, build_q2_k},
    {QType::Q3_K, "Q3_K", 110, 256, ref_dequant_q3_k, build_q3_k},
    {QType::Q4_K, "Q4_K", 144, 256, ref_dequant_q4_k, build_q4_k},
    {QType::Q5_K, "Q5_K", 176, 256, ref_dequant_q5_k, build_q5_k},
    {QType::Q6_K, "Q6_K", 210, 256, ref_dequant_q6_k, build_q6_k},
    {QType::Q8_K, "Q8_K", 292, 256, ref_dequant_q8_k, build_q8_k},
    {QType::IQ4_NL, "IQ4_NL", 18, 32, ref_dequant_iq4_nl, build_iq4_nl},
    {QType::IQ4_XS, "IQ4_XS", 136, 256, ref_dequant_iq4_xs, build_iq4_xs},
};

inline const FormatSpec& format_spec(QType qt) {
    for (const FormatSpec& f : kFormats)
        if (f.qt == qt)
            return f;
    ADD_FAILURE() << "no reference for QType " << static_cast<unsigned>(qt);
    return kFormats[0];
}

// Reference dequant of a whole [N,K] weight buffer into fp64.
inline void ref_dequant_all(const std::vector<uint8_t>& buf, int N, int K, QType qt,
                            std::vector<double>& out) {
    out.resize((size_t)N * K);
    const FormatSpec& f = format_spec(qt);
    const int bpr = K / f.block_elems;
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < bpr; ++b)
            f.ref(buf.data() + ((size_t)r * bpr + b) * f.block_bytes,
                  &out[(size_t)r * K + (size_t)b * f.block_elems]);
}

}  // namespace gguf_ref
}  // namespace imp
