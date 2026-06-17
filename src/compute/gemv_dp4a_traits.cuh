#pragma once
// Template infrastructure for dp4a-accelerated GEMV kernels.
// Replaces ~33 hand-written kernel functions with 6 template kernels.
// Each quant type provides a DequantTraits specialization with one dp4a_block() function.

#include "compute/gemm.h"
#include "runtime/pdl.h"
#include "compute/ptx92_utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <type_traits>

namespace imp {

// Smem Q8_1 stride: 9 int32s per block (8 data + 1 padding) to eliminate bank conflicts.
// With stride 8: lanes 0,4,8,... all hit the same shared memory bank → 8-way conflict.
// With stride 9: lane i starts at bank (i*9)%32 — all unique, zero conflicts.
static constexpr int kSmemQ8Stride = 9;

// File-local tag enum for template dispatch (distinct from imp::QType).
enum class DPQTag { Q4_0, Q8_0, Q6_K, Q4_K, Q5_K, Q2_K, Q3_K, Q5_1 };

// ============================================================================
// Helper device functions (moved from gemm.cu, unchanged)
// ============================================================================

__device__ __forceinline__ float q6k_dp4a_group_preloaded(
    const uint8_t* __restrict__ ql, const uint8_t* __restrict__ qh, const int8_t* __restrict__ sc, float d_w,
    const int* __restrict__ xqs_packed,  // [8] pre-loaded int32 from Q8_1
    float d_x, int g) {
    const int ql_base = (g / 4) * 64 + (g % 2) * 32;
    const int is_high = ((g % 4) >= 2);
    const int qh_base = (g < 4) ? 0 : 32;
    const int qh_shift = (g % 4) * 2;

    float group_sum = 0.0f;

#pragma unroll
    for (int sb = 0; sb < 2; sb++) {
        const int8_t sc_val = sc[2 * g + sb];
        const int sub_off = sb * 16;
        int32_t sumi = 0;

#pragma unroll
        for (int d4 = 0; d4 < 4; d4++) {
            const int k = sub_off + d4 * 4;

            uint32_t ql4;
            memcpy(&ql4, ql + ql_base + k, 4);
            const uint32_t lo4 = is_high ? ((ql4 >> 4) & 0x0F0F0F0FU) : (ql4 & 0x0F0F0F0FU);
            uint32_t qh4;
            memcpy(&qh4, qh + qh_base + k, 4);
            const uint32_t hi4 = ((qh4 >> qh_shift) & 0x03030303U) << 4;
            const int vi = __vsubss4(lo4 | hi4, 0x20202020U);
            sumi = __dp4a(vi, xqs_packed[sb * 4 + d4], sumi);
        }
        group_sum += d_w * d_x * (float)sc_val * (float)sumi;
    }
    return group_sum;
}

// 6-bit scale/min unpacker matching ggml get_scale_min_k4, register variant
// (#598): the 12 scale bytes arrive as three words (s0 = sc[0..3],
// s1 = sc[4..7], s2 = sc[8..11]) from a single uint4 header load instead of
// per-byte L1 traffic. Byte-pointer reference copies live in
// mmq_q4k_imma_layout.cu / gemv_ggml_compat.cu.
__device__ __forceinline__ void get_scale_min_k4_reg(uint32_t s0, uint32_t s1, uint32_t s2, int sub,
                                                     uint8_t& sc_val, uint8_t& min_val) {
    auto byte_of = [](uint32_t w, int i) -> uint32_t { return (w >> (8 * i)) & 0xFFu; };
    if (sub < 4) {
        sc_val = byte_of(s0, sub) & 63;
        min_val = byte_of(s1, sub) & 63;
    } else {
        sc_val = (byte_of(s2, sub - 4) & 0xF) | ((byte_of(s0, sub - 4) >> 6) << 4);
        min_val = (byte_of(s2, sub - 4) >> 4) | ((byte_of(s1, sub - 4) >> 6) << 4);
    }
}

__device__ __forceinline__ float q4k_dp4a_sub(const uint8_t* __restrict__ qs,  // Q4_K qs base (128 bytes)
                                              int sub,                         // sub-block index (0..7)
                                              float d_super,                   // super-block scale
                                              float dmin_super,                // super-block min
                                              uint8_t sc_val,                  // 6-bit sub-block scale
                                              uint8_t min_val,                 // 6-bit sub-block min
                                              const int* __restrict__ xi,      // [8] packed Q8_1 int32 values
                                              float dq) {                      // Q8_1 block scale
    const int qs_byte_offset = (sub / 2) * 32;
    const bool use_high = (sub & 1);
    const uint8_t* qs_base = qs + qs_byte_offset;

    // Two LDG.128 instead of eight LDG.32 (#598): this kernel class is
    // L1TEX-instruction-bound (L1 hit 98.7%, DRAM 42%), so load count is
    // the limiter. Alignment is guaranteed: the 144-byte superblock is
    // 16B-aligned (144 = 9*16, GGUF tensor alignment 32) and qs sits at
    // +16 + 32*k within it.
    const uint4 lo = *reinterpret_cast<const uint4*>(qs_base);
    const uint4 hi = *reinterpret_cast<const uint4*>(qs_base + 16);
    const uint32_t qsw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    int32_t sumi = 0;
    int q8_sum_int = 0;
    const int ones = 0x01010101;

#pragma unroll
    for (int j = 0; j < 8; j++) {
        uint32_t nibbles = use_high ? ((qsw[j] >> 4) & 0x0F0F0F0Fu) : (qsw[j] & 0x0F0F0F0Fu);
        sumi = __dp4a(static_cast<int>(nibbles), xi[j], sumi);
        q8_sum_int = __dp4a(xi[j], ones, q8_sum_int);
    }

    return dq * (d_super * (float)sc_val * (float)sumi - dmin_super * (float)min_val * (float)q8_sum_int);
}

__device__ __forceinline__ float q5k_dp4a_sub(
    const uint8_t* __restrict__ qs,  // Q5_K qs base (128 bytes, offset +48)
    const uint8_t* __restrict__ qh,  // Q5_K qh base (32 bytes, offset +16)
    int sub,                         // sub-block index (0..7)
    float d_super,                   // super-block scale
    float dmin_super,                // super-block min
    uint8_t sc_val,                  // 6-bit sub-block scale
    uint8_t min_val,                 // 6-bit sub-block min
    const int* __restrict__ xi,      // [8] packed Q8_1 int32 values
    float dq) {                      // Q8_1 block scale
    // Ref layout (ggml dequantize_row_q5_K): qh is 32 bytes shared across
    // all 8 subs. Element at position `i` within a sub uses bit `sub` of
    // qh[i]. i.e. qh[l] byte holds the 5th bit for element l of EVERY sub,
    // at bit position `sub`. Our prior code treated qh as `sub*4` private
    // bytes with bits 0..7 encoding 4 elements — completely wrong layout.
    const int qs_byte_offset = (sub / 2) * 32;
    const bool use_high = (sub & 1);
    const uint8_t* qs_base = qs + qs_byte_offset;

    // Vectorized loads (#598, same rationale as q4k_dp4a_sub): qs sits at
    // +48 + 32*k (16B-aligned), qh at +16 (16B-aligned, 32 bytes shared by
    // all subs). 4x LDG.128 replaces 16x LDG.32 per call.
    const uint4 qlo = *reinterpret_cast<const uint4*>(qs_base);
    const uint4 qhi = *reinterpret_cast<const uint4*>(qs_base + 16);
    const uint32_t qsw[8] = {qlo.x, qlo.y, qlo.z, qlo.w, qhi.x, qhi.y, qhi.z, qhi.w};
    const uint4 h0 = *reinterpret_cast<const uint4*>(qh);
    const uint4 h1 = *reinterpret_cast<const uint4*>(qh + 16);
    const uint32_t qhw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};

    int32_t sumi = 0;
    int32_t sumi_h = 0;  // 5th-bit correction
    int q8_sum_int = 0;
    const int ones = 0x01010101;

#pragma unroll
    for (int j = 0; j < 8; j++) {
        uint32_t nibbles = use_high ? ((qsw[j] >> 4) & 0x0F0F0F0Fu) : (qsw[j] & 0x0F0F0F0Fu);
        sumi = __dp4a(static_cast<int>(nibbles), xi[j], sumi);
        q8_sum_int = __dp4a(xi[j], ones, q8_sum_int);

        // Four consecutive elements l = j*4 .. j*4+3 live in word qhw[j]
        // (byte l%4). Each extracts bit `sub` and places the 0/1 into bit 4
        // (→ value 16) of the corresponding byte, matching how `nibbles`
        // was built from nibbles 0..15.
        const uint32_t w = qhw[j];
        uint32_t hbits = ((((w >> 0) >> sub) & 1u) << 4) | ((((w >> 8) >> sub) & 1u) << 12) |
                         ((((w >> 16) >> sub) & 1u) << 20) | ((((w >> 24) >> sub) & 1u) << 28);
        sumi_h = __dp4a(static_cast<int>(hbits), xi[j], sumi_h);
    }

    return dq * (d_super * (float)sc_val * (float)(sumi + sumi_h) -
                 dmin_super * (float)min_val * (float)q8_sum_int);
}

// ============================================================================
// DequantTraits<DPQTag> — compile-time constants + dp4a_block() per type
//
// dp4a_block(bp, sub, xi, dq, q8_sum):
//   bp      — pointer to the start of the weight block/super-block
//   sub     — sub-group index within super-block (0 for Q8_0/Q4_0; 0-7 for Q6_K/Q4_K/Q5_K)
//   xi[8]   — pre-loaded Q8_1 int32 packed values
//   dq      — Q8_1 block scale
//   q8_sum  — sum of Q8_1 int8 values (only used by Q4_0)
// ============================================================================

template <DPQTag Q>
struct DequantTraits;

template <>
struct DequantTraits<DPQTag::Q6_K> {
    static constexpr int kBlockBytes = 210;
    static constexpr int kBlockElems = 256;
    static constexpr int kQ8PerWeight = 8;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 2;        // NR=4 uses 64 regs → occupancy drop
    static constexpr bool kPreferKpar = true;  // compute-heavy dequant: K-par wins on ties

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int sub, const int* xi, float dq,
                                                       float /*q8_sum*/) {
        float d_w = __half2float(*reinterpret_cast<const half*>(bp + 208));
        return q6k_dp4a_group_preloaded(bp, bp + 128, reinterpret_cast<const int8_t*>(bp + 192), d_w, xi, dq,
                                        sub);
    }
};

template <>
struct DequantTraits<DPQTag::Q8_0> {
    static constexpr int kBlockBytes = 34;
    static constexpr int kBlockElems = 32;
    static constexpr int kQ8PerWeight = 1;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 2;         // NR=4 uses 59 regs → occupancy drop
    static constexpr bool kPreferKpar = false;  // simple dequant: row-par smem wins on ties

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int /*sub*/, const int* xi,
                                                       float dq, float /*q8_sum*/) {
        half d_w_h;
        memcpy(&d_w_h, bp, sizeof(half));
        float d_w = __half2float(d_w_h);
        int wi[8];
        memcpy(wi, bp + 2, 32);

        int32_t sumi = 0;
        sumi = __dp4a(wi[0], xi[0], sumi);
        sumi = __dp4a(wi[1], xi[1], sumi);
        sumi = __dp4a(wi[2], xi[2], sumi);
        sumi = __dp4a(wi[3], xi[3], sumi);
        sumi = __dp4a(wi[4], xi[4], sumi);
        sumi = __dp4a(wi[5], xi[5], sumi);
        sumi = __dp4a(wi[6], xi[6], sumi);
        sumi = __dp4a(wi[7], xi[7], sumi);

        return d_w * dq * (float)sumi;
    }
};

template <>
struct DequantTraits<DPQTag::Q4_0> {
    static constexpr int kBlockBytes = 18;
    static constexpr int kBlockElems = 32;
    static constexpr int kQ8PerWeight = 1;
    static constexpr bool kNeedsQ8Sum = false;  // computed internally (like every other type)
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = false;  // simple dequant: row-par smem wins on ties

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int /*sub*/, const int* xi,
                                                       float dq, float /*q8_sum*/) {
        half d_w_h;
        memcpy(&d_w_h, bp, sizeof(half));
        float d_w = __half2float(d_w_h);

        // ggml Q4_0 packs 32 elements as SPLIT nibbles: of the 16 qs bytes, the
        // low nibbles are elements 0..15 and the high nibbles are elements 16..31
        // (same convention as imp's dequant_q4_0_kernel). The Q8_1 activations
        // xi[0..7] are natural-order, so pair four consecutive low nibbles (then
        // four consecutive high nibbles) with each xi[j] — the proven Q4_K
        // extraction (q4k_dp4a_sub). Reading nibbles interleaved instead mispairs
        // weights with activations. q8_sum (for the -8 zero-point) is summed
        // internally from xi, like every other type, instead of trusting the
        // passed value (AUDIT.md F1).
        uint32_t w[4];
        memcpy(w, bp + 2, 16);

        const int ones = 0x01010101;
        int32_t sumi = 0;
        int32_t q8_sum_int = 0;
        sumi = __dp4a(static_cast<int>(w[0] & 0x0F0F0F0Fu), xi[0], sumi);         // elems 0..3
        sumi = __dp4a(static_cast<int>(w[1] & 0x0F0F0F0Fu), xi[1], sumi);         // elems 4..7
        sumi = __dp4a(static_cast<int>(w[2] & 0x0F0F0F0Fu), xi[2], sumi);         // elems 8..11
        sumi = __dp4a(static_cast<int>(w[3] & 0x0F0F0F0Fu), xi[3], sumi);         // elems 12..15
        sumi = __dp4a(static_cast<int>((w[0] >> 4) & 0x0F0F0F0Fu), xi[4], sumi);  // elems 16..19
        sumi = __dp4a(static_cast<int>((w[1] >> 4) & 0x0F0F0F0Fu), xi[5], sumi);  // elems 20..23
        sumi = __dp4a(static_cast<int>((w[2] >> 4) & 0x0F0F0F0Fu), xi[6], sumi);  // elems 24..27
        sumi = __dp4a(static_cast<int>((w[3] >> 4) & 0x0F0F0F0Fu), xi[7], sumi);  // elems 28..31
#pragma unroll
        for (int j = 0; j < 8; j++)
            q8_sum_int = __dp4a(xi[j], ones, q8_sum_int);

        return d_w * dq * ((float)sumi - 8.0f * (float)q8_sum_int);
    }
};

template <>
struct DequantTraits<DPQTag::Q4_K> {
    static constexpr int kBlockBytes = 144;
    static constexpr int kBlockElems = 256;
    static constexpr int kQ8PerWeight = 8;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = true;  // complex dequant: K-par wins on ties

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int sub, const int* xi, float dq,
                                                       float /*q8_sum*/) {
        // One LDG.128 for the whole 16-byte header (d, dmin, 12 scale
        // bytes) instead of 2x LD.16 + per-byte scale loads (#598). The
        // superblock is 16B-aligned (144 = 9*16).
        const uint4 hdr = *reinterpret_cast<const uint4*>(bp);
        const half2 dd = *reinterpret_cast<const half2*>(&hdr.x);
        float d_super = __half2float(__low2half(dd));
        float dmin_super = __half2float(__high2half(dd));
        const uint8_t* qs = bp + 16;

        uint8_t sc_val, min_val;
        get_scale_min_k4_reg(hdr.y, hdr.z, hdr.w, sub, sc_val, min_val);

        return q4k_dp4a_sub(qs, sub, d_super, dmin_super, sc_val, min_val, xi, dq);
    }
};

template <>
struct DequantTraits<DPQTag::Q5_K> {
    static constexpr int kBlockBytes = 176;
    static constexpr int kBlockElems = 256;
    static constexpr int kQ8PerWeight = 8;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = true;  // complex dequant: K-par wins on ties

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int sub, const int* xi, float dq,
                                                       float /*q8_sum*/) {
        // Header via one LDG.128 (#598); 176-byte superblock is 16B-aligned.
        const uint4 hdr = *reinterpret_cast<const uint4*>(bp);
        const half2 dd = *reinterpret_cast<const half2*>(&hdr.x);
        float d_super = __half2float(__low2half(dd));
        float dmin_super = __half2float(__high2half(dd));
        const uint8_t* qh = bp + 16;
        const uint8_t* qs = bp + 48;

        uint8_t sc_val, min_val;
        get_scale_min_k4_reg(hdr.y, hdr.z, hdr.w, sub, sc_val, min_val);

        return q5k_dp4a_sub(qs, qh, sub, d_super, dmin_super, sc_val, min_val, xi, dq);
    }
};

template <>
struct DequantTraits<DPQTag::Q2_K> {
    static constexpr int kBlockBytes = 84;
    static constexpr int kBlockElems = 256;
    static constexpr int kQ8PerWeight = 8;
    static constexpr bool kNeedsQ8Sum = false;  // partial sums computed inline
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = false;

    // Q2_K layout (84 bytes / 256 elements):
    //   scales[16]  : 4-bit packed (low=scale, high=min) per 16 elements
    //   qs[64]      : 2-bit packed (4 elements/byte), 2 halves × 4 shifts
    //   d(fp16)     : at offset 80
    //   dmin(fp16)  : at offset 82
    //
    // Each sub (0..7) covers 32 elements. Two 16-element scale groups per sub.
    // qs layout: same 32 bytes reused with shift 0,2,4,6 for 4 groups of 32.
    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int sub, const int* xi, float dq,
                                                       float /*q8_sum*/) {
        const uint8_t* scales = bp;
        const uint8_t* qs = bp + 16;
        float d_w = __half2float(*reinterpret_cast<const half*>(bp + 80));
        float dmin_w = __half2float(*reinterpret_cast<const half*>(bp + 82));

        int half_idx = sub / 4;
        int shift = (sub % 4) * 2;
        const uint8_t* qs_base = qs + half_idx * 32;

        uint8_t sc_byte0 = scales[sub * 2];
        uint8_t sc_byte1 = scales[sub * 2 + 1];
        float sc0 = (float)(sc_byte0 & 0xF);
        float mn0 = (float)(sc_byte0 >> 4);
        float sc1 = (float)(sc_byte1 & 0xF);
        float mn1 = (float)(sc_byte1 >> 4);

        const int ones = 0x01010101;
        int32_t sumi0 = 0, sumi1 = 0;
        int q8s0 = 0, q8s1 = 0;

#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t qb4;
            memcpy(&qb4, qs_base + j * 4, 4);
            uint32_t q2_4 = (qb4 >> shift) & 0x03030303u;
            int qi;
            memcpy(&qi, &q2_4, 4);
            sumi0 = __dp4a(qi, xi[j], sumi0);
            q8s0 = __dp4a(xi[j], ones, q8s0);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t qb4;
            memcpy(&qb4, qs_base + 16 + j * 4, 4);
            uint32_t q2_4 = (qb4 >> shift) & 0x03030303u;
            int qi;
            memcpy(&qi, &q2_4, 4);
            sumi1 = __dp4a(qi, xi[4 + j], sumi1);
            q8s1 = __dp4a(xi[4 + j], ones, q8s1);
        }

        return dq * (d_w * (sc0 * (float)sumi0 + sc1 * (float)sumi1) -
                     dmin_w * (mn0 * (float)q8s0 + mn1 * (float)q8s1));
    }
};

template <>
struct DequantTraits<DPQTag::Q3_K> {
    static constexpr int kBlockBytes = 110;
    static constexpr int kBlockElems = 256;
    static constexpr int kQ8PerWeight = 8;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 2;        // complex dequant → cap NR to avoid reg pressure
    static constexpr bool kPreferKpar = true;  // compute-heavy: K-par wins on ties

    // Q3_K layout (110 bytes / 256 elements):
    //   hmask[32]   : high bit (bit 2) for each of 256 elements
    //   qs[64]      : 2-bit packed (same layout as Q2_K)
    //   scales[12]  : packed 6-bit scales (complex GGML packing)
    //   d(fp16)     : at offset 108
    //
    // q3 = q2_lowbits + (hmask_bit ? 0 : -4), range [-4..3]
    // val = d * (scale6bit - 32) * q3
    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int sub, const int* xi, float dq,
                                                       float /*q8_sum*/) {
        const uint8_t* hmask = bp;
        const uint8_t* qs = bp + 32;
        const uint8_t* sc_raw = bp + 96;
        float d_all = __half2float(*reinterpret_cast<const half*>(bp + 108));

        int half_idx = sub / 4;
        int shift = (sub % 4) * 2;
        const uint8_t* qs_base = qs + half_idx * 32;
        const uint8_t* hm_base = hmask + sub * 4;  // 4 bytes = 32 bits

        // Unpack 16 6-bit scales from 12 packed bytes
        uint32_t aux0, aux1, aux2;
        memcpy(&aux0, sc_raw, 4);
        memcpy(&aux1, sc_raw + 4, 4);
        memcpy(&aux2, sc_raw + 8, 4);
        constexpr uint32_t kmask2 = 0x0f0f0f0fu;
        constexpr uint32_t kmask1 = 0x03030303u;
        uint32_t s[4];
        s[0] = (aux0 & kmask2) | (((aux2 >> 0) & kmask1) << 4);
        s[1] = (aux1 & kmask2) | (((aux2 >> 2) & kmask1) << 4);
        s[2] = ((aux0 >> 4) & kmask2) | (((aux2 >> 4) & kmask1) << 4);
        s[3] = ((aux1 >> 4) & kmask2) | (((aux2 >> 6) & kmask1) << 4);
        const int8_t* up = reinterpret_cast<const int8_t*>(s);
        float sc0 = (float)(up[sub * 2] - 32);
        float sc1 = (float)(up[sub * 2 + 1] - 32);

        // First 16 elements
        int32_t sumi0 = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t qb4;
            memcpy(&qb4, qs_base + j * 4, 4);
            uint32_t q2_4 = (qb4 >> shift) & 0x03030303u;
            // Extract 4 hmask bits → build subtraction mask
            uint8_t hm_byte = hm_base[j / 2];
            int bit_base = (j & 1) * 4;
            uint32_t hm4 = ((hm_byte >> (bit_base + 0)) & 1) | (((hm_byte >> (bit_base + 1)) & 1) << 8) |
                           (((hm_byte >> (bit_base + 2)) & 1) << 16) |
                           (((hm_byte >> (bit_base + 3)) & 1) << 24);
            // q3 = q2 - 4*(1-hm): subtract 4 from each byte where hm=0
            uint32_t sub_mask = (hm4 ^ 0x01010101u) * 4;
            int q3i = __vsubss4(q2_4, sub_mask);
            sumi0 = __dp4a(q3i, xi[j], sumi0);
        }

        // Last 16 elements
        int32_t sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t qb4;
            memcpy(&qb4, qs_base + 16 + j * 4, 4);
            uint32_t q2_4 = (qb4 >> shift) & 0x03030303u;
            uint8_t hm_byte = hm_base[2 + j / 2];
            int bit_base = (j & 1) * 4;
            uint32_t hm4 = ((hm_byte >> (bit_base + 0)) & 1) | (((hm_byte >> (bit_base + 1)) & 1) << 8) |
                           (((hm_byte >> (bit_base + 2)) & 1) << 16) |
                           (((hm_byte >> (bit_base + 3)) & 1) << 24);
            uint32_t sub_mask = (hm4 ^ 0x01010101u) * 4;
            int q3i = __vsubss4(q2_4, sub_mask);
            sumi1 = __dp4a(q3i, xi[4 + j], sumi1);
        }

        return d_all * dq * (sc0 * (float)sumi0 + sc1 * (float)sumi1);
    }
};

// Convenience aliases
// Q5_1: 24 bytes per 32 elements: [2B delta_fp16] [2B min_fp16] [16B low_nibbles] [4B high_bits]
// Dequant: val = q5 * delta + min, where q5 = low4 | (hi1 << 4), range 0..31
template <>
struct DequantTraits<DPQTag::Q5_1> {
    static constexpr int kBlockBytes = 24;
    static constexpr int kBlockElems = 32;
    static constexpr int kQ8PerWeight = 1;
    static constexpr bool kNeedsQ8Sum = false;  // compute q8_sum internally
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = false;

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int /*sub*/, const int* xi,
                                                       float dq, float /*q8_sum*/) {
        half d_w_h, m_w_h;
        memcpy(&d_w_h, bp, sizeof(half));
        memcpy(&m_w_h, bp + 2, sizeof(half));
        float d_w = __half2float(d_w_h);
        float m_w = __half2float(m_w_h);
        const uint8_t* qh = bp + 4;  // 4 bytes high bits (comes FIRST in Q5_1)
        const uint8_t* qs = bp + 8;  // 16 bytes low nibbles

        uint32_t hbits;
        memcpy(&hbits, qh, sizeof(uint32_t));

        int32_t sumi = 0;
        int32_t q8_sum_int = 0;
        const int ones = 0x01010101;

#pragma unroll
        for (int g = 0; g < 8; g++) {
            int base = g * 4;
            int32_t packed = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int idx = base + j;
                uint8_t byte = qs[idx / 2];
                int lo = (idx & 1) ? (byte >> 4) & 0xF : byte & 0xF;
                int hi = (hbits >> idx) & 1;
                int q5 = lo | (hi << 4);  // 0..31
                packed |= (q5 & 0xFF) << (j * 8);
            }
            sumi = __dp4a(packed, xi[g], sumi);
            q8_sum_int = __dp4a(xi[g], ones, q8_sum_int);
        }

        // val = q5 * d_w + m_w → sum = d_w * dq * sumi + m_w * dq * q8_sum_int
        return dq * (d_w * (float)sumi + m_w * (float)q8_sum_int);
    }
};

using Q5_1_Traits = DequantTraits<DPQTag::Q5_1>;
using Q4_0_Traits = DequantTraits<DPQTag::Q4_0>;
using Q8_0_Traits = DequantTraits<DPQTag::Q8_0>;
using Q6_K_Traits = DequantTraits<DPQTag::Q6_K>;
using Q4_K_Traits = DequantTraits<DPQTag::Q4_K>;
using Q5_K_Traits = DequantTraits<DPQTag::Q5_K>;
using Q2_K_Traits = DequantTraits<DPQTag::Q2_K>;
using Q3_K_Traits = DequantTraits<DPQTag::Q3_K>;

// ============================================================================
// K-parallel helpers and occupancy heuristic
// ============================================================================

// Detect GPU SM count for K-parallel occupancy decisions. Cached after first call.
static inline int kpar_n_sms() {
    static int n_sms = 0;
    if (__builtin_expect(n_sms == 0, 0)) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        n_sms = prop.multiProcessorCount;
    }
    return n_sms;
}

// Returns true if K-parallel GEMV gives more active warps/SM than row-parallel.
// K-parallel: 128 threads (4 warps), 1 row per block, Q8_1 from L2 cache.
// Row-parallel: 256 threads (8 warps), NR rows per warp, smem-cached Q8_1.
// PREFER_KPAR: when true, K-par wins on ties (>= comparison). Use for compute-
// heavy quant types (Q6_K, Q4_K, Q5_K) where warp-cooperative K-splitting and
// no-syncthreads access pattern outweigh smem's bandwidth advantage. When false,
// row-par wins on ties (> comparison) since smem Q8_1 caching is faster for
// bandwidth-bound quant types (Q8_0, Q4_0).
template <bool PREFER_KPAR>
static inline bool kpar_is_better(int M, int rpar_blocks) {
    const int n = kpar_n_sms();
    if (n < 1)
        return false;
    // K-parallel: max ~12 blocks/SM (128 threads × ~40 regs = 5120 regs/block)
    int kpar_bpsm = M / n;
    int kpar_warps = (kpar_bpsm < 12 ? kpar_bpsm : 12) * 4;
    // Row-parallel: max ~6 blocks/SM (256 threads × ~40 regs = 10240 regs/block)
    int rpar_bpsm = rpar_blocks / n;
    int rpar_warps = (rpar_bpsm < 6 ? rpar_bpsm : 6) * 8;
    if constexpr (PREFER_KPAR)
        return kpar_warps >= rpar_warps;
    else
        return kpar_warps > rpar_warps;
}

// ============================================================================
// K-parallel GEMV kernels: all warps cooperate on K-dimension for 1 row.
// 128 threads (4 warps), Q8_1 from L2 cache, static 16-byte smem for reduction.
// Used when M is small relative to GPU SMs (typical for d_model dimensions).
// This dramatically increases blocks/SM: e.g., M=3072 → 3072 blocks instead of
// 192 (NR=2, 256 threads), giving 48 warps/SM instead of 8.
// ============================================================================

template <typename QT, bool ADD_RESIDUAL>
__global__ void gemv_dp4a_kpar_kernel(const uint8_t* __restrict__ W, const block_q8_1* __restrict__ q8_1,
                                      const float* __restrict__ d8, half* y, const half* residual, int M,
                                      int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8];
        memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum)
            q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0)
        partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = partial[0] + partial[1] + partial[2] + partial[3];
        if constexpr (ADD_RESIDUAL)
            total += __half2float(residual[row]);
        y[row] = __float2half(total);
    }
}

template <typename QT>
__global__ void gemv_dp4a_kpar_fp32_kernel(const uint8_t* __restrict__ W, const block_q8_1* __restrict__ q8_1,
                                           const float* __restrict__ d8, float* __restrict__ y, int M,
                                           int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8];
        memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum)
            q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0)
        partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0)
        y[row] = partial[0] + partial[1] + partial[2] + partial[3];
}

template <typename QT>
__global__ void gemv_dp4a_kpar_qkv_kernel(const uint8_t* __restrict__ W_q, const uint8_t* __restrict__ W_k,
                                          const uint8_t* __restrict__ W_v,
                                          const block_q8_1* __restrict__ q8_1, const float* __restrict__ d8,
                                          half* __restrict__ y_q, half* __restrict__ y_k,
                                          half* __restrict__ y_v, int q_rows, int k_rows, int v_rows, int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int global_row = blockIdx.x;
    const int total_rows = q_rows + k_rows + v_rows;
    if (global_row >= total_rows)
        return;

    const uint8_t* W;
    half* y_out;
    int local_row;
    if (global_row < q_rows) {
        W = W_q;
        y_out = y_q;
        local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = W_k;
        y_out = y_k;
        local_row = global_row - q_rows;
    } else {
        W = W_v;
        y_out = y_v;
        local_row = global_row - q_rows - k_rows;
    }

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)local_row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8];
        memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum)
            q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0)
        partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        const float2 s = add_f32x2(make_float2(partial[0], partial[1]),
                                   make_float2(partial[2], partial[3]));
        y_out[local_row] = __float2half(s.x + s.y);
    }
}

template <typename QT>
__global__ void gemv_dp4a_kpar_gate_up_kernel(const uint8_t* __restrict__ gate_weights,
                                              const uint8_t* __restrict__ up_weights,
                                              const block_q8_1* __restrict__ q8_1,
                                              const float* __restrict__ d8, half* __restrict__ y_gate,
                                              half* __restrict__ y_up, int M, int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* W = is_up ? up_weights : gate_weights;
    half* y = is_up ? y_up : y_gate;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8];
        memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum)
            q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0)
        partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        const float2 s = add_f32x2(make_float2(partial[0], partial[1]),
                                   make_float2(partial[2], partial[3]));
        y[row] = __float2half(s.x + s.y);
    }
}

// ============================================================================
// Template kernel #1: Basic + Residual (replaces 10 hand-written kernels)
// ============================================================================

template <typename QT, int N_ROWS, bool ADD_RESIDUAL>
__global__ void gemv_dp4a_kernel(const uint8_t* __restrict__ W, const block_q8_1* __restrict__ q8_1,
                                 const float* __restrict__ d8, half* y, const half* residual, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = reinterpret_cast<int*>(smem_q8);
    float* smem_d = reinterpret_cast<float*>(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val;
        memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= M)
        return;

    float sum[N_ROWS];
#pragma unroll
    for (int r = 0; r < N_ROWS; r++)
        sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

#pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M)
                break;
            const uint8_t* bp = W + (size_t)row * row_bytes + (size_t)wb * QT::kBlockBytes;
            sum[r] += QT::dp4a_block(bp, sub, xi, dq, q8_sum);
        }
    }

#pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0) {
            const int row = row_base + r;
            if (row < M) {
                float s = sum[r];
                if constexpr (ADD_RESIDUAL)
                    s += __half2float(residual[row]);
                y[row] = __float2half(s);
            }
        }
    }
}

template <typename QT>
static void launch_gemv_dp4a(const uint8_t* W, const block_q8_1* q8_1, const float* d8, half* y,
                             const half* residual, bool add_residual, int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    // K-parallel check: compare against NR=1 (maximum occupancy row-par baseline).
    // NR=1 has the most blocks, giving the highest row-par occupancy estimate.
    // Using higher NR would undercount row-par warps (fewer blocks/SM), falsely
    // favoring K-par for large M (e.g., gate_up d_ff=14336 on 170 SMs).
    {
        int nr1_blocks = (M + warps_per_block - 1) / warps_per_block;
        if (kpar_is_better<QT::kPreferKpar>(M, nr1_blocks)) {
            if (add_residual)
                pdl::launch(gemv_dp4a_kpar_kernel<QT, true>, dim3(M), dim3(128), size_t(0), stream, W, q8_1,
                            d8, y, residual, M, K);
            else
                pdl::launch(gemv_dp4a_kpar_kernel<QT, false>, dim3(M), dim3(128), size_t(0), stream, W, q8_1,
                            d8, y, static_cast<const half*>(nullptr), M, K);
            return;
        }
    }

    // Row-parallel path (for large M where SMs are already well-utilized)
    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem_size = (size_t)total_q8 * (40 + QT::kSmemExtra);

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (M + rows_per_block - 1) / rows_per_block;
        if (add_residual)
            pdl::launch(gemv_dp4a_kernel<QT, NR, true>, dim3(blocks), dim3(threads_per_block), smem_size,
                        stream, W, q8_1, d8, y, residual, M, K);
        else
            pdl::launch(gemv_dp4a_kernel<QT, NR, false>, dim3(blocks), dim3(threads_per_block), smem_size,
                        stream, W, q8_1, d8, y, static_cast<const half*>(nullptr), M, K);
    };

    // Dispatch NR based on kMaxNRows (caps NR to avoid register pressure)
    constexpr int MAX_NR = QT::kMaxNRows;
    if constexpr (MAX_NR >= 4) {
        int nr4_blocks = (M + warps_per_block * 4 - 1) / (warps_per_block * 4);
        if (nr4_blocks >= 128) {
            launch(std::integral_constant<int, 4>{});
            return;
        }
    }
    if constexpr (MAX_NR >= 2) {
        int nr2_blocks = (M + warps_per_block * 2 - 1) / (warps_per_block * 2);
        if (nr2_blocks >= 64) {
            launch(std::integral_constant<int, 2>{});
            return;
        }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #2: FP32 Output (replaces 5 hand-written kernels)
// ============================================================================

template <typename QT, int N_ROWS>
__global__ void gemv_dp4a_fp32_kernel(const uint8_t* __restrict__ W, const block_q8_1* __restrict__ q8_1,
                                      const float* __restrict__ d8, float* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = reinterpret_cast<int*>(smem_q8);
    float* smem_d = reinterpret_cast<float*>(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val;
        memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= M)
        return;

    float sum[N_ROWS];
#pragma unroll
    for (int r = 0; r < N_ROWS; r++)
        sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

#pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M)
                break;
            const uint8_t* bp = W + (size_t)row * row_bytes + (size_t)wb * QT::kBlockBytes;
            sum[r] += QT::dp4a_block(bp, sub, xi, dq, q8_sum);
        }
    }

#pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0 && row_base + r < M)
            y[row_base + r] = sum[r];
    }
}

template <typename QT>
static void launch_gemv_dp4a_fp32(const uint8_t* W, const block_q8_1* q8_1, const float* d8, float* y, int M,
                                  int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    // K-parallel check (compare against NR=1 max-occupancy baseline)
    {
        int nr1_blocks = (M + warps_per_block - 1) / warps_per_block;
        if (kpar_is_better<QT::kPreferKpar>(M, nr1_blocks)) {
            pdl::launch(gemv_dp4a_kpar_fp32_kernel<QT>, dim3(M), dim3(128), size_t(0), stream, W, q8_1, d8, y,
                        M, K);
            return;
        }
    }

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem_size = (size_t)total_q8 * (40 + QT::kSmemExtra);

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (M + rows_per_block - 1) / rows_per_block;
        pdl::launch(gemv_dp4a_fp32_kernel<QT, NR>, dim3(blocks), dim3(threads_per_block), smem_size, stream,
                    W, q8_1, d8, y, M, K);
    };

    // Dispatch NR based on kMaxNRows (caps NR to avoid register pressure)
    constexpr int MAX_NR = QT::kMaxNRows;
    if constexpr (MAX_NR >= 4) {
        int nr4_blocks = (M + warps_per_block * 4 - 1) / (warps_per_block * 4);
        if (nr4_blocks >= 128) {
            launch(std::integral_constant<int, 4>{});
            return;
        }
    }
    if constexpr (MAX_NR >= 2) {
        int nr2_blocks = (M + warps_per_block * 2 - 1) / (warps_per_block * 2);
        if (nr2_blocks >= 64) {
            launch(std::integral_constant<int, 2>{});
            return;
        }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #3: QKV Fused (replaces 5 hand-written kernels)
// ============================================================================

template <typename QT, int N_ROWS>
__global__ void gemv_dp4a_qkv_kernel(const uint8_t* __restrict__ W_q, const uint8_t* __restrict__ W_k,
                                     const uint8_t* __restrict__ W_v, const block_q8_1* __restrict__ q8_1,
                                     const float* __restrict__ d8, half* __restrict__ y_q,
                                     half* __restrict__ y_k, half* __restrict__ y_v, int q_rows, int k_rows,
                                     int v_rows, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;
    const int total_rows = q_rows + k_rows + v_rows;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = reinterpret_cast<int*>(smem_q8);
    float* smem_d = reinterpret_cast<float*>(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val;
        memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= total_rows)
        return;

    float sum[N_ROWS];
#pragma unroll
    for (int r = 0; r < N_ROWS; r++)
        sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

#pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int global_row = row_base + r;
            if (global_row >= total_rows)
                break;

            const uint8_t* W;
            int local_row;
            if (global_row < q_rows) {
                local_row = global_row;
                W = W_q;
            } else if (global_row < q_rows + k_rows) {
                local_row = global_row - q_rows;
                W = W_k;
            } else {
                local_row = global_row - q_rows - k_rows;
                W = W_v;
            }

            const uint8_t* bp = W + (size_t)local_row * row_bytes + (size_t)wb * QT::kBlockBytes;
            sum[r] += QT::dp4a_block(bp, sub, xi, dq, q8_sum);
        }
    }

#pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0) {
            const int global_row = row_base + r;
            if (global_row >= total_rows)
                break;
            half* y;
            int local_row;
            if (global_row < q_rows) {
                local_row = global_row;
                y = y_q;
            } else if (global_row < q_rows + k_rows) {
                local_row = global_row - q_rows;
                y = y_k;
            } else {
                local_row = global_row - q_rows - k_rows;
                y = y_v;
            }
            y[local_row] = __float2half(sum[r]);
        }
    }
}

template <typename QT>
static void launch_gemv_dp4a_qkv(const uint8_t* W_q, const uint8_t* W_k, const uint8_t* W_v,
                                 const block_q8_1* q8_1, const float* d8, half* y_q, half* y_k, half* y_v,
                                 int q_rows, int k_rows, int v_rows, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int total = q_rows + k_rows + v_rows;

    // K-parallel check (compare against NR=1 max-occupancy baseline)
    {
        int nr1_blocks = (total + warps_per_block - 1) / warps_per_block;
        if (kpar_is_better<QT::kPreferKpar>(total, nr1_blocks)) {
            pdl::launch(gemv_dp4a_kpar_qkv_kernel<QT>, dim3(total), dim3(128), size_t(0), stream, W_q, W_k,
                        W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K);
            return;
        }
    }

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem = (size_t)total_q8 * (40 + QT::kSmemExtra);

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (total + rows_per_block - 1) / rows_per_block;
        pdl::launch(gemv_dp4a_qkv_kernel<QT, NR>, dim3(blocks), dim3(threads_per_block), smem, stream, W_q,
                    W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K);
    };

    // Dispatch NR based on kMaxNRows
    constexpr int MAX_NR = QT::kMaxNRows;
    if constexpr (MAX_NR >= 4) {
        int nr4_blocks = (total + warps_per_block * 4 - 1) / (warps_per_block * 4);
        if (nr4_blocks >= 128) {
            launch(std::integral_constant<int, 4>{});
            return;
        }
    }
    if constexpr (MAX_NR >= 2) {
        int nr2_blocks = (total + warps_per_block * 2 - 1) / (warps_per_block * 2);
        if (nr2_blocks >= 64) {
            launch(std::integral_constant<int, 2>{});
            return;
        }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #4: Gate+Up Fused (replaces 5 hand-written kernels)
// blockIdx.y: 0 = gate, 1 = up. N_ROWS rows per warp.
// ============================================================================

template <typename QT, int N_ROWS>
__global__ void gemv_dp4a_gate_up_kernel(const uint8_t* __restrict__ gate_weights,
                                         const uint8_t* __restrict__ up_weights,
                                         const block_q8_1* __restrict__ q8_1, const float* __restrict__ d8,
                                         half* __restrict__ y_gate, half* __restrict__ y_up, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = reinterpret_cast<int*>(smem_q8);
    float* smem_d = reinterpret_cast<float*>(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val;
        memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= M)
        return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* W = is_up ? up_weights : gate_weights;
    half* y = is_up ? y_up : y_gate;

    float sum[N_ROWS];
#pragma unroll
    for (int r = 0; r < N_ROWS; r++)
        sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

#pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M)
                break;
            const uint8_t* bp = W + (size_t)row * row_bytes + (size_t)wb * QT::kBlockBytes;
            sum[r] += QT::dp4a_block(bp, sub, xi, dq, q8_sum);
        }
    }

#pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0) {
            const int row = row_base + r;
            if (row < M)
                y[row] = __float2half(sum[r]);
        }
    }
}

// ============================================================================
// Template kernel #5: MoE Decode with NR (replaces 4 hand-written kernels)
// Q8_1 data cooperatively loaded into shared memory — all 8 warps share
// the same Q8_1 input per expert slot, eliminating 8x redundant L2 reads.
// NR>1: each warp handles multiple rows, halving CTAs and smem loads.
// ============================================================================

template <typename QT, int NR>
__global__ void gemv_dp4a_moe_decode_kernel(const uint8_t* __restrict__ packed_weights,
                                            const int32_t* __restrict__ expert_indices,
                                            const block_q8_1* __restrict__ q8_1, const float* __restrict__ d8,
                                            half* __restrict__ y, int rows, int K, size_t expert_stride_bytes,
                                            int q8_1_stride, int d8_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row_base = (local_block * warps_per_block + warp_id) * NR;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;

    // Cooperatively load Q8_1 into shared memory (same pattern as dense kernel)
    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;

    extern __shared__ char smem_q8[];
    int* smem_qs = reinterpret_cast<int*>(smem_q8);
    float* smem_d = reinterpret_cast<float*>(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val;
        memcpy(&val, x_q8[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = x_d8[i];
            smem_s[i] = x_q8[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = x_d8[i];
    }
    __syncthreads();

    if (row_base >= rows)
        return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    float sum[NR];
#pragma unroll
    for (int r = 0; r < NR; r++)
        sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int row = row_base + r;
            if (row >= rows)
                break;
            const uint8_t* bp = W + (size_t)row * row_bytes + (size_t)wb * QT::kBlockBytes;
            sum[r] += QT::dp4a_block(bp, sub, xi, dq, q8_sum);
        }
    }

#pragma unroll
    for (int r = 0; r < NR; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0) {
            const int row = row_base + r;
            if (row < rows)
                y[expert_slot * rows + row] = __float2half(sum[r]);
        }
    }
}

template <typename QT>
static void launch_gemv_dp4a_moe_decode(const uint8_t* packed_weights, const int32_t* expert_indices,
                                        const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                                        size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                        cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem_size = (size_t)total_q8 * (40 + QT::kSmemExtra);

    constexpr int MAX_NR = QT::kMaxNRows;
    auto launch = [&](auto nr_tag) {
        constexpr int NR = decltype(nr_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks_per_expert = (rows + rows_per_block - 1) / rows_per_block;
        const int total_blocks = top_k * blocks_per_expert;
        pdl::launch(gemv_dp4a_moe_decode_kernel<QT, NR>, dim3(total_blocks), dim3(threads_per_block),
                    smem_size, stream, packed_weights, expert_indices, q8_1, d8, y, rows, K,
                    expert_stride_bytes, q8_1_stride, d8_stride, blocks_per_expert);
    };

    // Use NR=2 when enough blocks to fill SMs (same threshold as dense kernel)
    if constexpr (MAX_NR >= 2) {
        int nr2_rows_per_block = warps_per_block * 2;
        int nr2_blocks = top_k * ((rows + nr2_rows_per_block - 1) / nr2_rows_per_block);
        if (nr2_blocks >= 64) {
            launch(std::integral_constant<int, 2>{});
            return;
        }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #6: MoE Gate+Up Dual-Matrix (replaces 4 hand-written kernels)
// Each warp computes BOTH gate and up projections for the same row, sharing
// Q8_1 from smem. Halves CTA count and smem loads vs separate gate/up blocks.
// ============================================================================

template <typename QT>
__global__ void gemv_dp4a_moe_gate_up_kernel(
    const uint8_t* __restrict__ gate_weights, const uint8_t* __restrict__ up_weights,
    const int32_t* __restrict__ expert_indices, const block_q8_1* __restrict__ q8_1,
    const float* __restrict__ d8, half* __restrict__ y_gate, half* __restrict__ y_up, int rows, int K,
    size_t gate_stride_bytes, size_t up_stride_bytes, int q8_1_stride, int d8_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;

    // Cooperatively load Q8_1 into shared memory (same pattern as dense kernel)
    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;

    extern __shared__ char smem_q8[];
    int* smem_qs = reinterpret_cast<int*>(smem_q8);
    float* smem_d = reinterpret_cast<float*>(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val;
        memcpy(&val, x_q8[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = x_d8[i];
            smem_s[i] = x_q8[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = x_d8[i];
    }
    __syncthreads();

    if (row >= rows)
        return;

    const int expert_id = expert_indices[expert_slot];
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* W_gate_row = gate_weights + (size_t)expert_id * gate_stride_bytes +
                                (size_t)row * row_bytes;
    const uint8_t* W_up_row = up_weights + (size_t)expert_id * up_stride_bytes + (size_t)row * row_bytes;

    float sum_gate = 0.0f, sum_up = 0.0f;
    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        const size_t block_off = (size_t)wb * QT::kBlockBytes;
        sum_gate += QT::dp4a_block(W_gate_row + block_off, sub, xi, dq, q8_sum);
        sum_up += QT::dp4a_block(W_up_row + block_off, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1) {
        sum_gate += __shfl_down_sync(0xFFFFFFFF, sum_gate, off);
        sum_up += __shfl_down_sync(0xFFFFFFFF, sum_up, off);
    }

    if (lane == 0) {
        const int out_idx = expert_slot * rows + row;
        y_gate[out_idx] = __float2half(sum_gate);
        y_up[out_idx] = __float2half(sum_up);
    }
}

// Multi-row variant: NR rows per warp, halves CTA count for small K.
template <typename QT, int NR>
__global__ void gemv_dp4a_moe_gate_up_mr_kernel(
    const uint8_t* __restrict__ gate_weights, const uint8_t* __restrict__ up_weights,
    const int32_t* __restrict__ expert_indices, const block_q8_1* __restrict__ q8_1,
    const float* __restrict__ d8, half* __restrict__ y_gate, half* __restrict__ y_up, int rows, int K,
    size_t gate_stride_bytes, size_t up_stride_bytes, int q8_1_stride, int d8_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row_base = (local_block * warps_per_block + warp_id) * NR;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;

    // Cooperatively load Q8_1 into shared memory
    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;

    extern __shared__ char smem_q8[];
    int* smem_qs = reinterpret_cast<int*>(smem_q8);
    float* smem_d = reinterpret_cast<float*>(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val;
        memcpy(&val, x_q8[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = x_d8[i];
            smem_s[i] = x_q8[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = x_d8[i];
    }
    __syncthreads();

    if (row_base >= rows)
        return;

    const int expert_id = expert_indices[expert_slot];
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    // Compute NR rows, each with gate+up accumulators
    float sum_gate[NR], sum_up[NR];
    const uint8_t* W_gate[NR];
    const uint8_t* W_up[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) {
        sum_gate[r] = 0.0f;
        sum_up[r] = 0.0f;
        W_gate[r] = gate_weights + (size_t)expert_id * gate_stride_bytes + (size_t)(row_base + r) * row_bytes;
        W_up[r] = up_weights + (size_t)expert_id * up_stride_bytes + (size_t)(row_base + r) * row_bytes;
    }

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = reinterpret_cast<half*>(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        const size_t block_off = (size_t)wb * QT::kBlockBytes;
#pragma unroll
        for (int r = 0; r < NR; r++) {
            if (row_base + r < rows) {
                sum_gate[r] += QT::dp4a_block(W_gate[r] + block_off, sub, xi, dq, q8_sum);
                sum_up[r] += QT::dp4a_block(W_up[r] + block_off, sub, xi, dq, q8_sum);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < NR; r++) {
        for (int off = 16; off > 0; off >>= 1) {
            sum_gate[r] += __shfl_down_sync(0xFFFFFFFF, sum_gate[r], off);
            sum_up[r] += __shfl_down_sync(0xFFFFFFFF, sum_up[r], off);
        }
        if (lane == 0 && row_base + r < rows) {
            const int out_idx = expert_slot * rows + row_base + r;
            y_gate[out_idx] = __float2half(sum_gate[r]);
            y_up[out_idx] = __float2half(sum_up[r]);
        }
    }
}

template <typename QT>
static void launch_gemv_dp4a_moe_gate_up(const uint8_t* gate_weights, const uint8_t* up_weights,
                                         const int32_t* expert_indices, const block_q8_1* q8_1,
                                         const float* d8, half* y_gate, half* y_up, int rows, int K,
                                         size_t gate_stride_bytes, size_t up_stride_bytes, int q8_1_stride,
                                         int d8_stride, int top_k, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;  // 8
    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem_size = (size_t)total_q8 * (40 + QT::kSmemExtra);

    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    pdl::launch(gemv_dp4a_moe_gate_up_kernel<QT>, dim3(top_k * blocks_per_expert), dim3(threads_per_block),
                smem_size, stream, gate_weights, up_weights, expert_indices, q8_1, d8, y_gate, y_up, rows, K,
                gate_stride_bytes, up_stride_bytes, q8_1_stride, d8_stride, blocks_per_expert);
}

}  // namespace imp
