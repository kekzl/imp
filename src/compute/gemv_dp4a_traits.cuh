#pragma once
// Template infrastructure for dp4a-accelerated GEMV kernels.
// Replaces ~33 hand-written kernel functions with 6 template kernels.
// Each quant type provides a DequantTraits specialization with one dp4a_block() function.

#include "compute/gemm.h"
#include "runtime/pdl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <type_traits>

namespace imp {

// Smem Q8_1 stride: 9 int32s per block (8 data + 1 padding) to eliminate bank conflicts.
// With stride 8: lanes 0,4,8,... all hit the same shared memory bank → 8-way conflict.
// With stride 9: lane i starts at bank (i*9)%32 — all unique, zero conflicts.
static constexpr int kSmemQ8Stride = 9;

// Tag enum for template dispatch (separate from GGMLQuantType)
enum class QType { Q4_0, Q8_0, Q6_K, Q4_K, Q5_K, Q2_K, Q3_K };

// ============================================================================
// Helper device functions (moved from gemm.cu, unchanged)
// ============================================================================

__device__ __forceinline__ float q6k_dp4a_group_preloaded(
        const uint8_t* __restrict__ ql,
        const uint8_t* __restrict__ qh,
        const int8_t* __restrict__ sc,
        float d_w,
        const int* __restrict__ xqs_packed,  // [8] pre-loaded int32 from Q8_1
        float d_x,
        int g) {
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
            const uint32_t lo4 = is_high ? ((ql4 >> 4) & 0x0F0F0F0FU)
                                         : (ql4 & 0x0F0F0F0FU);
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

__device__ __forceinline__ int unpack_nibbles_2(uint8_t b0, uint8_t b1) {
    int r;
    int8_t vals[4] = { static_cast<int8_t>(b0 & 0xF), static_cast<int8_t>(b0 >> 4),
                       static_cast<int8_t>(b1 & 0xF), static_cast<int8_t>(b1 >> 4) };
    memcpy(&r, vals, 4);
    return r;
}

__device__ __forceinline__ void get_scale_min_k4(
        const uint8_t* __restrict__ sc, int sub,
        uint8_t& sc_val, uint8_t& min_val) {
    if (sub < 4) {
        sc_val  = sc[sub] & 63;
        min_val = sc[sub + 4] & 63;
    } else {
        sc_val  = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        min_val = (sc[sub + 4] >> 4)   | ((sc[sub]     >> 6) << 4);
    }
}

__device__ __forceinline__ float q4k_dp4a_sub(
        const uint8_t* __restrict__ qs,   // Q4_K qs base (128 bytes)
        int sub,                           // sub-block index (0..7)
        float d_super,                     // super-block scale
        float dmin_super,                  // super-block min
        uint8_t sc_val,                    // 6-bit sub-block scale
        uint8_t min_val,                   // 6-bit sub-block min
        const int* __restrict__ xi,        // [8] packed Q8_1 int32 values
        float dq) {                        // Q8_1 block scale
    const int qs_byte_offset = (sub / 2) * 32;
    const bool use_high = (sub & 1);
    const uint8_t* qs_base = qs + qs_byte_offset;

    int32_t sumi = 0;
    int q8_sum_int = 0;
    const int ones = 0x01010101;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint32_t qs4;
        memcpy(&qs4, qs_base + j * 4, 4);
        uint32_t nibbles = use_high ? ((qs4 >> 4) & 0x0F0F0F0Fu)
                                    : (qs4 & 0x0F0F0F0Fu);
        int ni;
        memcpy(&ni, &nibbles, 4);
        sumi = __dp4a(ni, xi[j], sumi);
        q8_sum_int = __dp4a(xi[j], ones, q8_sum_int);
    }

    return dq * (d_super * (float)sc_val * (float)sumi
               - dmin_super * (float)min_val * (float)q8_sum_int);
}

__device__ __forceinline__ float q5k_dp4a_sub(
        const uint8_t* __restrict__ qs,   // Q5_K qs base (128 bytes, offset +48)
        const uint8_t* __restrict__ qh,   // Q5_K qh base (32 bytes, offset +16)
        int sub,                           // sub-block index (0..7)
        float d_super,                     // super-block scale
        float dmin_super,                  // super-block min
        uint8_t sc_val,                    // 6-bit sub-block scale
        uint8_t min_val,                   // 6-bit sub-block min
        const int* __restrict__ xi,        // [8] packed Q8_1 int32 values
        float dq) {                        // Q8_1 block scale
    const int qs_byte_offset = (sub / 2) * 32;
    const bool use_high = (sub & 1);
    const uint8_t* qs_base = qs + qs_byte_offset;

    const uint8_t* qh_sub = qh + sub * 4;

    int32_t sumi = 0;
    int32_t sumi_h = 0;   // 5th-bit correction
    int q8_sum_int = 0;
    const int ones = 0x01010101;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint32_t qs4;
        memcpy(&qs4, qs_base + j * 4, 4);
        uint32_t nibbles = use_high ? ((qs4 >> 4) & 0x0F0F0F0Fu)
                                    : (qs4 & 0x0F0F0F0Fu);
        int ni;
        memcpy(&ni, &nibbles, 4);
        sumi = __dp4a(ni, xi[j], sumi);
        q8_sum_int = __dp4a(xi[j], ones, q8_sum_int);

        uint8_t qh_byte = qh_sub[j / 2];
        int bit_base = (j & 1) * 4;
        uint32_t hbits = ((qh_byte >> (bit_base + 0)) & 1) |
                         (((qh_byte >> (bit_base + 1)) & 1) << 8) |
                         (((qh_byte >> (bit_base + 2)) & 1) << 16) |
                         (((qh_byte >> (bit_base + 3)) & 1) << 24);
        hbits *= 0x10;
        int hi;
        memcpy(&hi, &hbits, 4);
        sumi_h = __dp4a(hi, xi[j], sumi_h);
    }

    return dq * (d_super * (float)sc_val * (float)(sumi + sumi_h)
               - dmin_super * (float)min_val * (float)q8_sum_int);
}

// ============================================================================
// DequantTraits<QType> — compile-time constants + dp4a_block() per type
//
// dp4a_block(bp, sub, xi, dq, q8_sum):
//   bp      — pointer to the start of the weight block/super-block
//   sub     — sub-group index within super-block (0 for Q8_0/Q4_0; 0-7 for Q6_K/Q4_K/Q5_K)
//   xi[8]   — pre-loaded Q8_1 int32 packed values
//   dq      — Q8_1 block scale
//   q8_sum  — sum of Q8_1 int8 values (only used by Q4_0)
// ============================================================================

template<QType Q> struct DequantTraits;

template<> struct DequantTraits<QType::Q6_K> {
    static constexpr int kBlockBytes   = 210;
    static constexpr int kBlockElems   = 256;
    static constexpr int kQ8PerWeight  = 8;
    static constexpr bool kNeedsQ8Sum  = false;
    static constexpr int kSmemExtra    = 0;
    static constexpr int kMaxNRows     = 2;   // NR=4 uses 64 regs → occupancy drop
    static constexpr bool kPreferKpar  = true; // compute-heavy dequant: K-par wins on ties

    static __device__ __forceinline__ float
    dp4a_block(const uint8_t* bp, int sub,
               const int* xi, float dq, float /*q8_sum*/) {
        float d_w = __half2float(*(const half*)(bp + 208));
        return q6k_dp4a_group_preloaded(
            bp, bp + 128, (const int8_t*)(bp + 192), d_w, xi, dq, sub);
    }
};

template<> struct DequantTraits<QType::Q8_0> {
    static constexpr int kBlockBytes   = 34;
    static constexpr int kBlockElems   = 32;
    static constexpr int kQ8PerWeight  = 1;
    static constexpr bool kNeedsQ8Sum  = false;
    static constexpr int kSmemExtra    = 0;
    static constexpr int kMaxNRows     = 2;   // NR=4 uses 59 regs → occupancy drop
    static constexpr bool kPreferKpar  = false; // simple dequant: row-par smem wins on ties

    static __device__ __forceinline__ float
    dp4a_block(const uint8_t* bp, int /*sub*/,
               const int* xi, float dq, float /*q8_sum*/) {
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

template<> struct DequantTraits<QType::Q4_0> {
    static constexpr int kBlockBytes   = 18;
    static constexpr int kBlockElems   = 32;
    static constexpr int kQ8PerWeight  = 1;
    static constexpr bool kNeedsQ8Sum  = true;
    static constexpr int kSmemExtra    = 2;   // half s field per Q8_1 block
    static constexpr int kMaxNRows     = 4;
    static constexpr bool kPreferKpar  = false; // simple dequant: row-par smem wins on ties

    static __device__ __forceinline__ float
    dp4a_block(const uint8_t* bp, int /*sub*/,
               const int* xi, float dq, float q8_sum) {
        half d_w_h;
        memcpy(&d_w_h, bp, sizeof(half));
        float d_w = __half2float(d_w_h);
        const uint8_t* qs = bp + 2;

        int ni0 = unpack_nibbles_2(qs[0],  qs[1]);
        int ni1 = unpack_nibbles_2(qs[2],  qs[3]);
        int ni2 = unpack_nibbles_2(qs[4],  qs[5]);
        int ni3 = unpack_nibbles_2(qs[6],  qs[7]);
        int ni4 = unpack_nibbles_2(qs[8],  qs[9]);
        int ni5 = unpack_nibbles_2(qs[10], qs[11]);
        int ni6 = unpack_nibbles_2(qs[12], qs[13]);
        int ni7 = unpack_nibbles_2(qs[14], qs[15]);

        int32_t sumi = 0;
        sumi = __dp4a(ni0, xi[0], sumi);
        sumi = __dp4a(ni1, xi[1], sumi);
        sumi = __dp4a(ni2, xi[2], sumi);
        sumi = __dp4a(ni3, xi[3], sumi);
        sumi = __dp4a(ni4, xi[4], sumi);
        sumi = __dp4a(ni5, xi[5], sumi);
        sumi = __dp4a(ni6, xi[6], sumi);
        sumi = __dp4a(ni7, xi[7], sumi);

        return d_w * (dq * (float)sumi - 8.0f * q8_sum);
    }
};

template<> struct DequantTraits<QType::Q4_K> {
    static constexpr int kBlockBytes   = 144;
    static constexpr int kBlockElems   = 256;
    static constexpr int kQ8PerWeight  = 8;
    static constexpr bool kNeedsQ8Sum  = false;
    static constexpr int kSmemExtra    = 0;
    static constexpr int kMaxNRows     = 4;
    static constexpr bool kPreferKpar  = true; // complex dequant: K-par wins on ties

    static __device__ __forceinline__ float
    dp4a_block(const uint8_t* bp, int sub,
               const int* xi, float dq, float /*q8_sum*/) {
        float d_super = __half2float(*(const half*)bp);
        float dmin_super = __half2float(*(const half*)(bp + 2));
        const uint8_t* sc = bp + 4;
        const uint8_t* qs = bp + 16;

        uint8_t sc_val, min_val;
        get_scale_min_k4(sc, sub, sc_val, min_val);

        return q4k_dp4a_sub(qs, sub, d_super, dmin_super,
                             sc_val, min_val, xi, dq);
    }
};

template<> struct DequantTraits<QType::Q5_K> {
    static constexpr int kBlockBytes   = 176;
    static constexpr int kBlockElems   = 256;
    static constexpr int kQ8PerWeight  = 8;
    static constexpr bool kNeedsQ8Sum  = false;
    static constexpr int kSmemExtra    = 0;
    static constexpr int kMaxNRows     = 4;
    static constexpr bool kPreferKpar  = true; // complex dequant: K-par wins on ties

    static __device__ __forceinline__ float
    dp4a_block(const uint8_t* bp, int sub,
               const int* xi, float dq, float /*q8_sum*/) {
        float d_super = __half2float(*(const half*)bp);
        float dmin_super = __half2float(*(const half*)(bp + 2));
        const uint8_t* sc = bp + 4;
        const uint8_t* qh = bp + 16;
        const uint8_t* qs = bp + 48;

        uint8_t sc_val, min_val;
        get_scale_min_k4(sc, sub, sc_val, min_val);

        return q5k_dp4a_sub(qs, qh, sub, d_super, dmin_super,
                             sc_val, min_val, xi, dq);
    }
};

template<> struct DequantTraits<QType::Q2_K> {
    static constexpr int kBlockBytes   = 84;
    static constexpr int kBlockElems   = 256;
    static constexpr int kQ8PerWeight  = 8;
    static constexpr bool kNeedsQ8Sum  = false;  // partial sums computed inline
    static constexpr int kSmemExtra    = 0;
    static constexpr int kMaxNRows     = 4;
    static constexpr bool kPreferKpar  = false;

    // Q2_K layout (84 bytes / 256 elements):
    //   scales[16]  : 4-bit packed (low=scale, high=min) per 16 elements
    //   qs[64]      : 2-bit packed (4 elements/byte), 2 halves × 4 shifts
    //   d(fp16)     : at offset 80
    //   dmin(fp16)  : at offset 82
    //
    // Each sub (0..7) covers 32 elements. Two 16-element scale groups per sub.
    // qs layout: same 32 bytes reused with shift 0,2,4,6 for 4 groups of 32.
    static __device__ __forceinline__ float
    dp4a_block(const uint8_t* bp, int sub,
               const int* xi, float dq, float /*q8_sum*/) {
        const uint8_t* scales = bp;
        const uint8_t* qs     = bp + 16;
        float d_w    = __half2float(*(const half*)(bp + 80));
        float dmin_w = __half2float(*(const half*)(bp + 82));

        int half_idx = sub / 4;
        int shift    = (sub % 4) * 2;
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
            int qi; memcpy(&qi, &q2_4, 4);
            sumi0 = __dp4a(qi, xi[j], sumi0);
            q8s0 = __dp4a(xi[j], ones, q8s0);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t qb4;
            memcpy(&qb4, qs_base + 16 + j * 4, 4);
            uint32_t q2_4 = (qb4 >> shift) & 0x03030303u;
            int qi; memcpy(&qi, &q2_4, 4);
            sumi1 = __dp4a(qi, xi[4 + j], sumi1);
            q8s1 = __dp4a(xi[4 + j], ones, q8s1);
        }

        return dq * (d_w * (sc0 * (float)sumi0 + sc1 * (float)sumi1)
                    - dmin_w * (mn0 * (float)q8s0 + mn1 * (float)q8s1));
    }
};

template<> struct DequantTraits<QType::Q3_K> {
    static constexpr int kBlockBytes   = 110;
    static constexpr int kBlockElems   = 256;
    static constexpr int kQ8PerWeight  = 8;
    static constexpr bool kNeedsQ8Sum  = false;
    static constexpr int kSmemExtra    = 0;
    static constexpr int kMaxNRows     = 2;   // complex dequant → cap NR to avoid reg pressure
    static constexpr bool kPreferKpar  = true; // compute-heavy: K-par wins on ties

    // Q3_K layout (110 bytes / 256 elements):
    //   hmask[32]   : high bit (bit 2) for each of 256 elements
    //   qs[64]      : 2-bit packed (same layout as Q2_K)
    //   scales[12]  : packed 6-bit scales (complex GGML packing)
    //   d(fp16)     : at offset 108
    //
    // q3 = q2_lowbits + (hmask_bit ? 0 : -4), range [-4..3]
    // val = d * (scale6bit - 32) * q3
    static __device__ __forceinline__ float
    dp4a_block(const uint8_t* bp, int sub,
               const int* xi, float dq, float /*q8_sum*/) {
        const uint8_t* hmask  = bp;
        const uint8_t* qs     = bp + 32;
        const uint8_t* sc_raw = bp + 96;
        float d_all = __half2float(*(const half*)(bp + 108));

        int half_idx = sub / 4;
        int shift    = (sub % 4) * 2;
        const uint8_t* qs_base = qs + half_idx * 32;
        const uint8_t* hm_base = hmask + sub * 4;  // 4 bytes = 32 bits

        // Unpack 16 6-bit scales from 12 packed bytes
        uint32_t aux0, aux1, aux2;
        memcpy(&aux0, sc_raw,     4);
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
            uint32_t hm4 = ((hm_byte >> (bit_base + 0)) & 1) |
                           (((hm_byte >> (bit_base + 1)) & 1) << 8) |
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
            uint32_t hm4 = ((hm_byte >> (bit_base + 0)) & 1) |
                           (((hm_byte >> (bit_base + 1)) & 1) << 8) |
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
using Q4_0_Traits = DequantTraits<QType::Q4_0>;
using Q8_0_Traits = DequantTraits<QType::Q8_0>;
using Q6_K_Traits = DequantTraits<QType::Q6_K>;
using Q4_K_Traits = DequantTraits<QType::Q4_K>;
using Q5_K_Traits = DequantTraits<QType::Q5_K>;
using Q2_K_Traits = DequantTraits<QType::Q2_K>;
using Q3_K_Traits = DequantTraits<QType::Q3_K>;

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
template<bool PREFER_KPAR>
static inline bool kpar_is_better(int M, int rpar_blocks) {
    const int n = kpar_n_sms();
    if (n < 1) return false;
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

template<typename QT, bool ADD_RESIDUAL>
__global__ void gemv_dp4a_kpar_kernel(
        const uint8_t* __restrict__ W,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* y,
        const half* residual,
        int M, int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x;
    if (row >= M) return;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8]; memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = partial[0] + partial[1] + partial[2] + partial[3];
        if constexpr (ADD_RESIDUAL) total += __half2float(residual[row]);
        y[row] = __float2half(total);
    }
}

template<typename QT>
__global__ void gemv_dp4a_kpar_fp32_kernel(
        const uint8_t* __restrict__ W,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        float* __restrict__ y,
        int M, int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x;
    if (row >= M) return;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8]; memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0)
        y[row] = partial[0] + partial[1] + partial[2] + partial[3];
}

template<typename QT>
__global__ void gemv_dp4a_kpar_qkv_kernel(
        const uint8_t* __restrict__ W_q,
        const uint8_t* __restrict__ W_k,
        const uint8_t* __restrict__ W_v,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_q,
        half* __restrict__ y_k,
        half* __restrict__ y_v,
        int q_rows, int k_rows, int v_rows, int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int global_row = blockIdx.x;
    const int total_rows = q_rows + k_rows + v_rows;
    if (global_row >= total_rows) return;

    const uint8_t* W;
    half* y_out;
    int local_row;
    if (global_row < q_rows) {
        W = W_q; y_out = y_q; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = W_k; y_out = y_k; local_row = global_row - q_rows;
    } else {
        W = W_v; y_out = y_v; local_row = global_row - q_rows - k_rows;
    }

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)local_row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8]; memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0)
        y_out[local_row] = __float2half(partial[0] + partial[1] + partial[2] + partial[3]);
}

template<typename QT>
__global__ void gemv_dp4a_kpar_gate_up_kernel(
        const uint8_t* __restrict__ gate_weights,
        const uint8_t* __restrict__ up_weights,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_gate,
        half* __restrict__ y_up,
        int M, int K) {
    constexpr int NWARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x;
    if (row >= M) return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* W = is_up ? up_weights : gate_weights;
    half* y = is_up ? y_up : y_gate;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (size_t)row * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        int xi[8]; memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (size_t)wb * QT::kBlockBytes, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0)
        y[row] = __float2half(partial[0] + partial[1] + partial[2] + partial[3]);
}

// ============================================================================
// Template kernel #1: Basic + Residual (replaces 10 hand-written kernels)
// ============================================================================

template<typename QT, int N_ROWS, bool ADD_RESIDUAL>
__global__ void gemv_dp4a_kernel(
        const uint8_t* __restrict__ W,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* y,
        const half* residual,
        int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = (int*)smem_q8;
    float* smem_d = (float*)(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val; memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = (half*)(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= M) return;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = (half*)(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb  = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) break;
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
                if constexpr (ADD_RESIDUAL) s += __half2float(residual[row]);
                y[row] = __float2half(s);
            }
        }
    }
}

template<typename QT>
static void launch_gemv_dp4a(const uint8_t* W, const block_q8_1* q8_1, const float* d8,
                               half* y, const half* residual, bool add_residual,
                               int M, int K, cudaStream_t stream) {
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
                pdl::launch(gemv_dp4a_kpar_kernel<QT, true>,
                    dim3(M), dim3(128), size_t(0), stream,
                    W, q8_1, d8, y, residual, M, K);
            else
                pdl::launch(gemv_dp4a_kpar_kernel<QT, false>,
                    dim3(M), dim3(128), size_t(0), stream,
                    W, q8_1, d8, y, (const half*)nullptr, M, K);
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
            pdl::launch(gemv_dp4a_kernel<QT, NR, true>,
                dim3(blocks), dim3(threads_per_block), smem_size, stream,
                W, q8_1, d8, y, residual, M, K);
        else
            pdl::launch(gemv_dp4a_kernel<QT, NR, false>,
                dim3(blocks), dim3(threads_per_block), smem_size, stream,
                W, q8_1, d8, y, (const half*)nullptr, M, K);
    };

    // Dispatch NR based on kMaxNRows (caps NR to avoid register pressure)
    constexpr int MAX_NR = QT::kMaxNRows;
    if constexpr (MAX_NR >= 4) {
        int nr4_blocks = (M + warps_per_block * 4 - 1) / (warps_per_block * 4);
        if (nr4_blocks >= 128) { launch(std::integral_constant<int, 4>{}); return; }
    }
    if constexpr (MAX_NR >= 2) {
        int nr2_blocks = (M + warps_per_block * 2 - 1) / (warps_per_block * 2);
        if (nr2_blocks >= 64) { launch(std::integral_constant<int, 2>{}); return; }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #2: FP32 Output (replaces 5 hand-written kernels)
// ============================================================================

template<typename QT, int N_ROWS>
__global__ void gemv_dp4a_fp32_kernel(
        const uint8_t* __restrict__ W,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        float* __restrict__ y,
        int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = (int*)smem_q8;
    float* smem_d = (float*)(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val; memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = (half*)(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= M) return;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = (half*)(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb  = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) break;
            const uint8_t* bp = W + (size_t)row * row_bytes + (size_t)wb * QT::kBlockBytes;
            sum[r] += QT::dp4a_block(bp, sub, xi, dq, q8_sum);
        }
    }

    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0 && row_base + r < M) y[row_base + r] = sum[r];
    }
}

template<typename QT>
static void launch_gemv_dp4a_fp32(const uint8_t* W, const block_q8_1* q8_1, const float* d8,
                                    float* y, int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    // K-parallel check (compare against NR=1 max-occupancy baseline)
    {
        int nr1_blocks = (M + warps_per_block - 1) / warps_per_block;
        if (kpar_is_better<QT::kPreferKpar>(M, nr1_blocks)) {
            pdl::launch(gemv_dp4a_kpar_fp32_kernel<QT>,
                dim3(M), dim3(128), size_t(0), stream,
                W, q8_1, d8, y, M, K);
            return;
        }
    }

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem_size = (size_t)total_q8 * (40 + QT::kSmemExtra);

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (M + rows_per_block - 1) / rows_per_block;
        pdl::launch(gemv_dp4a_fp32_kernel<QT, NR>,
            dim3(blocks), dim3(threads_per_block), smem_size, stream,
            W, q8_1, d8, y, M, K);
    };

    // Dispatch NR based on kMaxNRows (caps NR to avoid register pressure)
    constexpr int MAX_NR = QT::kMaxNRows;
    if constexpr (MAX_NR >= 4) {
        int nr4_blocks = (M + warps_per_block * 4 - 1) / (warps_per_block * 4);
        if (nr4_blocks >= 128) { launch(std::integral_constant<int, 4>{}); return; }
    }
    if constexpr (MAX_NR >= 2) {
        int nr2_blocks = (M + warps_per_block * 2 - 1) / (warps_per_block * 2);
        if (nr2_blocks >= 64) { launch(std::integral_constant<int, 2>{}); return; }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #3: QKV Fused (replaces 5 hand-written kernels)
// ============================================================================

template<typename QT, int N_ROWS>
__global__ void gemv_dp4a_qkv_kernel(
        const uint8_t* __restrict__ W_q,
        const uint8_t* __restrict__ W_k,
        const uint8_t* __restrict__ W_v,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_q,
        half* __restrict__ y_k,
        half* __restrict__ y_v,
        int q_rows, int k_rows, int v_rows, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;
    const int total_rows = q_rows + k_rows + v_rows;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = (int*)smem_q8;
    float* smem_d = (float*)(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val; memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = (half*)(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= total_rows) return;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = (half*)(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb  = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int global_row = row_base + r;
            if (global_row >= total_rows) break;

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
            if (global_row >= total_rows) break;
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

template<typename QT>
static void launch_gemv_dp4a_qkv(const uint8_t* W_q, const uint8_t* W_k, const uint8_t* W_v,
                                    const block_q8_1* q8_1, const float* d8,
                                    half* y_q, half* y_k, half* y_v,
                                    int q_rows, int k_rows, int v_rows, int K,
                                    cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int total = q_rows + k_rows + v_rows;

    // K-parallel check (compare against NR=1 max-occupancy baseline)
    {
        int nr1_blocks = (total + warps_per_block - 1) / warps_per_block;
        if (kpar_is_better<QT::kPreferKpar>(total, nr1_blocks)) {
            pdl::launch(gemv_dp4a_kpar_qkv_kernel<QT>,
                dim3(total), dim3(128), size_t(0), stream,
                W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v,
                q_rows, k_rows, v_rows, K);
            return;
        }
    }

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem = (size_t)total_q8 * (40 + QT::kSmemExtra);

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (total + rows_per_block - 1) / rows_per_block;
        pdl::launch(gemv_dp4a_qkv_kernel<QT, NR>,
            dim3(blocks), dim3(threads_per_block), smem, stream,
            W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v,
            q_rows, k_rows, v_rows, K);
    };

    // Dispatch NR based on kMaxNRows
    constexpr int MAX_NR = QT::kMaxNRows;
    if constexpr (MAX_NR >= 4) {
        int nr4_blocks = (total + warps_per_block * 4 - 1) / (warps_per_block * 4);
        if (nr4_blocks >= 128) { launch(std::integral_constant<int, 4>{}); return; }
    }
    if constexpr (MAX_NR >= 2) {
        int nr2_blocks = (total + warps_per_block * 2 - 1) / (warps_per_block * 2);
        if (nr2_blocks >= 64) { launch(std::integral_constant<int, 2>{}); return; }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #4: Gate+Up Fused (replaces 5 hand-written kernels)
// blockIdx.y: 0 = gate, 1 = up. N_ROWS rows per warp.
// ============================================================================

template<typename QT, int N_ROWS>
__global__ void gemv_dp4a_gate_up_kernel(
        const uint8_t* __restrict__ gate_weights,
        const uint8_t* __restrict__ up_weights,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_gate,
        half* __restrict__ y_up,
        int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    extern __shared__ char smem_q8[];
    int* smem_qs = (int*)smem_q8;
    float* smem_d = (float*)(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val; memcpy(&val, q8_1[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = (half*)(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = d8[i];
            smem_s[i] = q8_1[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = d8[i];
    }
    __syncthreads();

    if (row_base >= M) return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* W = is_up ? up_weights : gate_weights;
    half* y = is_up ? y_up : y_gate;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = (half*)(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb  = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) break;
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
            if (row < M) y[row] = __float2half(sum[r]);
        }
    }
}

// ============================================================================
// Template kernel #5: MoE Decode with NR (replaces 4 hand-written kernels)
// Q8_1 data cooperatively loaded into shared memory — all 8 warps share
// the same Q8_1 input per expert slot, eliminating 8x redundant L2 reads.
// NR>1: each warp handles multiple rows, halving CTAs and smem loads.
// ============================================================================

template<typename QT, int NR>
__global__ void gemv_dp4a_moe_decode_kernel(
        const uint8_t* __restrict__ packed_weights,
        const int32_t* __restrict__ expert_indices,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y,
        int rows, int K,
        size_t expert_stride_bytes,
        int q8_1_stride,
        int d8_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row_base = (local_block * warps_per_block + warp_id) * NR;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;

    // Cooperatively load Q8_1 into shared memory (same pattern as dense kernel)
    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;

    extern __shared__ char smem_q8[];
    int* smem_qs = (int*)smem_q8;
    float* smem_d = (float*)(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val; memcpy(&val, x_q8[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = (half*)(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = x_d8[i];
            smem_s[i] = x_q8[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = x_d8[i];
    }
    __syncthreads();

    if (row_base >= rows) return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    float sum[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = (half*)(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb  = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const int row = row_base + r;
            if (row >= rows) break;
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

template<typename QT>
static void launch_gemv_dp4a_moe_decode(
        const uint8_t* packed_weights, const int32_t* expert_indices,
        const block_q8_1* q8_1, const float* d8,
        half* y, int rows, int K,
        size_t expert_stride_bytes,
        int q8_1_stride, int d8_stride, int top_k,
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
        pdl::launch(gemv_dp4a_moe_decode_kernel<QT, NR>,
            dim3(total_blocks), dim3(threads_per_block), smem_size, stream,
            packed_weights, expert_indices, q8_1, d8, y, rows, K,
            expert_stride_bytes, q8_1_stride, d8_stride, blocks_per_expert);
    };

    // Use NR=2 when enough blocks to fill SMs (same threshold as dense kernel)
    if constexpr (MAX_NR >= 2) {
        int nr2_rows_per_block = warps_per_block * 2;
        int nr2_blocks = top_k * ((rows + nr2_rows_per_block - 1) / nr2_rows_per_block);
        if (nr2_blocks >= 64) { launch(std::integral_constant<int, 2>{}); return; }
    }
    launch(std::integral_constant<int, 1>{});
}

// ============================================================================
// Template kernel #6: MoE Gate+Up Dual-Matrix (replaces 4 hand-written kernels)
// Each warp computes BOTH gate and up projections for the same row, sharing
// Q8_1 from smem. Halves CTA count and smem loads vs separate gate/up blocks.
// ============================================================================

template<typename QT>
__global__ void gemv_dp4a_moe_gate_up_kernel(
        const uint8_t* __restrict__ gate_weights,
        const uint8_t* __restrict__ up_weights,
        const int32_t* __restrict__ expert_indices,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_gate,
        half* __restrict__ y_up,
        int rows, int K,
        size_t gate_stride_bytes,
        size_t up_stride_bytes,
        int q8_1_stride,
        int d8_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;

    // Cooperatively load Q8_1 into shared memory (same pattern as dense kernel)
    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;

    extern __shared__ char smem_q8[];
    int* smem_qs = (int*)smem_q8;
    float* smem_d = (float*)(smem_q8 + total_q8 * 36);

    for (int i = threadIdx.x; i < total_q8 * 8; i += blockDim.x) {
        int blk = i >> 3, w = i & 7;
        int val; memcpy(&val, x_q8[blk].qs + w * 4, 4);
        smem_qs[blk * kSmemQ8Stride + w] = val;
    }
    if constexpr (QT::kNeedsQ8Sum) {
        half* smem_s = (half*)(smem_q8 + total_q8 * 40);
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x) {
            smem_d[i] = x_d8[i];
            smem_s[i] = x_q8[i].s;
        }
    } else {
        for (int i = threadIdx.x; i < total_q8; i += blockDim.x)
            smem_d[i] = x_d8[i];
    }
    __syncthreads();

    if (row >= rows) return;

    const int expert_id = expert_indices[expert_slot];
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* W_gate_row = gate_weights + (size_t)expert_id * gate_stride_bytes
                              + (size_t)row * row_bytes;
    const uint8_t* W_up_row = up_weights + (size_t)expert_id * up_stride_bytes
                            + (size_t)row * row_bytes;

    float sum_gate = 0.0f, sum_up = 0.0f;
    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum) {
            half* smem_s = (half*)(smem_q8 + total_q8 * 40);
            q8_sum = __half2float(smem_s[b]);
        }

        const int wb  = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        const size_t block_off = (size_t)wb * QT::kBlockBytes;
        sum_gate += QT::dp4a_block(W_gate_row + block_off, sub, xi, dq, q8_sum);
        sum_up   += QT::dp4a_block(W_up_row   + block_off, sub, xi, dq, q8_sum);
    }

    for (int off = 16; off > 0; off >>= 1) {
        sum_gate += __shfl_down_sync(0xFFFFFFFF, sum_gate, off);
        sum_up   += __shfl_down_sync(0xFFFFFFFF, sum_up,   off);
    }

    if (lane == 0) {
        const int out_idx = expert_slot * rows + row;
        y_gate[out_idx] = __float2half(sum_gate);
        y_up[out_idx]   = __float2half(sum_up);
    }
}

template<typename QT>
static void launch_gemv_dp4a_moe_gate_up(
        const uint8_t* gate_weights, const uint8_t* up_weights,
        const int32_t* expert_indices,
        const block_q8_1* q8_1, const float* d8,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int q8_1_stride, int d8_stride, int top_k,
        cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem_size = (size_t)total_q8 * (40 + QT::kSmemExtra);
    pdl::launch(gemv_dp4a_moe_gate_up_kernel<QT>,
        dim3(top_k * blocks_per_expert), dim3(threads_per_block), smem_size, stream,
        gate_weights, up_weights, expert_indices, q8_1, d8,
        y_gate, y_up, rows, K,
        gate_stride_bytes, up_stride_bytes,
        q8_1_stride, d8_stride, blocks_per_expert);
}

// ============================================================================
// Template kernel #7: Inline Q8_1 Quantization + GEMV + Residual
// Takes FP16 input, cooperatively quantizes to Q8_1 in shared memory,
// then runs dp4a GEMV. Eliminates separate quantize kernel + DRAM round-trip.
// ============================================================================

template<typename QT, int N_ROWS, bool ADD_RESIDUAL>
__global__ void gemv_dp4a_inline_quant_kernel(
        const uint8_t* __restrict__ W,
        const half* __restrict__ x_fp16,   // FP16 input vector [K]
        half* y,
        const half* residual,
        int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = (size_t)(K / QT::kBlockElems) * QT::kBlockBytes;

    // Shared memory layout: [qs: total_q8 * 36 bytes (padded)] [d: total_q8 * 4 bytes]
    extern __shared__ char smem_q8[];
    int* smem_qs = (int*)smem_q8;
    float* smem_d = (float*)(smem_q8 + total_q8 * 36);

    // Cooperative FP16 → Q8_1 quantization directly into shared memory.
    // Each iteration processes one Q8_1 block (32 FP16 elements).
    // We assign blocks round-robin across all threads.
    const int n_q8_blocks = K / 32;  // Q8_1 blocks from raw FP16 input
    for (int blk = threadIdx.x / 32; blk < n_q8_blocks; blk += blockDim.x / 32) {
        const int t = threadIdx.x % 32;  // lane within this warp
        const int base = blk * 32;

        // Load FP16 value
        float val = __half2float(x_fp16[base + t]);

        // Warp-wide max for quantization scale
        float amax = fabsf(val);
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, amax, off);
            amax = fmaxf(amax, other);
        }

        float d = amax / 127.0f;
        float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        int8_t q = static_cast<int8_t>(__float2int_rn(val * id));

        // Write quantized int8 to smem (pack 4 bytes into int32 for dp4a)
        // smem_qs layout: [blk * 8 + word] where each word is 4 packed int8s
        // We need smem_qs[blk * 8 + t/4] |= (q << (t%4)*8)
        // Simpler: write to a byte array, then the reads will use memcpy
        int8_t* smem_bytes = (int8_t*)smem_q8;

        // For kQ8PerWeight > 1 (Q6_K, Q4_K, Q5_K): each weight super-block
        // consumes kQ8PerWeight Q8_1 blocks. blk maps 1:1 to Q8_1 blocks.
        smem_bytes[blk * (kSmemQ8Stride * 4) + t] = q;
        if (t == 0) smem_d[blk] = d;
    }
    __syncthreads();

    if (row_base >= M) return;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int b = lane; b < total_q8; b += 32) {
        int xi[8];
        memcpy(xi, smem_qs + b * kSmemQ8Stride, 32);
        float dq = smem_d[b];

        const int wb  = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) break;
            const uint8_t* bp = W + (size_t)row * row_bytes + (size_t)wb * QT::kBlockBytes;
            sum[r] += QT::dp4a_block(bp, sub, xi, dq, 0.0f);
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
                if constexpr (ADD_RESIDUAL) s += __half2float(residual[row]);
                y[row] = __float2half(s);
            }
        }
    }
}

template<typename QT>
static void launch_gemv_dp4a_inline_quant(
        const uint8_t* W, const half* x_fp16,
        half* y, const half* residual, bool add_residual,
        int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t smem_size = (size_t)total_q8 * 40;  // 36 bytes qs (padded) + 4 bytes d per block

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (M + rows_per_block - 1) / rows_per_block;
        if (add_residual)
            pdl::launch(gemv_dp4a_inline_quant_kernel<QT, NR, true>,
                dim3(blocks), dim3(threads_per_block), smem_size, stream,
                W, x_fp16, y, residual, M, K);
        else
            pdl::launch(gemv_dp4a_inline_quant_kernel<QT, NR, false>,
                dim3(blocks), dim3(threads_per_block), smem_size, stream,
                W, x_fp16, y, (const half*)nullptr, M, K);
    };

    // Dispatch NR based on kMaxNRows
    constexpr int MAX_NR = QT::kMaxNRows;
    if constexpr (MAX_NR >= 4) {
        int nr4_blocks = (M + warps_per_block * 4 - 1) / (warps_per_block * 4);
        if (nr4_blocks >= 128) { launch(std::integral_constant<int, 4>{}); return; }
    }
    if constexpr (MAX_NR >= 2) {
        int nr2_blocks = (M + warps_per_block * 2 - 1) / (warps_per_block * 2);
        if (nr2_blocks >= 64) { launch(std::integral_constant<int, 2>{}); return; }
    }
    launch(std::integral_constant<int, 1>{});
}

} // namespace imp
