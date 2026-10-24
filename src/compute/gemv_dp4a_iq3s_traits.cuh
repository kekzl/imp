#pragma once
// DequantTraits for IQ3_S (dp4a GEMV family, gemv_dp4a_traits.cuh).

#include "compute/gemv_dp4a_traits.cuh"
#include "compute/iq3s_grid.cuh"

namespace imp {

// IQ3_S: 110-B super-block = fp16 d, qs[64], qh[8], signs[32], scales[4]; 2-aligned. Sub-block s
// (32 elements): group g = elements 4g..4g+3 = grid[qs[8s+g] | qh[s] bit g << 8], sign bit e of
// signs[4s..4s+3] (u32) negates element e, scale d * (1 + 2 * nibble s of scales).
template <>
struct DequantTraits<DPQTag::IQ3_S> {
    static constexpr int kBlockBytes = 110;
    static constexpr int kBlockElems = 256;
    static constexpr int kQ8PerWeight = 8;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = true;

    // One sub-block decoded: 32 signed int8 (groups 0..7) and d * (1 + 2 * scale).
    struct Block {
        int v[8];
        float sc;
    };

    static __device__ __forceinline__ Block decode(const uint8_t* bp, int sub) {
        uint32_t qw[2], sw[1];
        ldg_words_2a(bp + 2 + (8 * sub), qw);
        ldg_words_2a(bp + 74 + (4 * sub), sw);
        const uint32_t qh = __ldg(bp + 66 + sub);
        const uint32_t sc = (__ldg(bp + 106 + (sub >> 1)) >> (4 * (sub & 1))) & 0xFu;
        Block b;
#pragma unroll
        for (int g = 0; g < 8; g++) {
            const uint32_t q = qw[g >> 2];
            const int idx = static_cast<int>(((q >> (8 * (g & 3))) & 0xFFu) | (((qh >> g) & 1u) << 8));
            b.v[g] = iq3s_signed4(idx, sw[0] >> (4 * g));
        }
        const auto d = __ldg(reinterpret_cast<const unsigned short*>(bp));
        b.sc = __half2float(__ushort_as_half(d)) * static_cast<float>(1 + (2 * sc));
        return b;
    }

    static __device__ __forceinline__ float dot(const Block& b, const int* xi, float dq) {
        int sumi = 0;
#pragma unroll
        for (int g = 0; g < 8; g++)
            sumi = __dp4a(b.v[g], xi[g], sumi);
        return b.sc * dq * static_cast<float>(sumi);
    }

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int sub, const int* xi, float dq,
                                                       float /*q8_sum*/) {
        return dot(decode(bp, sub), xi, dq);
    }
};

using IQ3_S_Traits = DequantTraits<DPQTag::IQ3_S>;

}  // namespace imp
