#pragma once
// DequantTraits for IQ4_NL and IQ4_XS (dp4a GEMV family, gemv_dp4a_traits.cuh).

#include "compute/gemv_dp4a_traits.cuh"
#include "compute/iq4_lut.cuh"

namespace imp {

// One 32-element IQ4 group: qs[e] low nibble = element e, high nibble = e + 16 (IQ4_NL layout).
__device__ __forceinline__ int iq4_dp4a_group(const uint32_t (&w)[4], const int* xi) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int2 v = iq4nl_lut8(w[j]);
        sumi = __dp4a(v.x, xi[j], sumi);
        sumi = __dp4a(v.y, xi[j + 4], sumi);
    }
    return sumi;
}

// IQ4_NL: 18-B block = fp16 d + qs[16]; val = d * kvalues_iq4nl[nibble]. Blocks are 2-aligned.
template <>
struct DequantTraits<DPQTag::IQ4_NL> {
    static constexpr int kBlockBytes = 18;
    static constexpr int kBlockElems = 32;
    static constexpr int kQ8PerWeight = 1;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = false;

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int /*sub*/, const int* xi,
                                                       float dq, float /*q8_sum*/) {
        const auto* h = reinterpret_cast<const uint16_t*>(bp);
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; i++)
            w[i] = static_cast<uint32_t>(__ldg(h + 1 + (2 * i))) |
                   (static_cast<uint32_t>(__ldg(h + 2 + (2 * i))) << 16);
        return __half2float(__ushort_as_half(__ldg(h))) * dq * static_cast<float>(iq4_dp4a_group(w, xi));
    }
};

// IQ4_XS: 136-B super-block = fp16 d, scales_h u16 (2 bits/sub), scales_l[4] (nibble/sub),
// qs[128]; sub-block s: val = d * (ls - 32) * kvalues_iq4nl[nibble]. K % 256 == 0 keeps it 8-aligned.
template <>
struct DequantTraits<DPQTag::IQ4_XS> {
    static constexpr int kBlockBytes = 136;
    static constexpr int kBlockElems = 256;
    static constexpr int kQ8PerWeight = 8;
    static constexpr bool kNeedsQ8Sum = false;
    static constexpr int kSmemExtra = 0;
    static constexpr int kMaxNRows = 4;
    static constexpr bool kPreferKpar = true;

    static __device__ __forceinline__ float dp4a_block(const uint8_t* bp, int sub, const int* xi, float dq,
                                                       float /*q8_sum*/) {
        const uint2 hdr = __ldg(reinterpret_cast<const uint2*>(bp));
        const uint2 qa = __ldg(reinterpret_cast<const uint2*>(bp + 8 + (16 * sub)));
        const uint2 qb = __ldg(reinterpret_cast<const uint2*>(bp + 16 + (16 * sub)));
        const uint32_t w[4] = {qa.x, qa.y, qb.x, qb.y};
        const int ls = static_cast<int>(((hdr.y >> (4 * sub)) & 0xFu) |
                                        (((hdr.x >> (16 + (2 * sub))) & 3u) << 4));
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xFFFFu)));
        return d * static_cast<float>(ls - 32) * dq * static_cast<float>(iq4_dp4a_group(w, xi));
    }
};

using IQ4_NL_Traits = DequantTraits<DPQTag::IQ4_NL>;
using IQ4_XS_Traits = DequantTraits<DPQTag::IQ4_XS>;

}  // namespace imp
