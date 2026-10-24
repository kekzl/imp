#pragma once
// IQ3_S codebook (512 x u32, byte j = magnitude of element j) for dp4a and int8 tensor-core paths.
// Global memory read via __ldg: per-lane indices would serialize on __constant__.

#include <cstdint>

namespace imp {

static __device__ const uint32_t kIq3sGridLdg[512] = {
#include "quant/iq3s_grid.inc"
};

// Grid magnitudes with sign bit k of `sb` negating byte k: 4 signed int8.
__device__ __forceinline__ int iq3s_apply_signs(uint32_t grid, uint32_t sb) {
    const uint32_t mask = (((sb & 0xFu) * 0x10204080u) & 0x80808080u) >> 7;  // 0/1 per byte
    const uint32_t neg = mask * 0xFFu;                                       // 0x00/0xFF per byte
    return static_cast<int>(__vsub4(grid ^ neg, neg));
}

// 4 signed int8 of one IQ3_S grid group: magnitudes kIq3sGridLdg[idx], sign bit k of `sb` negates byte k.
__device__ __forceinline__ int iq3s_signed4(int idx, uint32_t sb) {
    return iq3s_apply_signs(__ldg(&kIq3sGridLdg[idx]), sb);
}

}  // namespace imp
