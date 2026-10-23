#pragma once
// IQ4_NL / IQ4_XS codebook lookup (kvalues_iq4nl) for int8 tensor-core and dp4a paths.

#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// 8 IQ4 codebook indices (4 qs bytes) -> int8 values: .x = the 4 low nibbles, .y = the 4 high
// nibbles, both in byte order. kvalues_iq4nl as 4 words; __byte_perm reads selector bits 0-2 only.
__device__ __forceinline__ int2 iq4nl_lut8(uint32_t q4) {
    constexpr uint32_t kT0 = 0xBFAD9881u, kT1 = 0xF6EADDCFu, kT2 = 0x26190D01u, kT3 = 0x71594535u;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const uint32_t s = q4 >> (16 * i);
        tmp[i] = __byte_perm(__byte_perm(kT0, kT1, s), __byte_perm(kT2, kT3, s), sel >> (16 * i));
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

}  // namespace imp
