#pragma once

// TurboQuant FP4 E2M1 + UE8M0 device helpers.
// Shared between KV cache write kernels and attention decode kernels.

#include <cstdint>

namespace imp {

// MXFP4 micro-scale group size (32 elements per UE8M0 scale)
static constexpr int kTQFP4GroupSize = 32;

// FP4 E2M1 dequant LUT: magnitude code [0-7] → float value
// Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
__constant__ float kTQFP4DequantLUT[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// Quantize float magnitude → FP4 E2M1 3-bit code (round-to-nearest)
__device__ __forceinline__ uint8_t tq_fp4_quantize_abs(float abs_val) {
    if (abs_val <= 0.0f) return 0;
    if (abs_val >= 6.0f) return 7;
    if (abs_val < 0.25f)  return 0;
    if (abs_val < 0.75f)  return 1;
    if (abs_val < 1.25f)  return 2;
    if (abs_val < 1.75f)  return 3;
    if (abs_val < 2.5f)   return 4;
    if (abs_val < 3.5f)   return 5;
    if (abs_val < 5.0f)   return 6;
    return 7;
}

// Float → UE8M0 (pure-exponent, value = 2^(bits-127), rounds up to next pow2)
__device__ __forceinline__ uint8_t tq_fp4_float_to_ue8m0(float val) {
    if (val <= 0.0f) return 0;
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(float));
    int f_exp = static_cast<int>((fbits >> 23) & 0xFF);
    if (fbits & 0x7FFFFF) f_exp++;
    if (f_exp < 0) return 0;
    if (f_exp > 254) return 254;
    return static_cast<uint8_t>(f_exp);
}

// UE8M0 → float: value = 2^(bits - 127)
__device__ __forceinline__ float tq_fp4_ue8m0_to_float(uint8_t bits) {
    if (bits == 0) return 0.0f;
    uint32_t fp32 = static_cast<uint32_t>(bits) << 23;
    return __uint_as_float(fp32);
}

// Dequant one FP4 E2M1 nibble (4 bits: sign[3] | code[2:0]) → float
__device__ __forceinline__ float tq_fp4_dequant_nibble(uint8_t nibble) {
    uint8_t sign = (nibble >> 3) & 1;
    uint8_t code = nibble & 0x7;
    float val = kTQFP4DequantLUT[code];
    return sign ? -val : val;
}

// Unpack + dequant low nibble (bits [3:0]) of a packed byte
__device__ __forceinline__ float tq_fp4_unpack_lo(uint8_t packed) {
    return tq_fp4_dequant_nibble(packed & 0xF);
}

// Unpack + dequant high nibble (bits [7:4]) of a packed byte
__device__ __forceinline__ float tq_fp4_unpack_hi(uint8_t packed) {
    return tq_fp4_dequant_nibble((packed >> 4) & 0xF);
}

} // namespace imp
