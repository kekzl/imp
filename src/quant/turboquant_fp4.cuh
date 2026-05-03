#pragma once

// TurboQuant FP4 E2M1 + UE8M0 device helpers.
// Shared between KV cache write kernels and attention decode kernels.

#include <cstdint>

namespace imp {

// MXFP4 micro-scale group size (32 elements per UE8M0 scale)
static constexpr int kTQFP4GroupSize = 32;

// FP4 E2M1 dequant LUT: magnitude code [0-7] → float value
// Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
static __constant__ float kTQFP4DequantLUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Quantize float magnitude → FP4 E2M1 3-bit code (round-to-nearest)
// Branchless version: sum of comparisons against midpoint thresholds.
// Thresholds between adjacent E2M1 values: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
// The code equals the count of thresholds the value exceeds.
__device__ __forceinline__ uint8_t tq_fp4_quantize_abs(float abs_val) {
    uint8_t code = (abs_val >= 0.25f) + (abs_val >= 0.75f) + (abs_val >= 1.25f) + (abs_val >= 1.75f) +
                   (abs_val >= 2.5f) + (abs_val >= 3.5f) + (abs_val >= 5.0f);
    return code;  // 0..7 maps directly to E2M1 magnitude codes
}

// Quantize a float (signed) to packed 4-bit FP4 E2M1 nibble [sign:1 | code:3]
__device__ __forceinline__ uint8_t tq_fp4_quantize_signed(float val) {
    uint8_t sign = (val < 0.0f) ? 1u : 0u;
    uint8_t code = tq_fp4_quantize_abs(fabsf(val));
    return (sign << 3) | code;
}

// Pack two signed FP4 nibbles into one byte (lo=even, hi=odd).
// Uses HW cvt.rn.satfinite.e2m1x2.f32 on sm_120+ (single PTX instruction,
// IEEE RNE rounding, saturates to ±6). SW fallback for older archs.
__device__ __forceinline__ uint8_t tq_fp4_pack_pair(float v0, float v1) {
#if __CUDA_ARCH__ >= 1200
    uint32_t out;
    asm volatile(
        "{ .reg .b8 b;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "  cvt.u32.u8 %0, b; }\n"
        : "=r"(out)
        : "f"(v0), "f"(v1));
    return static_cast<uint8_t>(out);
#else
    return tq_fp4_quantize_signed(v0) | (tq_fp4_quantize_signed(v1) << 4);
#endif
}

// Float → UE8M0 (pure-exponent, value = 2^(bits-127), rounds up to next pow2)
__device__ __forceinline__ uint8_t tq_fp4_float_to_ue8m0(float val) {
    if (val <= 0.0f)
        return 0;
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(float));
    int f_exp = static_cast<int>((fbits >> 23) & 0xFF);
    if (fbits & 0x7FFFFF)
        f_exp++;
    if (f_exp < 0)
        return 0;
    if (f_exp > 254)
        return 254;
    return static_cast<uint8_t>(f_exp);
}

// UE8M0 → float: value = 2^(bits - 127)
__device__ __forceinline__ float tq_fp4_ue8m0_to_float(uint8_t bits) {
    if (bits == 0)
        return 0.0f;
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

// Reciprocal of INT4 symmetric range max (7) for dequantization: val / 7.0
static constexpr float kTQINT4InvScale = 1.0f / 7.0f;

}  // namespace imp
