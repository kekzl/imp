#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// FP8 E4M3 software conversion device helpers.
//
// Shared by fp8_quant.cu, nvfp4_quant.cu, and any other CUDA code that needs
// scalar float <-> FP8 E4M3 conversion on the device.
//
// FP8 E4M3 layout (8 bits): 1 sign | 4 exponent | 3 mantissa, bias = 7
//   Normal value   : (-1)^s * 2^(e-7) * (1 + m/8)    e in [1,14]
//   Subnormal value: (-1)^s * 2^(-6)  * (m/8)         e == 0, m != 0
//   Zero           : s=0|1, e=0, m=0
//   Max normal     : +/- 448.0  (e=14, m=7)
//   Min subnormal  : +/- 2^(-9) = 1/512
//
// Uses round-to-nearest-even for float->FP8 and saturates to max normal
// (not NaN) on overflow.
// ---------------------------------------------------------------------------

static constexpr float kFP8E4M3MaxVal = 448.0f;

// ---------------------------------------------------------------------------
// FP32 -> FP8 E4M3 software conversion with saturation (no Inf in E4M3).
// Round-to-nearest-even.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val)
{
    const uint32_t sign = (val < 0.0f) ? 1u : 0u;
    float abs_val = fabsf(val);

    if (abs_val > kFP8E4M3MaxVal) abs_val = kFP8E4M3MaxVal;  // clamp

    // Smallest E4M3 subnormal: 2^(-9)
    if (abs_val < (1.0f / 512.0f)) {
        return (uint8_t)(sign << 7);
    }

    uint32_t fbits;
    memcpy(&fbits, &abs_val, sizeof(float));
    int f_exp = (int)((fbits >> 23) & 0xFF) - 127;
    uint32_t f_man = fbits & 0x7FFFFF;

    int e4 = f_exp + 7;  // E4M3 bias = 7

    uint8_t result;
    if (e4 <= 0) {
        // Subnormal in E4M3.
        int shift = 1 - e4;
        uint32_t full_man = (1u << 23) | f_man;
        int right_shift = 20 + shift;
        uint8_t m3;
        if (right_shift >= 32) {
            m3 = 0;
        } else {
            uint32_t shifted = full_man >> right_shift;
            uint32_t remainder = full_man & ((1u << right_shift) - 1);
            uint32_t half_point = 1u << (right_shift - 1);
            if (remainder > half_point ||
                (remainder == half_point && (shifted & 1))) {
                shifted += 1;
            }
            m3 = (uint8_t)(shifted & 0x07);
            if (shifted > 7) {
                result = (uint8_t)((sign << 7) | (1 << 3) | 0);
                return result;
            }
        }
        result = (uint8_t)((sign << 7) | m3);
    } else if (e4 >= 15) {
        // E4M3 has no Inf/NaN for saturation; clamp to max normal (e=14, m=7) = 448.
        result = (uint8_t)((sign << 7) | (14 << 3) | 7);
    } else {
        // Normal: round-to-nearest-even.
        uint32_t round_bit = (f_man >> 19) & 1;
        uint32_t sticky = (f_man & 0x7FFFF) ? 1 : 0;
        uint8_t m3 = (uint8_t)((f_man >> 20) & 0x07);
        if (round_bit && (sticky || (m3 & 1))) {
            m3 += 1;
            if (m3 > 7) {
                m3 = 0;
                e4 += 1;
                if (e4 >= 15) {
                    result = (uint8_t)((sign << 7) | (14 << 3) | 7);
                    return result;
                }
            }
        }
        result = (uint8_t)((sign << 7) | ((e4 & 0x0F) << 3) | (m3 & 0x07));
    }
    return result;
}

// ---------------------------------------------------------------------------
// FP8 E4M3 -> FP32 software conversion.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits)
{
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0x0F;
    uint32_t man  = bits & 0x07;

    float abs_val;
    if (exp == 0) {
        // Subnormal: value = 0.mantissa * 2^(1 - bias) = man * 2^(-9)
        abs_val = (float)man * (1.0f / 512.0f);
    } else {
        // Normal: value = 1.mantissa * 2^(exp - bias) = (8 + man) * 2^(exp - 10)
        abs_val = (float)(8 + man) * exp2f((float)(exp) - 10.0f);
    }
    return sign ? -abs_val : abs_val;
}

} // namespace imp
