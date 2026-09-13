#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace imp {

// FP8 E4M3 software conversion device helpers, shared by fp8_quant.cu, nvfp4_quant.cu, and
// any CUDA code needing scalar float<->FP8 E4M3 on device.
//   Normal: (-1)^s*2^(e-7)*(1+m/8), e in [1,14]. Subnormal: (-1)^s*2^-6*(m/8), e=0,m!=0.
//   Zero: e=0,m=0. Max normal: +-448.0 (e=14,m=7). Min subnormal: +-2^-9 = 1/512.
// Round-to-nearest-even float->FP8, saturates to max normal (not NaN) on overflow.

static constexpr float kFP8E4M3MaxVal = 448.0f;

// FP32->FP8 E4M3 software conversion with saturation (no Inf in E4M3), round-to-nearest-even.
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
    const uint32_t sign = (val < 0.0f) ? 1u : 0u;
    float abs_val = fabsf(val);

    if (abs_val > kFP8E4M3MaxVal)
        abs_val = kFP8E4M3MaxVal;  // clamp

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
            if (remainder > half_point || (remainder == half_point && (shifted & 1))) {
                shifted += 1;
            }
            m3 = (uint8_t)(shifted & 0x07);
            if (shifted > 7) {
                result = (uint8_t)((sign << 7) | (1 << 3) | 0);
                return result;
            }
        }
        result = (uint8_t)((sign << 7) | m3);
    } else if (e4 > 15) {
        // True overflow (input > E4M3-fn max range) saturates to max normal 448 = (1+6/8)*2^8,
        // bits 0x7E. Earlier code returned (14<<3)|7 = 0x77 (decode 240), a 0.536x squash for any
        // value >= 256: broke compressed-tensors NVFP4 prequant, outlier-block scales near 447 read
        // back as 240, halving GEMM output on affected rows.
        result = (uint8_t)((sign << 7) | (15 << 3) | 6);
    } else {
        // Normal range, including e4=15 (which is valid for m=0..6, encoding
        // the [256, 448] range; m=7 is the only NaN slot in E4M3-fn).
        // Round-to-nearest-even.
        uint32_t round_bit = (f_man >> 19) & 1;
        uint32_t sticky = (f_man & 0x7FFFF) ? 1 : 0;
        uint8_t m3 = (uint8_t)((f_man >> 20) & 0x07);
        if (round_bit && (sticky || (m3 & 1))) {
            m3 += 1;
            if (m3 > 7) {
                m3 = 0;
                e4 += 1;
                if (e4 > 15) {
                    // Round-up overflowed past the highest valid e_field.
                    result = (uint8_t)((sign << 7) | (15 << 3) | 6);
                    return result;
                }
            }
        }
        // Saturate the NaN slot (e=15, m=7) to max normal (e=15, m=6).
        if (e4 == 15 && m3 == 7)
            m3 = 6;
        result = (uint8_t)((sign << 7) | ((e4 & 0x0F) << 3) | (m3 & 0x07));
    }
    return result;
}

// FP8 E4M3->FP32 (fast, branchless bit repack). Normal (exp>0): (1+man/8)*2^(exp-7)
// (bias=7). Denorm (exp=0): man*2^-9. Sign bit (bit 7) applies to output sign.
// Consolidates two prior copies (a slow exp2f version here, a fast bit-repack one in
// nvfp4_gemm.cu) that disagreed on denorm values until the NVFP4 prequant debug (50x
// inflation) forced a fix; now this is the single correct fast implementation.
__device__ __forceinline__ float fp8_e4m3_to_float_fast(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 3) & 0x0F;
    uint32_t man = bits & 0x07;
    uint32_t fp32;
    if (exp == 0) {
        float v = (float)man * (1.0f / 512.0f);
        fp32 = (sign << 31) | __float_as_uint(v);
    } else {
        fp32 = (sign << 31) | ((exp + 120u) << 23) | (man << 20);
    }
    return __uint_as_float(fp32);
}

// Alias for code that used the slow exp2f-based name. Both refer to the same
// fast implementation now.
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) { return fp8_e4m3_to_float_fast(bits); }

}  // namespace imp
