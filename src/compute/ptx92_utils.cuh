#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstring>

namespace imp {

// Packed FP8 E4M3 conversion (sm_120+, Blackwell). cvt.e4m3x2 converts 2 elements/
// instruction (vs 1), halving instruction count in FP8 KV cache quant/dequant hot paths.

// Convert 2 packed FP16 (f16x2, 32-bit) → 2 packed FP8 E4M3 (e4m3x2, 16-bit).
// Applies round-to-nearest-even with saturation-to-finite (no inf/nan output).
__device__ __forceinline__ uint16_t cvt_f16x2_to_e4m3x2(uint32_t f16x2) {
    uint16_t result;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(result) : "r"(f16x2));
    return result;
}

// Convert 2 packed FP8 E4M3 (e4m3x2, 16-bit) → 2 packed FP16 (f16x2, 32-bit).
__device__ __forceinline__ uint32_t cvt_e4m3x2_to_f16x2(uint16_t e4m3x2) {
    uint32_t result;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(e4m3x2));
    return result;
}

// Convenience: convert 4 packed FP8 (uint32_t) → 4 floats via 2 paired cvt ops.
__device__ __forceinline__ void cvt_e4m3x4_to_f32x4(uint32_t packed_fp8, float& f0, float& f1, float& f2,
                                                    float& f3) {
    uint32_t f16x2_lo = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(packed_fp8 & 0xFFFF));
    uint32_t f16x2_hi = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>((packed_fp8 >> 16) & 0xFFFF));
    const half2* h2_lo = reinterpret_cast<const half2*>(&f16x2_lo);
    const half2* h2_hi = reinterpret_cast<const half2*>(&f16x2_hi);
    f0 = __half2float(h2_lo->x);
    f1 = __half2float(h2_lo->y);
    f2 = __half2float(h2_hi->x);
    f3 = __half2float(h2_hi->y);
}

// Packed FP4 E2M1 conversion. NOTE: FP16->FP4 packed conversion
// (cvt.rn.satfinite.e2m1x2.f16x2) is in PTX ISA 9.2 but ptxas rejects it on CUDA 13.2.
// Production paths use the FP32 variant (cvt.rn.satfinite.e2m1x2.f32, see
// src/quant/nvfp4_quant.cu:148); FP16x2 variant unwired, re-evaluate on next toolkit bump.

// Blackwell add.f32x2 PTX (sm_120a): 2-lane FP32 add. ptxas on consumer Blackwell
// decomposes this to 2x scalar FADD at SASS (vectorized HW path not exposed on sm_120);
// change is structural/forward-compat, not a perf delta.
// Bit-cast via uint64_t is the only register form ptxas accepts (verified against CUDA
// 13.2.1); natural {%0,%1},{%2,%3} form is rejected.
// Folds 4 floats as (a0+b0)+(a1+b1), not left-fold: FP add non-associative, so callers
// may see <=1-ULP FP32 difference on rounding-boundary inputs (sub-ULP after FP16 cast).
__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    uint64_t ar, br, sr;
    memcpy(&ar, &a, sizeof(float2));
    memcpy(&br, &b, sizeof(float2));
    asm("add.f32x2 %0, %1, %2;" : "=l"(sr) : "l"(ar), "l"(br));
    float2 s;
    memcpy(&s, &sr, sizeof(float2));
    return s;
}

}  // namespace imp
