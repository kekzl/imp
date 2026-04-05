#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// Packed FP8 E4M3 conversion helpers (sm_120+, Blackwell)
//
// cvt.e4m3x2 converts 2 elements per instruction instead of 1, halving
// instruction count in the FP8 KV cache quantize/dequantize hot paths.
//
// On sm_90a/sm_100 the scalar fallback uses __nv_fp8_e4m3 (hardware cvt
// per element). The packed variant saves one instruction per pair on sm_120+.
// ---------------------------------------------------------------------------

// Convert 2 packed FP16 (f16x2, 32-bit) → 2 packed FP8 E4M3 (e4m3x2, 16-bit).
// Applies round-to-nearest-even with saturation-to-finite (no inf/nan output).
__device__ __forceinline__ uint16_t cvt_f16x2_to_e4m3x2(uint32_t f16x2) {
#if __CUDA_ARCH__ >= 1200
    uint16_t result;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(result) : "r"(f16x2));
    return result;
#else
    const half* hp = reinterpret_cast<const half*>(&f16x2);
    uint8_t lo = static_cast<uint8_t>(__nv_fp8_e4m3(__half2float(hp[0])).__x);
    uint8_t hi = static_cast<uint8_t>(__nv_fp8_e4m3(__half2float(hp[1])).__x);
    return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
#endif
}

// Convert 2 packed FP8 E4M3 (e4m3x2, 16-bit) → 2 packed FP16 (f16x2, 32-bit).
__device__ __forceinline__ uint32_t cvt_e4m3x2_to_f16x2(uint16_t e4m3x2) {
#if __CUDA_ARCH__ >= 1200
    uint32_t result;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(e4m3x2));
    return result;
#else
    uint8_t lo = e4m3x2 & 0xFF;
    uint8_t hi = (e4m3x2 >> 8) & 0xFF;
    half2 result;
    __nv_fp8_e4m3 v0, v1;
    memcpy(&v0, &lo, 1);
    memcpy(&v1, &hi, 1);
    result.x = static_cast<half>(static_cast<float>(v0));
    result.y = static_cast<half>(static_cast<float>(v1));
    uint32_t r;
    memcpy(&r, &result, 4);
    return r;
#endif
}

// Convenience: convert 4 packed FP8 (uint32_t) → 4 floats via 2 paired cvt ops.
__device__ __forceinline__ void cvt_e4m3x4_to_f32x4(
    uint32_t packed_fp8, float& f0, float& f1, float& f2, float& f3)
{
    uint32_t f16x2_lo = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(packed_fp8 & 0xFFFF));
    uint32_t f16x2_hi = cvt_e4m3x2_to_f16x2(static_cast<uint16_t>((packed_fp8 >> 16) & 0xFFFF));
    const half2* h2_lo = reinterpret_cast<const half2*>(&f16x2_lo);
    const half2* h2_hi = reinterpret_cast<const half2*>(&f16x2_hi);
    f0 = __half2float(h2_lo->x);
    f1 = __half2float(h2_lo->y);
    f2 = __half2float(h2_hi->x);
    f3 = __half2float(h2_hi->y);
}

// ---------------------------------------------------------------------------
// Packed FP4 E2M1 conversion helpers
//
// NOTE: cvt.rn.satfinite.e2m1x2.f16x2 is documented in PTX ISA 9.2 but
// REJECTED by ptxas in CUDA 13.2.0. The PTX instruction would convert
// 2 FP16 values to 2 packed FP4 E2M1 nibbles in a single instruction.
// Once a future CUDA release supports it, enable the #if block below.
//
// Until then, the branchless scalar fallback in turboquant_fp4.cuh
// (tq_fp4_quantize_abs using comparison sums) provides the quantization.
// ---------------------------------------------------------------------------

#if 0  // BLOCKED: ptxas rejects in CUDA 13.2 — retry with CUDA 13.3+
__device__ __forceinline__ uint16_t cvt_f16x2_to_e2m1x2(uint32_t f16x2) {
    uint16_t result;
    asm("cvt.rn.satfinite.e2m1x2.f16x2 %0, %1;" : "=h"(result) : "r"(f16x2));
    return result;
}
#endif

} // namespace imp
