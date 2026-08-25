#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "quant/fp8_utils.cuh"

namespace imp {

// ---------------------------------------------------------------------------
// NVFP4 (FP4 E2M1) micro-block packing device helpers.
//
// Moved VERBATIM from nvfp4_quant.cu so producer-side kernels (fused
// RMSNorm+quantize, fused SwiGLU+quantize) share the exact arithmetic with
// quantize_fp16_to_nvfp4_into(): identical inputs must yield identical
// packed bytes and micro-scales.
//
// Packed format: 2 FP4 values per byte.
//   Low nibble  (bits 0-3) = even-indexed element
//   High nibble (bits 4-7) = odd-indexed element
// ---------------------------------------------------------------------------

static constexpr int kNvfp4MicroBlockSize = 16;   // micro-block: 16 values
static constexpr float kNvfp4FP4E2M1Max = 6.0f;   // max representable in FP4 E2M1
static constexpr float kNvfp4FP8E4M3Max = 448.0f;

// ---------------------------------------------------------------------------
// Device helper: quantize a single FP32 magnitude to FP4 E2M1 (3-bit code)
// Uses round-to-nearest-even among the 8 representable magnitudes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t float_abs_to_fp4_e2m1(float abs_val) {
    // Branchless: count of midpoint thresholds exceeded gives the E2M1 code.
    // Thresholds between adjacent representable values:
    //   0    0.5    1.0    1.5    2.0    3.0    4.0    6.0
    //     0.25  0.75  1.25  1.75  2.5   3.5    5.0
    uint8_t code = (abs_val >= 0.25f) + (abs_val >= 0.75f) + (abs_val >= 1.25f) + (abs_val >= 1.75f) +
                   (abs_val >= 2.5f) + (abs_val >= 3.5f) + (abs_val >= 5.0f);
    return code;  // 0..7
}

// HW FP32 pair → packed E2M1 byte (low = v0, high = v1). IEEE RNE rounding,
// saturates to ±6. Single PTX instruction on sm_120+.
__device__ __forceinline__ uint8_t nvfp4_pack_pair_hw(float v0, float v1) {
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
    uint8_t sign0 = (v0 < 0.0f) ? 1u : 0u;
    uint8_t sign1 = (v1 < 0.0f) ? 1u : 0u;
    uint8_t c0 = (sign0 << 3) | float_abs_to_fp4_e2m1(fabsf(v0));
    uint8_t c1 = (sign1 << 3) | float_abs_to_fp4_e2m1(fabsf(v1));
    return (c1 << 4) | c0;
#endif
}

// ---------------------------------------------------------------------------
// Micro-scale encode for one 16-value micro-block: local absmax → clamped
// FP8 E4M3 micro-scale. Returns the RECONSTRUCTED scale (the value the
// dequant will use), which is what the pack step must divide by. Extracted
// verbatim from quantize_micro_block_nvfp4 (steps 2 of nvfp4_quant.cu).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float nvfp4_encode_micro_scale(float local_absmax, float tensor_scale,
                                                          uint8_t* fp8_out) {
    float micro_scale_f = local_absmax / (tensor_scale * kNvfp4FP4E2M1Max);
    if (micro_scale_f < 1.0f / 512.0f)
        micro_scale_f = 1.0f / 512.0f;  // FP8 E4M3 min subnormal
    if (micro_scale_f > kNvfp4FP8E4M3Max)
        micro_scale_f = kNvfp4FP8E4M3Max;

    uint8_t micro_scale_fp8 = float_to_fp8_e4m3(micro_scale_f);
    *fp8_out = micro_scale_fp8;

    float micro_scale_actual = fp8_e4m3_to_float(micro_scale_fp8);
    if (micro_scale_actual == 0.0f)
        micro_scale_actual = 1.0f / 512.0f;
    return micro_scale_actual;
}

}  // namespace imp
