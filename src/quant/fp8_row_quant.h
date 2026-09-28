#pragma once

// Per-row FP8 E4M3 weight quantization, one definition for the device kernel
// (quantize_fp8_rows_kernel) and host tests: scale = row_absmax / 448, q = RNE-satfinite(v / scale).

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#if defined(__CUDACC__)
#define IMP_FP8_ROW_HD __host__ __device__
#else
#define IMP_FP8_ROW_HD
#endif

namespace imp::fp8_rows {

inline constexpr float kE4M3Max = 448.0f;

// All-zero row: scale 1 (every code stays 0).
IMP_FP8_ROW_HD inline float row_scale(float row_absmax) {
    return row_absmax > 0.0f ? row_absmax / kE4M3Max : 1.0f;
}

// E4M3 round-to-nearest-even, saturating at +-448 (cuda_fp8.h __NV_SATFINITE).
IMP_FP8_ROW_HD inline uint8_t encode(float v, float inv_scale) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3));
}

IMP_FP8_ROW_HD inline float decode(uint8_t code) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3)));
}

}  // namespace imp::fp8_rows
