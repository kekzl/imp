#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Small-M FP8 E4M3 x E4M3 GEMM for batched decode: y[M,N] fp16 = as(m) * *w_scale * A W^T + beta * y with
// as(m) = a_scale[m] when a_per_row (per-token dynamic scales, M floats), else *a_scale.
// A [M,K] / W [N,K] row-major E4M3, device scales (graph-capturable). M 1..32, K % 64 == 0,
// N % 16 == 0; refuses (false, y untouched) otherwise. A row's bits do not depend on M (fixed K order).
inline constexpr int kFp8SmallMMaxM = 32;
[[nodiscard]] bool gemm_fp8_smallm(const void* a_fp8, const void* w_fp8, const float* a_scale,
                                   const float* w_scale, float beta, half* y, int M, int N, int K,
                                   cudaStream_t stream, bool a_per_row = false);

// 2-3 weights on one activation (q|k|v, gate|up), beta 0, one launch; each tensor's bits equal
// gemm_fp8_smallm's. Same refusals per tensor.
inline constexpr int kFp8SmallMMaxSiblings = 3;
struct Fp8SmallMSibling {
    const void* w;
    const float* w_scale;
    half* y;
    int N;
};
[[nodiscard]] bool gemm_fp8_smallm_multi(const Fp8SmallMSibling* t, int count, const void* a_fp8,
                                         const float* a_scale, int M, int K, cudaStream_t stream,
                                         bool a_per_row = false);

// Bench/test hook: 8 warps per CTA, explicit k-steps per load round (1, 2, 4).
// Any other warps value is refused; production uses gemm_fp8_smallm (8 warps, 2 k-steps).
[[nodiscard]] bool gemm_fp8_smallm_tuned(const void* a_fp8, const void* w_fp8, const float* a_scale,
                                         const float* w_scale, float beta, half* y, int M, int N, int K,
                                         cudaStream_t stream, int warps, int unroll, bool a_per_row = false);

}  // namespace imp
