#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Small-M FP8 E4M3 x E4M3 GEMM for batched decode: y[M,N] fp16 = *a_scale * *w_scale * A W^T + beta * y.
// A [M,K] / W [N,K] row-major E4M3, per-tensor device scales (graph-capturable). M 1..32, K % 64 == 0,
// N % 16 == 0; refuses (false, y untouched) otherwise. A row's bits do not depend on M (fixed K order).
inline constexpr int kFp8SmallMMaxM = 32;
[[nodiscard]] bool gemm_fp8_smallm(const void* a_fp8, const void* w_fp8, const float* a_scale,
                                   const float* w_scale, float beta, half* y, int M, int N, int K,
                                   cudaStream_t stream);

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
                                         const float* a_scale, int M, int K, cudaStream_t stream);

// Bench/test hook: explicit warps per CTA (4, 8) and k-steps per load round (2, 4). warps changes the
// K summation order; production uses gemm_fp8_smallm's fixed choice.
[[nodiscard]] bool gemm_fp8_smallm_tuned(const void* a_fp8, const void* w_fp8, const float* a_scale,
                                         const float* w_scale, float beta, half* y, int M, int N, int K,
                                         cudaStream_t stream, int warps, int unroll);

}  // namespace imp
