#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Native Q4_K_M GEMV: computes y = x @ W^T where W is in Q4_K_M format.
// x: [M, K] FP16 activations
// w_raw: [N, K/256*144] raw Q4_K bytes on GPU
// y: [M, N] FP16 output
// All intermediate computation in FP32 (no FP16 weight rounding).
void gemv_q4k_native(
    const half* x,
    const uint8_t* w_raw,
    half* y,
    int M, int N, int K,
    cudaStream_t stream);

} // namespace imp
