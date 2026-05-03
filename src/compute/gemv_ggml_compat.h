#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// ggml-compatible Q4_K MoE GEMV: matches llama.cpp's accumulation order.
// W: [rows, K/256*144] raw Q4_K bytes on GPU
// x: [K] FP16 input vector
// y: [rows] FP16 output vector
void gemv_q4k_ggml_compat(const uint8_t* W, const half* x, half* y, int rows, int K, cudaStream_t stream);

}  // namespace imp
