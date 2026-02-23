#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// cuBLAS GEMM wrapper: C = alpha * A @ B^T + beta * C
// A [M, K]  B [N, K]  C [M, N]   -- all row-major
void gemm(const Tensor& A, const Tensor& B, Tensor& C,
          float alpha = 1.0f, float beta = 0.0f,
          cudaStream_t stream = nullptr);

// cuBLASLt GEMM with explicit algorithm selection and FP8 scale support.
// aScale/bScale are optional per-tensor FP32 scales for FP8 operands.
void gemm_cublaslt(const Tensor& A, const Tensor& B, Tensor& C,
                   float alpha = 1.0f, float beta = 0.0f,
                   const float* aScale = nullptr,
                   const float* bScale = nullptr,
                   cudaStream_t stream = nullptr);

// Small batch GEMV for batch_size 1-4
void gemv(const Tensor& A, const Tensor& x, Tensor& y,
          cudaStream_t stream = nullptr);

// FP8 E4M3 GEMV: y = A_fp8 @ x_fp16 (with per-tensor scale)
// A: [M, K] FP8_E4M3, x: [K] FP16, y: [M] FP16
void gemv_fp8(const Tensor& A, const Tensor& x, Tensor& y,
              float scale, cudaStream_t stream = nullptr);

} // namespace imp
