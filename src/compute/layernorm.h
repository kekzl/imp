#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Fused RMSNorm + residual add: out = norm(x + residual) * (weight + weight_offset)
// weight_offset: 0.0 for standard models, 1.0 for Gemma (which stores weights centered at 0)
void rmsnorm_residual(const Tensor& x, const Tensor& residual, const Tensor& weight, Tensor& out,
                      float eps = 1e-5f, cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// Simple RMSNorm: out = norm(x) * (weight + weight_offset)
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, float eps = 1e-5f,
             cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// RMSNorm with FP32 input and FP16 output. Used when the residual stream is
// kept in FP32 (Gemma-4 post-norm arch) but the downstream GEMM wants FP16.
// Avoids the FP32 → FP16 → RMSNorm rounding that would lose ~1-2% per layer.
void rmsnorm_fp32_to_fp16(const Tensor& x_fp32, const Tensor& weight, Tensor& out_fp16, float eps = 1e-5f,
                          cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// RMSNorm with FP32 input and FP32 output (FP16 weight). Used for Gemma-4 ggml
// MMVQ prefill to keep full FP32 precision through norm → Q8_1 quantization.
void rmsnorm_fp32_to_fp32(const Tensor& x_fp32, const Tensor& weight, float* out_fp32, int rows, int d_model,
                          float eps = 1e-5f, cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// Register layernorm kernels for PDL tail/head overlap.
void layernorm_pdl_register();

}  // namespace imp
