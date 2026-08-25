#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Fused RMSNorm + residual add: out = norm(x + residual) * (weight + weight_offset)
// weight_offset: 0.0 for standard models, 1.0 for Gemma (which stores weights centered at 0)
void rmsnorm_residual(const Tensor& x, const Tensor& residual, const Tensor& weight, Tensor& out,
                      float eps = 1e-5f, cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// Simple RMSNorm: out = norm(x) * (weight + weight_offset)
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, float eps = 1e-5f,
             cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// Batched-decode producer fusion: rmsnorm + NVFP4 activation quantize in one
// kernel. Writes the same FP16 `out` as rmsnorm() (bit-identical) plus the
// packed nibbles [rows, d/2] and FP8 micro-scales [rows, d/16] the small-M
// NVFP4 GEMM reads (plain layout, tensor_scale 1.0 — bit-identical to
// quantize_fp16_to_nvfp4_into on the stored FP16). Returns false when the
// shape is outside the fused envelope (F16, rows 2..64, d % 256 == 0,
// d <= 8192); the caller must then fall back to rmsnorm().
bool rmsnorm_nvfp4(const Tensor& x, const Tensor& weight, Tensor& out, uint8_t* xq_packed,
                   uint8_t* xq_scales, float eps = 1e-5f, cudaStream_t stream = nullptr,
                   float weight_offset = 0.0f);

// RMSNorm with FP32 input and FP16 output. Used when the residual stream is
// kept in FP32 (Gemma-4 post-norm arch) but the downstream GEMM wants FP16.
// Avoids the FP32 → FP16 → RMSNorm rounding that would lose ~1-2% per layer.
void rmsnorm_fp32_to_fp16(const Tensor& x_fp32, const Tensor& weight, Tensor& out_fp16, float eps = 1e-5f,
                          cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// RMSNorm with FP32 input and FP32 output (FP16 weight). Used for Gemma-4 ggml
// MMVQ prefill to keep full FP32 precision through norm → Q8_1 quantization.
void rmsnorm_fp32_to_fp32(const Tensor& x_fp32, const Tensor& weight, float* out_fp32, int rows, int d_model,
                          float eps = 1e-5f, cudaStream_t stream = nullptr, float weight_offset = 0.0f);

// True LayerNorm with residual add (#836, encoder post-LN):
//   out = ((x + residual) - mean) / sqrt(var + eps) * weight + bias
// x/residual/out: [rows, d_model] FP16; weight/bias: [d_model] F32 or F16.
// residual may be empty (Tensor{}) for the post-embedding norm.
void layernorm_residual(const Tensor& x, const Tensor& residual, const Tensor& weight,
                        const Tensor& bias, Tensor& out, float eps = 1e-12f,
                        cudaStream_t stream = nullptr);

// Register layernorm kernels for PDL tail/head overlap.
void layernorm_pdl_register();

}  // namespace imp
