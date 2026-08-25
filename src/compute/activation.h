#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Fused SwiGLU: out = silu(gate) * up
void swiglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream = nullptr);

// Batched-decode producer fusion: swiglu + NVFP4 activation quantize in one
// kernel. Writes the same FP16 `out` as swiglu() (bit-identical) plus the
// packed nibbles [n/2 bytes] and FP8 micro-scales [n/16] the small-M NVFP4
// GEMM reads (plain layout, tensor_scale 1.0 — bit-identical to
// quantize_fp16_to_nvfp4_into on the stored FP16). Returns false outside the
// fused envelope (all F16, numel % 16 == 0); caller falls back to swiglu().
bool swiglu_quantize_nvfp4(const Tensor& gate, const Tensor& up, Tensor& out, uint8_t* xq_packed,
                           uint8_t* xq_scales, cudaStream_t stream = nullptr);

// Fused GeGLU: out = gelu_tanh(gate) * up  (Gemma-3)
void geglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream = nullptr);

// gpt-oss clamped GLU (issue #547), HF GptOssExperts semantics:
//   gate_c = min(gate, 7);  up_c = clamp(up, -7, 7)
//   out = (up_c + 1) * gate_c * sigmoid(1.702 * gate_c)
void gpt_oss_glu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream = nullptr);

void gelu(const Tensor& x, Tensor& out, cudaStream_t stream = nullptr);

// Qwen3-Next / Qwen3.6 shared-expert sigmoid gate:
//   gate[r]    = sigmoid(sum_d x[r, d] * W[d])        // one scalar per row
//   y[r, j]   *= gate[r]                              // scale shared output in-place
// x: [n, d_model] FP16. W: [d_model] FP16 (GGUF stores F32 but weight_upload
// converts to FP16 on the way to GPU — same as other norm weights). y: [n, d]
// FP16 (mutated).
void shared_expert_gate_scale(const void* x_fp16, const void* W_fp16, void* y_fp16_inout, int n, int d_model,
                              int d, cudaStream_t stream = nullptr);

// Register activation kernels for PDL tail/head overlap.
void activation_pdl_register();

}  // namespace imp
