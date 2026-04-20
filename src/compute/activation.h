#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Fused SwiGLU: out = silu(gate) * up
void swiglu(const Tensor& gate, const Tensor& up, Tensor& out,
            cudaStream_t stream = nullptr);

// Fused GeGLU: out = gelu_tanh(gate) * up  (Gemma-3)
void geglu(const Tensor& gate, const Tensor& up, Tensor& out,
           cudaStream_t stream = nullptr);

void gelu(const Tensor& x, Tensor& out, cudaStream_t stream = nullptr);

// Qwen3-Next / Qwen3.6 shared-expert sigmoid gate:
//   gate[r]    = sigmoid(sum_d x[r, d] * W[d])        // one scalar per row
//   y[r, j]   *= gate[r]                              // scale shared output in-place
// x: [n, d_model] FP16. W: [d_model] FP16 (GGUF stores F32 but weight_upload
// converts to FP16 on the way to GPU — same as other norm weights). y: [n, d]
// FP16 (mutated).
void shared_expert_gate_scale(const void* x_fp16, const void* W_fp16,
                               void* y_fp16_inout,
                               int n, int d_model, int d,
                               cudaStream_t stream = nullptr);

// Register activation kernels for PDL tail/head overlap.
void activation_pdl_register();

} // namespace imp
