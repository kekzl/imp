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

// Register activation kernels for PDL tail/head overlap.
void activation_pdl_register();

} // namespace imp
