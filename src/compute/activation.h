#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Fused SwiGLU: out = silu(gate) * up
void swiglu(const Tensor& gate, const Tensor& up, Tensor& out,
            cudaStream_t stream = nullptr);

void gelu(const Tensor& x, Tensor& out, cudaStream_t stream = nullptr);

} // namespace imp
