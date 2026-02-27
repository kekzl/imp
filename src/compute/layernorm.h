#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Fused RMSNorm + residual add: out = norm(x + residual) * (weight + weight_offset)
// weight_offset: 0.0 for standard models, 1.0 for Gemma (which stores weights centered at 0)
void rmsnorm_residual(const Tensor& x, const Tensor& residual,
                      const Tensor& weight, Tensor& out,
                      float eps = 1e-5f,
                      cudaStream_t stream = nullptr,
                      float weight_offset = 0.0f);

// Simple RMSNorm: out = norm(x) * (weight + weight_offset)
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out,
             float eps = 1e-5f, cudaStream_t stream = nullptr,
             float weight_offset = 0.0f);

} // namespace imp
