#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Fused RMSNorm + residual add: out = norm(x + residual) * weight
void rmsnorm_residual(const Tensor& x, const Tensor& residual,
                      const Tensor& weight, Tensor& out,
                      float eps = 1e-5f,
                      cudaStream_t stream = nullptr);

// Simple RMSNorm: out = norm(x) * weight
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out,
             float eps = 1e-5f, cudaStream_t stream = nullptr);

} // namespace imp
