#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Fused RoPE on Q and K tensors in-place
void rope_forward(Tensor& Q, Tensor& K,
                  const int* positions, int head_dim,
                  float theta = 10000.0f, float scaling = 1.0f,
                  cudaStream_t stream = nullptr);

} // namespace imp
