#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Fused RoPE on Q and K tensors in-place
// rope_dim: if > 0, only rotate the first rope_dim dimensions; rest unchanged.
//           if 0, rotate all head_dim dimensions (default).
void rope_forward(Tensor& Q, Tensor& K,
                  const int* positions, int head_dim,
                  float theta = 10000.0f, float scaling = 1.0f,
                  int rope_dim = 0,
                  cudaStream_t stream = nullptr);

} // namespace imp
