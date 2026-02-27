#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Fused RoPE on Q and K tensors in-place
// rope_dim: if > 0, only rotate the first rope_dim dimensions; rest unchanged.
//           if 0, rotate all head_dim dimensions (default).
// neox: true = NeoX/split style pairs (i, i+d/2), false = interleaved pairs (2i, 2i+1)
void rope_forward(Tensor& Q, Tensor& K,
                  const int* positions, int head_dim,
                  float theta = 10000.0f, float scaling = 1.0f,
                  int rope_dim = 0, bool neox = false,
                  cudaStream_t stream = nullptr);

// Fused QK-norm + RoPE for decode (n=1, FP16).
// Applies per-head RMSNorm on Q and K, then RoPE, in a single kernel launch.
// Q: [n_heads * head_dim], K: [n_kv_heads * head_dim].
// q_norm_weight, k_norm_weight: [head_dim] RMSNorm weights.
// positions: device pointer to [1] int (decode only, n=1).
void qknorm_rope_fused(half* Q, half* K,
                        const half* q_norm_weight, const half* k_norm_weight,
                        int n_heads, int n_kv_heads, int head_dim,
                        float eps, const int* positions,
                        float theta = 10000.0f, float scaling = 1.0f,
                        int rope_dim = 0, bool neox = false,
                        cudaStream_t stream = nullptr,
                        float weight_offset = 0.0f);

// Register RoPE kernels for PDL tail/head overlap.
void rope_pdl_register();

} // namespace imp
