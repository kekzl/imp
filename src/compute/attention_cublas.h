#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// cuBLAS batched-GEMM attention for prefill.
//
// Computes standard scaled dot-product attention using cuBLAS for the two
// matrix multiplications (S = Q*K^T, O = P*V), with a custom fused
// causal-mask + softmax kernel in between.
//
// This achieves much higher TC utilization than the hand-written WMMA flash
// attention kernel, because cuBLAS can tile and schedule the GEMMs optimally.
//
// Q:   [n_tokens, n_heads * head_dim]   FP16 contiguous
// K:   [n_tokens, n_kv_heads * head_dim] FP16 contiguous
// V:   [n_tokens, n_kv_heads * head_dim] FP16 contiguous
// O:   [n_tokens, n_heads * head_dim]   FP16 contiguous (output)
// S:   [n_heads, n_tokens, n_tokens]    FP16 workspace (caller-provided)
//
// n_heads, n_kv_heads, head_dim: model dimensions
// scale: 1/sqrt(head_dim)
// causal: apply causal mask
void attention_cublas_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    Tensor& O, Tensor& S,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, bool causal, float softcap = 0.0f,
    cudaStream_t stream = nullptr);

} // namespace imp
