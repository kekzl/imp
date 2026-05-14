#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Prefill attention via cuBLAS materialized QK^T + softmax + PV.
//
// Q: [q_len, n_heads * head_dim] FP16
// K: [kv_len, n_kv_heads * head_dim] FP16
// V: [kv_len, n_kv_heads * head_dim] FP16
// O: [q_len, n_heads * head_dim] FP16
// S: workspace, sized for [n_heads * q_len * kv_len] in FP16 or FP32 (FP32 picked when buffer fits).
//
// q_offset is the absolute position of Q[0] in the full sequence. When causal=true,
// Q[i] (abs pos = q_offset + i) is masked against K[j] for j > q_offset + i.
// q_offset=0 reproduces the historic square path exactly.
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap = 0.0f, int q_offset = 0, cudaStream_t stream = nullptr);

// Force-create the static cuBLAS handle. Safe to call multiple times.
// Engine init calls this so the first attention_cublas_prefill invocation
// inside a captured stream can reuse the handle without cublasCreate
// (which does internal cudaMalloc for workspace and is illegal under capture).
void attention_cublas_prewarm();

}  // namespace imp
