#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Flash Attention prefill: Q,K,V -> O
// Q: [batch, seq_q, n_heads, head_dim]
// K,V: [batch, seq_kv, n_kv_heads, head_dim]
// O: [batch, seq_q, n_heads, head_dim]
void flash_attention_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal = true,
    cudaStream_t stream = nullptr);

// Runtime-dispatched attention prefill.
// Selects the best kernel based on compute capability:
//   sm_120+ (Blackwell)  -> TCGEN05 systolic attention (attention_blackwell.cu)
//   sm_90+  (Hopper)     -> WMMA tensor-core attention (attention_tc.cu)
//   <sm_90               -> Scalar Flash Attention 2 (attention.cu)
void attention_prefill_dispatch(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal = true,
    cudaStream_t stream = nullptr);

// Query the compute capability of the current device.
int get_device_sm_version();

} // namespace imp
