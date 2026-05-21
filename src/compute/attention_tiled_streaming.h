#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Hand-written FA2-style tiled streaming attention for sm_120a.
// 1 producer + 7 consumer warps. FP16 KV + NVFP4 KV via runtime dispatch.
// Returns true on success, false if config unsupported (caller falls back).
//
// Q:    [batch, seq_q, n_heads, head_dim]            FP16
// K, V: [batch, seq_kv, n_kv_heads, head_dim]        FP16 or NVFP4 (K.scales set)
// O:    [batch, seq_q, n_heads, head_dim]            FP16
//
// q_offset: absolute position of Q[0] (for chunked prefill causal alignment).
bool attention_tiled_streaming_prefill(const Tensor& Q, const Tensor& K,
                                       const Tensor& V, Tensor& O, float scale,
                                       bool causal, int sliding_window,
                                       float softcap, int q_offset,
                                       cudaStream_t stream);

}  // namespace imp
