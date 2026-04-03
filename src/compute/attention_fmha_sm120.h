#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Native sm_120 FMHA using WGMMA (Warp Group MMA) for prefill attention.
//
// Uses wgmma.mma_async PTX instructions for ~2x tensor core throughput vs WMMA.
// Supports: FP16, causal masking, softcap, sliding window, GQA.
// Head dims: 64, 96, 128, 256. Falls back for unsupported configs.
//
// Q: [batch, seq_q, n_heads, head_dim]
// K,V: [batch, seq_kv, n_kv_heads, head_dim]
// O: [batch, seq_q, n_heads, head_dim]
//
// Returns true on success, false if config unsupported (caller falls back).
bool fmha_sm120_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap,
    cudaStream_t stream);

} // namespace imp
