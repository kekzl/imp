#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Flash Attention 2 using WMMA (Warp Matrix Multiply Accumulate) tensor cores.
// Requires sm_90+ (Hopper/Blackwell). Falls back to scalar on older GPUs.
// Q: [batch, seq_q, n_heads, head_dim]
// K,V: [batch, seq_kv, n_kv_heads, head_dim]
// O: [batch, seq_q, n_heads, head_dim]
// head_dim must be 64 or 128 (multiples of 16 for WMMA).
// sliding_window: 0 = disabled, >0 = only attend to last N KV positions
void flash_attention_prefill_tc(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal = true, int sliding_window = 0,
    cudaStream_t stream = nullptr);

// Check if tensor-core attention is available on current device.
bool tc_attention_available();

// Optimized WMMA attention for Blackwell (sm_120+) with 128x64 tiles.
// Same interface as flash_attention_prefill_tc but with larger tiles and
// double-buffered KV pipeline for improved compute-to-memory ratio.
void flash_attention_blackwell(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal = true, int sliding_window = 0,
    cudaStream_t stream = nullptr);

} // namespace imp
