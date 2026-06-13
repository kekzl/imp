#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// WMMA Flash Attention 2 for Blackwell (sm_120) with 128x64 tiles and a
// double-buffered KV pipeline.
// Q: [batch, seq_q, n_heads, head_dim]
// K,V: [batch, seq_kv, n_kv_heads, head_dim]
// O: [batch, seq_q, n_heads, head_dim]
// sliding_window: 0 = disabled, >0 = only attend to last N KV positions.
// Returns false (launches nothing) when no template fits the config —
// hd ∉ {64,96,128,256}, or smem over the device opt-in (hd=256 needs ~176 KB
// at Br=64 vs 99 KB on sm_120) — or when the launch itself errors. Callers
// must handle the decline; the old silent tc fallback was #654.
bool flash_attention_blackwell(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                               bool causal = true, int sliding_window = 0, float softcap = 0.0f,
                               cudaStream_t stream = nullptr, int q_offset = 0);

}  // namespace imp
