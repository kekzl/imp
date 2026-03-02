#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// CUTLASS Hopper FMHA for prefill attention (sm_90+).
// Uses WGMMA + TMA for ~2x throughput vs WMMA kernels.
//
// Q: [batch, seq_q, n_heads, head_dim]
// K,V: [batch, seq_kv, n_kv_heads, head_dim]
// O: [batch, seq_q, n_heads, head_dim]
//
// Returns true on success, false if configuration is unsupported
// (unsupported head_dim, smem too large, etc.) — caller should fall back.
bool cutlass_fmha_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, cudaStream_t stream);

// Pre-allocate FMHA workspace (LSE buffer + kernel workspace) at max dimensions.
// Call during engine init so these allocations are tracked in the VRAM budget.
// Returns total bytes allocated.
size_t cutlass_fmha_init_workspace(int max_batch, int max_seq, int n_heads, int head_dim);

// Estimate VRAM needed for FMHA workspace (for budget calculation before allocation).
size_t cutlass_fmha_workspace_estimate(int max_batch, int max_seq, int n_heads, int head_dim);

// Free pre-allocated workspace. Call during engine shutdown.
void cutlass_fmha_free_workspace();

} // namespace imp
