#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// MXFP4 tensor core attention for prefill on sm_120 (Blackwell).
//
// Uses CUTLASS block-scaled MXFP4×MXFP4 GEMM for Q·K^T, giving ~2x
// throughput over FP16 tensor cores on Blackwell's 5th-gen Tensor Cores.
// P·V uses standard cuBLAS FP16 GEMM.
//
// This is "chunked" attention — it materializes the full S = Q·K^T score
// matrix per head, so memory usage is O(seq²). For long sequences, flash
// attention (attention_blackwell.cu) is the better choice.
// The decode path remains scalar software-dequant (GEMV is memory-bound).
//
// Q:  [batch, seq_q, n_heads, head_dim]     FP16
// K:  [batch, seq_kv, n_kv_heads, head_dim] FP16
// V:  [batch, seq_kv, n_kv_heads, head_dim] FP16
// O:  [batch, seq_q, n_heads, head_dim]     FP16
//
// Requirements:
//   - sm_120+ (Blackwell block-scaled tensor cores)
//   - head_dim must be multiple of 32 (MXFP4 group size)
//   - IMP_USE_CUTLASS enabled at compile time
//   - IMP_MXFP4_ATTENTION=1 environment variable at runtime
//
// Returns false if the configuration is unsupported or GEMM fails.
bool attention_mxfp4_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, float softcap, cudaStream_t stream);

// Check if MXFP4 attention is available and enabled.
bool attention_mxfp4_available();

// Workspace estimate for VRAM budget planning.
size_t attention_mxfp4_workspace_estimate(int seq_q, int seq_kv, int head_dim);

} // namespace imp
