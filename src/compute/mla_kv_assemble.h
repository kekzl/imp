#pragma once
// MLA KV-buffer assembly kernels (Task 2.3).
//
// Two scatter operations used by the MLA materialized-KV projection:
//   mla_assemble_kv — scatter kv_b + k_rope into K[pe|nope] and V
//   mla_reorder_q   — reorder Q from HF [nope|pe] to imp [pe|nope]
//
// RoPE layout choice (b): pe FIRST in each K (and Q) head so that the
// existing rope kernel (which rotates the first rope_dim dims) applies
// unchanged to both Q and K.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Scatter kv_b + k_rope into K and V buffers.
//
//   kv_b:   [n_tokens, n_heads*(nope_dim+v_head_dim)]  FP16
//   k_rope: [n_tokens, rope_dim]                       FP16  (shared MQA-style)
//   K_out:  [n_tokens, n_heads, rope_dim+nope_dim]     FP16  layout [pe|nope]
//   V_out:  [n_tokens, n_heads, v_head_dim]            FP16
//
void mla_assemble_kv(const half* kv_b, const half* k_rope,
                     half* K_out, half* V_out,
                     int n_tokens, int n_heads,
                     int nope_dim, int v_head_dim, int rope_dim,
                     cudaStream_t stream = nullptr);

// Reorder Q in-place: [n_tokens, n_heads, nope_dim+rope_dim]
// HF layout: [nope_dim | rope_dim] -> imp layout: [rope_dim | nope_dim]
//
void mla_reorder_q(half* q_data, int n_tokens, int n_heads,
                   int nope_dim, int rope_dim,
                   cudaStream_t stream = nullptr);

}  // namespace imp
