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
//   V_out:  [n_tokens, n_heads, v_dst_head_dim]        FP16
//
// v_dst_head_dim controls the per-head destination stride of V_out:
//   0 (default): compact — V_out is [n_tokens, n_heads, v_head_dim].
//   > v_head_dim: padded — V_out is [n_tokens, n_heads, v_dst_head_dim] with the
//     real v_head_dim values first and the (v_dst_head_dim - v_head_dim) tail
//     elements zeroed. Used by the MLA prefill/decode path so V shares K's
//     head_dim layout (over-allocation): downstream attention kernels can read
//     V at head_dim stride; the zero tail contributes nothing to the P·V sum.
//
void mla_assemble_kv(const half* kv_b, const half* k_rope,
                     half* K_out, half* V_out,
                     int n_tokens, int n_heads,
                     int nope_dim, int v_head_dim, int rope_dim,
                     cudaStream_t stream = nullptr,
                     int v_dst_head_dim = 0);

// Reorder Q in-place: [n_tokens, n_heads, nope_dim+rope_dim]
// HF layout: [nope_dim | rope_dim] -> imp layout: [rope_dim | nope_dim]
//
void mla_reorder_q(half* q_data, int n_tokens, int n_heads,
                   int nope_dim, int rope_dim,
                   cudaStream_t stream = nullptr);

// Compact per-head MLA attention output from head_dim-strided to v_head_dim.
//
// The prefill attention kernels (cuBLAS/FA2/FMHA) accumulate V at K's head_dim
// because the materialized V is over-allocated/zero-padded to head_dim. Their
// output is therefore [n_tokens, n_heads, head_dim] with the real value in the
// first v_head_dim dims of each head and zeros in the tail. This compacts it to
// [n_tokens, n_heads, v_head_dim] so the downstream o_proj sees the correct
// n_heads*v_head_dim input width.
//
//   src: [n_tokens, n_heads, head_dim]   (head_dim-strided per head)
//   dst: [n_tokens, n_heads, v_head_dim] (compact)
// src and dst MUST NOT alias (use a separate buffer).
void mla_compact_attn_output(const half* src, half* dst,
                             int n_tokens, int n_heads,
                             int head_dim, int v_head_dim,
                             cudaStream_t stream = nullptr);

}  // namespace imp
