#pragma once
// MLA KV-buffer assembly kernels (Task 2.3): two scatter operations used by the MLA
// materialized-KV projection: mla_assemble_kv (scatter kv_b+k_rope into K[pe|nope] and V),
// mla_reorder_q (reorder Q from HF [nope|pe] to imp [pe|nope]). RoPE layout choice (b): pe
// FIRST in each K/Q head so the existing rope kernel (rotating the first rope_dim dims)
// applies unchanged to both.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Scatters kv_b + k_rope into K and V buffers.
//   kv_b: [n_tokens,n_heads*(nope_dim+v_head_dim)] FP16; k_rope: [n_tokens,rope_dim] FP16
//   (shared MQA-style); K_out: [n_tokens,n_heads,rope_dim+nope_dim] FP16 [pe|nope];
//   V_out: [n_tokens,n_heads,v_dst_head_dim] FP16
// v_dst_head_dim: 0 (default) = compact V_out [n_tokens,n_heads,v_head_dim]; > v_head_dim =
// padded, real values first and the tail zeroed, so V shares K's head_dim stride for
// downstream attention kernels (the zero tail contributes nothing to P.V).
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

// Compacts per-head MLA attention output from head_dim-strided to v_head_dim. Prefill
// attention kernels (cuBLAS/FA2/FMHA) accumulate V at K's head_dim because materialized V is
// over-allocated/zero-padded to head_dim, so their output is [n_tokens,n_heads,head_dim] with
// the real value in the first v_head_dim dims and zeros in the tail. This compacts it to
// [n_tokens,n_heads,v_head_dim] so o_proj sees the correct n_heads*v_head_dim width.
//   src: [n_tokens,n_heads,head_dim]; dst: [n_tokens,n_heads,v_head_dim]; must not alias.
void mla_compact_attn_output(const half* src, half* dst,
                             int n_tokens, int n_heads,
                             int head_dim, int v_head_dim,
                             cudaStream_t stream = nullptr);

}  // namespace imp
