#pragma once
// MLA absorbed-decode latent KV cache (Phase 3, opt-in via attention.mla_absorb).
//
// Stage A (materialized) reconstructs full per-head K[rope+nope] / V[v_head_dim]
// from the compressed latent at projection time and runs standard paged
// attention. Phase 3 stores ONLY the compressed latent + decoupled RoPE key in a
// dedicated cache and runs the mathematically-equivalent "absorbed" attention
// at decode, where W_UK is folded into Q and W_UV into the output. The per-token
// cache footprint is (kv_lora_rank + qk_rope_head_dim) halfs vs the materialized
// n_heads*(qk+v head dims) — ~9x smaller for DeepSeek-V2-Lite.
//
// Both routines are correctness-first (one CUDA block per head, scalar dot
// products). They are NOT a fused single-kernel optimization.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Append the current step's RMSNorm'd latent + post-RoPE decoupled key into the
// per-layer latent cache.
//
//   latent:      [n_tokens, kv_lora_rank]            FP16  (mla_latent_buf_)
//   k_assembled: [n_tokens, n_heads, head_dim]       FP16  (post-RoPE K; the
//                decoupled RoPE key is replicated across heads MQA-style, so
//                head 0's first rope_dim dims are read)
//   cache:       [max_seq, kv_lora_rank + rope_dim]  FP16  (per-layer slice)
//   positions:   [n_tokens]                          INT32 device — absolute
//                token position == destination row in the cache.
//
void mla_latent_cache_write(const half* latent, const half* k_assembled, half* cache,
                            const int* positions, int n_tokens, int n_heads, int head_dim,
                            int rope_dim, int kv_lora_rank, int max_seq,
                            cudaStream_t stream = nullptr);

// Absorbed MLA decode attention for a single sequence (one query token).
//
//   q:            [n_heads, head_dim]                  FP16  per head [pe|nope]
//   kv_b:         [n_heads*(nope_dim+v_head_dim), kv_lora_rank] FP16 row-major
//                 (ly.kv_b_proj, a PyTorch Linear weight [out,in]).
//   cache:        [max_seq, kv_lora_rank + rope_dim]   FP16  (per-layer slice)
//   out:          [n_heads, v_head_dim]                FP16  (compact, written
//                 directly — matches the paged decode kernel output layout)
//   scores:       [n_heads, max_seq]                   FP32  scratch
//   context_lens: [1]                                  INT32 device — number of
//                 cached tokens (incl. the current one, already written).
//   scale:        attention logit scale (1/sqrt(qk_head_dim) * MLA mscale^2).
//
void mla_absorbed_decode(const half* q, const half* kv_b, const half* cache, half* out,
                         float* scores, const int* context_lens, int n_heads, int head_dim,
                         int rope_dim, int nope_dim, int kv_lora_rank, int v_head_dim,
                         int max_seq, float scale, cudaStream_t stream = nullptr);

}  // namespace imp
