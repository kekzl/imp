#pragma once
// MLA absorbed-decode latent KV cache (Phase 3, opt-in via attention.mla_absorb). Stage A
// (materialized) reconstructs full per-head K[rope+nope]/V[v_head_dim] from the compressed
// latent at projection time and runs standard paged attention. Phase 3 stores ONLY the
// compressed latent + decoupled RoPE key and runs the mathematically-equivalent "absorbed"
// attention at decode, where W_UK folds into Q and W_UV into the output: per-token footprint
// is (kv_lora_rank+qk_rope_head_dim) halfs vs the materialized n_heads*(qk+v head dims).
// Both routines are correctness-first (one CUDA block per head, scalar dot products), not a
// fused single-kernel optimization.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Appends the current step's RMSNorm'd latent + post-RoPE decoupled key into the per-layer
// latent cache.
//   latent: [n_tokens,kv_lora_rank] FP16 (mla_latent_buf_)
//   k_assembled: [n_tokens,n_heads,head_dim] FP16 (post-RoPE K; the decoupled RoPE key is
//   replicated across heads MQA-style, so head 0's first rope_dim dims are read)
//   cache: [max_seq,kv_lora_rank+rope_dim] FP16 (per-layer slice)
//   positions: [n_tokens] INT32 device, absolute token position = destination row in cache
void mla_latent_cache_write(const half* latent, const half* k_assembled, half* cache,
                            const int* positions, int n_tokens, int n_heads, int head_dim,
                            int rope_dim, int kv_lora_rank, int max_seq,
                            cudaStream_t stream = nullptr);

// Absorbed MLA decode attention for a single sequence (one query token).
//   q: [n_heads,head_dim] FP16 per head [pe|nope]
//   kv_b: [n_heads*(nope_dim+v_head_dim),kv_lora_rank] FP16 row-major (ly.kv_b_proj)
//   cache: [max_seq,kv_lora_rank+rope_dim] FP16 (per-layer slice)
//   out: [n_heads,v_head_dim] FP16 (compact, matches the paged decode kernel output layout)
//   scores: [n_heads,max_seq] FP32 scratch
//   context_lens: [1] INT32 device, cached tokens incl. the current one
//   scale: attention logit scale (1/sqrt(qk_head_dim) * MLA mscale^2)
// Arguments in the old parameter order (audit D-11).
struct MlaAbsorbedDecodeArgs {
    const half* q = nullptr;
    const half* kv_b = nullptr;
    const half* cache = nullptr;
    half* out = nullptr;
    float* scores = nullptr;
    const int* context_lens = nullptr;
    int n_heads = 0;
    int head_dim = 0;
    int rope_dim = 0;
    int nope_dim = 0;
    int kv_lora_rank = 0;
    int v_head_dim = 0;
    int max_seq = 0;
    float scale = 0.0f;
    cudaStream_t stream = nullptr;
};
void mla_absorbed_decode(const MlaAbsorbedDecodeArgs& a);

}  // namespace imp
