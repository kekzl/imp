#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Gated DeltaNet decode (single token, n=1).
// Updates recurrent state S and computes output y.
//
// State S[n_kv_heads, head_dim, head_dim] — one matrix per KV head.
// Alpha/beta have n_alpha_heads values, mapped to n_kv_heads by averaging.
//
// q:      [n_q_heads * head_dim] FP16 (GQA: q_per_kv = n_q_heads / n_kv_heads)
// k:      [n_kv_heads * head_dim] FP16
// v:      [n_kv_heads * head_dim] FP16
// alpha:  [n_alpha_heads] FP16 (decay gate, sigmoid applied internally)
// beta:   [n_alpha_heads] FP16 (learning rate, sigmoid applied internally)
// s_state:[n_kv_heads, head_dim, head_dim] FP32 (updated in-place)
// y:      [n_q_heads * head_dim] FP16 (output)
// gate:   [n_q_heads * head_dim] FP16 (sigmoid gate, nullptr = no gating)
void gdn_decode(const half* q, const half* k, const half* v,
                const half* alpha, const half* beta,
                float* s_state, half* y, const half* gate,
                int n_q_heads, int n_kv_heads, int head_dim, int n_alpha_heads,
                cudaStream_t stream);

// Gated DeltaNet prefill (sequential per-token processing).
void gdn_prefill(const half* q, const half* k, const half* v,
                 const half* alpha, const half* beta,
                 float* s_state, half* y, const half* gate,
                 int n_tokens, int n_q_heads, int n_kv_heads,
                 int head_dim, int n_alpha_heads,
                 cudaStream_t stream);

} // namespace imp
