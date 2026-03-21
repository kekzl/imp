#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// Fused multi-token GDN scan.
// Processes all tokens in a SINGLE kernel launch with register-cached state.
// conv_f32: [n_tokens, conv_channels] FP32 — full conv+SiLU output per token
//           layout per token: [Q(BC_size), K(BC_size), V(inner)]
// ---------------------------------------------------------------------------
void gdn_scan_fused_f32(const float* conv_f32, int conv_channels,
                         const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias,
                         float* h_state, half* y,
                         int n_tokens, int n_heads, int head_dim_ssm,
                         int state_size, int n_groups,
                         cudaStream_t stream);

// Fused RMSNormGated + SiLU: y = rmsnorm(y) * silu(gate)
// Processes all tokens × heads in one launch.
void gdn_rmsnorm_gated_silu(half* y, const half* gate, const half* weight,
                              float eps, int n_tokens, int n_heads, int head_dim,
                              cudaStream_t stream);

// ---------------------------------------------------------------------------
// Legacy per-token interfaces (kept for fallback / testing)
// ---------------------------------------------------------------------------
void gdn_scan_decode_f32(const float* x, const float* B, const float* C,
                         const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias,
                         float* h_state, half* y, const half* z,
                         int n_heads, int head_dim_ssm,
                         int state_size, int n_groups,
                         cudaStream_t stream);

void gdn_scan_prefill_f32(const float* x, const float* B, const float* C,
                          const half* alpha, const half* beta,
                          const float* A_log, const float* dt_bias,
                          float* h_state, half* y, const half* z,
                          int n_tokens, int n_heads, int head_dim_ssm,
                          int state_size, int n_groups,
                          cudaStream_t stream);

// V-head reorder: tiled → grouped
void vhead_tiled_to_grouped(const half* src, half* dst,
                             int n_tokens, int n_heads, int head_dim, int n_groups,
                             cudaStream_t stream);

// Legacy stubs
void gdn_scan_decode(const half*, const half*, const half*,
                     const half*, const half*, const float*, const float*,
                     float*, half*, const half*,
                     int, int, int, int, cudaStream_t);
void gdn_scan_prefill(const half*, const half*, const half*,
                      const half*, const half*, const float*, const float*,
                      float*, half*, const half*,
                      int, int, int, int, int, cudaStream_t);
void gdn_decode(const half*, const half*, const half*,
                const half*, const half*, float*, half*, const half*,
                int, int, int, int, cudaStream_t);
void gdn_prefill(const half*, const half*, const half*,
                 const half*, const half*, float*, half*, const half*,
                 int, int, int, int, int, cudaStream_t);

} // namespace imp
