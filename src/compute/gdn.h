#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// GDN Delta Rule scan — decode (single token).
// Replaces ssm_scan_decode for GDN layers.
// Same parameter layout but uses delta rule instead of selective scan.
//
// x (V):   [inner_size] FP16 — value to store
// B (K):   [n_groups * state_size] FP16 — key for addressing
// C (Q):   [n_groups * state_size] FP16 — query for readout
// alpha:   [n_heads] FP16 — decay gate (pre-softplus, combined with A_log/dt_bias)
// beta:    [n_heads] FP16 — learning rate (pre-sigmoid)
// A_log:   [n_heads] FP32 — log decay parameter
// dt_bias: [n_heads] FP32 — bias for alpha
// h_state: [n_heads, state_size, head_dim_ssm] FP32 — recurrent state
// y:       [inner_size] FP16 — output
// z:       [inner_size] FP16 — gate (nullptr = no fusion, otherwise y *= SiLU(z))
void gdn_scan_decode(const half* x, const half* B, const half* C,
                     const half* alpha, const half* beta,
                     const float* A_log, const float* dt_bias,
                     float* h_state, half* y, const half* z,
                     int n_heads, int head_dim_ssm,
                     int state_size, int n_groups,
                     cudaStream_t stream);

// GDN Delta Rule scan — prefill (sequential per-token).
void gdn_scan_prefill(const half* x, const half* B, const half* C,
                      const half* alpha, const half* beta,
                      const float* A_log, const float* dt_bias,
                      float* h_state, half* y, const half* z,
                      int n_tokens, int n_heads, int head_dim_ssm,
                      int state_size, int n_groups,
                      cudaStream_t stream);

// Legacy API stubs (kept for compatibility, not used by run_gdn)
void gdn_decode(const half* q, const half* k, const half* v,
                const half* alpha, const half* beta,
                float* s_state, half* y, const half* gate,
                int n_q_heads, int n_kv_heads, int head_dim, int n_alpha_heads,
                cudaStream_t stream);

void gdn_prefill(const half* q, const half* k, const half* v,
                 const half* alpha, const half* beta,
                 float* s_state, half* y, const half* gate,
                 int n_tokens, int n_q_heads, int n_kv_heads,
                 int head_dim, int n_alpha_heads,
                 cudaStream_t stream);

} // namespace imp
