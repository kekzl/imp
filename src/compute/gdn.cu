#include "compute/gdn.h"
#include "core/logging.h"
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// GDN Delta Rule Scan — decode kernel (single token per head)
//
// One block per head. Each thread handles one d ∈ [0, head_dim_ssm).
// State h[state_size, head_dim_ssm] stored in global memory (FP32).
//
// Mapping from Mamba2 terminology to Delta Rule:
//   x → V (value to store), B → K (key for addressing), C → Q (query for readout)
//
// Delta rule per head h:
//   a_bar = exp(A_log[h] * softplus(alpha[h] + dt_bias[h]))  // decay
//   beta_h = sigmoid(beta[h])                                  // learning rate
//   predicted[d] = sum_s(h[s,d] * K[s])                       // predict from state
//   error[d] = V[d] - a_bar * predicted[d]                    // prediction error
//   h[s,d] = a_bar * h[s,d] + beta_h * K[s] * error[d]       // rank-1 update
//   y[d] = sum_s(h[s,d] * Q[s])                               // readout
//   if z: y[d] *= sigmoid(z[d])                                // gating
// ---------------------------------------------------------------------------

__global__ void gdn_scan_decode_kernel(
    const half* __restrict__ x,          // [inner_size] = V
    const half* __restrict__ B_in,       // [n_groups * state_size] = K
    const half* __restrict__ C_in,       // [n_groups * state_size] = Q
    const half* __restrict__ alpha_raw,  // [n_heads] decay gate (pre-softplus)
    const half* __restrict__ beta_raw,   // [n_heads] learning rate (pre-sigmoid)
    const float* __restrict__ A_log,     // [n_heads]
    const float* __restrict__ dt_bias,   // [n_heads]
    float*       __restrict__ h_state,   // [n_heads, state_size, head_dim_ssm]
    half*        __restrict__ y,         // [inner_size]
    const half*  __restrict__ z,         // [inner_size] gate (nullptr = no fusion)
    int n_heads, int head_dim_ssm, int state_size, int n_groups)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int d = threadIdx.x;
    if (d >= head_dim_ssm) return;

    const int g = h / (n_heads / n_groups);  // group for this head
    const int heads_per_group = n_heads / n_groups;

    // Pointers for this head
    float* H = h_state + static_cast<size_t>(h) * state_size * head_dim_ssm;
    const half* K_g = B_in + g * state_size;   // key [state_size]
    const half* Q_g = C_in + g * state_size;   // query [state_size]

    // V (value) for this head's d-th dimension
    float v_d = __half2float(x[h * head_dim_ssm + d]);

    // Decay gate: g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    // exp(A_log) converts from log-space → linear decay rate
    // The negative sign ensures g ∈ (0, 1]
    float alpha_h = __half2float(alpha_raw[h]);
    float dt_val = alpha_h + dt_bias[h];
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));  // softplus
    float decay_rate = expf(A_log[h]);  // A_log → linear (typically A_log < 0 → rate < 1)
    float exponent = -decay_rate * dt_val;
    exponent = fmaxf(exponent, -20.0f);  // clamp to prevent underflow
    float g_t = expf(exponent);  // decay gate ∈ (0, 1]

    // Learning rate: beta = sigmoid(beta_raw) ∈ (0, 1)
    float beta_h = __half2float(beta_raw[h]);
    beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

    // L2-normalize K and Q (critical for delta rule stability and correctness)
    __shared__ float s_k_inv, s_q_inv;
    if (d == 0) {
        float k_sq = 0.0f, q_sq = 0.0f;
        for (int s = 0; s < state_size; s++) {
            float ks = __half2float(K_g[s]);
            float qs = __half2float(Q_g[s]);
            k_sq += ks * ks;
            q_sq += qs * qs;
        }
        s_k_inv = (k_sq > 1e-12f) ? rsqrtf(k_sq) : 0.0f;
        s_q_inv = (q_sq > 1e-12f) ? rsqrtf(q_sq) : 0.0f;
    }
    __syncthreads();

    // Step 1: Predict — kv[d] = (g * S)^T @ k_norm = g * sum_s(S[s,d] * k_norm[s])
    float kv_d = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float k_s = __half2float(K_g[s]) * s_k_inv;
        kv_d += H[s * head_dim_ssm + d] * k_s;
    }
    kv_d *= g_t;  // prediction from decayed state

    // Step 2: Delta = (v - kv) * beta
    float delta_d = (v_d - kv_d) * beta_h;

    // Step 3: Update state + compute output
    //   S[s,d] = g * S[s,d] + k_norm[s] * delta[d]
    //   y[d] = sum_s(S[s,d] * q_norm[s])
    float y_partial = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float k_s = __half2float(K_g[s]) * s_k_inv;
        float q_s = __half2float(Q_g[s]) * s_q_inv;

        float h_new = g_t * H[s * head_dim_ssm + d] + k_s * delta_d;
        H[s * head_dim_ssm + d] = h_new;

        y_partial += h_new * q_s;
    }

    // Note: output scaling by 1/√head_dim is NOT applied here.
    // The GroupNorm + output projection handle the scaling implicitly.

    // Gating: y *= SiLU(z) if z is provided
    int out_idx = h * head_dim_ssm + d;
    if (z) {
        float z_val = __half2float(z[out_idx]);
        y_partial *= z_val / (1.0f + expf(-z_val));  // SiLU
    }

    y[out_idx] = __float2half(y_partial);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void gdn_decode(const half* q, const half* k, const half* v,
                const half* alpha, const half* beta,
                float* s_state, half* y, const half* gate,
                int n_q_heads, int n_kv_heads, int head_dim, int n_alpha_heads,
                cudaStream_t stream)
{
    // For GDN scan: use Mamba2-compatible interface
    // n_q_heads here is actually n_heads (from ssm_dt_rank)
    // head_dim is head_dim_ssm (inner_size / n_heads)
    // n_alpha_heads = n_heads (same as n_q_heads for GDN)
    // This function is kept for API compatibility but is NOT used by run_gdn.
    // run_gdn calls gdn_scan_decode directly.
    (void)q; (void)k; (void)v; (void)alpha; (void)beta;
    (void)s_state; (void)y; (void)gate;
    (void)n_q_heads; (void)n_kv_heads; (void)head_dim; (void)n_alpha_heads;
    (void)stream;
}

void gdn_prefill(const half* q, const half* k, const half* v,
                 const half* alpha, const half* beta,
                 float* s_state, half* y, const half* gate,
                 int n_tokens, int n_q_heads, int n_kv_heads,
                 int head_dim, int n_alpha_heads,
                 cudaStream_t stream)
{
    (void)q; (void)k; (void)v; (void)alpha; (void)beta;
    (void)s_state; (void)y; (void)gate;
    (void)n_tokens; (void)n_q_heads; (void)n_kv_heads;
    (void)head_dim; (void)n_alpha_heads;
    (void)stream;
}

// ---------------------------------------------------------------------------
// GDN scan decode: called from run_gdn() as replacement for ssm_scan_decode.
// Same parameter layout as ssm_scan_decode but uses delta rule math.
// ---------------------------------------------------------------------------
void gdn_scan_decode(const half* x, const half* B, const half* C,
                     const half* alpha, const half* beta,
                     const float* A_log, const float* dt_bias,
                     float* h_state, half* y, const half* z,
                     int n_heads, int head_dim_ssm,
                     int state_size, int n_groups,
                     cudaStream_t stream)
{
    // One block per head, head_dim_ssm threads per block
    gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, 0, stream>>>(
        x, B, C, alpha, beta, A_log, dt_bias,
        h_state, y, z,
        n_heads, head_dim_ssm, state_size, n_groups);
}

// ---------------------------------------------------------------------------
// GDN scan prefill: sequential per-token, same kernel.
// ---------------------------------------------------------------------------
void gdn_scan_prefill(const half* x, const half* B, const half* C,
                      const half* alpha, const half* beta,
                      const float* A_log, const float* dt_bias,
                      float* h_state, half* y, const half* z,
                      int n_tokens, int n_heads, int head_dim_ssm,
                      int state_size, int n_groups,
                      cudaStream_t stream)
{
    int inner = n_heads * head_dim_ssm;
    int BC_size = n_groups * state_size;

    for (int t = 0; t < n_tokens; t++) {
        gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, 0, stream>>>(
            x + t * inner,
            B + t * BC_size,
            C + t * BC_size,
            alpha + t * n_heads,
            beta + t * n_heads,
            A_log, dt_bias,
            h_state,
            y + t * inner,
            z ? z + t * inner : nullptr,
            n_heads, head_dim_ssm, state_size, n_groups);
    }
}

} // namespace imp
