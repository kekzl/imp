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

    // Compute decay gate: a_bar = exp(A_log * softplus(alpha + dt_bias))
    // A_log is typically negative → a_bar ∈ (0, 1]
    float alpha_h = __half2float(alpha_raw[h]);
    float dt_val = alpha_h + dt_bias[h];
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));  // softplus
    float exponent = dt_val * A_log[h];
    exponent = fmaxf(fminf(exponent, 0.0f), -20.0f);  // clamp to [-20, 0]
    float a_bar = expf(exponent);

    // Learning rate: beta ∈ (0, 1)
    float beta_h = __half2float(beta_raw[h]);
    beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

    // L2-normalize K (critical for delta rule stability)
    // Each thread computes partial norm, then warp-reduce
    float k_norm_sq = 0.0f;
    // Note: all threads in the block share the same K_g (per group).
    // Thread 0 computes the norm, others read it.
    __shared__ float s_k_norm_inv;
    if (d == 0) {
        for (int s = 0; s < state_size; s++) {
            float ks = __half2float(K_g[s]);
            k_norm_sq += ks * ks;
        }
        s_k_norm_inv = (k_norm_sq > 1e-12f) ? rsqrtf(k_norm_sq) : 1.0f;
    }
    __syncthreads();
    float k_inv = s_k_norm_inv;

    // Step 1: Predict — predicted[d] = sum_s(h[s,d] * K_norm[s])
    float predicted = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float k_s = __half2float(K_g[s]) * k_inv;
        predicted += H[s * head_dim_ssm + d] * k_s;
    }

    // Step 2: Error (no a_bar on prediction — a_bar only decays the state)
    float error_d = v_d - predicted;

    // Step 3 + 4: Update state + compute output
    float y_partial = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float k_s = __half2float(K_g[s]) * k_inv;  // normalized key
        float q_s = __half2float(Q_g[s]);
        float h_old = H[s * head_dim_ssm + d];

        // Delta rule: S = decay * S + lr * outer(K_norm, error)
        float h_new = a_bar * h_old + beta_h * k_s * error_d;
        H[s * head_dim_ssm + d] = h_new;

        // Output: y[d] += h[s,d] * Q[s]
        y_partial += h_new * q_s;
    }

    // Gating: y *= SiLU(z) if z is provided
    int out_idx = h * head_dim_ssm + d;
    if (z) {
        float z_val = __half2float(z[out_idx]);
        float silu = z_val / (1.0f + expf(-z_val));
        y_partial *= silu;
    }
    // TODO: remove debug — test without gate
    // y_partial *= 10.0f;  // amplify for testing

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
