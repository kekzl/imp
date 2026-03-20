#include "compute/gdn.h"
#include "core/logging.h"
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// GDN Delta Rule Scan kernel.
// Matches llama.cpp's gated_delta_net kernel behavior:
//   kv = S^T @ k  (prediction)
//   delta = (v - g*kv) * beta
//   S = g*S + outer(k, delta)
//   y = S^T @ q
//   y *= scale (1/√state_size)
// K and Q are L2-normalized inside the kernel.
// ---------------------------------------------------------------------------
__global__ void gdn_scan_decode_kernel(
    const half* __restrict__ x,          // [inner_size] = V
    const half* __restrict__ B_in,       // [n_groups * state_size] = K
    const half* __restrict__ C_in,       // [n_groups * state_size] = Q
    const half* __restrict__ alpha_raw,  // [n_heads]
    const half* __restrict__ beta_raw,   // [n_heads]
    const float* __restrict__ A_log,     // [n_heads]
    const float* __restrict__ dt_bias,   // [n_heads]
    float*       __restrict__ h_state,   // [n_heads, state_size, head_dim_ssm]
    half*        __restrict__ y,         // [inner_size]
    const half*  __restrict__ z,         // unused (gate applied externally)
    int n_heads, int head_dim_ssm, int state_size, int n_groups)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int d = threadIdx.x;
    if (d >= head_dim_ssm) return;

    // V-heads are stored in TILED order (from GGUF converter _LinearAttentionVReorderBase):
    //   Grouped (HF): [G0_v0, G0_v1, G0_v2, G1_v0, G1_v1, G1_v2, ...]
    //   Tiled (GGUF):  [G0_v0, G1_v0, ..., G0_v1, G1_v1, ..., G0_v2, G1_v2, ...]
    // For tiled layout: V-head h maps to K-group g = h % n_groups (not h / ratio).
    const int g = h % n_groups;

    float* H = h_state + static_cast<size_t>(h) * state_size * head_dim_ssm;
    const half* K_g = B_in + g * state_size;
    const half* Q_g = C_in + g * state_size;

    float v_d = __half2float(x[h * head_dim_ssm + d]);

    // Decay gate: g_t = exp(A_gguf * softplus(alpha + dt_bias))
    // GGUF stores A as -exp(A_log_original), NOT A_log itself!
    // So A_gguf is already the negative decay rate. No exp() needed.
    float alpha_h = __half2float(alpha_raw[h]);
    float dt_val = alpha_h + dt_bias[h];
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    float g_t = expf(fmaxf(A_log[h] * dt_val, -20.0f));

    // Beta = sigmoid(beta_raw)
    float beta_h = __half2float(beta_raw[h]);
    beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

    // K and Q are used WITHOUT normalization here.
    // L2-norm is applied as a separate graph op in llama.cpp.
    // We apply it here for stability, matching llama.cpp's pre-kernel norm.
    // NOTE: removing normalization causes NaN explosion (tested).
    __shared__ float s_k_inv, s_q_inv;
    if (d == 0) {
        float k_sq = 0.0f, q_sq = 0.0f;
        for (int s = 0; s < state_size; s++) {
            float ks = __half2float(K_g[s]);
            float qs = __half2float(Q_g[s]);
            k_sq += ks * ks;
            q_sq += qs * qs;
        }
        // Option: L2-normalize K and Q (matching llama.cpp pre-kernel l2_norm)
        // TEST: try WITHOUT normalization — llama.cpp normalizes BEFORE kernel
        // but the normalization might be interacting badly with our kernel's
        // different state layout or timing.
        // L2-normalize K and Q (matching llama.cpp ggml_l2_norm)
        const float eps_l2 = 1e-6f;
        s_k_inv = rsqrtf(k_sq + eps_l2);
        s_q_inv = rsqrtf(q_sq + eps_l2);
    }
    __syncthreads();

    // Step 1: kv[d] = sum_s(S[s,d] * k_norm[s])
    float kv_d = 0.0f;
    for (int s = 0; s < state_size; s++)
        kv_d += H[s * head_dim_ssm + d] * __half2float(K_g[s]) * s_k_inv;

    // Step 2: delta[d] = (v[d] - g * kv[d]) * beta
    float delta_d = (v_d - g_t * kv_d) * beta_h;

    // Step 3: Update S + compute output
    float y_partial = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float k_s = __half2float(K_g[s]) * s_k_inv;
        float q_s = __half2float(Q_g[s]) * s_q_inv;
        float h_new = g_t * H[s * head_dim_ssm + d] + k_s * delta_d;
        H[s * head_dim_ssm + d] = h_new;
        y_partial += h_new * q_s;
    }

    // Output scale matching llama.cpp: attn_data[col] = attn_col * (1/√S_v)
    // The model was TRAINED with this scale — ssm_norm and ssm_out weights
    // expect 1/√128-scaled inputs. Without it, magnitudes are 11.3x too large.
    y[h * head_dim_ssm + d] = __float2half(y_partial * rsqrtf(static_cast<float>(head_dim_ssm)));
}

// Host launchers
void gdn_scan_decode(const half* x, const half* B, const half* C,
                     const half* alpha, const half* beta,
                     const float* A_log, const float* dt_bias,
                     float* h_state, half* y, const half* z,
                     int n_heads, int head_dim_ssm,
                     int state_size, int n_groups,
                     cudaStream_t stream) {
    gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, 2 * sizeof(float), stream>>>(
        x, B, C, alpha, beta, A_log, dt_bias, h_state, y, z,
        n_heads, head_dim_ssm, state_size, n_groups);
}

void gdn_scan_prefill(const half* x, const half* B, const half* C,
                      const half* alpha, const half* beta,
                      const float* A_log, const float* dt_bias,
                      float* h_state, half* y, const half* z,
                      int n_tokens, int n_heads, int head_dim_ssm,
                      int state_size, int n_groups,
                      cudaStream_t stream) {
    int inner = n_heads * head_dim_ssm;
    int BC_size = n_groups * state_size;
    for (int t = 0; t < n_tokens; t++) {
        gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, 2 * sizeof(float), stream>>>(
            x + t * inner, B + t * BC_size, C + t * BC_size,
            alpha + t * n_heads, beta + t * n_heads,
            A_log, dt_bias, h_state,
            y + t * inner, nullptr,
            n_heads, head_dim_ssm, state_size, n_groups);
    }
}

// Legacy stubs
void gdn_decode(const half*, const half*, const half*,
                const half*, const half*, float*, half*, const half*,
                int, int, int, int, cudaStream_t) {}
void gdn_prefill(const half*, const half*, const half*,
                 const half*, const half*, float*, half*, const half*,
                 int, int, int, int, int, cudaStream_t) {}

} // namespace imp
