#include "compute/gdn.h"
#include "core/logging.h"
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// GDN Delta Rule Scan — matching llama.cpp's gated_delta_net_cuda kernel.
//
// Key differences from our previous kernel:
// 1. State processed in registers (shard per warp lane)
// 2. KV dot product via warp_reduce_sum (not per-thread accumulation)
// 3. Q and K are L2-normalized before this kernel (we do it inside)
// 4. Output scaled by 1/√S_v
// ---------------------------------------------------------------------------

template <int WARP_SIZE_T>
__device__ __forceinline__ float warp_reduce_sum_gdn(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE_T / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Kernel: one block per head, one warp per column (d dimension).
// Each lane owns a shard of the state column (state_size / warp_size rows).
// This matches llama.cpp's kernel structure.
__global__ void gdn_scan_decode_kernel(
    const half* __restrict__ x,          // V: [inner_size]
    const half* __restrict__ B_in,       // K: [n_groups * state_size]
    const half* __restrict__ C_in,       // Q: [n_groups * state_size]
    const half* __restrict__ alpha_raw,  // [n_heads]
    const half* __restrict__ beta_raw,   // [n_heads]
    const float* __restrict__ A_log,     // [n_heads] = -exp(A_log_original)
    const float* __restrict__ dt_bias,   // [n_heads]
    float*       __restrict__ h_state,   // [n_heads, state_size, head_dim_ssm]
    half*        __restrict__ y,         // [inner_size]
    const half*  __restrict__ z,         // unused
    int n_heads, int head_dim_ssm, int state_size, int n_groups)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int d = threadIdx.x;  // column index (value dimension)
    if (d >= head_dim_ssm) return;

    const int g = h % n_groups;

    float* H = h_state + static_cast<size_t>(h) * state_size * head_dim_ssm;
    const half* K_g = B_in + g * state_size;
    const half* Q_g = C_in + g * state_size;

    float v_d = __half2float(x[h * head_dim_ssm + d]);

    // Decay: g_t = exp(A_gguf * softplus(alpha + dt_bias))
    float alpha_h = __half2float(alpha_raw[h]);
    float dt_val = alpha_h + dt_bias[h];
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    float g_t = expf(fmaxf(A_log[h] * dt_val, -20.0f));

    // Beta = sigmoid(beta_raw)
    float beta_h = __half2float(beta_raw[h]);
    beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

    // Cache K and Q in shared memory as FP32 (converted from FP16 once).
    // This matches llama.cpp which receives FP32 Q/K from the graph.
    // Also compute L2-norms for normalization.
    extern __shared__ float smem[];
    float* s_k = smem;                    // [state_size]
    float* s_q = smem + state_size;       // [state_size]
    // Thread 0 loads and normalizes K/Q
    __shared__ float s_k_inv, s_q_inv;
    if (d == 0) {
        float k_sq = 0.0f, q_sq = 0.0f;
        for (int s = 0; s < state_size; s++) {
            float ks = __half2float(K_g[s]);
            float qs = __half2float(Q_g[s]);
            s_k[s] = ks;
            s_q[s] = qs;
            k_sq += ks * ks;
            q_sq += qs * qs;
        }
        s_k_inv = rsqrtf(k_sq + 1e-6f);
        s_q_inv = rsqrtf(q_sq + 1e-6f);
        // Normalize in-place in smem
        for (int s = 0; s < state_size; s++) {
            s_k[s] *= s_k_inv;
            s_q[s] *= s_q_inv;
        }
    }
    __syncthreads();

    // Scale factor for output (matching llama.cpp: scale = 1/√S_v)
    const float scale = rsqrtf(static_cast<float>(head_dim_ssm));

    // Step 1: kv[d] = sum_s(S[s,d] * k_norm[s]) — using cached FP32 K
    float kv_d = 0.0f;
    for (int s = 0; s < state_size; s++) {
        kv_d += H[s * head_dim_ssm + d] * s_k[s];
    }

    // Step 2: delta[d] = (v[d] - g * kv[d]) * beta
    float delta_d = (v_d - g_t * kv_d) * beta_h;

    // Step 3: Update state + compute output — using cached FP32 K/Q
    float y_partial = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float h_new = g_t * H[s * head_dim_ssm + d] + s_k[s] * delta_d;
        H[s * head_dim_ssm + d] = h_new;
        y_partial += h_new * s_q[s];
    }

    // Apply scale and write output
    y[h * head_dim_ssm + d] = __float2half(y_partial * scale);
}

// Host launchers
void gdn_scan_decode(const half* x, const half* B, const half* C,
                     const half* alpha, const half* beta,
                     const float* A_log, const float* dt_bias,
                     float* h_state, half* y, const half* z,
                     int n_heads, int head_dim_ssm,
                     int state_size, int n_groups,
                     cudaStream_t stream) {
    size_t smem_sz = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem_sz, stream>>>(
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
        size_t smem_sz = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem_sz, stream>>>(
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
