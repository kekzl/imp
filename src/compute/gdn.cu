#include "compute/gdn.h"
#include "core/logging.h"
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// GDN decode kernel: one block per KV head.
// State S[head_dim, head_dim] updated via simplified delta rule:
//   S = sigmoid(alpha) * S + sigmoid(beta) * outer(k, v)
//   y[q_head] = S^T @ q[q_head]
//
// GQA: q_per_kv Q heads share one KV head's state.
// Alpha/beta: n_alpha_heads values averaged per KV head.
// ---------------------------------------------------------------------------
__global__ void gdn_decode_kernel(
    const half* __restrict__ q,        // [n_q_heads * head_dim]
    const half* __restrict__ k,        // [n_kv_heads * head_dim]
    const half* __restrict__ v,        // [n_kv_heads * head_dim]
    const half* __restrict__ alpha,    // [n_alpha_heads]
    const half* __restrict__ beta,     // [n_alpha_heads]
    float*      __restrict__ s_state,  // [n_kv_heads, head_dim, head_dim]
    half*       __restrict__ y,        // [n_q_heads * head_dim]
    const half* __restrict__ gate,     // [n_q_heads * head_dim] or nullptr
    int n_q_heads, int n_kv_heads, int n_alpha_heads, int head_dim)
{
    const int kv_head = blockIdx.x;
    if (kv_head >= n_kv_heads) return;

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    const int q_per_kv = n_q_heads / n_kv_heads;
    const int alpha_per_kv = n_alpha_heads / n_kv_heads;

    // State for this KV head: S[head_dim, head_dim]
    const int state_elems = head_dim * head_dim;
    float* S = s_state + kv_head * state_elems;

    // K and V for this KV head
    const half* k_h = k + kv_head * head_dim;
    const half* v_h = v + kv_head * head_dim;

    // Average alpha/beta across mapped heads
    float g = 0.0f, b = 0.0f;
    int alpha_base = kv_head * alpha_per_kv;
    for (int ai = 0; ai < alpha_per_kv; ai++) {
        g += 1.0f / (1.0f + expf(-__half2float(alpha[alpha_base + ai])));
        b += 1.0f / (1.0f + expf(-__half2float(beta[alpha_base + ai])));
    }
    g /= alpha_per_kv;
    b /= alpha_per_kv;

    // Update S: S[i,j] = g * S[i,j] + b * k[i] * v[j]
    for (int idx = tid; idx < state_elems; idx += n_threads) {
        int i = idx / head_dim;
        int j = idx % head_dim;
        float k_i = __half2float(k_h[i]);
        float v_j = __half2float(v_h[j]);
        S[idx] = g * S[idx] + b * k_i * v_j;
    }

    __syncthreads();

    // Output: for each Q head in the GQA group, compute y = S^T @ q
    for (int qi = 0; qi < q_per_kv; qi++) {
        int q_idx = kv_head * q_per_kv + qi;
        const half* q_h = q + q_idx * head_dim;

        for (int j = tid; j < head_dim; j += n_threads) {
            float sum = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                sum += S[i * head_dim + j] * __half2float(q_h[i]);
            }

            int out_idx = q_idx * head_dim + j;
            if (gate) {
                float g_val = 1.0f / (1.0f + expf(-__half2float(gate[out_idx])));
                sum *= g_val;
            }
            y[out_idx] = __float2half(sum);
        }
    }
}

void gdn_decode(const half* q, const half* k, const half* v,
                const half* alpha, const half* beta,
                float* s_state, half* y, const half* gate,
                int n_q_heads, int n_kv_heads, int head_dim, int n_alpha_heads,
                cudaStream_t stream)
{
    int threads = 256;
    gdn_decode_kernel<<<n_kv_heads, threads, 0, stream>>>(
        q, k, v, alpha, beta, s_state, y, gate,
        n_q_heads, n_kv_heads, n_alpha_heads, head_dim);
}

// ---------------------------------------------------------------------------
// GDN prefill: iterate decode over all tokens sequentially.
// ---------------------------------------------------------------------------
void gdn_prefill(const half* q, const half* k, const half* v,
                 const half* alpha, const half* beta,
                 float* s_state, half* y, const half* gate,
                 int n_tokens, int n_q_heads, int n_kv_heads,
                 int head_dim, int n_alpha_heads,
                 cudaStream_t stream)
{
    int q_stride = n_q_heads * head_dim;
    int k_stride = n_kv_heads * head_dim;
    int alpha_stride = n_alpha_heads;

    for (int t = 0; t < n_tokens; t++) {
        int threads = 256;
        gdn_decode_kernel<<<n_kv_heads, threads, 0, stream>>>(
            q + t * q_stride,
            k + t * k_stride,
            v + t * k_stride,
            alpha + t * alpha_stride,
            beta + t * alpha_stride,
            s_state,
            y + t * q_stride,
            gate ? gate + t * q_stride : nullptr,
            n_q_heads, n_kv_heads, n_alpha_heads, head_dim);
    }
}

} // namespace imp
