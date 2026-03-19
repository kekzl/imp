#include "compute/gdn.h"
#include "core/logging.h"
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// GDN decode kernel: one block per KV head, all Q heads in GQA group.
//
// Delta rule with error correction:
//   g = sigmoid(avg_alpha)                    // decay
//   b = sigmoid(avg_beta)                     // learning rate
//   error = v - S @ k                         // prediction error [hd]
//   S = g * S + b * outer(k, error)           // state update [hd, hd]
//   y[q_head] = S @ q[q_head]                 // output [hd]
//
// Gate applied as SiLU (Mamba-style): y *= gate * sigmoid(gate)
// ---------------------------------------------------------------------------

// Shared memory layout: float error[head_dim] + float s_k[head_dim] + float s_v[head_dim]
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
    extern __shared__ float smem[];
    float* s_error = smem;                        // [head_dim]
    float* s_k_vec = smem + head_dim;             // [head_dim]

    const int kv_head = blockIdx.x;
    if (kv_head >= n_kv_heads) return;

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    const int q_per_kv = n_q_heads / n_kv_heads;
    const int alpha_per_kv = n_alpha_heads / n_kv_heads;

    const int state_elems = head_dim * head_dim;
    float* S = s_state + kv_head * state_elems;

    const half* k_h = k + kv_head * head_dim;
    const half* v_h = v + kv_head * head_dim;

    // Load k vector to shared memory
    for (int i = tid; i < head_dim; i += n_threads) {
        s_k_vec[i] = __half2float(k_h[i]);
    }
    __syncthreads();

    // Average alpha/beta across mapped heads
    float g = 0.0f, b = 0.0f;
    int alpha_base = kv_head * alpha_per_kv;
    for (int ai = 0; ai < alpha_per_kv; ai++) {
        g += 1.0f / (1.0f + expf(-__half2float(alpha[alpha_base + ai])));
        b += 1.0f / (1.0f + expf(-__half2float(beta[alpha_base + ai])));
    }
    g /= alpha_per_kv;
    b /= alpha_per_kv;

    // Step 1: Compute error = v - S @ k (where S @ k = matrix-vector product)
    // error[j] = v[j] - sum_i(S[j * head_dim + i] * k[i])
    // Note: S is stored row-major as [head_dim, head_dim], S[row=j, col=i]
    // But for delta rule, S maps k->v, so S @ k means: for each output dim j,
    // predicted_v[j] = sum_i S[j,i] * k[i]
    for (int j = tid; j < head_dim; j += n_threads) {
        float predicted = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            predicted += S[j * head_dim + i] * s_k_vec[i];
        }
        s_error[j] = __half2float(v_h[j]) - predicted;
    }
    __syncthreads();

    // Step 2: Update S[j,i] = g * S[j,i] + b * k[i] * error[j]
    for (int idx = tid; idx < state_elems; idx += n_threads) {
        int j = idx / head_dim;  // output (value) dim
        int i = idx % head_dim;  // input (key) dim
        S[idx] = g * S[idx] + b * s_k_vec[i] * s_error[j];
    }
    __syncthreads();

    // Step 3: Output y[q_head] = S @ q[q_head]
    // y[q_head, j] = sum_i(S[j, i] * q[i])
    for (int qi = 0; qi < q_per_kv; qi++) {
        int q_idx = kv_head * q_per_kv + qi;
        const half* q_h = q + q_idx * head_dim;

        for (int j = tid; j < head_dim; j += n_threads) {
            float sum = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                sum += S[j * head_dim + i] * __half2float(q_h[i]);
            }

            int out_idx = q_idx * head_dim + j;
            // SiLU gating: y * gate * sigmoid(gate) = y * SiLU(gate)
            if (gate) {
                float g_val = __half2float(gate[out_idx]);
                float silu = g_val / (1.0f + expf(-g_val));  // x * sigmoid(x)
                sum *= silu;
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
    // Shared memory for error vector + k vector
    size_t smem = 2 * head_dim * sizeof(float);
    int threads = 256;
    gdn_decode_kernel<<<n_kv_heads, threads, smem, stream>>>(
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

    size_t smem = 2 * head_dim * sizeof(float);

    for (int t = 0; t < n_tokens; t++) {
        gdn_decode_kernel<<<n_kv_heads, 256, smem, stream>>>(
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
