// Naive reference attention kernel for debugging.
// No optimizations — pure FP32 accumulation, one thread block per (head, query).
// Activated by IMP_NAIVE_ATTN=1 env var.
//
// Computes: O[q] = softmax(scale * Q[q] @ K^T, causal) @ V
// Handles GQA (multiple Q heads share one KV head).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace imp {

// Grid: (n_heads, seq_len)  — one block per (head, query_position)
// Block: 256 threads
// Each block computes one output row: O[head, query_pos, 0..head_dim-1]
//
// Memory layout (row-major, interleaved heads):
//   Q[pos * n_heads * hd + head * hd + d]
//   K[pos * n_kv_heads * hd + kv_head * hd + d]
//   V[pos * n_kv_heads * hd + kv_head * hd + d]
//   O[pos * n_heads * hd + head * hd + d]
__global__ void naive_attention_prefill_kernel(
    const half* __restrict__ Q,      // [seq_len, n_heads * head_dim]
    const half* __restrict__ K,      // [seq_len, n_kv_heads * head_dim]
    const half* __restrict__ V,      // [seq_len, n_kv_heads * head_dim]
    half* __restrict__ O,            // [seq_len, n_heads * head_dim]
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    float scale, float softcap, int sliding_window)
{
    const int head = blockIdx.x;
    const int q_pos = blockIdx.y;
    const int tid = threadIdx.x;
    const int gqa_group = head / (n_heads / n_kv_heads);  // which KV head

    if (head >= n_heads || q_pos >= seq_len) return;

    // Pointers to this head's Q row and output row
    const half* q_row = Q + (int64_t)q_pos * n_heads * head_dim + head * head_dim;
    half* o_row = O + (int64_t)q_pos * n_heads * head_dim + head * head_dim;

    // --- Phase 1: Compute attention scores (FP32) ---
    // Each thread handles a subset of key positions
    extern __shared__ float smem[];  // [seq_len] scores + [1] max + [1] sum
    float* scores = smem;

    for (int k_pos = tid; k_pos < seq_len; k_pos += blockDim.x) {
        if (k_pos > q_pos) {
            // Causal mask: future positions get -inf
            scores[k_pos] = -FLT_MAX;
        } else if (sliding_window > 0 && (q_pos - k_pos) >= sliding_window) {
            // Sliding window: positions outside the window get -inf
            scores[k_pos] = -FLT_MAX;
        } else {
            // Dot product: Q[q_pos, head] . K[k_pos, gqa_group]
            const half* k_row = K + (int64_t)k_pos * n_kv_heads * head_dim + gqa_group * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += __half2float(q_row[d]) * __half2float(k_row[d]);
            }
            dot *= scale;

            // Optional softcap: score = cap * tanh(score / cap)
            if (softcap > 0.0f) {
                dot = softcap * tanhf(dot / softcap);
            }

            scores[k_pos] = dot;
        }
    }
    __syncthreads();

    // --- Phase 2: Softmax (block-wide reduction) ---
    // Find max
    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);

    // Warp reduce max
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));

    __shared__ float s_max_vals[8];  // up to 8 warps (256 threads)
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) s_max_vals[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = s_max_vals[0];
        for (int w = 1; w < (blockDim.x + 31) / 32; w++)
            m = fmaxf(m, s_max_vals[w]);
        s_max_vals[0] = m;
    }
    __syncthreads();
    float max_val = s_max_vals[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float e = (scores[j] > -FLT_MAX + 1.0f) ? expf(scores[j] - max_val) : 0.0f;
        scores[j] = e;
        local_sum += e;
    }

    // Warp reduce sum
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);

    __shared__ float s_sum_vals[8];
    if (lane == 0) s_sum_vals[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++)
            s += s_sum_vals[w];
        s_sum_vals[0] = (s > 0.0f) ? (1.0f / s) : 0.0f;
    }
    __syncthreads();
    float inv_sum = s_sum_vals[0];

    // Normalize scores in-place
    for (int j = tid; j < seq_len; j += blockDim.x)
        scores[j] *= inv_sum;
    __syncthreads();

    // --- Phase 3: Weighted sum of V vectors ---
    // Each thread computes a subset of output dimensions
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            if (scores[k_pos] > 0.0f) {
                const half* v_row = V + (int64_t)k_pos * n_kv_heads * head_dim + gqa_group * head_dim;
                acc += scores[k_pos] * __half2float(v_row[d]);
            }
        }
        o_row[d] = __float2half(acc);
    }
}

void naive_attention_prefill(
    const half* Q, const half* K, const half* V, half* O,
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    float scale, float softcap, cudaStream_t stream,
    int sliding_window)
{
    int threads = 256;
    dim3 grid(n_heads, seq_len);
    size_t smem = seq_len * sizeof(float);  // scores array

    naive_attention_prefill_kernel<<<grid, threads, smem, stream>>>(
        Q, K, V, O, seq_len, n_heads, n_kv_heads, head_dim, scale, softcap, sliding_window);
}

} // namespace imp
