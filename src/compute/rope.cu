#include "compute/rope.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// --------------------------------------------------------------------------
// RoPE kernel for FP32
//
// Q: [batch, seq_len, n_heads,    head_dim]
// K: [batch, seq_len, n_kv_heads, head_dim]
// positions: [batch * seq_len]
//
// Grid.x  = batch * seq_len  (one block per token position)
// Grid.y  = max(n_heads, n_kv_heads)
// Block.x = head_dim / 2  (one thread per rotation pair)
//
// Each thread handles both Q and K if its head index is in range.
// --------------------------------------------------------------------------
__global__ void rope_forward_fp32_kernel(
    float* __restrict__ Q,
    float* __restrict__ K,
    const int* __restrict__ positions,
    int batch,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float theta,
    float inv_scaling)
{
    const int token_idx = blockIdx.x;  // flattened batch*seq index
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x; // which rotation pair (0..head_dim/2-1)

    if (pair_idx >= head_dim / 2) return;

    const int pos = positions[token_idx];

    // Compute frequency: 1.0 / (theta ^ (2*pair_idx / head_dim)) / scaling
    float freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
    freq *= inv_scaling;
    float angle = static_cast<float>(pos) * freq;
    float cos_val = __cosf(angle);
    float sin_val = __sinf(angle);

    // Apply to Q if head_idx < n_heads
    if (head_idx < n_heads) {
        // Q layout: [batch*seq_len, n_heads, head_dim]
        int64_t base = static_cast<int64_t>(token_idx) * n_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float q0 = Q[base + 2 * pair_idx];
        float q1 = Q[base + 2 * pair_idx + 1];
        Q[base + 2 * pair_idx]     = q0 * cos_val - q1 * sin_val;
        Q[base + 2 * pair_idx + 1] = q0 * sin_val + q1 * cos_val;
    }

    // Apply to K if head_idx < n_kv_heads
    if (head_idx < n_kv_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_kv_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float k0 = K[base + 2 * pair_idx];
        float k1 = K[base + 2 * pair_idx + 1];
        K[base + 2 * pair_idx]     = k0 * cos_val - k1 * sin_val;
        K[base + 2 * pair_idx + 1] = k0 * sin_val + k1 * cos_val;
    }
}

// --------------------------------------------------------------------------
// RoPE kernel for FP16 (load half -> compute float -> store half)
// --------------------------------------------------------------------------
__global__ void rope_forward_fp16_kernel(
    __half* __restrict__ Q,
    __half* __restrict__ K,
    const int* __restrict__ positions,
    int batch,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float theta,
    float inv_scaling)
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int pos = positions[token_idx];

    float freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
    freq *= inv_scaling;
    float angle = static_cast<float>(pos) * freq;
    float cos_val = __cosf(angle);
    float sin_val = __sinf(angle);

    if (head_idx < n_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float q0 = __half2float(Q[base + 2 * pair_idx]);
        float q1 = __half2float(Q[base + 2 * pair_idx + 1]);
        Q[base + 2 * pair_idx]     = __float2half(q0 * cos_val - q1 * sin_val);
        Q[base + 2 * pair_idx + 1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    if (head_idx < n_kv_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_kv_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float k0 = __half2float(K[base + 2 * pair_idx]);
        float k1 = __half2float(K[base + 2 * pair_idx + 1]);
        K[base + 2 * pair_idx]     = __float2half(k0 * cos_val - k1 * sin_val);
        K[base + 2 * pair_idx + 1] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------
void rope_forward(Tensor& Q, Tensor& K,
                  const int* positions, int head_dim,
                  float theta, float scaling,
                  cudaStream_t stream)
{
    // Q: [batch, seq_len, n_heads, head_dim]
    // K: [batch, seq_len, n_kv_heads, head_dim]
    const int batch     = static_cast<int>(Q.shape[0]);
    const int seq_len   = static_cast<int>(Q.shape[1]);
    const int n_heads   = static_cast<int>(Q.shape[2]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);
    const int total_tokens = batch * seq_len;
    const int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;

    if (total_tokens == 0 || head_dim == 0) return;

    // Block: head_dim/2 threads (one per rotation pair), capped at 512
    const int pairs   = head_dim / 2;
    const int block_x = (pairs <= 512) ? pairs : 512;

    dim3 grid(total_tokens, max_heads);
    dim3 block(block_x);

    const float inv_scaling = 1.0f / scaling;

    switch (Q.dtype) {
        case DType::FP32:
            rope_forward_fp32_kernel<<<grid, block, 0, stream>>>(
                static_cast<float*>(Q.data),
                static_cast<float*>(K.data),
                positions,
                batch, seq_len, n_heads, n_kv_heads, head_dim,
                theta, inv_scaling);
            break;
        case DType::FP16:
            rope_forward_fp16_kernel<<<grid, block, 0, stream>>>(
                static_cast<__half*>(Q.data),
                static_cast<__half*>(K.data),
                positions,
                batch, seq_len, n_heads, n_kv_heads, head_dim,
                theta, inv_scaling);
            break;
        default:
            break;
    }
}

} // namespace imp
