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
// neox=true:  NeoX/split pairs   (i, i + half_dim)
// neox=false: interleaved pairs  (2i, 2i+1)
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
    float inv_scaling,
    int rope_pairs,  // number of pairs to rotate (rope_dim/2 or head_dim/2)
    bool neox)
{
    const int token_idx = blockIdx.x;  // flattened batch*seq index
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x; // which rotation pair (0..rope_pairs-1)

    if (pair_idx >= rope_pairs) return;

    const int pos = positions[token_idx];

    // Compute frequency: 1.0 / (theta ^ (2*pair_idx / head_dim)) / scaling
    float freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
    freq *= inv_scaling;
    float angle = static_cast<float>(pos) * freq;
    float cos_val = __cosf(angle);
    float sin_val = __sinf(angle);

    // Index offsets within head: NeoX pairs (i, i+half) vs interleaved (2i, 2i+1)
    const int idx0 = neox ? pair_idx : (2 * pair_idx);
    const int idx1 = neox ? (pair_idx + rope_pairs) : (2 * pair_idx + 1);

    // Apply to Q if head_idx < n_heads
    if (head_idx < n_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float q0 = Q[base + idx0];
        float q1 = Q[base + idx1];
        Q[base + idx0] = q0 * cos_val - q1 * sin_val;
        Q[base + idx1] = q0 * sin_val + q1 * cos_val;
    }

    // Apply to K if head_idx < n_kv_heads
    if (head_idx < n_kv_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_kv_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float k0 = K[base + idx0];
        float k1 = K[base + idx1];
        K[base + idx0] = k0 * cos_val - k1 * sin_val;
        K[base + idx1] = k0 * sin_val + k1 * cos_val;
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
    float inv_scaling,
    int rope_pairs,
    bool neox)
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;

    if (pair_idx >= rope_pairs) return;

    const int pos = positions[token_idx];

    float freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
    freq *= inv_scaling;
    float angle = static_cast<float>(pos) * freq;
    float cos_val = __cosf(angle);
    float sin_val = __sinf(angle);

    const int idx0 = neox ? pair_idx : (2 * pair_idx);
    const int idx1 = neox ? (pair_idx + rope_pairs) : (2 * pair_idx + 1);

    if (head_idx < n_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float q0 = __half2float(Q[base + idx0]);
        float q1 = __half2float(Q[base + idx1]);
        Q[base + idx0] = __float2half(q0 * cos_val - q1 * sin_val);
        Q[base + idx1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    if (head_idx < n_kv_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_kv_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float k0 = __half2float(K[base + idx0]);
        float k1 = __half2float(K[base + idx1]);
        K[base + idx0] = __float2half(k0 * cos_val - k1 * sin_val);
        K[base + idx1] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------
void rope_forward(Tensor& Q, Tensor& K,
                  const int* positions, int head_dim,
                  float theta, float scaling,
                  int rope_dim, bool neox,
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

    // Partial RoPE: only rotate first rope_dim dimensions (or all if 0)
    const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
    const int pairs = effective_rope_dim / 2;
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
                theta, inv_scaling, pairs, neox);
            break;
        case DType::FP16:
            rope_forward_fp16_kernel<<<grid, block, 0, stream>>>(
                static_cast<__half*>(Q.data),
                static_cast<__half*>(K.data),
                positions,
                batch, seq_len, n_heads, n_kv_heads, head_dim,
                theta, inv_scaling, pairs, neox);
            break;
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// Fused QK-norm + RoPE kernel (decode-only, n=1, FP16).
//
// Combines per-head RMSNorm on Q and K with RoPE in a single kernel launch.
// One block per head (grid = max(n_heads, n_kv_heads)).
//
// Phase 1: RMSNorm(Q_head) using block_reduce, store normed values in shared.
// Phase 2: Apply RoPE to Q from shared, write to global.
// Phase 3: RMSNorm(K_head) if head_idx < n_kv_heads.
// Phase 4: Apply RoPE to K from shared, write to global.
//
// Saves 2 kernel launches vs separate Q-norm + K-norm + RoPE.
// --------------------------------------------------------------------------

__device__ __forceinline__ float rope_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float rope_block_reduce_sum(float val, float* shared_buf) {
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = rope_warp_reduce_sum(val);
    if (lane == 0) shared_buf[warp_id] = val;
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared_buf[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = rope_warp_reduce_sum(val);
    return val;
}

__global__ void qknorm_rope_fused_fp16_kernel(
    __half* __restrict__ Q,              // [n_heads * head_dim]
    __half* __restrict__ K,              // [n_kv_heads * head_dim]
    const __half* __restrict__ q_norm_w, // [head_dim]
    const __half* __restrict__ k_norm_w, // [head_dim]
    const int* __restrict__ positions,   // [1] on device
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float eps,
    float theta,
    float inv_scaling,
    int rope_pairs,
    bool neox,
    float weight_offset)
{
    const int head_idx = blockIdx.x;
    const int pos = positions[0];  // read from device memory

    // Shared memory layout: [8 floats for reduce] + [head_dim floats for normed values]
    extern __shared__ float smem[];
    float* reduce_buf = smem;            // [8]
    float* normed_vals = smem + 8;       // [head_dim]

    // --- Process Q head (always, since grid.x = max(n_heads, n_kv_heads) >= n_heads) ---
    if (head_idx < n_heads) {
        __half* q_head = Q + head_idx * head_dim;

        // Phase 1: Compute RMS
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __half2float(q_head[i]);
            sum_sq += v * v;
        }
        sum_sq = rope_block_reduce_sum(sum_sq, reduce_buf);

        float inv_rms;
        if (threadIdx.x == 0) {
            reduce_buf[0] = rsqrtf(sum_sq / (float)head_dim + eps);
        }
        __syncthreads();
        inv_rms = reduce_buf[0];

        // Phase 2: Normalize and store to shared memory
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __half2float(q_head[i]);
            float w = __half2float(q_norm_w[i]) + weight_offset;
            normed_vals[i] = v * inv_rms * w;
        }
        __syncthreads();

        // Phase 3: RoPE from shared memory, write to global
        for (int pair = threadIdx.x; pair < rope_pairs; pair += blockDim.x) {
            float freq = 1.0f / powf(theta, (2.0f * pair) / (float)head_dim) * inv_scaling;
            float angle = (float)pos * freq;
            float cos_val = __cosf(angle);
            float sin_val = __sinf(angle);

            int idx0 = neox ? pair : (2 * pair);
            int idx1 = neox ? (pair + rope_pairs) : (2 * pair + 1);

            float q0 = normed_vals[idx0];
            float q1 = normed_vals[idx1];
            q_head[idx0] = __float2half(q0 * cos_val - q1 * sin_val);
            q_head[idx1] = __float2half(q0 * sin_val + q1 * cos_val);
        }
        // Write un-rotated elements (if rope_pairs < head_dim/2)
        int rotated_end = neox ? (2 * rope_pairs) : (2 * rope_pairs);
        for (int i = rotated_end + threadIdx.x; i < head_dim; i += blockDim.x) {
            q_head[i] = __float2half(normed_vals[i]);
        }
    }

    // Need a barrier before reusing shared memory for K
    __syncthreads();

    // --- Process K head (only for heads < n_kv_heads) ---
    if (head_idx < n_kv_heads) {
        __half* k_head = K + head_idx * head_dim;

        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __half2float(k_head[i]);
            sum_sq += v * v;
        }
        sum_sq = rope_block_reduce_sum(sum_sq, reduce_buf);

        float inv_rms;
        if (threadIdx.x == 0) {
            reduce_buf[0] = rsqrtf(sum_sq / (float)head_dim + eps);
        }
        __syncthreads();
        inv_rms = reduce_buf[0];

        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __half2float(k_head[i]);
            float w = __half2float(k_norm_w[i]) + weight_offset;
            normed_vals[i] = v * inv_rms * w;
        }
        __syncthreads();

        for (int pair = threadIdx.x; pair < rope_pairs; pair += blockDim.x) {
            float freq = 1.0f / powf(theta, (2.0f * pair) / (float)head_dim) * inv_scaling;
            float angle = (float)pos * freq;
            float cos_val = __cosf(angle);
            float sin_val = __sinf(angle);

            int idx0 = neox ? pair : (2 * pair);
            int idx1 = neox ? (pair + rope_pairs) : (2 * pair + 1);

            float k0 = normed_vals[idx0];
            float k1 = normed_vals[idx1];
            k_head[idx0] = __float2half(k0 * cos_val - k1 * sin_val);
            k_head[idx1] = __float2half(k0 * sin_val + k1 * cos_val);
        }
        int rotated_end = neox ? (2 * rope_pairs) : (2 * rope_pairs);
        for (int i = rotated_end + threadIdx.x; i < head_dim; i += blockDim.x) {
            k_head[i] = __float2half(normed_vals[i]);
        }
    }
}

void qknorm_rope_fused(half* Q, half* K,
                        const half* q_norm_weight, const half* k_norm_weight,
                        int n_heads, int n_kv_heads, int head_dim,
                        float eps, const int* positions,
                        float theta, float scaling,
                        int rope_dim, bool neox,
                        cudaStream_t stream,
                        float weight_offset) {
    const int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;
    if (max_heads == 0 || head_dim == 0) return;

    const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
    const int rope_pairs = effective_rope_dim / 2;
    const float inv_scaling = 1.0f / scaling;

    // Use 128 threads (enough for head_dim up to 256 with 2 iterations)
    const int block_size = 128;
    // Shared memory: 8 floats for reduction + head_dim floats for normed values
    const int smem_bytes = (8 + head_dim) * sizeof(float);

    qknorm_rope_fused_fp16_kernel<<<max_heads, block_size, smem_bytes, stream>>>(
        reinterpret_cast<__half*>(Q),
        reinterpret_cast<__half*>(K),
        reinterpret_cast<const __half*>(q_norm_weight),
        reinterpret_cast<const __half*>(k_norm_weight),
        positions,
        n_heads, n_kv_heads, head_dim, eps,
        theta, inv_scaling, rope_pairs, neox, weight_offset);
}

} // namespace imp
