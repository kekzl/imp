#include "compute/rope.h"
#include "runtime/pdl.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace imp {

// --------------------------------------------------------------------------
// YaRN device helpers
// --------------------------------------------------------------------------

// Linear ramp: 1.0 when i0/2 <= low, 0.0 when i0/2 >= high, linear blend between
static __device__ __forceinline__ float rope_yarn_ramp(float low, float high, int i0) {
    float y = (i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// YaRN frequency blending: blends between interpolated (freq_scale * theta_extrap)
// and extrapolated (theta_extrap) based on the correction dimension ramp.
// When ext_factor == 0, reduces to pure linear scaling (theta = freq_scale * theta_extrap).
static __device__ __forceinline__ void rope_yarn(
        float theta_extrap, float freq_scale,
        float corr_dim_0, float corr_dim_1, int i0,
        float ext_factor, float mscale,
        float& cos_theta, float& sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dim_0, corr_dim_1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    cos_theta = __cosf(theta) * mscale;
    sin_theta = __sinf(theta) * mscale;
}

// --------------------------------------------------------------------------
// RoPE kernel for FP32 with YaRN support
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
    int rope_pairs,
    bool neox,
    float ext_factor,
    float attn_factor,
    float corr_dim_0,
    float corr_dim_1,
    const float* __restrict__ longrope_inv_freqs)
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;

    if (pair_idx >= rope_pairs) return;

    const int pos = positions[token_idx];

    float cos_val, sin_val;
    if (longrope_inv_freqs) {
        float freq = longrope_inv_freqs[pair_idx];
        float angle = static_cast<float>(pos) * freq;
        cos_val = __cosf(angle);
        sin_val = __sinf(angle);
    } else if (ext_factor != 0.0f) {
        // YaRN mode: per-dimension frequency blending
        float theta_extrap = static_cast<float>(pos) / powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim));
        rope_yarn(theta_extrap, inv_scaling, corr_dim_0, corr_dim_1,
                  2 * pair_idx, ext_factor, attn_factor, cos_val, sin_val);
    } else {
        // Linear mode: simple frequency * inv_scaling
        float freq = 1.0f / powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim));
        freq *= inv_scaling;
        float angle = static_cast<float>(pos) * freq;
        cos_val = __cosf(angle);
        sin_val = __sinf(angle);
    }

    const int idx0 = neox ? pair_idx : (2 * pair_idx);
    const int idx1 = neox ? (pair_idx + rope_pairs) : (2 * pair_idx + 1);

    if (head_idx < n_heads) {
        int64_t base = static_cast<int64_t>(token_idx) * n_heads * head_dim
                     + static_cast<int64_t>(head_idx) * head_dim;
        float q0 = Q[base + idx0];
        float q1 = Q[base + idx1];
        Q[base + idx0] = q0 * cos_val - q1 * sin_val;
        Q[base + idx1] = q0 * sin_val + q1 * cos_val;
    }

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
// RoPE kernel for FP16 with YaRN support
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
    bool neox,
    float ext_factor,
    float attn_factor,
    float corr_dim_0,
    float corr_dim_1,
    const float* __restrict__ longrope_inv_freqs)
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;

    if (pair_idx >= rope_pairs) return;

    const int pos = positions[token_idx];

    float cos_val, sin_val;
    if (longrope_inv_freqs) {
        float freq = longrope_inv_freqs[pair_idx];
        float angle = static_cast<float>(pos) * freq;
        cos_val = __cosf(angle);
        sin_val = __sinf(angle);
    } else if (ext_factor != 0.0f) {
        float theta_extrap = static_cast<float>(pos) / powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim));
        rope_yarn(theta_extrap, inv_scaling, corr_dim_0, corr_dim_1,
                  2 * pair_idx, ext_factor, attn_factor, cos_val, sin_val);
    } else {
        float freq = 1.0f / powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim));
        freq *= inv_scaling;
        float angle = static_cast<float>(pos) * freq;
        cos_val = __cosf(angle);
        sin_val = __sinf(angle);
    }

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
                  float ext_factor, float attn_factor,
                  const float* corr_dims,
                  cudaStream_t stream,
                  const float* longrope_inv_freqs)
{
    const int batch     = static_cast<int>(Q.shape[0]);
    const int seq_len   = static_cast<int>(Q.shape[1]);
    const int n_heads   = static_cast<int>(Q.shape[2]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);
    const int total_tokens = batch * seq_len;
    const int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;

    if (total_tokens == 0 || head_dim == 0) return;

    const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
    const int pairs = effective_rope_dim / 2;
    const int block_x = (pairs <= 512) ? pairs : 512;

    dim3 grid(total_tokens, max_heads);
    dim3 block(block_x);

    const float inv_scaling = 1.0f / scaling;
    float cd0 = 0.0f, cd1 = 0.0f;
    if (corr_dims) {
        cd0 = corr_dims[0];
        cd1 = corr_dims[1];
    }

    switch (Q.dtype) {
        case DType::FP32:
            pdl::launch(rope_forward_fp32_kernel, grid, block, 0, stream,
                static_cast<float*>(Q.data),
                static_cast<float*>(K.data),
                positions,
                batch, seq_len, n_heads, n_kv_heads, head_dim,
                theta, inv_scaling, pairs, neox,
                ext_factor, attn_factor, cd0, cd1,
                longrope_inv_freqs);
            break;
        case DType::FP16:
            pdl::launch(rope_forward_fp16_kernel, grid, block, 0, stream,
                static_cast<__half*>(Q.data),
                static_cast<__half*>(K.data),
                positions,
                batch, seq_len, n_heads, n_kv_heads, head_dim,
                theta, inv_scaling, pairs, neox,
                ext_factor, attn_factor, cd0, cd1,
                longrope_inv_freqs);
            break;
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// Fused QK-norm + RoPE kernel (decode-only, n=1, FP16) with YaRN support
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
    __half* __restrict__ Q,
    __half* __restrict__ K,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float eps,
    float theta,
    float inv_scaling,
    int rope_pairs,
    bool neox,
    float weight_offset,
    float ext_factor,
    float attn_factor,
    float corr_dim_0,
    float corr_dim_1,
    const float* __restrict__ longrope_inv_freqs)
{
    const int head_idx = blockIdx.x;
    const int pos = positions[0];

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* normed_vals = smem + 8;

    // --- Process Q head ---
    if (head_idx < n_heads) {
        __half* q_head = Q + head_idx * head_dim;

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

        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __half2float(q_head[i]);
            float w = __half2float(q_norm_w[i]) + weight_offset;
            normed_vals[i] = v * inv_rms * w;
        }
        __syncthreads();

        for (int pair = threadIdx.x; pair < rope_pairs; pair += blockDim.x) {
            float cos_val, sin_val;
            if (longrope_inv_freqs) {
                float freq = longrope_inv_freqs[pair];
                float angle = (float)pos * freq;
                cos_val = __cosf(angle);
                sin_val = __sinf(angle);
            } else if (ext_factor != 0.0f) {
                float theta_extrap = (float)pos / powf(theta, (2.0f * pair) / (float)head_dim);
                rope_yarn(theta_extrap, inv_scaling, corr_dim_0, corr_dim_1,
                          2 * pair, ext_factor, attn_factor, cos_val, sin_val);
            } else {
                float freq = 1.0f / powf(theta, (2.0f * pair) / (float)head_dim) * inv_scaling;
                float angle = (float)pos * freq;
                cos_val = __cosf(angle);
                sin_val = __sinf(angle);
            }

            int idx0 = neox ? pair : (2 * pair);
            int idx1 = neox ? (pair + rope_pairs) : (2 * pair + 1);

            float q0 = normed_vals[idx0];
            float q1 = normed_vals[idx1];
            q_head[idx0] = __float2half(q0 * cos_val - q1 * sin_val);
            q_head[idx1] = __float2half(q0 * sin_val + q1 * cos_val);
        }
        int rotated_end = neox ? (2 * rope_pairs) : (2 * rope_pairs);
        for (int i = rotated_end + threadIdx.x; i < head_dim; i += blockDim.x) {
            q_head[i] = __float2half(normed_vals[i]);
        }
    }

    __syncthreads();

    // --- Process K head ---
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
            float cos_val, sin_val;
            if (longrope_inv_freqs) {
                float freq = longrope_inv_freqs[pair];
                float angle = (float)pos * freq;
                cos_val = __cosf(angle);
                sin_val = __sinf(angle);
            } else if (ext_factor != 0.0f) {
                float theta_extrap = (float)pos / powf(theta, (2.0f * pair) / (float)head_dim);
                rope_yarn(theta_extrap, inv_scaling, corr_dim_0, corr_dim_1,
                          2 * pair, ext_factor, attn_factor, cos_val, sin_val);
            } else {
                float freq = 1.0f / powf(theta, (2.0f * pair) / (float)head_dim) * inv_scaling;
                float angle = (float)pos * freq;
                cos_val = __cosf(angle);
                sin_val = __sinf(angle);
            }

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
                        float weight_offset,
                        float ext_factor, float attn_factor,
                        const float* corr_dims,
                        const float* longrope_inv_freqs) {
    const int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;
    if (max_heads == 0 || head_dim == 0) return;

    const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
    const int rope_pairs = effective_rope_dim / 2;
    const float inv_scaling = 1.0f / scaling;

    float cd0 = 0.0f, cd1 = 0.0f;
    if (corr_dims) {
        cd0 = corr_dims[0];
        cd1 = corr_dims[1];
    }

    const int block_size = 128;
    const int smem_bytes = (8 + head_dim) * sizeof(float);

    pdl::launch(qknorm_rope_fused_fp16_kernel, dim3(max_heads), dim3(block_size), smem_bytes, stream,
        reinterpret_cast<__half*>(Q),
        reinterpret_cast<__half*>(K),
        reinterpret_cast<const __half*>(q_norm_weight),
        reinterpret_cast<const __half*>(k_norm_weight),
        positions,
        n_heads, n_kv_heads, head_dim, eps,
        theta, inv_scaling, rope_pairs, neox, weight_offset,
        ext_factor, attn_factor, cd0, cd1,
        longrope_inv_freqs);
}

// --------------------------------------------------------------------------
// YaRN correction dimensions computation (host-side)
// --------------------------------------------------------------------------

static float rope_yarn_corr_dim_impl(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return static_cast<float>(n_dims) * logf(static_cast<float>(n_ctx_orig) / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * logf(base));
}

void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                         float beta_fast, float beta_slow, float dims[2]) {
    float start = floorf(rope_yarn_corr_dim_impl(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(rope_yarn_corr_dim_impl(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf(static_cast<float>(n_dims - 1), end);
}

// --------------------------------------------------------------------------
// PDL registration
// --------------------------------------------------------------------------
void rope_pdl_register() {
    pdl::enable(reinterpret_cast<const void*>(&rope_forward_fp16_kernel));
    pdl::enable(reinterpret_cast<const void*>(&rope_forward_fp32_kernel));
    pdl::enable(reinterpret_cast<const void*>(&qknorm_rope_fused_fp16_kernel));
}

} // namespace imp
