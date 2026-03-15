#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace imp {

// ---------------------------------------------------------------------------
// INT4 Paged Attention Decode Kernel
//
// KV cache stores 2 INT4 values per byte (low nibble = even, high nibble = odd).
// Per-head FP16 scales stored separately. Dequant: val = int4_value * scale.
// ---------------------------------------------------------------------------

// Unpack INT4 nibble to signed integer [-8, 7]
__device__ __forceinline__ int unpack_int4_lo(uint8_t packed) {
    int val = packed & 0xF;
    return (val >= 8) ? (val - 16) : val;  // sign extend 4-bit
}

__device__ __forceinline__ int unpack_int4_hi(uint8_t packed) {
    int val = (packed >> 4) & 0xF;
    return (val >= 8) ? (val - 16) : val;
}

template<int HEAD_DIM>
__global__ void paged_attention_decode_int4_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,   // packed INT4 pairs
    const uint8_t* __restrict__ V_cache,
    const half* __restrict__ K_scales,     // [total_blocks, block_size, n_kv_heads]
    const half* __restrict__ V_scales,
    half* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    int max_context_len,
    int max_num_blocks,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0);
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // Load Q into registers
    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * HEAD_DIM
                          + (int64_t)head_idx * HEAD_DIM;
    float q_reg[ELEMS];
    {
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q_ptr + lane_offset);
        #pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            half2 h2 = Q_ptr2[i];
            q_reg[2*i]   = __half2float(h2.x);
            q_reg[2*i+1] = __half2float(h2.y);
        }
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;  // bytes per head per token (INT4 packed)
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride  = n_kv_heads * kv_head_bytes;
    const int scale_block_stride = block_size * n_kv_heads;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const half* K_sc_block = K_scales + (int64_t)phys_block * scale_block_stride;
        const half* V_sc_block = V_scales + (int64_t)phys_block * scale_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float k_scale = __half2float(K_sc_block[t * n_kv_heads + kv_head]);

            // Q·K dot product: unpack INT4, dequant, multiply with Q
            float dot = 0.0f;
            {
                // Each lane handles ELEMS elements starting at lane_offset
                // INT4: 2 elements per byte, so lane reads ELEMS/2 bytes
                const uint8_t* k_bytes = K_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_bytes[i];
                    float k0 = static_cast<float>(unpack_int4_lo(packed)) * k_scale;
                    float k1 = static_cast<float>(unpack_int4_hi(packed)) * k_scale;
                    dot += q_reg[2*i]   * k0;
                    dot += q_reg[2*i+1] * k1;
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            if (softcap > 0.0f) dot = softcap * tanhf(dot / softcap);

            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            // V accumulation: unpack INT4, dequant, weighted sum
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float v_scale = __half2float(V_sc_block[t * n_kv_heads + kv_head]);
            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_bytes[i];
                    float v0 = static_cast<float>(unpack_int4_lo(packed)) * v_scale;
                    float v1 = static_cast<float>(unpack_int4_hi(packed)) * v_scale;
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * v0;
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * v1;
                }
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    // Cross-warp reduction
    extern __shared__ char smem_int4[];
    float* warp_max = reinterpret_cast<float*>(smem_int4);
    float* warp_l   = warp_max + NUM_WARPS;
    float* warp_o   = warp_l   + NUM_WARPS;

    if (lane_id == 0) {
        warp_max[warp_id] = m_w;
        warp_l[warp_id]   = l_w;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        warp_o[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, warp_max[w]);

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            global_l += expf(warp_max[w] - global_max) * warp_l[w];

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max[w] - global_max) * warp_l[w];
                o_val += weight * warp_o[w * HEAD_DIM + d];
            }
            if (global_l > 0.0f) o_val /= global_l;

            int out_idx = batch_idx * n_heads * HEAD_DIM
                        + head_idx * HEAD_DIM + d;
            O[out_idx] = __float2half(o_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void paged_attention_decode_int4(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const half* K_scales, const half* V_scales,
    const int* block_tables, const int* context_lens,
    int block_size, float scale,
    int max_context_len, int sliding_window,
    float softcap, cudaStream_t stream,
    int max_blocks_per_seq)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq : (max_context_len + block_size - 1) / block_size;

    size_t smem_bytes = NUM_WARPS * sizeof(float)
                      + NUM_WARPS * sizeof(float)
                      + NUM_WARPS * head_dim * sizeof(float);

    dim3 grid(batch_size, n_heads);
    dim3 block(BLOCK_THREADS);

    #define LAUNCH_INT4(HD) \
        paged_attention_decode_int4_kernel<HD><<<grid, block, smem_bytes, stream>>>( \
            reinterpret_cast<const half*>(Q.data), \
            reinterpret_cast<const uint8_t*>(K_cache.data), \
            reinterpret_cast<const uint8_t*>(V_cache.data), \
            K_scales, V_scales, \
            reinterpret_cast<half*>(O.data), \
            block_tables, context_lens, \
            batch_size, n_heads, n_kv_heads, \
            block_size, scale, max_context_len, max_num_blocks, \
            sliding_window, softcap)

    switch (head_dim) {
        case 64:  LAUNCH_INT4(64);  break;
        case 96:  LAUNCH_INT4(96);  break;
        case 128: LAUNCH_INT4(128); break;
        case 256: LAUNCH_INT4(256); break;
        default:
            IMP_LOG_ERROR("paged_attention_decode_int4: unsupported head_dim %d", head_dim);
            return;
    }
    #undef LAUNCH_INT4
}

} // namespace imp
