#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "compute/attention.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>

namespace imp {

// ---------------------------------------------------------------------------
// NVFP4 Paged Attention Decode
//
// KV cache stores 2 FP4 (E2M1) values per byte (low nibble = even, high = odd).
// Per-token-head-group_of_16 UE4M3 (FP8 E4M3) scales stored separately.
//
// Layout per cache block:
//   K_cache : [block, t_in_block, kv_head, head_dim/2]    uint8_t  (packed FP4)
//   V_cache : same shape as K_cache
//   K_scales: [block, t_in_block, kv_head, head_dim/16]   uint8_t  (UE4M3)
//   V_scales: same shape as K_scales
//
// Dequant: val = e2m1_decode(nibble) * ue4m3_decode(scale_byte)
//
// Each lane covers ELEMS = HEAD_DIM/32 contiguous elems. For all imp head_dims
// (64/128/256/512), ELEMS ≤ 16 so a lane covers a slice of exactly ONE
// 16-element scale group. lane scale index = lane_offset / 16.
// ---------------------------------------------------------------------------

// E2M1 4-bit decode: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
__device__ __forceinline__ float e2m1_decode(uint8_t nib) {
    static const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float m = mags[nib & 0x7];
    return (nib & 0x8) ? -m : m;
}

// UE4M3 byte → float (standard FP8 E4M3, sign always 0 in NVFP4 scale role).
__device__ __forceinline__ float ue4m3_decode(uint8_t bits) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &bits, 1);
    return static_cast<float>(v);
}

// ---------------------------------------------------------------------------
// Non-Split-K NVFP4 decode kernel
// ---------------------------------------------------------------------------

template <int HEAD_DIM>
__global__ void paged_attention_decode_nvfp4_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,    // packed FP4 pairs
    const uint8_t* __restrict__ V_cache,    // packed FP4 pairs
    const uint8_t* __restrict__ K_scales,   // UE4M3 per group
    const uint8_t* __restrict__ V_scales,   // UE4M3 per group
    half* __restrict__ O, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, int max_context_len, int max_num_blocks, int sliding_window, float softcap) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be divisible by 16 (NVFP4 group size)");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;
    const int lane_group = lane_offset / 16;  // scale group covering this lane's elems

    // Q into registers
    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * HEAD_DIM + (int64_t)head_idx * HEAD_DIM;
    float q_reg[ELEMS];
    {
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q_ptr + lane_offset);
#pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            half2 h2 = Q_ptr2[i];
            q_reg[2 * i] = __half2float(h2.x);
            q_reg[2 * i + 1] = __half2float(h2.y);
        }
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_block_stride = block_size * n_kv_heads * sc_groups;
    const int sc_slot_stride = n_kv_heads * sc_groups;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* K_sc_block = K_scales + (int64_t)phys_block * sc_block_stride;
        const uint8_t* V_sc_block = V_scales + (int64_t)phys_block * sc_block_stride;

        int tok_start = blk * block_size;
        int tok_end = tok_start + block_size;
        if (tok_end > ctx_len)
            tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;

            // Per-lane scale (one group covers all ELEMS for this lane)
            float k_scale = ue4m3_decode(
                K_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            float v_scale = ue4m3_decode(
                V_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);

            // Q.K dot
            float dot = 0.0f;
            {
                const uint8_t* k_bytes = K_tok + lane_offset / 2;
#pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_bytes[i];
                    float k0 = e2m1_decode(packed & 0xF) * k_scale;
                    float k1 = e2m1_decode((packed >> 4) & 0xF) * k_scale;
                    dot += q_reg[2 * i] * k0;
                    dot += q_reg[2 * i + 1] * k1;
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            // V accumulation
            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
#pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_bytes[i];
                    float v0 = e2m1_decode(packed & 0xF) * v_scale;
                    float v1 = e2m1_decode((packed >> 4) & 0xF) * v_scale;
                    o_reg[2 * i] = rescale * o_reg[2 * i] + w_new * v0;
                    o_reg[2 * i + 1] = rescale * o_reg[2 * i + 1] + w_new * v1;
                }
            }
        }
    }

    extern __shared__ char smem_nvfp4[];
    crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_nvfp4), m_w, l_w, o_reg, warp_id,
                                         lane_id, lane_offset, O, batch_idx, n_heads, head_idx);
}

// ---------------------------------------------------------------------------
// Split-K NVFP4 decode kernel
// ---------------------------------------------------------------------------

template <int HEAD_DIM>
__global__ void paged_attention_splitk_nvfp4_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be divisible by 16 (NVFP4 group size)");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;
    const int lane_group = lane_offset / 16;

    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * HEAD_DIM + (int64_t)head_idx * HEAD_DIM;
    float q_reg[ELEMS];
    {
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q_ptr + lane_offset);
#pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            half2 h2 = Q_ptr2[i];
            q_reg[2 * i] = __half2float(h2.x);
            q_reg[2 * i + 1] = __half2float(h2.y);
        }
    }

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    const int total_blocks = num_ctx_blocks - first_block;

    int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    int split_start = first_block + split_idx * blocks_per_split;
    int split_end = split_start + blocks_per_split;
    if (split_end > num_ctx_blocks)
        split_end = num_ctx_blocks;

    if (split_start >= split_end) {
        write_empty_split_sentinel<HEAD_DIM>(partial_out, batch_idx, n_heads, head_idx, num_splits, split_idx,
                                             lane_offset);
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_block_stride = block_size * n_kv_heads * sc_groups;
    const int sc_slot_stride = n_kv_heads * sc_groups;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* K_sc_block = K_scales + (int64_t)phys_block * sc_block_stride;
        const uint8_t* V_sc_block = V_scales + (int64_t)phys_block * sc_block_stride;

        int tok_start = blk * block_size;
        int tok_end = tok_start + block_size;
        if (tok_end > ctx_len)
            tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;

            float k_scale = ue4m3_decode(
                K_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            float v_scale = ue4m3_decode(
                V_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);

            float dot = 0.0f;
            {
                const uint8_t* k_bytes = K_tok + lane_offset / 2;
#pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_bytes[i];
                    float k0 = e2m1_decode(packed & 0xF) * k_scale;
                    float k1 = e2m1_decode((packed >> 4) & 0xF) * k_scale;
                    dot += q_reg[2 * i] * k0;
                    dot += q_reg[2 * i + 1] * k1;
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
#pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_bytes[i];
                    float v0 = e2m1_decode(packed & 0xF) * v_scale;
                    float v1 = e2m1_decode((packed >> 4) & 0xF) * v_scale;
                    o_reg[2 * i] = rescale * o_reg[2 * i] + w_new * v0;
                    o_reg[2 * i + 1] = rescale * o_reg[2 * i + 1] + w_new * v1;
                }
            }
        }
    }

    extern __shared__ char smem_sk_nvfp4[];
    crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_sk_nvfp4), m_w, l_w, o_reg, warp_id,
                                      lane_id, lane_offset, partial_out, batch_idx, n_heads, head_idx,
                                      num_splits, split_idx);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void paged_attention_decode_nvfp4(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                  const uint8_t* K_scales, const uint8_t* V_scales, const int* block_tables,
                                  const int* context_lens, int block_size, float scale, int max_context_len,
                                  int sliding_window, float softcap, cudaStream_t stream,
                                  int max_blocks_per_seq, int n_sinks) {
    (void)n_sinks;  // streaming not yet wired
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                                                        : (max_context_len + block_size - 1) / block_size;

    size_t smem_bytes = NUM_WARPS * sizeof(float) + NUM_WARPS * sizeof(float) +
                        NUM_WARPS * head_dim * sizeof(float);

    void* scratch_ptr = nullptr;
    int num_splits = compute_splitk_splits(batch_size, n_heads, head_dim, max_context_len, block_size,
                                           &scratch_ptr);

    if (num_splits > 1) {
        float* partial = static_cast<float*>(scratch_ptr);
        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

#define LAUNCH_SPLITK_NVFP4(HD)                                                                            \
    paged_attention_splitk_nvfp4_kernel<HD>                                                                \
        <<<grid1, block1, smem_bytes, stream>>>(reinterpret_cast<const half*>(Q.data),                     \
                                                reinterpret_cast<const uint8_t*>(K_cache.data),            \
                                                reinterpret_cast<const uint8_t*>(V_cache.data), K_scales,  \
                                                V_scales, partial, block_tables, context_lens, batch_size, \
                                                n_heads, n_kv_heads, block_size, scale, max_num_blocks,    \
                                                num_splits, sliding_window, softcap)

        switch (head_dim) {
            case 64:
                LAUNCH_SPLITK_NVFP4(64);
                break;
            case 128:
                LAUNCH_SPLITK_NVFP4(128);
                break;
            case 256:
                LAUNCH_SPLITK_NVFP4(256);
                break;
            case 512:
                LAUNCH_SPLITK_NVFP4(512);
                break;
            default:
                IMP_LOG_ERROR("paged_attention_decode_nvfp4 splitk: unsupported head_dim %d", head_dim);
                return;
        }
#undef LAUNCH_SPLITK_NVFP4

        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data), batch_size, n_heads, head_dim,
                                      num_splits, stream);
    } else {
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

#define LAUNCH_NVFP4(HD)                                                                                     \
    paged_attention_decode_nvfp4_kernel<HD><<<grid, block, smem_bytes, stream>>>(                            \
        reinterpret_cast<const half*>(Q.data), reinterpret_cast<const uint8_t*>(K_cache.data),               \
        reinterpret_cast<const uint8_t*>(V_cache.data), K_scales, V_scales, reinterpret_cast<half*>(O.data), \
        block_tables, context_lens, batch_size, n_heads, n_kv_heads, block_size, scale, max_context_len,     \
        max_num_blocks, sliding_window, softcap)

        switch (head_dim) {
            case 64:
                LAUNCH_NVFP4(64);
                break;
            case 128:
                LAUNCH_NVFP4(128);
                break;
            case 256:
                LAUNCH_NVFP4(256);
                break;
            case 512:
                LAUNCH_NVFP4(512);
                break;
            default:
                IMP_LOG_ERROR("paged_attention_decode_nvfp4: unsupported head_dim %d", head_dim);
                return;
        }
#undef LAUNCH_NVFP4
    }
}

}  // namespace imp
