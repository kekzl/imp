#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "compute/attention.h"
#include "quant/turboquant_fp4.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace imp {

// ---------------------------------------------------------------------------
// TurboQuant Paged Attention Decode Kernel
//
// K cache: PolarQuant INT4 directions + FP16 norms + QJL 1-bit sketches
// V cache: Standard INT4 with per-head FP16 scales
//
// Q.K estimation:
//   1. PolarQuant: qk_polar = ||k|| * (q . dir_k_quantized)
//      where dir_k_quantized is INT4-dequantized unit vector
//   2. QJL correction: qk_qjl = ||q|| * ||k|| * (2*popcount(XNOR(sketch_q, sketch_k)) - sketch_dim) / sketch_dim
//   3. Combined: qk = (1 - lambda) * qk_polar + lambda * qk_qjl
//      lambda = 0.1 (small correction weight from paper)
//
// V accumulation: same as INT4 kernel (unpack, dequant with per-head scale)
// ---------------------------------------------------------------------------

// Unpack INT4 nibble to signed integer [-8, 7]
__device__ __forceinline__ int tq_unpack_int4_lo(uint8_t packed) {
    int val = packed & 0xF;
    return (val >= 8) ? (val - 16) : val;
}

__device__ __forceinline__ int tq_unpack_int4_hi(uint8_t packed) {
    int val = (packed >> 4) & 0xF;
    return (val >= 8) ? (val - 16) : val;
}

// QJL correction weight (from TurboQuant paper)
static constexpr float kQJLLambda = 0.1f;

template<int HEAD_DIM, bool USE_MXFP4 = false>
__global__ void paged_attention_decode_turboquant_kernel(
    const half* __restrict__ Q,                // [batch, n_heads, HEAD_DIM]
    const uint8_t* __restrict__ K_dir_cache,   // INT4 or FP4 E2M1 packed normalized directions
    const uint8_t* __restrict__ V_cache,       // INT4 packed values
    const half* __restrict__ K_norms,          // [total_blocks, block_size, n_kv_heads] FP16 norms
    const half* __restrict__ V_scales,         // [total_blocks, block_size, n_kv_heads] FP16 scales
    const uint8_t* __restrict__ K_sketches,    // [total_blocks, block_size, n_kv_heads, sketch_dim/8] packed bits
    const uint8_t* __restrict__ qjl_matrix,    // [sketch_dim, head_dim/8] packed Rademacher signs
    const uint8_t* __restrict__ K_mscales,     // MXFP4: UE8M0 micro-scales [blocks, block_size, n_kv_heads, hd/32] (nullptr if !USE_MXFP4)
    half* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    int sketch_dim,
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

    // Compute Q norm for QJL correction
    float q_norm_sq = 0.0f;
    for (int i = 0; i < ELEMS; i++) q_norm_sq += q_reg[i] * q_reg[i];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        q_norm_sq += __shfl_xor_sync(0xFFFFFFFF, q_norm_sq, offset);
    float q_norm = sqrtf(q_norm_sq);

    // Compute Q's QJL sketch: sign(R @ q) for each sketch row
    // Each lane stores a portion of the sketch in registers.
    // sketch_dim <= HEAD_DIM typically, and we store as packed uint32 chunks.
    // We'll compute the sketch in shared memory for broadcasting.
    extern __shared__ char smem_tq[];
    // Layout: [sketch_dim/8 bytes for Q sketch] [warp_max + warp_l + warp_o]
    const int sketch_bytes = sketch_dim / 8;
    uint8_t* q_sketch = reinterpret_cast<uint8_t*>(smem_tq);
    float* warp_max_ptr = reinterpret_cast<float*>(q_sketch + ((sketch_bytes + 3) & ~3));  // align to 4
    float* warp_l_ptr   = warp_max_ptr + NUM_WARPS;
    float* warp_o_ptr   = warp_l_ptr + NUM_WARPS;

    // Initialize Q sketch to zero
    for (int i = threadIdx.x; i < sketch_bytes; i += blockDim.x)
        q_sketch[i] = 0;
    __syncthreads();

    // Compute Q sketch: each thread handles a subset of sketch rows
    {
        const int bytes_per_qjl_row = HEAD_DIM / 8;
        for (int sr = threadIdx.x; sr < sketch_dim; sr += blockDim.x) {
            const uint8_t* R_row = qjl_matrix + sr * bytes_per_qjl_row;

            // Compute dot product R[sr] . Q (full head dim)
            // Each thread needs all Q elements - broadcast from lane 0 via shfl
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                // Get Q[d] from the lane that owns it
                int owning_lane = d / ELEMS;
                int local_idx = d % ELEMS;
                float q_val = __shfl_sync(0xFFFFFFFF, q_reg[local_idx], owning_lane);

                // Get R sign bit
                uint8_t r_byte = __ldg(&R_row[d / 8]);
                float r_sign = (r_byte & (1u << (d % 8))) ? 1.0f : -1.0f;
                dot += r_sign * q_val;
            }

            // Store sign bit atomically
            int byte_idx = sr / 8;
            int bit_idx = sr % 8;
            if (dot >= 0.0f) {
                atomicOr(reinterpret_cast<unsigned int*>(&q_sketch[byte_idx & ~3]),
                         static_cast<unsigned int>(1u << bit_idx) << (8 * (byte_idx & 3)));
            }
        }
    }
    __syncthreads();

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;  // bytes per head per token (INT4/FP4 packed)
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride  = n_kv_heads * kv_head_bytes;
    const int scale_block_stride = block_size * n_kv_heads;
    const int sketch_head_bytes = sketch_dim / 8;
    const int sketch_slot_stride = n_kv_heads * sketch_head_bytes;
    const int sketch_kv_block_stride = block_size * sketch_slot_stride;

    // MXFP4 micro-scale strides (only used when USE_MXFP4 == true)
    constexpr int N_GROUPS = HEAD_DIM / 32;
    const int mscale_slot_stride = USE_MXFP4 ? (n_kv_heads * N_GROUPS) : 0;
    const int mscale_block_stride_v = USE_MXFP4 ? (block_size * mscale_slot_stride) : 0;

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
        const uint8_t* K_dir_block = K_dir_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block     = V_cache     + (int64_t)phys_block * kv_block_stride;
        const half* K_norm_block   = K_norms     + (int64_t)phys_block * scale_block_stride;
        const half* V_sc_block     = V_scales    + (int64_t)phys_block * scale_block_stride;
        const uint8_t* K_sk_block  = K_sketches  + (int64_t)phys_block * sketch_kv_block_stride;
        // MXFP4 micro-scale block pointer (only valid when USE_MXFP4)
        const uint8_t* K_ms_block  = USE_MXFP4
            ? (K_mscales + (int64_t)phys_block * mscale_block_stride_v)
            : nullptr;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_dir_tok = K_dir_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float k_norm = __half2float(K_norm_block[t * n_kv_heads + kv_head]);

            // PolarQuant Q.K: q . (norm * dir_quantized) = norm * q . dir_quantized
            float dot_polar = 0.0f;
            if constexpr (USE_MXFP4) {
                // MXFP4 path: FP4 E2M1 nibbles + per-32-element UE8M0 micro-scales
                const uint8_t* k_bytes = K_dir_tok + lane_offset / 2;
                const uint8_t* k_ms = K_ms_block + t * mscale_slot_stride + kv_head * N_GROUPS;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_bytes[i];
                    int elem_idx = lane_offset + 2 * i;
                    int group_idx = elem_idx / 32;
                    float ms = tq_fp4_ue8m0_to_float(k_ms[group_idx]);
                    float d0 = tq_fp4_unpack_lo(packed) * ms;
                    float d1 = tq_fp4_unpack_hi(packed) * ms;
                    dot_polar += q_reg[2*i]   * d0;
                    dot_polar += q_reg[2*i+1] * d1;
                }
            } else {
                // INT4 uniform path: signed INT4 / 7.0
                const uint8_t* k_bytes = K_dir_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_bytes[i];
                    float d0 = static_cast<float>(tq_unpack_int4_lo(packed)) / 7.0f;
                    float d1 = static_cast<float>(tq_unpack_int4_hi(packed)) / 7.0f;
                    dot_polar += q_reg[2*i]   * d0;
                    dot_polar += q_reg[2*i+1] * d1;
                }
            }
            dot_polar = warp_reduce_sum(dot_polar);
            dot_polar *= k_norm;

            // QJL correction: compare Q sketch with K sketch
            float dot_qjl = 0.0f;
            if (lane_id == 0) {
                const uint8_t* k_sketch = K_sk_block + t * sketch_slot_stride + kv_head * sketch_head_bytes;
                int match_count = 0;
                // XNOR + popcount over sketch_dim bits
                for (int sb = 0; sb < sketch_bytes / 4; sb++) {
                    uint32_t q_word = reinterpret_cast<const uint32_t*>(q_sketch)[sb];
                    uint32_t k_word;
                    // Safely load potentially unaligned K sketch
                    memcpy(&k_word, k_sketch + sb * 4, sizeof(uint32_t));
                    uint32_t xnor = ~(q_word ^ k_word);
                    match_count += __popc(xnor);
                }
                // Handle remaining bytes
                for (int sb = (sketch_bytes / 4) * 4; sb < sketch_bytes; sb++) {
                    uint8_t xnor = ~(q_sketch[sb] ^ k_sketch[sb]);
                    match_count += __popc(static_cast<unsigned int>(xnor) & 0xFF);
                }
                // QJL estimator: ||q|| * ||k|| * (2*matches - sketch_dim) / sketch_dim
                dot_qjl = q_norm * k_norm * static_cast<float>(2 * match_count - sketch_dim)
                          / static_cast<float>(sketch_dim);
            }
            dot_qjl = __shfl_sync(0xFFFFFFFF, dot_qjl, 0);

            // Combined estimate with QJL correction
            float dot = (1.0f - kQJLLambda) * dot_polar + kQJLLambda * dot_qjl;

            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            // V accumulation: standard INT4 path
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float v_scale = __half2float(V_sc_block[t * n_kv_heads + kv_head]);
            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_bytes[i];
                    float v0 = static_cast<float>(tq_unpack_int4_lo(packed)) * v_scale;
                    float v1 = static_cast<float>(tq_unpack_int4_hi(packed)) * v_scale;
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * v0;
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * v1;
                }
            }
        }
    }

    // Cross-warp reduction
    if (lane_id == 0) {
        warp_max_ptr[warp_id] = m_w;
        warp_l_ptr[warp_id]   = l_w;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        warp_o_ptr[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, warp_max_ptr[w]);

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            global_l += expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];
                o_val += weight * warp_o_ptr[w * HEAD_DIM + d];
            }
            if (global_l > 0.0f) o_val /= global_l;

            int out_idx = batch_idx * n_heads * HEAD_DIM
                        + head_idx * HEAD_DIM + d;
            O[out_idx] = __float2half(o_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K Phase 1: TurboQuant variant
// ---------------------------------------------------------------------------

template<int HEAD_DIM, bool USE_MXFP4 = false>
__global__ void paged_attention_splitk_turboquant_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_dir_cache,
    const uint8_t* __restrict__ V_cache,
    const half* __restrict__ K_norms,
    const half* __restrict__ V_scales,
    const uint8_t* __restrict__ K_sketches,
    const uint8_t* __restrict__ qjl_matrix,
    const uint8_t* __restrict__ K_mscales,     // MXFP4 UE8M0 micro-scales (nullptr if !USE_MXFP4)
    float* __restrict__ partial_out,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    int sketch_dim,
    int max_num_blocks,
    int num_splits,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0);
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
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

    // Q norm for QJL
    float q_norm_sq = 0.0f;
    for (int i = 0; i < ELEMS; i++) q_norm_sq += q_reg[i] * q_reg[i];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        q_norm_sq += __shfl_xor_sync(0xFFFFFFFF, q_norm_sq, offset);
    float q_norm = sqrtf(q_norm_sq);

    // Q sketch in shared memory
    extern __shared__ char smem_tq_sk[];
    const int sketch_bytes = sketch_dim / 8;
    uint8_t* q_sketch = reinterpret_cast<uint8_t*>(smem_tq_sk);
    float* warp_max_ptr = reinterpret_cast<float*>(q_sketch + ((sketch_bytes + 3) & ~3));
    float* warp_l_ptr   = warp_max_ptr + NUM_WARPS;
    float* warp_o_ptr   = warp_l_ptr + NUM_WARPS;

    for (int i = threadIdx.x; i < sketch_bytes; i += blockDim.x)
        q_sketch[i] = 0;
    __syncthreads();

    // Compute Q sketch
    {
        const int bytes_per_qjl_row = HEAD_DIM / 8;
        for (int sr = threadIdx.x; sr < sketch_dim; sr += blockDim.x) {
            const uint8_t* R_row = qjl_matrix + sr * bytes_per_qjl_row;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                int owning_lane = d / ELEMS;
                int local_idx = d % ELEMS;
                float q_val = __shfl_sync(0xFFFFFFFF, q_reg[local_idx], owning_lane);
                uint8_t r_byte = __ldg(&R_row[d / 8]);
                float r_sign = (r_byte & (1u << (d % 8))) ? 1.0f : -1.0f;
                dot += r_sign * q_val;
            }
            int byte_idx = sr / 8;
            int bit_idx = sr % 8;
            if (dot >= 0.0f) {
                atomicOr(reinterpret_cast<unsigned int*>(&q_sketch[byte_idx & ~3]),
                         static_cast<unsigned int>(1u << bit_idx) << (8 * (byte_idx & 3)));
            }
        }
    }
    __syncthreads();

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride  = n_kv_heads * kv_head_bytes;
    const int scale_block_stride = block_size * n_kv_heads;
    const int sketch_head_bytes = sketch_dim / 8;
    const int sketch_slot_stride = n_kv_heads * sketch_head_bytes;
    const int sketch_kv_block_stride = block_size * sketch_slot_stride;

    constexpr int N_GROUPS_SK = HEAD_DIM / 32;
    const int mscale_slot_stride_sk = USE_MXFP4 ? (n_kv_heads * N_GROUPS_SK) : 0;
    const int mscale_block_stride_sk = USE_MXFP4 ? (block_size * mscale_slot_stride_sk) : 0;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    // Split-K: this split handles blocks [split_start, split_end)
    int blocks_per_split = (num_ctx_blocks - first_block + num_splits - 1) / num_splits;
    int split_start = first_block + split_idx * blocks_per_split;
    int split_end   = min(split_start + blocks_per_split, num_ctx_blocks);

    if (split_start >= num_ctx_blocks) {
        write_empty_split_sentinel<HEAD_DIM>(partial_out, batch_idx, n_heads, head_idx,
                                              num_splits, split_idx, lane_offset);
        return;
    }

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_dir_block = K_dir_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block     = V_cache     + (int64_t)phys_block * kv_block_stride;
        const half* K_norm_block   = K_norms     + (int64_t)phys_block * scale_block_stride;
        const half* V_sc_block     = V_scales    + (int64_t)phys_block * scale_block_stride;
        const uint8_t* K_sk_block  = K_sketches  + (int64_t)phys_block * sketch_kv_block_stride;
        const uint8_t* K_ms_block_sk = USE_MXFP4
            ? (K_mscales + (int64_t)phys_block * mscale_block_stride_sk)
            : nullptr;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_dir_tok = K_dir_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float k_norm = __half2float(K_norm_block[t * n_kv_heads + kv_head]);

            // PolarQuant dot product
            float dot_polar = 0.0f;
            if constexpr (USE_MXFP4) {
                const uint8_t* k_bytes = K_dir_tok + lane_offset / 2;
                const uint8_t* k_ms = K_ms_block_sk + t * mscale_slot_stride_sk + kv_head * N_GROUPS_SK;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_bytes[i];
                    int elem_idx = lane_offset + 2 * i;
                    int group_idx = elem_idx / 32;
                    float ms = tq_fp4_ue8m0_to_float(k_ms[group_idx]);
                    float d0 = tq_fp4_unpack_lo(packed) * ms;
                    float d1 = tq_fp4_unpack_hi(packed) * ms;
                    dot_polar += q_reg[2*i]   * d0;
                    dot_polar += q_reg[2*i+1] * d1;
                }
            } else {
                const uint8_t* k_bytes = K_dir_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_bytes[i];
                    float d0 = static_cast<float>(tq_unpack_int4_lo(packed)) / 7.0f;
                    float d1 = static_cast<float>(tq_unpack_int4_hi(packed)) / 7.0f;
                    dot_polar += q_reg[2*i]   * d0;
                    dot_polar += q_reg[2*i+1] * d1;
                }
            }
            dot_polar = warp_reduce_sum(dot_polar);
            dot_polar *= k_norm;

            // QJL correction
            float dot_qjl = 0.0f;
            if (lane_id == 0) {
                const uint8_t* k_sketch = K_sk_block + t * sketch_slot_stride + kv_head * sketch_head_bytes;
                int match_count = 0;
                for (int sb = 0; sb < sketch_bytes / 4; sb++) {
                    uint32_t q_word = reinterpret_cast<const uint32_t*>(q_sketch)[sb];
                    uint32_t k_word;
                    memcpy(&k_word, k_sketch + sb * 4, sizeof(uint32_t));
                    match_count += __popc(~(q_word ^ k_word));
                }
                for (int sb = (sketch_bytes / 4) * 4; sb < sketch_bytes; sb++) {
                    uint8_t xnor = ~(q_sketch[sb] ^ k_sketch[sb]);
                    match_count += __popc(static_cast<unsigned int>(xnor) & 0xFF);
                }
                dot_qjl = q_norm * k_norm * static_cast<float>(2 * match_count - sketch_dim)
                          / static_cast<float>(sketch_dim);
            }
            dot_qjl = __shfl_sync(0xFFFFFFFF, dot_qjl, 0);

            float dot = (1.0f - kQJLLambda) * dot_polar + kQJLLambda * dot_qjl;
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            // V accumulation
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float v_scale_f = __half2float(V_sc_block[t * n_kv_heads + kv_head]);
            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_bytes[i];
                    float v0 = static_cast<float>(tq_unpack_int4_lo(packed)) * v_scale_f;
                    float v1 = static_cast<float>(tq_unpack_int4_hi(packed)) * v_scale_f;
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * v0;
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * v1;
                }
            }
        }
    }

    // Cross-warp reduction → write partial for split-K reduce
    __syncthreads();
    if (lane_id == 0) {
        warp_max_ptr[warp_id] = m_w;
        warp_l_ptr[warp_id]   = l_w;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        warp_o_ptr[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, warp_max_ptr[w]);
        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            global_l += expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];

        int partial_idx = ((batch_idx * n_heads + head_idx) * num_splits + split_idx);
        constexpr int partial_stride = 2 + HEAD_DIM;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;

        if (lane_id == 0) { out[0] = global_max; out[1] = global_l; }

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];
                o_val += weight * warp_o_ptr[w * HEAD_DIM + d];
            }
            out[2 + d] = o_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher -- TurboQuant variant (with Split-K support)
// ---------------------------------------------------------------------------
void paged_attention_decode_turboquant(
    const Tensor& Q, const Tensor& K_dir_cache, const Tensor& V_cache,
    Tensor& O,
    const half* K_norms, const half* V_scales,
    const uint8_t* K_sketches, const uint8_t* qjl_matrix,
    const int* block_tables, const int* context_lens,
    int block_size, float scale, int sketch_dim,
    int max_context_len, int sliding_window,
    float softcap, cudaStream_t stream,
    int max_blocks_per_seq,
    const uint8_t* K_mscales)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_dir_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                               : (max_context_len + block_size - 1) / block_size;

    const int sketch_bytes = sketch_dim / 8;
    const int sketch_aligned = (sketch_bytes + 3) & ~3;
    size_t smem_bytes = sketch_aligned
                      + NUM_WARPS * sizeof(float)
                      + NUM_WARPS * sizeof(float)
                      + NUM_WARPS * head_dim * sizeof(float);

    void* scratch_ptr = nullptr;
    int num_splits = compute_splitk_splits(
        batch_size, n_heads, head_dim, max_context_len, block_size, &scratch_ptr);

    const bool use_mxfp4 = (K_mscales != nullptr);

    if (num_splits > 1) {
        float* partial = static_cast<float*>(scratch_ptr);
        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        // Macro for dispatching split-K kernel with optional MXFP4 template
        #define LAUNCH_SPLITK_TQ(HD, MX) \
            paged_attention_splitk_turboquant_kernel<HD, MX><<<grid1, block1, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const uint8_t*>(K_dir_cache.data), \
                reinterpret_cast<const uint8_t*>(V_cache.data), \
                K_norms, V_scales, \
                K_sketches, qjl_matrix, K_mscales, \
                partial, \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, sketch_dim, \
                max_num_blocks, num_splits, \
                sliding_window, softcap)

        #define DISPATCH_SPLITK_TQ(HD) \
            if (use_mxfp4) { LAUNCH_SPLITK_TQ(HD, true); } \
            else            { LAUNCH_SPLITK_TQ(HD, false); }

        switch (head_dim) {
            case 64:  DISPATCH_SPLITK_TQ(64);  break;
            case 96:  DISPATCH_SPLITK_TQ(96);  break;
            case 128: DISPATCH_SPLITK_TQ(128); break;
            case 256: DISPATCH_SPLITK_TQ(256); break;
            default:
                IMP_LOG_ERROR("paged_attention_splitk_turboquant: unsupported head_dim %d", head_dim);
                return;
        }
        #undef DISPATCH_SPLITK_TQ
        #undef LAUNCH_SPLITK_TQ

        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data),
                                      batch_size, n_heads, head_dim, num_splits, stream);
    } else {
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

        #define LAUNCH_TQ(HD, MX) \
            paged_attention_decode_turboquant_kernel<HD, MX><<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const uint8_t*>(K_dir_cache.data), \
                reinterpret_cast<const uint8_t*>(V_cache.data), \
                K_norms, V_scales, \
                K_sketches, qjl_matrix, K_mscales, \
                reinterpret_cast<half*>(O.data), \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, sketch_dim, max_context_len, max_num_blocks, \
                sliding_window, softcap)

        #define DISPATCH_TQ(HD) \
            if (use_mxfp4) { LAUNCH_TQ(HD, true); } \
            else            { LAUNCH_TQ(HD, false); }

        switch (head_dim) {
            case 64:  DISPATCH_TQ(64);  break;
            case 96:  DISPATCH_TQ(96);  break;
            case 128: DISPATCH_TQ(128); break;
            case 256: DISPATCH_TQ(256); break;
            default:
                IMP_LOG_ERROR("paged_attention_decode_turboquant: unsupported head_dim %d", head_dim);
                return;
        }
        #undef DISPATCH_TQ
        #undef LAUNCH_TQ
    }
}

// ===========================================================================
// TurboQuant Lite: QJL sketch-only K + INT4 V
//
// Q.K is estimated purely via QJL (no PolarQuant component):
//   qk = ||q|| * ||k|| * (2*popcount(XNOR(sketch_q, sketch_k)) - sketch_dim) / sketch_dim
//
// V accumulation: same INT4 dequant as standard TurboQuant/INT4 path.
// ===========================================================================

template<int HEAD_DIM>
__global__ void paged_attention_decode_turboquant_lite_kernel(
    const half* __restrict__ Q,                // [batch, n_heads, HEAD_DIM]
    const uint8_t* __restrict__ V_cache,       // INT4 packed values
    const half* __restrict__ K_norms,          // [total_blocks, block_size, n_kv_heads] FP16 norms
    const half* __restrict__ V_scales,         // [total_blocks, block_size, n_kv_heads] FP16 scales
    const uint8_t* __restrict__ K_sketches,    // [total_blocks, block_size, n_kv_heads, sketch_dim/8]
    const uint8_t* __restrict__ qjl_matrix,    // [sketch_dim, head_dim/8] packed Rademacher signs
    half* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    int sketch_dim,
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

    // Compute Q norm for QJL
    float q_norm_sq = 0.0f;
    for (int i = 0; i < ELEMS; i++) q_norm_sq += q_reg[i] * q_reg[i];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        q_norm_sq += __shfl_xor_sync(0xFFFFFFFF, q_norm_sq, offset);
    float q_norm = sqrtf(q_norm_sq);

    // Compute Q's QJL sketch in shared memory
    extern __shared__ char smem_tql[];
    const int sketch_bytes = sketch_dim / 8;
    uint8_t* q_sketch = reinterpret_cast<uint8_t*>(smem_tql);
    float* warp_max_ptr = reinterpret_cast<float*>(q_sketch + ((sketch_bytes + 3) & ~3));
    float* warp_l_ptr   = warp_max_ptr + NUM_WARPS;
    float* warp_o_ptr   = warp_l_ptr + NUM_WARPS;

    for (int i = threadIdx.x; i < sketch_bytes; i += blockDim.x)
        q_sketch[i] = 0;
    __syncthreads();

    // Compute Q sketch
    {
        const int bytes_per_qjl_row = HEAD_DIM / 8;
        for (int sr = threadIdx.x; sr < sketch_dim; sr += blockDim.x) {
            const uint8_t* R_row = qjl_matrix + sr * bytes_per_qjl_row;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                int owning_lane = d / ELEMS;
                int local_idx = d % ELEMS;
                float q_val = __shfl_sync(0xFFFFFFFF, q_reg[local_idx], owning_lane);
                uint8_t r_byte = __ldg(&R_row[d / 8]);
                float r_sign = (r_byte & (1u << (d % 8))) ? 1.0f : -1.0f;
                dot += r_sign * q_val;
            }
            int byte_idx = sr / 8;
            int bit_idx = sr % 8;
            if (dot >= 0.0f) {
                atomicOr(reinterpret_cast<unsigned int*>(&q_sketch[byte_idx & ~3]),
                         static_cast<unsigned int>(1u << bit_idx) << (8 * (byte_idx & 3)));
            }
        }
    }
    __syncthreads();

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;  // V: INT4 packed bytes per head per token
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride  = n_kv_heads * kv_head_bytes;
    const int scale_block_stride = block_size * n_kv_heads;
    const int sketch_head_bytes = sketch_dim / 8;
    const int sketch_slot_stride = n_kv_heads * sketch_head_bytes;
    const int sketch_kv_block_stride = block_size * sketch_slot_stride;

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
        const uint8_t* V_block     = V_cache     + (int64_t)phys_block * kv_block_stride;
        const half* K_norm_block   = K_norms     + (int64_t)phys_block * scale_block_stride;
        const half* V_sc_block     = V_scales    + (int64_t)phys_block * scale_block_stride;
        const uint8_t* K_sk_block  = K_sketches  + (int64_t)phys_block * sketch_kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            float k_norm = __half2float(K_norm_block[t * n_kv_heads + kv_head]);

            // Pure QJL dot product estimation
            float dot_qjl = 0.0f;
            if (lane_id == 0) {
                const uint8_t* k_sketch = K_sk_block + t * sketch_slot_stride + kv_head * sketch_head_bytes;
                int match_count = 0;
                for (int sb = 0; sb < sketch_bytes / 4; sb++) {
                    uint32_t q_word = reinterpret_cast<const uint32_t*>(q_sketch)[sb];
                    uint32_t k_word;
                    memcpy(&k_word, k_sketch + sb * 4, sizeof(uint32_t));
                    uint32_t xnor = ~(q_word ^ k_word);
                    match_count += __popc(xnor);
                }
                for (int sb = (sketch_bytes / 4) * 4; sb < sketch_bytes; sb++) {
                    uint8_t xnor = ~(q_sketch[sb] ^ k_sketch[sb]);
                    match_count += __popc(static_cast<unsigned int>(xnor) & 0xFF);
                }
                dot_qjl = q_norm * k_norm * static_cast<float>(2 * match_count - sketch_dim)
                          / static_cast<float>(sketch_dim);
            }
            dot_qjl = __shfl_sync(0xFFFFFFFF, dot_qjl, 0);

            float dot = dot_qjl * scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            // V accumulation: standard INT4 path
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float v_scale_f = __half2float(V_sc_block[t * n_kv_heads + kv_head]);
            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_bytes[i];
                    float v0 = static_cast<float>(tq_unpack_int4_lo(packed)) * v_scale_f;
                    float v1 = static_cast<float>(tq_unpack_int4_hi(packed)) * v_scale_f;
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * v0;
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * v1;
                }
            }
        }
    }

    // Cross-warp reduction
    if (lane_id == 0) {
        warp_max_ptr[warp_id] = m_w;
        warp_l_ptr[warp_id]   = l_w;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        warp_o_ptr[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, warp_max_ptr[w]);

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            global_l += expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];
                o_val += weight * warp_o_ptr[w * HEAD_DIM + d];
            }
            if (global_l > 0.0f) o_val /= global_l;

            int out_idx = batch_idx * n_heads * HEAD_DIM
                        + head_idx * HEAD_DIM + d;
            O[out_idx] = __float2half(o_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K Phase 1: TurboQuant Lite variant
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_splitk_turboquant_lite_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ V_cache,
    const half* __restrict__ K_norms,
    const half* __restrict__ V_scales,
    const uint8_t* __restrict__ K_sketches,
    const uint8_t* __restrict__ qjl_matrix,
    float* __restrict__ partial_out,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    int sketch_dim,
    int max_num_blocks,
    int num_splits,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0);
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // Load Q
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

    // Q norm
    float q_norm_sq = 0.0f;
    for (int i = 0; i < ELEMS; i++) q_norm_sq += q_reg[i] * q_reg[i];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        q_norm_sq += __shfl_xor_sync(0xFFFFFFFF, q_norm_sq, offset);
    float q_norm = sqrtf(q_norm_sq);

    // Q sketch
    extern __shared__ char smem_tql_sk[];
    const int sketch_bytes = sketch_dim / 8;
    uint8_t* q_sketch = reinterpret_cast<uint8_t*>(smem_tql_sk);
    float* warp_max_ptr = reinterpret_cast<float*>(q_sketch + ((sketch_bytes + 3) & ~3));
    float* warp_l_ptr   = warp_max_ptr + NUM_WARPS;
    float* warp_o_ptr   = warp_l_ptr + NUM_WARPS;

    for (int i = threadIdx.x; i < sketch_bytes; i += blockDim.x)
        q_sketch[i] = 0;
    __syncthreads();

    {
        const int bytes_per_qjl_row = HEAD_DIM / 8;
        for (int sr = threadIdx.x; sr < sketch_dim; sr += blockDim.x) {
            const uint8_t* R_row = qjl_matrix + sr * bytes_per_qjl_row;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                int owning_lane = d / ELEMS;
                int local_idx = d % ELEMS;
                float q_val = __shfl_sync(0xFFFFFFFF, q_reg[local_idx], owning_lane);
                uint8_t r_byte = __ldg(&R_row[d / 8]);
                float r_sign = (r_byte & (1u << (d % 8))) ? 1.0f : -1.0f;
                dot += r_sign * q_val;
            }
            int byte_idx = sr / 8;
            int bit_idx = sr % 8;
            if (dot >= 0.0f) {
                atomicOr(reinterpret_cast<unsigned int*>(&q_sketch[byte_idx & ~3]),
                         static_cast<unsigned int>(1u << bit_idx) << (8 * (byte_idx & 3)));
            }
        }
    }
    __syncthreads();

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride  = n_kv_heads * kv_head_bytes;
    const int scale_block_stride = block_size * n_kv_heads;
    const int sketch_head_bytes = sketch_dim / 8;
    const int sketch_slot_stride = n_kv_heads * sketch_head_bytes;
    const int sketch_kv_block_stride = block_size * sketch_slot_stride;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    int blocks_per_split = (num_ctx_blocks - first_block + num_splits - 1) / num_splits;
    int split_start = first_block + split_idx * blocks_per_split;
    int split_end   = min(split_start + blocks_per_split, num_ctx_blocks);

    if (split_start >= num_ctx_blocks) {
        write_empty_split_sentinel<HEAD_DIM>(partial_out, batch_idx, n_heads, head_idx,
                                              num_splits, split_idx, lane_offset);
        return;
    }

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* V_block     = V_cache     + (int64_t)phys_block * kv_block_stride;
        const half* K_norm_block   = K_norms     + (int64_t)phys_block * scale_block_stride;
        const half* V_sc_block     = V_scales    + (int64_t)phys_block * scale_block_stride;
        const uint8_t* K_sk_block  = K_sketches  + (int64_t)phys_block * sketch_kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            float k_norm = __half2float(K_norm_block[t * n_kv_heads + kv_head]);

            // Pure QJL dot product
            float dot_qjl = 0.0f;
            if (lane_id == 0) {
                const uint8_t* k_sketch = K_sk_block + t * sketch_slot_stride + kv_head * sketch_head_bytes;
                int match_count = 0;
                for (int sb = 0; sb < sketch_bytes / 4; sb++) {
                    uint32_t q_word = reinterpret_cast<const uint32_t*>(q_sketch)[sb];
                    uint32_t k_word;
                    memcpy(&k_word, k_sketch + sb * 4, sizeof(uint32_t));
                    match_count += __popc(~(q_word ^ k_word));
                }
                for (int sb = (sketch_bytes / 4) * 4; sb < sketch_bytes; sb++) {
                    uint8_t xnor = ~(q_sketch[sb] ^ k_sketch[sb]);
                    match_count += __popc(static_cast<unsigned int>(xnor) & 0xFF);
                }
                dot_qjl = q_norm * k_norm * static_cast<float>(2 * match_count - sketch_dim)
                          / static_cast<float>(sketch_dim);
            }
            dot_qjl = __shfl_sync(0xFFFFFFFF, dot_qjl, 0);

            float dot = dot_qjl * scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            // V accumulation
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float v_scale_f = __half2float(V_sc_block[t * n_kv_heads + kv_head]);
            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_bytes[i];
                    float v0 = static_cast<float>(tq_unpack_int4_lo(packed)) * v_scale_f;
                    float v1 = static_cast<float>(tq_unpack_int4_hi(packed)) * v_scale_f;
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * v0;
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * v1;
                }
            }
        }
    }

    // Cross-warp reduction → write partial
    __syncthreads();
    if (lane_id == 0) {
        warp_max_ptr[warp_id] = m_w;
        warp_l_ptr[warp_id]   = l_w;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        warp_o_ptr[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, warp_max_ptr[w]);
        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            global_l += expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];

        int partial_idx = ((batch_idx * n_heads + head_idx) * num_splits + split_idx);
        constexpr int partial_stride = 2 + HEAD_DIM;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;

        if (lane_id == 0) { out[0] = global_max; out[1] = global_l; }

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max_ptr[w] - global_max) * warp_l_ptr[w];
                o_val += weight * warp_o_ptr[w * HEAD_DIM + d];
            }
            out[2 + d] = o_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher -- TurboQuant Lite variant (with Split-K support)
// ---------------------------------------------------------------------------
void paged_attention_decode_turboquant_lite(
    const Tensor& Q, const Tensor& V_cache,
    Tensor& O,
    const half* K_norms, const half* V_scales,
    const uint8_t* K_sketches, const uint8_t* qjl_matrix,
    const int* block_tables, const int* context_lens,
    int block_size, float scale, int sketch_dim,
    int max_context_len, int sliding_window,
    float softcap, cudaStream_t stream,
    int max_blocks_per_seq)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(V_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                               : (max_context_len + block_size - 1) / block_size;

    const int sketch_bytes = sketch_dim / 8;
    const int sketch_aligned = (sketch_bytes + 3) & ~3;
    size_t smem_bytes = sketch_aligned
                      + NUM_WARPS * sizeof(float)
                      + NUM_WARPS * sizeof(float)
                      + NUM_WARPS * head_dim * sizeof(float);

    void* scratch_ptr = nullptr;
    int num_splits = compute_splitk_splits(
        batch_size, n_heads, head_dim, max_context_len, block_size, &scratch_ptr);

    if (num_splits > 1) {
        float* partial = static_cast<float*>(scratch_ptr);

        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        #define LAUNCH_SPLITK_TQL(HD) \
            paged_attention_splitk_turboquant_lite_kernel<HD><<<grid1, block1, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const uint8_t*>(V_cache.data), \
                K_norms, V_scales, \
                K_sketches, qjl_matrix, \
                partial, \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, sketch_dim, \
                max_num_blocks, num_splits, \
                sliding_window, softcap)

        switch (head_dim) {
            case 64:  LAUNCH_SPLITK_TQL(64);  break;
            case 96:  LAUNCH_SPLITK_TQL(96);  break;
            case 128: LAUNCH_SPLITK_TQL(128); break;
            case 256: LAUNCH_SPLITK_TQL(256); break;
            default:
                IMP_LOG_ERROR("paged_attention_splitk_turboquant_lite: unsupported head_dim %d", head_dim);
                return;
        }
        #undef LAUNCH_SPLITK_TQL

        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data),
                                      batch_size, n_heads, head_dim, num_splits, stream);
    } else {
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

        #define LAUNCH_TQL(HD) \
            paged_attention_decode_turboquant_lite_kernel<HD><<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const uint8_t*>(V_cache.data), \
                K_norms, V_scales, \
                K_sketches, qjl_matrix, \
                reinterpret_cast<half*>(O.data), \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, sketch_dim, max_context_len, max_num_blocks, \
                sliding_window, softcap)

        switch (head_dim) {
            case 64:  LAUNCH_TQL(64);  break;
            case 96:  LAUNCH_TQL(96);  break;
            case 128: LAUNCH_TQL(128); break;
            case 256: LAUNCH_TQL(256); break;
            default:
                IMP_LOG_ERROR("paged_attention_decode_turboquant_lite: unsupported head_dim %d", head_dim);
                return;
        }
        #undef LAUNCH_TQL
    }
}

} // namespace imp
