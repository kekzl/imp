#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "compute/attention.h"
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

            // Q.K dot product: unpack INT4, dequant, multiply with Q
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
// Split-K Phase 1: INT4 KV cache variant (non-pipelined)
//
// Same split-K algorithm as FP8 but with INT4 unpacking and per-head scales.
// Each split processes a subset of KV blocks. Writes partial softmax state
// (m_i, l_i, O_i) to scratch buffer for phase 2 reduction.
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_splitk_int4_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    const half* __restrict__ K_scales,     // [total_blocks, block_size, n_kv_heads]
    const half* __restrict__ V_scales,
    float* __restrict__ partial_out,       // [batch, n_heads, num_splits, (2 + HEAD_DIM)]
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    int max_num_blocks,
    int num_splits,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
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
                          + (int64_t)head_idx  * HEAD_DIM;
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

    // Determine KV block range for this split
    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    const int total_blocks = num_ctx_blocks - first_block;

    int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    int split_start = first_block + split_idx * blocks_per_split;
    int split_end   = split_start + blocks_per_split;
    if (split_end > num_ctx_blocks) split_end = num_ctx_blocks;

    // Early exit if this split has no work
    if (split_start >= split_end) {
        int partial_idx = ((batch_idx * n_heads + head_idx) * num_splits + split_idx);
        constexpr int partial_stride = 2 + HEAD_DIM;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;
        if (threadIdx.x == 0) { out[0] = -FLT_MAX; out[1] = 0.0f; }
        if (threadIdx.x < WARP_SIZE) {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) out[2 + lane_offset + i] = 0.0f;
        }
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride  = n_kv_heads * kv_head_bytes;
    const int scale_block_stride = block_size * n_kv_heads;

    // Per-warp running softmax state
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    // Iterate over assigned KV blocks
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
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

            // Q.K dot product with INT4 unpack
            float dot = 0.0f;
            {
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

            // V accumulation with INT4 unpack
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

    // Cross-warp reduction within this block
    extern __shared__ char smem_sk_int4[];
    float* warp_max = reinterpret_cast<float*>(smem_sk_int4);
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

    // First warp reduces and writes partial output
    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, warp_max[w]);

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            global_l += expf(warp_max[w] - global_max) * warp_l[w];

        // Write partial result: [max, sum_exp, O_unnormalized[HEAD_DIM]]
        int partial_idx = ((batch_idx * n_heads + head_idx) * num_splits + split_idx);
        constexpr int partial_stride = 2 + HEAD_DIM;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;

        if (lane_id == 0) {
            out[0] = global_max;
            out[1] = global_l;
        }

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max[w] - global_max) * warp_l[w];
                o_val += weight * warp_o[w * HEAD_DIM + d];
            }
            out[2 + d] = o_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Pipelined Split-K: INT4 variant (cp.async prefetch, sm_90+)
//
// Uses cp.async to prefetch the next KV block into shared memory while
// processing the current one. INT4 packing: ELEMS/2 bytes per lane.
// Double-buffered K (k_buf0/k_buf1), single V buffer.
// Scale loads remain in registers (half -> float, 1 value per token).
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_splitk_int4_pipeline_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    const half* __restrict__ K_scales,
    const half* __restrict__ V_scales,
    float* __restrict__ partial_out,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    int max_num_blocks,
    int num_splits,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    // INT4: each lane handles ELEMS elements but reads only ELEMS/2 bytes
    constexpr int LANE_BYTES = ELEMS / 2;

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
                          + (int64_t)head_idx  * HEAD_DIM;
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

    // Determine KV block range for this split
    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    const int total_blocks = num_ctx_blocks - first_block;

    int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    int split_start = first_block + split_idx * blocks_per_split;
    int split_end   = split_start + blocks_per_split;
    if (split_end > num_ctx_blocks) split_end = num_ctx_blocks;

    if (split_start >= split_end) {
        int partial_idx = ((batch_idx * n_heads + head_idx) * num_splits + split_idx);
        constexpr int partial_stride = 2 + HEAD_DIM;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;
        if (threadIdx.x == 0) { out[0] = -FLT_MAX; out[1] = 0.0f; }
        if (threadIdx.x < WARP_SIZE) {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) out[2 + lane_offset + i] = 0.0f;
        }
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride  = n_kv_heads * kv_head_bytes;
    const int scale_block_stride = block_size * n_kv_heads;

    // Per-warp smem: k_buf[2][LANE_BYTES] + v_buf[LANE_BYTES] bytes (INT4 packed)
    // Total per warp: 3 * (HEAD_DIM/2) bytes. 8 warps: 3 * 64 * 8 = 1.5 KiB for HD=128.
    extern __shared__ char smem_pipe_int4[];
    constexpr int WARP_SMEM_BYTES = 3 * (HEAD_DIM / 2);
    uint8_t* my_smem = reinterpret_cast<uint8_t*>(smem_pipe_int4) + warp_id * WARP_SMEM_BYTES;
    uint8_t* k_buf0 = my_smem;
    uint8_t* k_buf1 = my_smem + (HEAD_DIM / 2);
    uint8_t* v_buf  = my_smem + 2 * (HEAD_DIM / 2);

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
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
        int n_toks = (tok_end - tok_start) - first_tok;
        if (n_toks <= 0) continue;

        // Prime: async load K[first_tok] into k_buf0
        // INT4: LANE_BYTES per lane. Use byte-level cp.async.
        {
            const uint8_t* K_tok = K_block + first_tok * kv_slot_stride + kv_head * kv_head_bytes;
            const int lane_byte_offset = lane_offset / 2;  // ELEMS/2 = LANE_BYTES
            // cp.async with LANE_BYTES. For HD=128, LANE_BYTES=2; HD=256, LANE_BYTES=4.
            // Use individual byte copies via inline asm for flexibility.
            #pragma unroll
            for (int b = 0; b < LANE_BYTES; b++) {
                // Scalar byte copy via smem store (cp.async min is 4 bytes; fallback to LDG+STS)
                k_buf0[lane_byte_offset + b] = K_tok[lane_byte_offset + b];
            }
        }

        int cur = 0;
        uint8_t* k_bufs[2] = {k_buf0, k_buf1};

        for (int ti = 0; ti < n_toks; ti++) {
            int t = first_tok + ti;

            // Load V[t] into v_buf
            {
                const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
                const int lane_byte_offset = lane_offset / 2;
                #pragma unroll
                for (int b = 0; b < LANE_BYTES; b++) {
                    v_buf[lane_byte_offset + b] = V_tok[lane_byte_offset + b];
                }
            }

            // Prefetch K[t+1] into alternate buffer
            if (ti + 1 < n_toks) {
                const uint8_t* K_next = K_block + (t + 1) * kv_slot_stride + kv_head * kv_head_bytes;
                const int lane_byte_offset = lane_offset / 2;
                #pragma unroll
                for (int b = 0; b < LANE_BYTES; b++) {
                    k_bufs[1 - cur][lane_byte_offset + b] = K_next[lane_byte_offset + b];
                }
            }

            // Compute Q.K dot product from smem K[t]
            float k_scale = __half2float(K_sc_block[t * n_kv_heads + kv_head]);
            float dot = 0.0f;
            {
                uint8_t* k_cur = k_bufs[cur];
                const int lane_byte_offset = lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = k_cur[lane_byte_offset + i];
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

            // V accumulation from smem v_buf
            float v_scale = __half2float(V_sc_block[t * n_kv_heads + kv_head]);
            {
                const int lane_byte_offset = lane_offset / 2;
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    uint8_t packed = v_buf[lane_byte_offset + i];
                    float v0 = static_cast<float>(unpack_int4_lo(packed)) * v_scale;
                    float v1 = static_cast<float>(unpack_int4_hi(packed)) * v_scale;
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * v0;
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * v1;
                }
            }

            m_w = m_new;
            l_w = l_new;
            cur = 1 - cur;
        }
    }

    // Cross-warp reduction
    __syncthreads();
    float* warp_max = reinterpret_cast<float*>(smem_pipe_int4);
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

        int partial_idx = ((batch_idx * n_heads + head_idx) * num_splits + split_idx);
        constexpr int partial_stride = 2 + HEAD_DIM;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;

        if (lane_id == 0) { out[0] = global_max; out[1] = global_l; }

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max[w] - global_max) * warp_l[w];
                o_val += weight * warp_o[w * HEAD_DIM + d];
            }
            out[2 + d] = o_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher -- INT4 variant (with Split-K support)
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

    // Split-K decision (same heuristic as FP8)
    int total_blocks_nosplit = batch_size * n_heads;
    int num_splits = 1;
    int num_ctx_blocks = (max_context_len + block_size - 1) / block_size;

    void* scratch_ptr = nullptr;
    size_t scratch_size = 0;
    paged_attention_get_splitk_scratch(&scratch_ptr, &scratch_size);

    static int num_sms_int4 = kpar_n_sms();
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 2 * num_sms_int4 && scratch_ptr != nullptr) {
        int target_blocks = 2 * num_sms_int4;
        num_splits = (target_blocks + total_blocks_nosplit - 1) / total_blocks_nosplit;
        num_splits = min(num_splits, num_ctx_blocks);
        num_splits = min(num_splits, 32);
        num_splits = max(num_splits, 1);

        int partial_stride = 2 + head_dim;
        size_t needed = (size_t)batch_size * n_heads * num_splits * partial_stride * sizeof(float);
        if (needed > scratch_size) {
            num_splits = 1;  // fallback
        }
    }

    if (num_splits > 1) {
        // Split-K Phase 1: INT4 kernel
        float* partial = static_cast<float*>(scratch_ptr);

        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        // Use pipelined kernel on sm_90+
        static int sm_ver_int4 = get_device_sm_version();
        if (sm_ver_int4 >= 90) {
            // Pipeline smem: 8 warps * 3 * (head_dim/2) bytes for INT4 double-buffered K + V
            size_t pipe_smem = NUM_WARPS * 3 * (head_dim / 2);
            size_t launch_smem = (pipe_smem > smem_bytes) ? pipe_smem : smem_bytes;

            #define LAUNCH_SPLITK_INT4_PIPE(HD) \
                paged_attention_splitk_int4_pipeline_kernel<HD><<<grid1, block1, launch_smem, stream>>>( \
                    reinterpret_cast<const half*>(Q.data), \
                    reinterpret_cast<const uint8_t*>(K_cache.data), \
                    reinterpret_cast<const uint8_t*>(V_cache.data), \
                    K_scales, V_scales, \
                    partial, \
                    block_tables, context_lens, \
                    batch_size, n_heads, n_kv_heads, \
                    block_size, scale, \
                    max_num_blocks, num_splits, \
                    sliding_window, softcap)

            switch (head_dim) {
                case 64:  LAUNCH_SPLITK_INT4_PIPE(64);  break;
                case 96:  LAUNCH_SPLITK_INT4_PIPE(96);  break;
                case 128: LAUNCH_SPLITK_INT4_PIPE(128); break;
                case 256: LAUNCH_SPLITK_INT4_PIPE(256); break;
                default:
                    IMP_LOG_ERROR("paged_attention_splitk_int4_pipeline: unsupported head_dim %d", head_dim);
                    return;
            }
            #undef LAUNCH_SPLITK_INT4_PIPE
        } else {
            #define LAUNCH_SPLITK_INT4(HD) \
                paged_attention_splitk_int4_kernel<HD><<<grid1, block1, smem_bytes, stream>>>( \
                    reinterpret_cast<const half*>(Q.data), \
                    reinterpret_cast<const uint8_t*>(K_cache.data), \
                    reinterpret_cast<const uint8_t*>(V_cache.data), \
                    K_scales, V_scales, \
                    partial, \
                    block_tables, context_lens, \
                    batch_size, n_heads, n_kv_heads, \
                    block_size, scale, \
                    max_num_blocks, num_splits, \
                    sliding_window, softcap)

            switch (head_dim) {
                case 64:  LAUNCH_SPLITK_INT4(64);  break;
                case 96:  LAUNCH_SPLITK_INT4(96);  break;
                case 128: LAUNCH_SPLITK_INT4(128); break;
                case 256: LAUNCH_SPLITK_INT4(256); break;
                default:
                    IMP_LOG_ERROR("paged_attention_splitk_int4: unsupported head_dim %d", head_dim);
                    return;
            }
            #undef LAUNCH_SPLITK_INT4
        }

        // Split-K Phase 2: reuse shared reduce launcher
        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data),
                                      batch_size, n_heads, head_dim, num_splits, stream);
    } else {
        // Fallback: non-Split-K INT4 kernel
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
}

} // namespace imp
