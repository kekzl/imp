#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace imp {

// ===========================================================================
// INT8 dp4a Paged Attention — Split-K kernel
// ===========================================================================
//
// Q·K uses dp4a: Q is quantized to INT8 in registers once per kernel, then
// __dp4a(K_int8x4, Q_int8x4, acc) computes 4 multiply-adds in 1 instruction.
// V accumulation uses trivial int8→float: (float)(int8_t)byte = 1 CVT instruction.
// Per-head scales from the INT8 KV cache write kernel handle dequantization.
//
// Grid: (batch, n_heads, num_splits)
// Block: BLOCK_THREADS (256 = 8 warps)
// ===========================================================================

template<int HEAD_DIM>
__global__ void paged_attention_splitk_int8_kernel(
    const half* __restrict__ Q,
    const int8_t* __restrict__ K_cache,
    const int8_t* __restrict__ V_cache,
    const half* __restrict__ K_scales,    // [total_blocks, block_size, n_kv_heads]
    const half* __restrict__ V_scales,
    float* __restrict__ partial_out,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size, int n_heads, int n_kv_heads,
    int block_size, float scale,
    int max_num_blocks, int num_splits, int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int DP4A_CALLS = (ELEMS + 3) / 4;
    constexpr int DP4A_REM   = ELEMS % 4;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q as FP16 → float ----
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

    // ---- Quantize Q to INT8 in registers (once per kernel) ----
    float q_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) q_amax = fmaxf(q_amax, fabsf(q_reg[i]));
    q_amax = warp_reduce_max(q_amax);

    float q_scale = q_amax / 127.0f;
    float q_inv_scale = (q_amax > 1e-8f) ? (127.0f / q_amax) : 0.0f;

    int8_t q_i8[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        q_i8[i] = (int8_t)__float2int_rn(q_reg[i] * q_inv_scale);

    // Pack Q into int32 for dp4a
    int q_packed[DP4A_CALLS];
    #pragma unroll
    for (int i = 0; i < DP4A_CALLS; i++) {
        uint32_t p = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int idx = i * 4 + j;
            uint8_t byte = (idx < ELEMS) ? static_cast<uint8_t>(q_i8[idx]) : 0;
            p |= (static_cast<uint32_t>(byte) << (j * 8));
        }
        q_packed[i] = static_cast<int>(p);
    }

    // ---- Determine KV block range for this split ----
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
        if (threadIdx.x == 0) {
            out[0] = -FLT_MAX;
            out[1] = 0.0f;
        }
        if (threadIdx.x < WARP_SIZE) {
            #pragma unroll
            for (int i = 0; i < ELEMS; i++)
                out[2 + lane_offset + i] = 0.0f;
        }
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride  = n_kv_heads * HEAD_DIM;
    const int scale_block_stride = block_size * n_kv_heads;

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    // ---- Iterate over assigned KV blocks ----
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const int8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const int8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const half* K_sc_block = K_scales + (int64_t)phys_block * scale_block_stride;
        const half* V_sc_block = V_scales + (int64_t)phys_block * scale_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            // ---- Q·K with dp4a ----
            const int8_t* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            int32_t sumi = 0;
            {
                const int* K_v = reinterpret_cast<const int*>(K_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < DP4A_CALLS; i++) {
                    int k_val;
                    if constexpr (DP4A_REM > 0) {
                        if (i == DP4A_CALLS - 1) {
                            // Last chunk may be partial — load byte-by-byte and pack
                            uint32_t p = 0;
                            const int8_t* K_rem = K_tok + lane_offset + i * 4;
                            #pragma unroll
                            for (int j = 0; j < DP4A_REM; j++)
                                p |= (static_cast<uint32_t>(static_cast<uint8_t>(K_rem[j])) << (j * 8));
                            k_val = static_cast<int>(p);
                        } else {
                            k_val = __ldcs(&K_v[i]);
                        }
                    } else {
                        k_val = __ldcs(&K_v[i]);
                    }
                    sumi = __dp4a(k_val, q_packed[i], sumi);
                }
            }
            float dot = warp_reduce_sum(static_cast<float>(sumi));

            // Load K scale and fuse all scales: softmax_scale * q_scale * k_scale
            half k_sc = K_sc_block[t * n_kv_heads + kv_head];
            dot *= scale * q_scale * __half2float(k_sc);
            if (softcap > 0.0f) dot = softcap * tanhf(dot / softcap);

            // ---- Online softmax update ----
            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            // ---- V accumulation with trivial int8→float dequant ----
            const int8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            half v_sc = V_sc_block[t * n_kv_heads + kv_head];
            float v_scale_f = __half2float(v_sc);
            float w_new_scaled = w_new * v_scale_f;

            // Vectorized int8 loads via uint32_t
            if constexpr (DP4A_CALLS > 0) {
                const uint32_t* V_v = reinterpret_cast<const uint32_t*>(V_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < DP4A_CALLS; i++) {
                    uint32_t packed;
                    int count = 4;
                    if constexpr (DP4A_REM > 0) {
                        if (i == DP4A_CALLS - 1) {
                            // Partial last chunk
                            packed = 0;
                            const uint8_t* V_rem = reinterpret_cast<const uint8_t*>(V_tok + lane_offset + i * 4);
                            #pragma unroll
                            for (int j = 0; j < DP4A_REM; j++)
                                packed |= (static_cast<uint32_t>(V_rem[j]) << (j * 8));
                            count = DP4A_REM;
                        } else {
                            packed = __ldcs(&V_v[i]);
                        }
                    } else {
                        packed = __ldcs(&V_v[i]);
                    }

                    if (count > 0) {
                        o_reg[i*4+0] = rescale * o_reg[i*4+0]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>(packed & 0xFF));
                    }
                    if (count > 1) {
                        o_reg[i*4+1] = rescale * o_reg[i*4+1]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xFF));
                    }
                    if (count > 2) {
                        o_reg[i*4+2] = rescale * o_reg[i*4+2]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xFF));
                    }
                    if (count > 3) {
                        o_reg[i*4+3] = rescale * o_reg[i*4+3]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xFF));
                    }
                }
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    // ---- Cross-warp reduction ----
    extern __shared__ char smem_sk_int8[];
    float* warp_max = reinterpret_cast<float*>(smem_sk_int8);
    float* warp_l   = warp_max + NUM_WARPS;
    float* warp_o   = warp_l   + NUM_WARPS;

    if (lane_id == 0) {
        warp_max[warp_id] = m_w;
        warp_l[warp_id]   = l_w;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        warp_o[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
    }
    __syncthreads();

    // First warp reduces and writes partial output
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

// ===========================================================================
// INT8 dp4a Paged Attention — Non-Split-K fallback kernel (templated)
// ===========================================================================

template<int HEAD_DIM>
__global__ void paged_attention_decode_int8_kernel(
    const half* __restrict__ Q,
    const int8_t* __restrict__ K_cache,
    const int8_t* __restrict__ V_cache,
    const half* __restrict__ K_scales,
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
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int DP4A_CALLS = (ELEMS + 3) / 4;
    constexpr int DP4A_REM   = ELEMS % 4;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q into registers using half2 vectorized loads ----
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

    // ---- Quantize Q to INT8 in registers ----
    float q_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) q_amax = fmaxf(q_amax, fabsf(q_reg[i]));
    q_amax = warp_reduce_max(q_amax);

    float q_scale = q_amax / 127.0f;
    float q_inv_scale = (q_amax > 1e-8f) ? (127.0f / q_amax) : 0.0f;

    int8_t q_i8[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        q_i8[i] = (int8_t)__float2int_rn(q_reg[i] * q_inv_scale);

    // Pack Q into int32 for dp4a
    int q_packed[DP4A_CALLS];
    #pragma unroll
    for (int i = 0; i < DP4A_CALLS; i++) {
        uint32_t p = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int idx = i * 4 + j;
            uint8_t byte = (idx < ELEMS) ? static_cast<uint8_t>(q_i8[idx]) : 0;
            p |= (static_cast<uint32_t>(byte) << (j * 8));
        }
        q_packed[i] = static_cast<int>(p);
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride  = n_kv_heads * HEAD_DIM;
    const int scale_block_stride = block_size * n_kv_heads;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const int8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const int8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const half* K_sc_block = K_scales + (int64_t)phys_block * scale_block_stride;
        const half* V_sc_block = V_scales + (int64_t)phys_block * scale_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            // ---- Q·K with dp4a ----
            const int8_t* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            int32_t sumi = 0;
            {
                const int* K_v = reinterpret_cast<const int*>(K_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < DP4A_CALLS; i++) {
                    int k_val;
                    if constexpr (DP4A_REM > 0) {
                        if (i == DP4A_CALLS - 1) {
                            uint32_t p = 0;
                            const int8_t* K_rem = K_tok + lane_offset + i * 4;
                            #pragma unroll
                            for (int j = 0; j < DP4A_REM; j++)
                                p |= (static_cast<uint32_t>(static_cast<uint8_t>(K_rem[j])) << (j * 8));
                            k_val = static_cast<int>(p);
                        } else {
                            k_val = __ldcs(&K_v[i]);
                        }
                    } else {
                        k_val = __ldcs(&K_v[i]);
                    }
                    sumi = __dp4a(k_val, q_packed[i], sumi);
                }
            }
            float dot = warp_reduce_sum(static_cast<float>(sumi));

            half k_sc = K_sc_block[t * n_kv_heads + kv_head];
            dot *= scale * q_scale * __half2float(k_sc);
            if (softcap > 0.0f) dot = softcap * tanhf(dot / softcap);

            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            // ---- V accumulation with vectorized int8 loads ----
            const int8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            half v_sc = V_sc_block[t * n_kv_heads + kv_head];
            float w_new_scaled = w_new * __half2float(v_sc);

            if constexpr (DP4A_CALLS > 0) {
                const uint32_t* V_v = reinterpret_cast<const uint32_t*>(V_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < DP4A_CALLS; i++) {
                    uint32_t packed;
                    int count = 4;
                    if constexpr (DP4A_REM > 0) {
                        if (i == DP4A_CALLS - 1) {
                            packed = 0;
                            const uint8_t* V_rem = reinterpret_cast<const uint8_t*>(V_tok + lane_offset + i * 4);
                            #pragma unroll
                            for (int j = 0; j < DP4A_REM; j++)
                                packed |= (static_cast<uint32_t>(V_rem[j]) << (j * 8));
                            count = DP4A_REM;
                        } else {
                            packed = __ldcs(&V_v[i]);
                        }
                    } else {
                        packed = __ldcs(&V_v[i]);
                    }

                    if (count > 0)
                        o_reg[i*4+0] = rescale * o_reg[i*4+0]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>(packed & 0xFF));
                    if (count > 1)
                        o_reg[i*4+1] = rescale * o_reg[i*4+1]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xFF));
                    if (count > 2)
                        o_reg[i*4+2] = rescale * o_reg[i*4+2]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xFF));
                    if (count > 3)
                        o_reg[i*4+3] = rescale * o_reg[i*4+3]
                            + w_new_scaled * static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xFF));
                }
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    // ---- Cross-warp reduction ----
    extern __shared__ char smem_int8[];
    float* warp_max = reinterpret_cast<float*>(smem_int8);
    float* warp_l   = warp_max + NUM_WARPS;
    float* warp_o   = warp_l   + NUM_WARPS;

    if (lane_id == 0) {
        warp_max[warp_id] = m_w;
        warp_l[warp_id]   = l_w;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        warp_o[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
    }
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
            stcs_half(&O[out_idx], __float2half(o_val));
        }
    }
}

// ===========================================================================
// INT8 dp4a Paged Attention — Host launcher
// ===========================================================================

void paged_attention_decode_int8(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O,
    const half* K_scales, const half* V_scales,
    const int* block_tables, const int* context_lens,
    int block_size, float scale,
    int max_context_len, int sliding_window,
    float softcap, cudaStream_t stream)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_context_len + block_size - 1) / block_size;

    size_t smem_bytes = NUM_WARPS * sizeof(float)
                      + NUM_WARPS * sizeof(float)
                      + NUM_WARPS * head_dim * sizeof(float);

    // ---- Split-K decision (same heuristic as FP16/FP8) ----
    int total_blocks_nosplit = batch_size * n_heads;
    int num_splits = 1;
    int num_ctx_blocks = (max_context_len + block_size - 1) / block_size;

    void* scratch_ptr = nullptr;
    size_t scratch_size = 0;
    paged_attention_get_splitk_scratch(&scratch_ptr, &scratch_size);

    static int num_sms_int8 = kpar_n_sms();
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 2 * num_sms_int8 && scratch_ptr != nullptr) {
        int target_blocks = 2 * num_sms_int8;
        num_splits = (target_blocks + total_blocks_nosplit - 1) / total_blocks_nosplit;
        num_splits = min(num_splits, num_ctx_blocks);
        num_splits = min(num_splits, 32);
        num_splits = max(num_splits, 1);

        int partial_stride = 2 + head_dim;
        size_t needed = (size_t)batch_size * n_heads * num_splits * partial_stride * sizeof(float);
        if (needed > scratch_size) {
            num_splits = 1;
        }
    }

    if (num_splits > 1) {
        float* partial = static_cast<float*>(scratch_ptr);

        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        #define LAUNCH_SPLITK_INT8(HD) \
            paged_attention_splitk_int8_kernel<HD><<<grid1, block1, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const int8_t*>(K_cache.data), \
                reinterpret_cast<const int8_t*>(V_cache.data), \
                K_scales, V_scales, \
                partial, \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, \
                max_num_blocks, num_splits, \
                sliding_window, softcap)

        switch (head_dim) {
            case 64:  LAUNCH_SPLITK_INT8(64);  break;
            case 96:  LAUNCH_SPLITK_INT8(96);  break;
            case 128: LAUNCH_SPLITK_INT8(128); break;
            case 256: LAUNCH_SPLITK_INT8(256); break;
            default:
                IMP_LOG_ERROR("paged_attention_splitk_int8: unsupported head_dim %d", head_dim);
                return;
        }
        #undef LAUNCH_SPLITK_INT8

        // Phase 2: reuse shared reduce launcher
        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data),
                                      batch_size, n_heads, head_dim, num_splits, stream);
    } else {
        // Non-Split-K INT8 fallback (templated + vectorized dp4a)
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

        #define LAUNCH_INT8_FALLBACK(HD) \
            paged_attention_decode_int8_kernel<HD><<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const int8_t*>(K_cache.data), \
                reinterpret_cast<const int8_t*>(V_cache.data), \
                K_scales, V_scales, \
                reinterpret_cast<half*>(O.data), \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, max_context_len, max_num_blocks, \
                sliding_window, softcap)

        switch (head_dim) {
            case 64:  LAUNCH_INT8_FALLBACK(64);  break;
            case 96:  LAUNCH_INT8_FALLBACK(96);  break;
            case 128: LAUNCH_INT8_FALLBACK(128); break;
            case 256: LAUNCH_INT8_FALLBACK(256); break;
            default:
                IMP_LOG_ERROR("paged_attention_decode_int8: unsupported head_dim %d", head_dim);
                return;
        }
        #undef LAUNCH_INT8_FALLBACK
    }
}

} // namespace imp
