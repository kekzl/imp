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
// FP8 E4M3 helper: convert a single FP8 byte to float
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
#ifdef __CUDA_FP8_TYPES_EXIST__
    __nv_fp8_e4m3 val;
    memcpy(&val, &bits, 1);
    return static_cast<float>(val);
#else
    // Software fallback: E4M3 (1 sign, 4 exponent, 3 mantissa, bias=7)
    uint32_t sign = (bits >> 7) & 1u;
    uint32_t exp  = (bits >> 3) & 0xFu;
    uint32_t mant = bits & 0x7u;

    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;

    float result;
    if (exp == 0) {
        // Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mantissa)
        result = ldexpf((float)mant / 8.0f, 1 - 7);
    } else if (exp == 0xFu && mant == 0x7u) {
        // NaN in E4M3 (no inf; max exp with max mantissa = NaN)
        result = __int_as_float(0x7FC00000);  // quiet NaN
    } else {
        // Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mantissa/8)
        result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -result : result;
#endif
}
// ---------------------------------------------------------------------------
// Pipelined Split-K: FP8 E4M3 variant
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_splitk_fp8_pipeline_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    float* __restrict__ partial_out,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    float kv_scale,
    int max_num_blocks,
    int num_splits,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int FP8_VEC4 = ELEMS / 4;
    constexpr int FP8_REM  = ELEMS % 4;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

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
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride  = n_kv_heads * HEAD_DIM;
    const float fused_scale = scale * kv_scale;

    // Per-warp smem: k_buf[2][ELEMS] + v_buf[ELEMS] bytes (FP8: 1 byte each)
    // Total per warp: 3 * HEAD_DIM bytes. 8 warps: 3 * 128 * 8 = 3 KiB for HD=128.
    extern __shared__ char smem_pipe_fp8[];
    constexpr int WARP_SMEM_BYTES = 3 * HEAD_DIM;
    uint8_t* my_smem = reinterpret_cast<uint8_t*>(smem_pipe_fp8) + warp_id * WARP_SMEM_BYTES;
    uint8_t* k_buf0 = my_smem;
    uint8_t* k_buf1 = my_smem + HEAD_DIM;
    uint8_t* v_buf  = my_smem + 2 * HEAD_DIM;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;
        int n_toks = (tok_end - tok_start) - first_tok;
        if (n_toks <= 0) continue;

        // Prime: async load K[first_tok] into k_buf0
        // FP8: 4 bytes per cp.async (ELEMS bytes per thread = 4 for HD=128)
        {
            const uint8_t* K_tok = K_block + first_tok * kv_slot_stride + kv_head * HEAD_DIM;
            // Use cp.async.cg 4-byte for FP8 (ELEMS=4 for HD=128)
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&k_buf0[lane_offset]))),
                   "l"(&K_tok[lane_offset]));
            cp_async_commit();
        }

        int cur = 0;
        uint8_t* k_bufs[2] = {k_buf0, k_buf1};

        for (int ti = 0; ti < n_toks; ti++) {
            int t = first_tok + ti;
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // Start async V[t] load (4 bytes)
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&v_buf[lane_offset]))),
                   "l"(&V_tok[lane_offset]));

            if (ti + 1 < n_toks) {
                const uint8_t* K_next = K_block + (t + 1) * kv_slot_stride + kv_head * HEAD_DIM;
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4;\n"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&k_bufs[1 - cur][lane_offset]))),
                       "l"(&K_next[lane_offset]));
            }
            cp_async_commit();
            cp_async_wait_group<1>();

            // Compute dot product from smem K[t]
            uint8_t* k_cur = k_bufs[cur];
            float dot = 0.0f;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* K_v = reinterpret_cast<const uint32_t*>(k_cur + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = K_v[i];
                        dot += q_reg[i*4 + 0] * fp8_e4m3_to_float(packed & 0xFF);
                        dot += q_reg[i*4 + 1] * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        dot += q_reg[i*4 + 2] * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        dot += q_reg[i*4 + 3] * fp8_e4m3_to_float((packed >> 24) & 0xFF);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    #pragma unroll
                    for (int i = 0; i < FP8_REM; i++)
                        dot += q_reg[done + i] * fp8_e4m3_to_float(k_cur[lane_offset + done + i]);
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= fused_scale;
            if (softcap > 0.0f) dot = softcap * tanhf(dot / softcap);

            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;
            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            cp_async_wait_group<0>();

            float w_new_scaled = w_new * kv_scale;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* V_v = reinterpret_cast<const uint32_t*>(v_buf + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = V_v[i];
                        o_reg[i*4 + 0] = rescale * o_reg[i*4 + 0] + w_new_scaled * fp8_e4m3_to_float(packed & 0xFF);
                        o_reg[i*4 + 1] = rescale * o_reg[i*4 + 1] + w_new_scaled * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        o_reg[i*4 + 2] = rescale * o_reg[i*4 + 2] + w_new_scaled * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        o_reg[i*4 + 3] = rescale * o_reg[i*4 + 3] + w_new_scaled * fp8_e4m3_to_float((packed >> 24) & 0xFF);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    #pragma unroll
                    for (int i = 0; i < FP8_REM; i++)
                        o_reg[done + i] = rescale * o_reg[done + i]
                            + w_new_scaled * fp8_e4m3_to_float(v_buf[lane_offset + done + i]);
                }
            }

            m_w = m_new;
            l_w = l_new;
            cur = 1 - cur;
        }
    }

    // ---- Cross-warp reduction ----
    __syncthreads();
    float* warp_max = reinterpret_cast<float*>(smem_pipe_fp8);
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
// Split-K Paged Attention -- Phase 1: FP8 E4M3 KV cache variant
// ---------------------------------------------------------------------------
//
// Same algorithm as paged_attention_splitk_kernel but K_cache/V_cache are FP8.
// Key optimizations over paged_attention_decode_fp8_kernel:
//   1. Split-K parallelism: grid.z = num_splits → full SM utilization
//   2. Vectorized uint32_t loads: 4 FP8 bytes per load, perfect coalescing
//   3. Scale fusion: kv_scale applied once after warp_reduce_sum (K) and
//      folded into w_new (V), eliminating 2*HEAD_DIM scalar muls per token
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_splitk_fp8_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    float* __restrict__ partial_out,     // [batch, n_heads, num_splits, (2 + HEAD_DIM)]
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    float kv_scale,
    int max_num_blocks,
    int num_splits,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    // FP8 vectorization: 4 bytes per uint32_t load
    constexpr int FP8_VEC4 = ELEMS / 4;   // # of uint32_t loads per thread
    constexpr int FP8_REM  = ELEMS % 4;   // remaining bytes

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // ---- Contiguous thread-to-element mapping ----
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q vector into registers using half2 vectorized loads ----
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

    // ---- Determine KV block range for this split ----
    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window) {
        effective_start = ctx_len - sliding_window;
    }
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    const int total_blocks = num_ctx_blocks - first_block;

    // Divide blocks among splits
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
            for (int i = 0; i < ELEMS; i++) {
                out[2 + lane_offset + i] = 0.0f;
            }
        }
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride  = n_kv_heads * HEAD_DIM;

    // Fused scale: apply kv_scale together with softmax scale after dot product
    const float fused_scale = scale * kv_scale;

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    // ---- Iterate over assigned KV blocks ----
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // ---- Vectorized Q.K dot product with uint32_t FP8 loads ----
            float dot = 0.0f;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* K_v = reinterpret_cast<const uint32_t*>(K_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = K_v[i];
                        uint8_t b0 = packed & 0xFF;
                        uint8_t b1 = (packed >> 8) & 0xFF;
                        uint8_t b2 = (packed >> 16) & 0xFF;
                        uint8_t b3 = (packed >> 24) & 0xFF;
                        dot += q_reg[i*4 + 0] * fp8_e4m3_to_float(b0);
                        dot += q_reg[i*4 + 1] * fp8_e4m3_to_float(b1);
                        dot += q_reg[i*4 + 2] * fp8_e4m3_to_float(b2);
                        dot += q_reg[i*4 + 3] * fp8_e4m3_to_float(b3);
                    }
                }
                // Handle remainder for ELEMS not divisible by 4 (e.g. HD=64, ELEMS=2)
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    const uint8_t* K_rem = K_tok + lane_offset + done;
                    #pragma unroll
                    for (int i = 0; i < FP8_REM; i++) {
                        dot += q_reg[done + i] * fp8_e4m3_to_float(K_rem[i]);
                    }
                }
            }
            dot = warp_reduce_sum(dot);
            // Scale fusion: apply both softmax scale and kv_scale once
            dot *= fused_scale;
            if (softcap > 0.0f) dot = softcap * tanhf(dot / softcap);

            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            // ---- Vectorized V accumulation with scale folded into weight ----
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            float w_new_scaled = w_new * kv_scale;  // fuse kv_scale into weight
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* V_v = reinterpret_cast<const uint32_t*>(V_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = V_v[i];
                        uint8_t b0 = packed & 0xFF;
                        uint8_t b1 = (packed >> 8) & 0xFF;
                        uint8_t b2 = (packed >> 16) & 0xFF;
                        uint8_t b3 = (packed >> 24) & 0xFF;
                        o_reg[i*4 + 0] = rescale * o_reg[i*4 + 0] + w_new_scaled * fp8_e4m3_to_float(b0);
                        o_reg[i*4 + 1] = rescale * o_reg[i*4 + 1] + w_new_scaled * fp8_e4m3_to_float(b1);
                        o_reg[i*4 + 2] = rescale * o_reg[i*4 + 2] + w_new_scaled * fp8_e4m3_to_float(b2);
                        o_reg[i*4 + 3] = rescale * o_reg[i*4 + 3] + w_new_scaled * fp8_e4m3_to_float(b3);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    const uint8_t* V_rem = V_tok + lane_offset + done;
                    #pragma unroll
                    for (int i = 0; i < FP8_REM; i++) {
                        o_reg[done + i] = rescale * o_reg[done + i] + w_new_scaled * fp8_e4m3_to_float(V_rem[i]);
                    }
                }
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    // ---- Cross-warp reduction within this block ----
    extern __shared__ char smem_sk_fp8[];
    float* warp_max = reinterpret_cast<float*>(smem_sk_fp8);
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


template<int HEAD_DIM>
__global__ void paged_attention_decode_fp8_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,   // FP8 E4M3 raw bytes
    const uint8_t* __restrict__ V_cache,   // FP8 E4M3 raw bytes
    half* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int block_size,
    float scale,
    float kv_scale,
    int max_context_len,
    int max_num_blocks,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int FP8_VEC4 = ELEMS / 4;
    constexpr int FP8_REM  = ELEMS % 4;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q vector into registers using half2 vectorized loads ----
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

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride  = n_kv_heads * HEAD_DIM;
    const float fused_scale = scale * kv_scale;

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
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // Vectorized Q.K dot product with uint32_t FP8 loads
            float dot = 0.0f;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* K_v = reinterpret_cast<const uint32_t*>(K_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = K_v[i];
                        dot += q_reg[i*4 + 0] * fp8_e4m3_to_float(packed & 0xFF);
                        dot += q_reg[i*4 + 1] * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        dot += q_reg[i*4 + 2] * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        dot += q_reg[i*4 + 3] * fp8_e4m3_to_float((packed >> 24) & 0xFF);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    const uint8_t* K_rem = K_tok + lane_offset + done;
                    #pragma unroll
                    for (int i = 0; i < FP8_REM; i++)
                        dot += q_reg[done + i] * fp8_e4m3_to_float(K_rem[i]);
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= fused_scale;
            if (softcap > 0.0f) dot = softcap * tanhf(dot / softcap);

            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            float w_new_scaled = w_new * kv_scale;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* V_v = reinterpret_cast<const uint32_t*>(V_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = V_v[i];
                        o_reg[i*4 + 0] = rescale * o_reg[i*4 + 0] + w_new_scaled * fp8_e4m3_to_float(packed & 0xFF);
                        o_reg[i*4 + 1] = rescale * o_reg[i*4 + 1] + w_new_scaled * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        o_reg[i*4 + 2] = rescale * o_reg[i*4 + 2] + w_new_scaled * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        o_reg[i*4 + 3] = rescale * o_reg[i*4 + 3] + w_new_scaled * fp8_e4m3_to_float((packed >> 24) & 0xFF);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    const uint8_t* V_rem = V_tok + lane_offset + done;
                    #pragma unroll
                    for (int i = 0; i < FP8_REM; i++)
                        o_reg[done + i] = rescale * o_reg[done + i] + w_new_scaled * fp8_e4m3_to_float(V_rem[i]);
                }
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    // ---- Cross-warp reduction ----
    extern __shared__ char smem_fp8[];
    float* warp_max = reinterpret_cast<float*>(smem_fp8);
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
            O[out_idx] = __float2half(o_val);
        }
    }
}


// ---------------------------------------------------------------------------
// Host launcher -- FP8 E4M3 variant (with Split-K support)
// ---------------------------------------------------------------------------
void paged_attention_decode_fp8(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const int* block_tables, const int* context_lens,
    int block_size, float scale, float kv_scale,
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

    // ---- Split-K decision (same heuristic as FP16) ----
    int total_blocks_nosplit = batch_size * n_heads;
    int num_splits = 1;
    int num_ctx_blocks = (max_context_len + block_size - 1) / block_size;

    void* scratch_ptr = nullptr;
    size_t scratch_size = 0;
    paged_attention_get_splitk_scratch(&scratch_ptr, &scratch_size);

    static int num_sms_fp8 = kpar_n_sms();
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 2 * num_sms_fp8 && scratch_ptr != nullptr) {
        int target_blocks = 2 * num_sms_fp8;
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
        // Split-K Phase 1: FP8 kernel
        float* partial = static_cast<float*>(scratch_ptr);

        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        // Use pipelined cp.async kernel on sm_90+
        static int sm_ver_fp8 = get_device_sm_version();
        if (sm_ver_fp8 >= 90) {
            size_t pipe_smem = NUM_WARPS * 3 * head_dim;
            size_t launch_smem = (pipe_smem > smem_bytes) ? pipe_smem : smem_bytes;

            #define LAUNCH_SPLITK_FP8_PIPE(HD) \
                paged_attention_splitk_fp8_pipeline_kernel<HD><<<grid1, block1, launch_smem, stream>>>( \
                    reinterpret_cast<const half*>(Q.data), \
                    reinterpret_cast<const uint8_t*>(K_cache.data), \
                    reinterpret_cast<const uint8_t*>(V_cache.data), \
                    partial, \
                    block_tables, context_lens, \
                    batch_size, n_heads, n_kv_heads, \
                    block_size, scale, kv_scale, \
                    max_num_blocks, num_splits, \
                    sliding_window, softcap)

            switch (head_dim) {
                case 64:  LAUNCH_SPLITK_FP8_PIPE(64);  break;
                case 96:  LAUNCH_SPLITK_FP8_PIPE(96);  break;
                case 128: LAUNCH_SPLITK_FP8_PIPE(128); break;
                case 256: LAUNCH_SPLITK_FP8_PIPE(256); break;
                default:
                    IMP_LOG_ERROR("paged_attention_splitk_fp8_pipeline: unsupported head_dim %d", head_dim);
                    return;
            }
            #undef LAUNCH_SPLITK_FP8_PIPE
        } else {
            #define LAUNCH_SPLITK_FP8(HD) \
                paged_attention_splitk_fp8_kernel<HD><<<grid1, block1, smem_bytes, stream>>>( \
                    reinterpret_cast<const half*>(Q.data), \
                    reinterpret_cast<const uint8_t*>(K_cache.data), \
                    reinterpret_cast<const uint8_t*>(V_cache.data), \
                    partial, \
                    block_tables, context_lens, \
                    batch_size, n_heads, n_kv_heads, \
                    block_size, scale, kv_scale, \
                    max_num_blocks, num_splits, \
                    sliding_window, softcap)

            switch (head_dim) {
                case 64:  LAUNCH_SPLITK_FP8(64);  break;
                case 96:  LAUNCH_SPLITK_FP8(96);  break;
                case 128: LAUNCH_SPLITK_FP8(128); break;
                case 256: LAUNCH_SPLITK_FP8(256); break;
                default:
                    IMP_LOG_ERROR("paged_attention_splitk_fp8: unsupported head_dim %d", head_dim);
                    return;
            }
            #undef LAUNCH_SPLITK_FP8
        }

        // Split-K Phase 2: reuse shared reduce launcher
        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data),
                                      batch_size, n_heads, head_dim, num_splits, stream);
    } else {
        // Fallback: non-Split-K FP8 kernel (templated + vectorized)
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

        #define LAUNCH_FP8_FALLBACK(HD) \
            paged_attention_decode_fp8_kernel<HD><<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const uint8_t*>(K_cache.data), \
                reinterpret_cast<const uint8_t*>(V_cache.data), \
                reinterpret_cast<half*>(O.data), \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, kv_scale, max_context_len, max_num_blocks, \
                sliding_window, softcap)

        switch (head_dim) {
            case 64:  LAUNCH_FP8_FALLBACK(64);  break;
            case 96:  LAUNCH_FP8_FALLBACK(96);  break;
            case 128: LAUNCH_FP8_FALLBACK(128); break;
            case 256: LAUNCH_FP8_FALLBACK(256); break;
            default:
                IMP_LOG_ERROR("paged_attention_decode_fp8: unsupported head_dim %d", head_dim);
                return;
        }
        #undef LAUNCH_FP8_FALLBACK
    }
}

} // namespace imp
