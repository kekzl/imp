#include "compute/attention_paged.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>

namespace imp {

// ---------------------------------------------------------------------------
// Paged Attention -- Decode Kernel (single-query per sequence)
// ---------------------------------------------------------------------------
//
// Each thread block handles one (batch, head) pair.
// Q:       [batch, 1, n_heads, head_dim]           -- FP16
// K_cache: [num_blocks, n_kv_heads, block_size, head_dim] -- FP16 (paged)
// V_cache: [num_blocks, n_kv_heads, block_size, head_dim] -- FP16 (paged)
// O:       [batch, 1, n_heads, head_dim]           -- FP16
//
// block_tables:  [batch, max_num_blocks] int32
// context_lens:  [batch] int32
//
// Algorithm (two-pass online softmax):
//   Pass 1 -- compute Q.K scores and track running max per warp group,
//             then do a cross-warp reduction to get the global max and
//             the global sum-of-exp.
//   Pass 2 -- recompute exp(score-max) * V and accumulate, then normalise.
//
// For simplicity and clarity we use a single-pass approach where each
// thread group iterates over KV blocks, maintains running softmax state
// (max, sum-of-exp, weighted-V accumulator), and then does a final
// cross-thread reduction.
//
// Thread mapping:
//   256 threads = 8 warps.
//   head_dim elements are distributed across threads for dot products.
//   Each thread handles head_dim / 256 elements (if hd=128: not evenly
//   divisible). Better: each warp handles a range of KV positions and
//   all threads in the warp cooperate on the dot product across head_dim.
//
// Practical design:
//   BLOCK_THREADS = 256, WARP_SIZE = 32, NUM_WARPS = 8.
//   Distribute KV tokens across warps: each warp processes a strided
//   subset of the context tokens.
//   Within a warp, the 32 threads cooperate on the head_dim dot product:
//     each thread handles ceil(head_dim/32) elements.
//   After the dot product, a warp reduction gives the full score.
//   Each warp tracks its own (max, l, O_acc[head_dim]).
//   After iterating all assigned tokens, cross-warp reduction merges the
//   8 partial softmax states into the final result.
// ---------------------------------------------------------------------------

static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_THREADS = 256;
static constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;  // 8

// Warp-level reductions
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__global__ void paged_attention_decode_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int block_size,
    float scale,
    int max_context_len,
    int max_num_blocks,
    int sliding_window)
{
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;   // [0, NUM_WARPS)
    const int lane_id = threadIdx.x % WARP_SIZE;   // [0, 32)

    // ---- Load Q vector into registers ----
    // Q: [batch, 1, n_heads, head_dim]
    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * head_dim
                          + (int64_t)head_idx  * head_dim;

    // Each thread in the warp handles a strided subset of head_dim.
    // We accumulate in float registers.
    // Max head_dim we support in registers: 256 (generous).
    // For head_dim = 128, each thread (of 32 in warp) handles 4 elements.
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    // Store Q values in registers
    float q_reg[8];  // support up to head_dim = 256 (8 * 32)
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        q_reg[i] = (d < head_dim) ? __half2float(Q_ptr[d]) : 0.0f;
    }

    // ---- block_tables pointer for this batch ----
    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;

    // K_cache / V_cache layout: [num_blocks_total, block_size, n_kv_heads, head_dim]
    // Written by write_kv_cache_kernel which stores [slot, kv_head, hd] per block.
    const int kv_block_stride = block_size * n_kv_heads * head_dim;  // stride per physical block
    const int kv_slot_stride  = n_kv_heads * head_dim;               // stride per slot within block

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;  // running max
    float l_w = 0.0f;      // running sum of exp

    // Per-warp O accumulator in registers (same layout as q_reg)
    float o_reg[8];
    for (int i = 0; i < elems_per_thread; i++) o_reg[i] = 0.0f;

    // ---- Iterate over KV tokens, striped across warps ----
    // With sliding window, only attend to the last `sliding_window` tokens.
    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window) {
        effective_start = ctx_len - sliding_window;
    }
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];  // physical block index

        // Base pointer for K and V in this physical block
        const half* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const half* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        // Number of valid tokens in this block
        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        // Skip tokens before the sliding window within the first relevant block
        int first_tok = 0;
        if (tok_start < effective_start) {
            first_tok = effective_start - tok_start;
        }

        // Process each token in the block
        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            // K for this token: [slot=t, kv_head, head_dim]
            const half* K_tok = K_block + t * kv_slot_stride + kv_head * head_dim;

            // Compute dot(Q, K) -- distributed across warp lanes
            float dot = 0.0f;
            for (int i = 0; i < elems_per_thread; i++) {
                int d = lane_id + i * WARP_SIZE;
                if (d < head_dim) {
                    dot += q_reg[i] * __half2float(K_tok[d]);
                }
            }
            // Warp reduction to get the full dot product
            dot = warp_reduce_sum(dot);
            dot *= scale;

            // ---- Online softmax update ----
            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            // Rescale existing O accumulator
            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            // V for this token: [slot=t, kv_head, head_dim]
            const half* V_tok = V_block + t * kv_slot_stride + kv_head * head_dim;

            for (int i = 0; i < elems_per_thread; i++) {
                int d = lane_id + i * WARP_SIZE;
                float v_val = (d < head_dim) ? __half2float(V_tok[d]) : 0.0f;
                o_reg[i] = rescale * o_reg[i] + w_new * v_val;
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    // ---- Cross-warp reduction ----
    // We need to merge NUM_WARPS partial softmax results.
    // Use shared memory for the reduction.
    //
    // Shared memory layout:
    //   float warp_max[NUM_WARPS]
    //   float warp_l  [NUM_WARPS]
    //   float warp_o  [NUM_WARPS * head_dim]

    extern __shared__ char smem[];
    float* warp_max = reinterpret_cast<float*>(smem);                          // [NUM_WARPS]
    float* warp_l   = warp_max + NUM_WARPS;                                    // [NUM_WARPS]
    float* warp_o   = warp_l   + NUM_WARPS;                                    // [NUM_WARPS * head_dim]

    // Lane 0 of each warp writes warp state
    if (lane_id == 0) {
        warp_max[warp_id] = m_w;
        warp_l[warp_id]   = l_w;
    }
    // All lanes write their portion of o_reg
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
            warp_o[warp_id * head_dim + d] = o_reg[i];
        }
    }
    __syncthreads();

    // First warp performs the final reduction across all warps
    if (warp_id == 0) {
        // Find global max
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++) {
            global_max = fmaxf(global_max, warp_max[w]);
        }

        // Compute global denominator
        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) {
            global_l += expf(warp_max[w] - global_max) * warp_l[w];
        }

        // Merge weighted O accumulators -- each lane handles its head_dim slice
        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
                float o_val = 0.0f;
                for (int w = 0; w < NUM_WARPS; w++) {
                    float weight = expf(warp_max[w] - global_max) * warp_l[w];
                    o_val += weight * warp_o[w * head_dim + d];
                }
                if (global_l > 0.0f) {
                    o_val /= global_l;
                }

                // Write to global output
                // O: [batch, 1, n_heads, head_dim]
                int out_idx = batch_idx * n_heads * head_dim
                            + head_idx * head_dim
                            + d;
                O[out_idx] = __float2half(o_val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void paged_attention_decode(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const int* block_tables, const int* context_lens,
    int block_size, float scale, int max_context_len,
    int sliding_window, cudaStream_t stream)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    // Q.shape[1] == 1  (decode: single query token)
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    // K_cache layout: [num_blocks, block_size, n_kv_heads, head_dim]
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_context_len + block_size - 1) / block_size;

    dim3 grid(batch_size, n_heads);
    dim3 block(BLOCK_THREADS);

    // Shared memory: warp_max[NW] + warp_l[NW] + warp_o[NW * head_dim]
    size_t smem_bytes = NUM_WARPS * sizeof(float)                 // warp_max
                      + NUM_WARPS * sizeof(float)                 // warp_l
                      + NUM_WARPS * head_dim * sizeof(float);     // warp_o

    paged_attention_decode_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half*>(Q.data),
        reinterpret_cast<const half*>(K_cache.data),
        reinterpret_cast<const half*>(V_cache.data),
        reinterpret_cast<half*>(O.data),
        block_tables,
        context_lens,
        batch_size, n_heads, n_kv_heads, head_dim,
        block_size, scale, max_context_len, max_num_blocks,
        sliding_window);
}

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
// Paged Attention -- Decode Kernel with FP8 E4M3 KV cache
// ---------------------------------------------------------------------------
//
// Identical algorithm to paged_attention_decode_kernel but K_cache and V_cache
// are stored in FP8 E4M3 format (1 byte per element).  On-the-fly
// dequantisation is applied: float val = fp8_to_float(raw_byte) * kv_scale.
//
// Q and O remain FP16 (half).
// ---------------------------------------------------------------------------

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
    int head_dim,
    int block_size,
    float scale,
    float kv_scale,
    int max_context_len,
    int max_num_blocks,
    int sliding_window)
{
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;   // [0, NUM_WARPS)
    const int lane_id = threadIdx.x % WARP_SIZE;   // [0, 32)

    // ---- Load Q vector into registers ----
    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * head_dim
                          + (int64_t)head_idx  * head_dim;

    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    float q_reg[8];  // support up to head_dim = 256 (8 * 32)
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        q_reg[i] = (d < head_dim) ? __half2float(Q_ptr[d]) : 0.0f;
    }

    // ---- block_tables pointer for this batch ----
    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;

    // K_cache / V_cache layout: [num_blocks_total, block_size, n_kv_heads, head_dim]
    // Each element is 1 byte (FP8), so strides are in bytes.
    const int kv_block_stride = block_size * n_kv_heads * head_dim;
    const int kv_slot_stride  = n_kv_heads * head_dim;

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;

    float o_reg[8];
    for (int i = 0; i < elems_per_thread; i++) o_reg[i] = 0.0f;

    // ---- Iterate over KV tokens, striped across warps ----
    int effective_start_fp8 = 0;
    if (sliding_window > 0 && ctx_len > sliding_window) {
        effective_start_fp8 = ctx_len - sliding_window;
    }
    const int first_block_fp8 = effective_start_fp8 / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block_fp8 + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];

        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok_fp8 = 0;
        if (tok_start < effective_start_fp8) {
            first_tok_fp8 = effective_start_fp8 - tok_start;
        }

        for (int t = first_tok_fp8; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * head_dim;

            // Compute dot(Q, K) with FP8 dequant
            float dot = 0.0f;
            for (int i = 0; i < elems_per_thread; i++) {
                int d = lane_id + i * WARP_SIZE;
                if (d < head_dim) {
                    float k_val = fp8_e4m3_to_float(K_tok[d]) * kv_scale;
                    dot += q_reg[i] * k_val;
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;

            // ---- Online softmax update ----
            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * head_dim;

            for (int i = 0; i < elems_per_thread; i++) {
                int d = lane_id + i * WARP_SIZE;
                float v_val = (d < head_dim)
                    ? fp8_e4m3_to_float(V_tok[d]) * kv_scale
                    : 0.0f;
                o_reg[i] = rescale * o_reg[i] + w_new * v_val;
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
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
            warp_o[warp_id * head_dim + d] = o_reg[i];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++) {
            global_max = fmaxf(global_max, warp_max[w]);
        }

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) {
            global_l += expf(warp_max[w] - global_max) * warp_l[w];
        }

        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
                float o_val = 0.0f;
                for (int w = 0; w < NUM_WARPS; w++) {
                    float weight = expf(warp_max[w] - global_max) * warp_l[w];
                    o_val += weight * warp_o[w * head_dim + d];
                }
                if (global_l > 0.0f) {
                    o_val /= global_l;
                }

                int out_idx = batch_idx * n_heads * head_dim
                            + head_idx * head_dim
                            + d;
                O[out_idx] = __float2half(o_val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher -- FP8 E4M3 variant
// ---------------------------------------------------------------------------
void paged_attention_decode_fp8(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const int* block_tables, const int* context_lens,
    int block_size, float scale, float kv_scale,
    int max_context_len, int sliding_window,
    cudaStream_t stream)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    // K_cache layout: [num_blocks, block_size, n_kv_heads, head_dim]
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_context_len + block_size - 1) / block_size;

    dim3 grid(batch_size, n_heads);
    dim3 block(BLOCK_THREADS);

    size_t smem_bytes = NUM_WARPS * sizeof(float)                 // warp_max
                      + NUM_WARPS * sizeof(float)                 // warp_l
                      + NUM_WARPS * head_dim * sizeof(float);     // warp_o

    paged_attention_decode_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half*>(Q.data),
        reinterpret_cast<const uint8_t*>(K_cache.data),
        reinterpret_cast<const uint8_t*>(V_cache.data),
        reinterpret_cast<half*>(O.data),
        block_tables,
        context_lens,
        batch_size, n_heads, n_kv_heads, head_dim,
        block_size, scale, kv_scale, max_context_len, max_num_blocks,
        sliding_window);
}

} // namespace imp
