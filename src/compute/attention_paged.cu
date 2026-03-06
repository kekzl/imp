#include "compute/attention_paged.h"
#include "compute/attention.h"
#include "compute/gemv_dp4a_traits.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
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

// ---------------------------------------------------------------------------
// GQA-aware Paged Attention Decode Kernel
// ---------------------------------------------------------------------------
//
// Key optimization: with GQA (e.g. 32 Q heads, 4 KV heads, ratio=8), the
// original kernel launches 32 blocks where groups of 8 read the exact same
// K/V data independently. This kernel instead launches per KV head and
// processes all Q heads sharing that KV head, loading K/V into shared memory
// once and reusing it across all Q heads.
//
// Grid: (batch, n_kv_heads)
// Block: GQA_BLOCK_THREADS threads
//
// Thread mapping:
//   n_q_per_kv = n_heads / n_kv_heads (e.g. 8 or 16)
//   Each Q head gets warps_per_q warps (4 for ratio<=8, 2 for ratio>8)
//   Total warps = n_q_per_kv * warps_per_q (always <= 32)
//   Total threads <= 1024
//
// Shared memory: K tile [block_size, head_dim] + V tile [block_size, head_dim]
//   loaded cooperatively by all threads, then each Q head's warps compute
//   dot products and accumulate from the shared tile.
// ---------------------------------------------------------------------------

// For GQA kernel: up to 16 Q heads per KV head
// warps_per_q is a runtime parameter: 4 for ratio<=8, 2 for ratio>8
static constexpr int MAX_Q_PER_KV = 16;

__global__ void __launch_bounds__(1024)
paged_attention_gqa_kernel(
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
    int n_q_per_kv,
    int warps_per_q,
    int sliding_window,
    float softcap)
{
    const int batch_idx = blockIdx.x;
    const int kv_head   = blockIdx.y;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int total_warps = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Which Q head does this warp belong to?
    const int q_local = warp_id / warps_per_q;  // [0, n_q_per_kv)
    const int warp_in_q = warp_id % warps_per_q;  // [0, warps_per_q)
    const int head_idx = kv_head * n_q_per_kv + q_local;

    // Skip if this warp's Q head is out of bounds (when n_q_per_kv < MAX_Q_PER_KV)
    const bool active = (q_local < n_q_per_kv);

    // ---- Load Q vector into registers ----
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    float q_reg[8];
    if (active) {
        const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * head_dim
                              + (int64_t)head_idx * head_dim;
        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            q_reg[i] = (d < head_dim) ? __half2float(Q_ptr[d]) : 0.0f;
        }
    } else {
        for (int i = 0; i < elems_per_thread; i++) q_reg[i] = 0.0f;
    }

    // ---- block_tables and KV layout ----
    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * head_dim;
    const int kv_slot_stride  = n_kv_heads * head_dim;

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[8];
    for (int i = 0; i < elems_per_thread; i++) o_reg[i] = 0.0f;

    // ---- Shared memory for K/V tile (double-buffered FP16) ----
    // Two FP16 buffers use the same total smem as one FP32 buffer:
    //   FP32 single: 2 * block_size * head_dim * 4 = 16 KiB (bs=16, hd=128)
    //   FP16 double: 4 * block_size * head_dim * 2 = 16 KiB
    // FP16→FP32 conversion happens during compute (negligible cost on Hopper+).
    extern __shared__ __align__(32) char smem_gqa[];
    half* s_kv_h = reinterpret_cast<half*>(smem_gqa);
    const int tile_elems = block_size * head_dim;
    // Buffer layout: [buf0_K, buf0_V, buf1_K, buf1_V], each tile_elems halfs
    // s_k(buf) = s_kv_h + buf * 2 * tile_elems
    // s_v(buf) = s_kv_h + buf * 2 * tile_elems + tile_elems

    // ---- Context range ----
    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window) {
        effective_start = ctx_len - sliding_window;
    }
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    // ---- Double-buffered KV block iteration ----
    int buf = 0;

    // Prefetch first block into buffer 0
    if (first_block < num_ctx_blocks) {
        int phys_block = bt[first_block];
        const half* K_block_base = K_cache + (int64_t)phys_block * kv_block_stride
                                   + kv_head * head_dim;
        const half* V_block_base = V_cache + (int64_t)phys_block * kv_block_stride
                                   + kv_head * head_dim;
        half* s_k = s_kv_h;
        half* s_v = s_kv_h + tile_elems;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int slot = idx / head_dim;
            int d    = idx % head_dim;
            int src_offset = slot * kv_slot_stride + d;
            s_k[idx] = K_block_base[src_offset];
            s_v[idx] = V_block_base[src_offset];
        }
    }
    __syncthreads();

    for (int blk = first_block; blk < num_ctx_blocks; blk++) {
        // Current data is in buffer `buf`
        const half* s_k_cur = s_kv_h + buf * 2 * tile_elems;
        const half* s_v_cur = s_k_cur + tile_elems;

        // Start loading next block into the other buffer (overlaps with compute)
        int next_buf = 1 - buf;
        if (blk + 1 < num_ctx_blocks) {
            int next_phys = bt[blk + 1];
            const half* K_next = K_cache + (int64_t)next_phys * kv_block_stride
                                 + kv_head * head_dim;
            const half* V_next = V_cache + (int64_t)next_phys * kv_block_stride
                                 + kv_head * head_dim;
            half* s_k_next = s_kv_h + next_buf * 2 * tile_elems;
            half* s_v_next = s_k_next + tile_elems;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int slot = idx / head_dim;
                int d    = idx % head_dim;
                int src_offset = slot * kv_slot_stride + d;
                s_k_next[idx] = K_next[src_offset];
                s_v_next[idx] = V_next[src_offset];
            }
        }

        // === Per-Q-head attention computation (on current buffer) ===
        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        if (active) {
            for (int ti = warp_in_q + first_tok; ti < (tok_end - tok_start); ti += warps_per_q) {
                float dot = 0.0f;
                for (int i = 0; i < elems_per_thread; i++) {
                    int d = lane_id + i * WARP_SIZE;
                    if (d < head_dim) {
                        dot += q_reg[i] * __half2float(s_k_cur[ti * head_dim + d]);
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

                for (int i = 0; i < elems_per_thread; i++) {
                    int d = lane_id + i * WARP_SIZE;
                    float v_val = (d < head_dim) ? __half2float(s_v_cur[ti * head_dim + d]) : 0.0f;
                    o_reg[i] = rescale * o_reg[i] + w_new * v_val;
                }

                m_w = m_new;
                l_w = l_new;
            }
        }

        __syncthreads();  // Wait for both next-block load and current compute
        buf = next_buf;
    }

    if (!active) return;

    // ---- Cross-warp reduction within each Q head group ----
    float* red_max = reinterpret_cast<float*>(smem_gqa);
    float* red_l   = red_max + total_warps;
    float* red_o   = red_l   + total_warps;

    if (lane_id == 0) {
        red_max[warp_id] = m_w;
        red_l[warp_id]   = l_w;
    }
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
            red_o[warp_id * head_dim + d] = o_reg[i];
        }
    }
    __syncthreads();

    if (warp_in_q == 0) {
        int base_w = q_local * warps_per_q;

        float global_max = -FLT_MAX;
        for (int w = 0; w < warps_per_q; w++) {
            global_max = fmaxf(global_max, red_max[base_w + w]);
        }

        float global_l = 0.0f;
        for (int w = 0; w < warps_per_q; w++) {
            global_l += expf(red_max[base_w + w] - global_max) * red_l[base_w + w];
        }

        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
                float o_val = 0.0f;
                for (int w = 0; w < warps_per_q; w++) {
                    float weight = expf(red_max[base_w + w] - global_max) * red_l[base_w + w];
                    o_val += weight * red_o[(base_w + w) * head_dim + d];
                }
                if (global_l > 0.0f) o_val /= global_l;

                int out_idx = batch_idx * n_heads * head_dim
                            + head_idx * head_dim + d;
                O[out_idx] = __float2half(o_val);
            }
        }
    }
}

// Fallback for non-GQA models (n_heads == n_kv_heads): per-head kernel
// Templated on HEAD_DIM with contiguous lane mapping and half2 vectorized loads.
template<int HEAD_DIM>
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
    int block_size,
    float scale,
    int max_context_len,
    int max_num_blocks,
    int sliding_window,
    float softcap)
{
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int kv_head   = head_idx;  // MHA: kv_head == head_idx

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

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window) {
        effective_start = ctx_len - sliding_window;
    }
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const half* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const half* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const half* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // Vectorized Q.K dot product using half2 loads
            float dot = 0.0f;
            {
                const half2* K_tok2 = reinterpret_cast<const half2*>(K_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    half2 k2 = K_tok2[i];
                    dot += q_reg[2*i]   * __half2float(k2.x);
                    dot += q_reg[2*i+1] * __half2float(k2.y);
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

            // Vectorized V accumulation using half2 loads
            const half* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            {
                const half2* V_tok2 = reinterpret_cast<const half2*>(V_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    half2 v2 = V_tok2[i];
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * __half2float(v2.x);
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * __half2float(v2.y);
                }
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    extern __shared__ char smem[];
    float* warp_max = reinterpret_cast<float*>(smem);
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

// Generic non-templated fallback for arbitrary head_dim (e.g. tests with head_dim=8).
// Uses strided lane mapping with bounds checks. No vectorization.
__global__ void paged_attention_decode_kernel_generic(
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
    int sliding_window,
    float softcap)
{
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int kv_head   = head_idx;  // MHA: kv_head == head_idx

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    // Load Q into registers (strided mapping with bounds check)
    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * head_dim
                          + (int64_t)head_idx  * head_dim;
    float q_reg[8]; // max elems_per_thread for reasonable head_dim
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        q_reg[i] = (d < head_dim) ? __half2float(Q_ptr[d]) : 0.0f;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * head_dim;
    const int kv_slot_stride  = n_kv_heads * head_dim;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[8];
    for (int i = 0; i < elems_per_thread; i++) o_reg[i] = 0.0f;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window) {
        effective_start = ctx_len - sliding_window;
    }
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const half* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const half* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const half* K_tok = K_block + t * kv_slot_stride + kv_head * head_dim;

            float dot = 0.0f;
            for (int i = 0; i < elems_per_thread; i++) {
                int d = lane_id + i * WARP_SIZE;
                if (d < head_dim)
                    dot += q_reg[i] * __half2float(K_tok[d]);
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

            const half* V_tok = V_block + t * kv_slot_stride + kv_head * head_dim;
            for (int i = 0; i < elems_per_thread; i++) {
                int d = lane_id + i * WARP_SIZE;
                if (d < head_dim)
                    o_reg[i] = rescale * o_reg[i] + w_new * __half2float(V_tok[d]);
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    extern __shared__ char smem[];
    float* warp_max = reinterpret_cast<float*>(smem);
    float* warp_l   = warp_max + NUM_WARPS;
    float* warp_o   = warp_l   + NUM_WARPS;

    if (lane_id == 0) {
        warp_max[warp_id] = m_w;
        warp_l[warp_id]   = l_w;
    }
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim)
            warp_o[warp_id * head_dim + d] = o_reg[i];
    }
    __syncthreads();

    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, warp_max[w]);

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            global_l += expf(warp_max[w] - global_max) * warp_l[w];

        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
                float o_val = 0.0f;
                for (int w = 0; w < NUM_WARPS; w++) {
                    float weight = expf(warp_max[w] - global_max) * warp_l[w];
                    o_val += weight * warp_o[w * head_dim + d];
                }
                if (global_l > 0.0f) o_val /= global_l;

                int out_idx = batch_idx * n_heads * head_dim
                            + head_idx * head_dim + d;
                O[out_idx] = __float2half(o_val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K Paged Attention -- Phase 1: partial attention over context splits
// ---------------------------------------------------------------------------
//
// Grid: (batch, n_heads, num_splits)
// Each block processes a subset of KV blocks and writes partial softmax state
// (max, log_sum_exp, O_accumulator) to global scratch buffers.
// Phase 2 reduces these partials into the final output.
//
// This increases SM utilization from n_heads blocks to n_heads * num_splits.
// E.g. 32 heads * 8 splits = 256 blocks >> 170 SMs on RTX 5090.
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_splitk_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    float* __restrict__ partial_out,     // [batch, n_heads, num_splits, (1 + 1 + head_dim)]
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
    constexpr int K_VEC8 = ELEMS / 8;   // # of float4 (128-bit) loads per thread
    constexpr int K_REM  = ELEMS - K_VEC8 * 8;  // remaining elements for half2 path

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // ---- Contiguous thread-to-element mapping for vectorized loads ----
    constexpr int lane_elems = ELEMS;
    const int lane_offset = lane_id * lane_elems;

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
        // Write sentinel partial: max=-inf, sum=0
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

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    // ---- Iterate over assigned KV blocks ----
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const half* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const half* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const half* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // Vectorized Q.K dot product
            float dot = 0.0f;
            {
                if constexpr (K_VEC8 > 0) {
                    // float4 (128-bit, 8 halves) loads for HEAD_DIM >= 256
                    const float4* K_v = reinterpret_cast<const float4*>(K_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < K_VEC8; i++) {
                        float4 k_raw = K_v[i];
                        const half2* k_h2 = reinterpret_cast<const half2*>(&k_raw);
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            dot += q_reg[i*8 + 2*j]   * __half2float(k_h2[j].x);
                            dot += q_reg[i*8 + 2*j+1] * __half2float(k_h2[j].y);
                        }
                    }
                    // Remainder via half2
                    if constexpr (K_REM > 0) {
                        constexpr int done = K_VEC8 * 8;
                        const half2* K_rem = reinterpret_cast<const half2*>(K_tok + lane_offset + done);
                        #pragma unroll
                        for (int i = 0; i < K_REM / 2; i++) {
                            half2 k2 = K_rem[i];
                            dot += q_reg[done + 2*i]   * __half2float(k2.x);
                            dot += q_reg[done + 2*i+1] * __half2float(k2.y);
                        }
                    }
                } else {
                    // Small head_dim (< 256): use half2 path
                    const half2* K_tok2 = reinterpret_cast<const half2*>(K_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < ELEMS / 2; i++) {
                        half2 k2 = K_tok2[i];
                        dot += q_reg[2*i]   * __half2float(k2.x);
                        dot += q_reg[2*i+1] * __half2float(k2.y);
                    }
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

            // Vectorized V accumulation
            const half* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            {
                if constexpr (K_VEC8 > 0) {
                    const float4* V_v = reinterpret_cast<const float4*>(V_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < K_VEC8; i++) {
                        float4 v_raw = V_v[i];
                        const half2* v_h2 = reinterpret_cast<const half2*>(&v_raw);
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            o_reg[i*8 + 2*j]   = rescale * o_reg[i*8 + 2*j]   + w_new * __half2float(v_h2[j].x);
                            o_reg[i*8 + 2*j+1] = rescale * o_reg[i*8 + 2*j+1] + w_new * __half2float(v_h2[j].y);
                        }
                    }
                    if constexpr (K_REM > 0) {
                        constexpr int done = K_VEC8 * 8;
                        const half2* V_rem = reinterpret_cast<const half2*>(V_tok + lane_offset + done);
                        #pragma unroll
                        for (int i = 0; i < K_REM / 2; i++) {
                            half2 v2 = V_rem[i];
                            o_reg[done + 2*i]   = rescale * o_reg[done + 2*i]   + w_new * __half2float(v2.x);
                            o_reg[done + 2*i+1] = rescale * o_reg[done + 2*i+1] + w_new * __half2float(v2.y);
                        }
                    }
                } else {
                    const half2* V_tok2 = reinterpret_cast<const half2*>(V_tok + lane_offset);
                    #pragma unroll
                    for (int i = 0; i < ELEMS / 2; i++) {
                        half2 v2 = V_tok2[i];
                        o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * __half2float(v2.x);
                        o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * __half2float(v2.y);
                    }
                }
            }

            m_w = m_new;
            l_w = l_new;
        }
    }

    // ---- Cross-warp reduction within this block ----
    extern __shared__ char smem_sk[];
    float* warp_max = reinterpret_cast<float*>(smem_sk);
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

        // Write partial result: [max, sum_exp, O_unnormalized[head_dim]]
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
            // Store unnormalized: sum_w(exp(m_w-gmax)*l_w * O_w)
            // The reduction kernel will divide by global_l across all splits.
            out[2 + d] = o_val;
        }
    }
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
// cp.async helpers for pipelined Split-K attention
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_ca_8(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(glob));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---------------------------------------------------------------------------
// Pipelined Split-K: overlaps V[t]+K[t+1] loads with K[t] dot product
// ---------------------------------------------------------------------------
//
// Each thread copies 8 bytes (4 halves) per cp.async call.
// Per-warp smem: k_buf[2][HEAD_DIM] + v_buf[HEAD_DIM] = 3 * HEAD_DIM halfs
// 8 warps × 3 × HEAD_DIM × 2B = 6 KiB for HD=128 (trivial vs reduction smem)
//
// Pipeline stages per token iteration:
//   1. cp.async(smem_v, V[t])        — start V[t] load
//   2. cp.async(smem_k[next], K[t+1]) — start K[t+1] load
//   3. commit; wait_group<1>          — wait for K[t] (from prior iter)
//   4. dot = Q · smem_k[cur]         — compute while V[t] + K[t+1] in flight
//   5. softmax update
//   6. wait_group<0>                  — wait for V[t]
//   7. O += weight * smem_v           — accumulate V
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_splitk_pipeline_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
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

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q vector into registers ----
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

    // ---- Per-warp smem for cp.async pipeline ----
    // Layout: [NUM_WARPS][3][HEAD_DIM] halfs = k_buf[0], k_buf[1], v_buf
    extern __shared__ char smem_pipe[];
    constexpr int WARP_SMEM = 3 * HEAD_DIM;  // halfs per warp
    half* my_smem = reinterpret_cast<half*>(smem_pipe) + warp_id * WARP_SMEM;
    half* k_buf0 = my_smem;
    half* k_buf1 = my_smem + HEAD_DIM;
    half* v_buf  = my_smem + 2 * HEAD_DIM;

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    // ---- Iterate over assigned KV blocks ----
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const half* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const half* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;
        int n_toks = (tok_end - tok_start) - first_tok;
        if (n_toks <= 0) continue;

        // Prime: async load K[first_tok] into k_buf0
        {
            const half* K_tok = K_block + first_tok * kv_slot_stride + kv_head * HEAD_DIM;
            // Each thread loads 8 bytes (4 halves = ELEMS halves for HD=128)
            // lane_offset gives contiguous mapping: lane_id * ELEMS
            cp_async_ca_8(&k_buf0[lane_offset], &K_tok[lane_offset]);
            cp_async_commit();
        }

        int cur = 0;  // current k_buf index (0 or 1)
        half* k_bufs[2] = {k_buf0, k_buf1};

        for (int ti = 0; ti < n_toks; ti++) {
            int t = first_tok + ti;
            const half* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // Start async V[t] load
            cp_async_ca_8(&v_buf[lane_offset], &V_tok[lane_offset]);

            // Start async K[t+1] load (if exists)
            if (ti + 1 < n_toks) {
                const half* K_next = K_block + (t + 1) * kv_slot_stride + kv_head * HEAD_DIM;
                cp_async_ca_8(&k_bufs[1 - cur][lane_offset], &K_next[lane_offset]);
            }
            cp_async_commit();

            // Wait for K[t] (the previous commit group)
            cp_async_wait_group<1>();

            // ---- Compute dot product from smem K[t] ----
            half* k_cur = k_bufs[cur];
            float dot = 0.0f;
            {
                const half2* K_tok2 = reinterpret_cast<const half2*>(k_cur + lane_offset);
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    half2 k2 = K_tok2[i];
                    dot += q_reg[2*i]   * __half2float(k2.x);
                    dot += q_reg[2*i+1] * __half2float(k2.y);
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

            // Wait for V[t]
            cp_async_wait_group<0>();

            // ---- Accumulate V from smem ----
            {
                const half2* V_tok2 = reinterpret_cast<const half2*>(v_buf + lane_offset);
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    half2 v2 = V_tok2[i];
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * __half2float(v2.x);
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * __half2float(v2.y);
                }
            }

            m_w = m_new;
            l_w = l_new;
            cur = 1 - cur;  // swap K double-buffer
        }
    }

    // ---- Cross-warp reduction ----
    // Reuse pipe smem as reduction buffer (it's no longer needed)
    __syncthreads();
    float* warp_max = reinterpret_cast<float*>(smem_pipe);
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

// ---------------------------------------------------------------------------
// Split-K Phase 2: reduce partial results across splits
// ---------------------------------------------------------------------------
// Grid: (batch, n_heads), Block: 128 threads
// Each block merges num_splits partial results for one (batch, head) pair.

__global__ void paged_attention_reduce_kernel(
    const float* __restrict__ partial_out,  // [batch, n_heads, num_splits, (2+head_dim)]
    half* __restrict__ O,                   // [batch, 1, n_heads, head_dim]
    int n_heads,
    int head_dim,
    int num_splits)
{
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid = threadIdx.x;

    const int partial_stride = 2 + head_dim;
    const float* base = partial_out
        + (int64_t)((batch_idx * n_heads + head_idx) * num_splits) * partial_stride;

    // Step 1: Find global max across all splits (thread 0)
    __shared__ float s_global_max;
    __shared__ float s_global_l;

    if (tid == 0) {
        float gmax = -FLT_MAX;
        for (int s = 0; s < num_splits; s++) {
            float m = base[s * partial_stride];
            gmax = fmaxf(gmax, m);
        }
        s_global_max = gmax;

        // Step 2: Compute global denominator
        float gl = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = base[s * partial_stride];
            float l = base[s * partial_stride + 1];
            gl += expf(m - gmax) * l;
        }
        s_global_l = gl;
    }
    __syncthreads();

    float gmax = s_global_max;
    float gl   = s_global_l;
    float inv_gl = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

    // Step 3: Each thread handles a subset of head_dim elements
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float o_val = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = base[s * partial_stride];
            float weight = expf(m - gmax);
            float o_s = base[s * partial_stride + 2 + d];
            o_val += weight * o_s;
        }
        o_val *= inv_gl;

        int out_idx = batch_idx * n_heads * head_dim
                    + head_idx * head_dim + d;
        O[out_idx] = __float2half(o_val);
    }
}

// ---------------------------------------------------------------------------
// Scratch buffer management for split-K
// ---------------------------------------------------------------------------

// Thread-local scratch buffer (managed by the caller — GraphExecutor)
static thread_local void* s_splitk_scratch = nullptr;
static thread_local size_t s_splitk_scratch_size = 0;

void paged_attention_set_splitk_scratch(void* ptr, size_t size) {
    s_splitk_scratch = ptr;
    s_splitk_scratch_size = size;
}

// ---------------------------------------------------------------------------
// Cluster-based GQA decode attention: shares KV via DSMEM across Q-heads.
// ---------------------------------------------------------------------------
//
// For GQA models with n_q_per_kv > 1, multiple Q-heads share the same KV-head.
// Without clusters, each Q-head's block independently reads KV from global memory.
// With clusters: blocks sharing a KV-head form a cluster, block 0 loads KV into
// shared memory, and other blocks read via distributed shared memory (DSMEM).
// This reduces KV global memory reads by n_q_per_kv× (4-8× typically).
//
// Grid: (batch_size, n_kv_heads)
// Cluster: (n_q_per_kv, 1, 1) — up to 8 blocks per cluster
// Each block: 256 threads (8 warps), handles one Q-head
// Block 0 in cluster: loads KV tiles into its shared memory
// All blocks: read from block 0's DSMEM for attention computation
// ---------------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_attention_cluster_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
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
    int n_q_per_kv,
    int sliding_window,
    float softcap)
{
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    const int q_local   = cluster.block_rank();  // which Q-head in this KV group [0, n_q_per_kv)
    const int batch_idx = blockIdx.x / n_q_per_kv;  // grid.x = batch_size * n_q_per_kv
    const int kv_head   = blockIdx.y;
    const int head_idx  = kv_head * n_q_per_kv + q_local;

    if (batch_idx >= batch_size || q_local >= n_q_per_kv) return;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q vector into registers ----
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

    // ---- Paged KV layout ----
    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride  = n_kv_heads * HEAD_DIM;

    // ---- Shared memory layout ----
    // Block 0: KV double-buffer [2 * block_size * HEAD_DIM] halfs
    // All blocks: reduction [NUM_WARPS * (2 + HEAD_DIM)] floats
    extern __shared__ char smem_cluster[];

    // KV tiles in block 0's shared memory (other blocks access via DSMEM)
    half* kv_tile_local = reinterpret_cast<half*>(smem_cluster);
    // Layout: for each token t in [0, block_size): K[t*HEAD_DIM..] then V[t*HEAD_DIM..]
    // Total: block_size * 2 * HEAD_DIM halfs (double-buffered: 2 buffers)
    const int per_buf = block_size * 2 * HEAD_DIM;  // K+V interleaved per buffer
    half* buf0 = kv_tile_local;

    // Reduction area (after KV tiles for all blocks, but only block 0 writes KV)
    // For block 0: after 2 * per_buf halfs. For others: at offset 0 (no KV tiles needed locally).
    // To keep it simple, put reduction at a fixed offset after the max KV tile area.
    const int kv_total_halfs = 2 * per_buf;  // two buffers
    float* red_base = reinterpret_cast<float*>(
        reinterpret_cast<char*>(smem_cluster) +
        kv_total_halfs * sizeof(half));

    float* warp_max = red_base;
    float* warp_l   = warp_max + NUM_WARPS;
    float* warp_o   = warp_l + NUM_WARPS;

    // Get DSMEM pointer to block 0's KV tiles
    half* kv_remote = cluster.map_shared_rank(kv_tile_local, 0);

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

    // ---- Context range ----
    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    // ---- Prefetch first KV block into buffer 0 (block 0 only) ----
    int cur_buf = 0;
    if (q_local == 0 && first_block < num_ctx_blocks) {
        int phys_block = bt[first_block];
        const half* K_base = K_cache + (int64_t)phys_block * kv_block_stride
                             + kv_head * HEAD_DIM;
        const half* V_base = V_cache + (int64_t)phys_block * kv_block_stride
                             + kv_head * HEAD_DIM;
        for (int idx = threadIdx.x; idx < block_size * HEAD_DIM; idx += BLOCK_THREADS) {
            int slot = idx / HEAD_DIM;
            int d    = idx % HEAD_DIM;
            buf0[slot * 2 * HEAD_DIM + d]             = K_base[slot * kv_slot_stride + d];
            buf0[slot * 2 * HEAD_DIM + HEAD_DIM + d]  = V_base[slot * kv_slot_stride + d];
        }
    }
    cluster.sync();

    // ---- Main iteration over KV blocks ----
    for (int blk = first_block; blk < num_ctx_blocks; blk++) {
        half* cur = (cur_buf == 0) ? kv_remote : (kv_remote + per_buf);

        // Start prefetching next block into other buffer (block 0 only)
        int next_buf = 1 - cur_buf;
        half* next_ptr = (next_buf == 0) ? kv_tile_local : (kv_tile_local + per_buf);
        if (q_local == 0 && blk + 1 < num_ctx_blocks) {
            int next_phys = bt[blk + 1];
            const half* K_next = K_cache + (int64_t)next_phys * kv_block_stride
                                 + kv_head * HEAD_DIM;
            const half* V_next = V_cache + (int64_t)next_phys * kv_block_stride
                                 + kv_head * HEAD_DIM;
            for (int idx = threadIdx.x; idx < block_size * HEAD_DIM; idx += BLOCK_THREADS) {
                int slot = idx / HEAD_DIM;
                int d    = idx % HEAD_DIM;
                next_ptr[slot * 2 * HEAD_DIM + d]             = K_next[slot * kv_slot_stride + d];
                next_ptr[slot * 2 * HEAD_DIM + HEAD_DIM + d]  = V_next[slot * kv_slot_stride + d];
            }
        }

        // All blocks compute attention from current buffer via DSMEM
        int tok_start = blk * block_size;
        int tok_end   = tok_start + block_size;
        if (tok_end > ctx_len) tok_end = ctx_len;
        int first_tok = 0;
        if (tok_start < effective_start) first_tok = effective_start - tok_start;

        for (int ti = warp_id + first_tok; ti < (tok_end - tok_start); ti += NUM_WARPS) {
            // Read K token from DSMEM (block 0's shared memory)
            const half* K_tok = cur + ti * 2 * HEAD_DIM;

            float dot = 0.0f;
            {
                const half2* K_tok2 = reinterpret_cast<const half2*>(K_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    half2 k2 = K_tok2[i];
                    dot += q_reg[2*i]   * __half2float(k2.x);
                    dot += q_reg[2*i+1] * __half2float(k2.y);
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

            // Read V token from DSMEM
            const half* V_tok = cur + ti * 2 * HEAD_DIM + HEAD_DIM;
            {
                const half2* V_tok2 = reinterpret_cast<const half2*>(V_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    half2 v2 = V_tok2[i];
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * __half2float(v2.x);
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * __half2float(v2.y);
                }
            }

            m_w = m_new;
            l_w = l_new;
        }

        cluster.sync();  // wait for next-block prefetch + current compute
        cur_buf = next_buf;
    }

    // ---- Cross-warp reduction within this block ----
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

        float inv_gl = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;

        half* O_ptr = O + (int64_t)batch_idx * n_heads * HEAD_DIM
                        + (int64_t)head_idx * HEAD_DIM;

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max[w] - global_max) * warp_l[w];
                o_val += weight * warp_o[w * HEAD_DIM + d];
            }
            O_ptr[d] = __float2half(o_val * inv_gl);
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
    int sliding_window, float softcap, cudaStream_t stream)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_context_len + block_size - 1) / block_size;

    // Shared memory: warp_max[NW] + warp_l[NW] + warp_o[NW * head_dim]
    size_t smem_bytes = NUM_WARPS * sizeof(float)
                      + NUM_WARPS * sizeof(float)
                      + NUM_WARPS * head_dim * sizeof(float);

    // ---- Decide whether to use split-K ----
    // Split-K improves SM utilization when n_heads * batch_size is small
    // relative to the GPU SM count. This handles both GQA and MHA models —
    // the split-K kernel maps kv_head = head_idx / (n_heads / n_kv_heads).
    int total_blocks_nosplit = batch_size * n_heads;
    int num_splits = 1;

    // Use split-K when we have spare SMs and enough context to split
    int num_ctx_blocks = (max_context_len + block_size - 1) / block_size;
    static int num_sms = kpar_n_sms();  // cached SM count query
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 2 * num_sms && s_splitk_scratch != nullptr) {
        // Target: enough blocks to keep all SMs busy (aim for ~2 blocks/SM)
        int target_blocks = 2 * num_sms;
        num_splits = (target_blocks + total_blocks_nosplit - 1) / total_blocks_nosplit;
        // Clamp: don't create more splits than KV blocks, and cap at 32
        num_splits = min(num_splits, num_ctx_blocks);
        num_splits = min(num_splits, 32);
        num_splits = max(num_splits, 1);

        // Check scratch buffer is large enough
        int partial_stride = 2 + head_dim;
        size_t needed = (size_t)batch_size * n_heads * num_splits * partial_stride * sizeof(float);
        if (needed > s_splitk_scratch_size) {
            num_splits = 1;  // fallback
        }
    }

    // Dispatch priority: split-K > GQA > MHA fallback
    // Split-K is preferred when it provides more parallelism (handles both GQA and MHA).
    // GQA kernel is used as short-context fallback when split-K isn't beneficial.
    int n_q_per_kv = n_heads / n_kv_heads;

    if (num_splits > 1) {
        // Split-K path: Phase 1 + Phase 2 (handles both GQA and MHA)
        float* partial = static_cast<float*>(s_splitk_scratch);

        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        // Use pipelined cp.async kernel on sm_90+ for better memory/compute overlap
        static int sm_ver = get_device_sm_version();
        if (sm_ver >= 90) {
            // Pipeline smem: 8 warps * 3 * head_dim * 2B (FP16)
            // Must be at least as large as reduction smem
            size_t pipe_smem = NUM_WARPS * 3 * head_dim * sizeof(half);
            size_t launch_smem = (pipe_smem > smem_bytes) ? pipe_smem : smem_bytes;

            #define LAUNCH_SPLITK_PIPE(HD) \
                paged_attention_splitk_pipeline_kernel<HD><<<grid1, block1, launch_smem, stream>>>( \
                    reinterpret_cast<const half*>(Q.data), \
                    reinterpret_cast<const half*>(K_cache.data), \
                    reinterpret_cast<const half*>(V_cache.data), \
                    partial, \
                    block_tables, context_lens, \
                    batch_size, n_heads, n_kv_heads, \
                    block_size, scale, max_num_blocks, num_splits, \
                    sliding_window, softcap)

            switch (head_dim) {
                case 64:  LAUNCH_SPLITK_PIPE(64);  break;
                case 96:  LAUNCH_SPLITK_PIPE(96);  break;
                case 128: LAUNCH_SPLITK_PIPE(128); break;
                case 256: LAUNCH_SPLITK_PIPE(256); break;
                default:
                    IMP_LOG_ERROR("paged_attention_splitk_pipeline: unsupported head_dim %d", head_dim);
                    return;
            }
            #undef LAUNCH_SPLITK_PIPE
        } else {
            #define LAUNCH_SPLITK(HD) \
                paged_attention_splitk_kernel<HD><<<grid1, block1, smem_bytes, stream>>>( \
                    reinterpret_cast<const half*>(Q.data), \
                    reinterpret_cast<const half*>(K_cache.data), \
                    reinterpret_cast<const half*>(V_cache.data), \
                    partial, \
                    block_tables, context_lens, \
                    batch_size, n_heads, n_kv_heads, \
                    block_size, scale, max_num_blocks, num_splits, \
                    sliding_window, softcap)

            switch (head_dim) {
                case 64:  LAUNCH_SPLITK(64);  break;
                case 96:  LAUNCH_SPLITK(96);  break;
                case 128: LAUNCH_SPLITK(128); break;
                case 256: LAUNCH_SPLITK(256); break;
                default:
                    IMP_LOG_ERROR("paged_attention_splitk: unsupported head_dim %d", head_dim);
                    return;
            }
            #undef LAUNCH_SPLITK
        }

        // Phase 2: reduce partials
        dim3 grid2(batch_size, n_heads);
        dim3 block2(128);

        paged_attention_reduce_kernel<<<grid2, block2, 0, stream>>>(
            partial,
            reinterpret_cast<half*>(O.data),
            n_heads, head_dim, num_splits);
    } else if (n_q_per_kv > 1) {
        // GQA path: try cluster kernel first, then fall back to cooperative GQA
        bool used_cluster = false;

        // Cluster GQA: share KV via DSMEM across Q-heads (sm_90+ only).
        // Reduces KV global memory reads by n_q_per_kv× (4-8×).
        // Cluster dimension must be a power of 2 (CUDA requirement).
        bool valid_cluster_dim = (n_q_per_kv == 2 || n_q_per_kv == 4 || n_q_per_kv == 8);
        if (valid_cluster_dim && num_ctx_blocks >= 8 &&
            (head_dim == 64 || head_dim == 96 || head_dim == 128 || head_dim == 256)) {
            size_t cluster_smem = 2 * block_size * 2 * head_dim * sizeof(half)
                                + NUM_WARPS * sizeof(float) * 2
                                + NUM_WARPS * head_dim * sizeof(float);

            dim3 cluster_grid(batch_size * n_q_per_kv, n_kv_heads);
            dim3 cluster_block(BLOCK_THREADS);

            cudaLaunchConfig_t config = {};
            config.gridDim = cluster_grid;
            config.blockDim = cluster_block;
            config.dynamicSmemBytes = cluster_smem;
            config.stream = stream;

            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim = {(unsigned int)n_q_per_kv, 1, 1};
            config.attrs = attrs;
            config.numAttrs = 1;

            #define LAUNCH_CLUSTER(HD) \
                cudaLaunchKernelEx(&config, paged_attention_cluster_kernel<HD>, \
                    reinterpret_cast<const half*>(Q.data), \
                    reinterpret_cast<const half*>(K_cache.data), \
                    reinterpret_cast<const half*>(V_cache.data), \
                    reinterpret_cast<half*>(O.data), \
                    block_tables, context_lens, \
                    batch_size, n_heads, n_kv_heads, \
                    block_size, scale, max_context_len, max_num_blocks, \
                    n_q_per_kv, sliding_window, softcap)

            switch (head_dim) {
                case 64:  LAUNCH_CLUSTER(64);  used_cluster = true; break;
                case 96:  LAUNCH_CLUSTER(96);  used_cluster = true; break;
                case 128: LAUNCH_CLUSTER(128); used_cluster = true; break;
                case 256: LAUNCH_CLUSTER(256); used_cluster = true; break;
                default: break;
            }
            #undef LAUNCH_CLUSTER
        }

        if (!used_cluster && n_q_per_kv <= MAX_Q_PER_KV) {
            // GQA-aware kernel (short context / non-cluster fallback)
            // Use 4 warps per Q head for ratio<=8 (1024 threads max),
            // 2 warps per Q head for ratio>8 (e.g. ratio=16: 16*2*32=1024)
            int warps_per_q = (n_q_per_kv <= 8) ? 4 : 2;
            int total_warps_gqa = n_q_per_kv * warps_per_q;
            int gqa_threads = total_warps_gqa * WARP_SIZE;

            size_t kv_tile_bytes = 4 * block_size * head_dim * sizeof(half);
            size_t red_bytes = total_warps_gqa * sizeof(float) * 2
                             + total_warps_gqa * head_dim * sizeof(float);
            size_t gqa_smem = (kv_tile_bytes > red_bytes) ? kv_tile_bytes : red_bytes;

            dim3 grid(batch_size, n_kv_heads);
            dim3 block(gqa_threads);

            paged_attention_gqa_kernel<<<grid, block, gqa_smem, stream>>>(
                reinterpret_cast<const half*>(Q.data),
                reinterpret_cast<const half*>(K_cache.data),
                reinterpret_cast<const half*>(V_cache.data),
                reinterpret_cast<half*>(O.data),
                block_tables, context_lens,
                batch_size, n_heads, n_kv_heads, head_dim,
                block_size, scale, max_context_len, max_num_blocks,
                n_q_per_kv, warps_per_q, sliding_window, softcap);
        } else if (!used_cluster) {
            // MHA fallback for n_q_per_kv > MAX_Q_PER_KV without cluster
            goto mha_fallback;
        }
    } else {
    mha_fallback:
        // MHA fallback: simple per-head kernel (templated + vectorized)
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

        #define LAUNCH_MHA(HD) \
            paged_attention_decode_kernel<HD><<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<const half*>(Q.data), \
                reinterpret_cast<const half*>(K_cache.data), \
                reinterpret_cast<const half*>(V_cache.data), \
                reinterpret_cast<half*>(O.data), \
                block_tables, context_lens, \
                batch_size, n_heads, n_kv_heads, \
                block_size, scale, max_context_len, max_num_blocks, \
                sliding_window, softcap)

        switch (head_dim) {
            case 64:  LAUNCH_MHA(64);  break;
            case 96:  LAUNCH_MHA(96);  break;
            case 128: LAUNCH_MHA(128); break;
            case 256: LAUNCH_MHA(256); break;
            default:
                // Generic non-templated fallback for non-standard head_dim (e.g. tests)
                paged_attention_decode_kernel_generic<<<grid, block, smem_bytes, stream>>>(
                    reinterpret_cast<const half*>(Q.data),
                    reinterpret_cast<const half*>(K_cache.data),
                    reinterpret_cast<const half*>(V_cache.data),
                    reinterpret_cast<half*>(O.data),
                    block_tables, context_lens,
                    batch_size, n_heads, n_kv_heads, head_dim,
                    block_size, scale, max_context_len, max_num_blocks,
                    sliding_window, softcap);
                break;
        }
        #undef LAUNCH_MHA
    }
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
    // K_cache layout: [num_blocks, block_size, n_kv_heads, head_dim]
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_context_len + block_size - 1) / block_size;

    size_t smem_bytes = NUM_WARPS * sizeof(float)                 // warp_max
                      + NUM_WARPS * sizeof(float)                 // warp_l
                      + NUM_WARPS * head_dim * sizeof(float);     // warp_o

    // ---- Split-K decision (same heuristic as FP16) ----
    int total_blocks_nosplit = batch_size * n_heads;
    int num_splits = 1;
    int num_ctx_blocks = (max_context_len + block_size - 1) / block_size;

    static int num_sms_fp8 = kpar_n_sms();
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 2 * num_sms_fp8 && s_splitk_scratch != nullptr) {
        int target_blocks = 2 * num_sms_fp8;
        num_splits = (target_blocks + total_blocks_nosplit - 1) / total_blocks_nosplit;
        num_splits = min(num_splits, num_ctx_blocks);
        num_splits = min(num_splits, 32);
        num_splits = max(num_splits, 1);

        int partial_stride = 2 + head_dim;
        size_t needed = (size_t)batch_size * n_heads * num_splits * partial_stride * sizeof(float);
        if (needed > s_splitk_scratch_size) {
            num_splits = 1;  // fallback
        }
    }

    if (num_splits > 1) {
        // Split-K Phase 1: FP8 kernel
        float* partial = static_cast<float*>(s_splitk_scratch);

        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        // Use pipelined cp.async kernel on sm_90+
        static int sm_ver_fp8 = get_device_sm_version();
        if (sm_ver_fp8 >= 90) {
            // Pipeline smem: 8 warps * 3 * head_dim bytes (FP8)
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

        // Split-K Phase 2: reuse existing reduce kernel (dtype-agnostic)
        dim3 grid2(batch_size, n_heads);
        dim3 block2(128);

        paged_attention_reduce_kernel<<<grid2, block2, 0, stream>>>(
            partial,
            reinterpret_cast<half*>(O.data),
            n_heads, head_dim, num_splits);
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
                            k_val = K_v[i];
                        }
                    } else {
                        k_val = K_v[i];
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
                            packed = V_v[i];
                        }
                    } else {
                        packed = V_v[i];
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
                            k_val = K_v[i];
                        }
                    } else {
                        k_val = K_v[i];
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
                            packed = V_v[i];
                        }
                    } else {
                        packed = V_v[i];
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
            O[out_idx] = __float2half(o_val);
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

    static int num_sms_int8 = kpar_n_sms();
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 2 * num_sms_int8 && s_splitk_scratch != nullptr) {
        int target_blocks = 2 * num_sms_int8;
        num_splits = (target_blocks + total_blocks_nosplit - 1) / total_blocks_nosplit;
        num_splits = min(num_splits, num_ctx_blocks);
        num_splits = min(num_splits, 32);
        num_splits = max(num_splits, 1);

        int partial_stride = 2 + head_dim;
        size_t needed = (size_t)batch_size * n_heads * num_splits * partial_stride * sizeof(float);
        if (needed > s_splitk_scratch_size) {
            num_splits = 1;
        }
    }

    if (num_splits > 1) {
        float* partial = static_cast<float*>(s_splitk_scratch);

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

        // Phase 2: reuse existing reduce kernel (dtype-agnostic)
        dim3 grid2(batch_size, n_heads);
        dim3 block2(128);

        paged_attention_reduce_kernel<<<grid2, block2, 0, stream>>>(
            partial,
            reinterpret_cast<half*>(O.data),
            n_heads, head_dim, num_splits);
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
