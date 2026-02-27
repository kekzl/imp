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
//   n_q_per_kv = n_heads / n_kv_heads (e.g. 8)
//   Each Q head gets NUM_WARPS_PER_Q warps (e.g. 4 warps per Q head)
//   Total warps = n_q_per_kv * NUM_WARPS_PER_Q (e.g. 8 * 4 = 32)
//   Total threads = 32 * 32 = 1024
//
// Shared memory: K tile [block_size, head_dim] + V tile [block_size, head_dim]
//   loaded cooperatively by all threads, then each Q head's warps compute
//   dot products and accumulate from the shared tile.
// ---------------------------------------------------------------------------

// For GQA kernel: 4 warps per Q head, up to 8 Q heads per KV head
static constexpr int NUM_WARPS_PER_Q = 4;
static constexpr int MAX_Q_PER_KV = 8;

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
    int sliding_window)
{
    const int batch_idx = blockIdx.x;
    const int kv_head   = blockIdx.y;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int total_warps = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Which Q head does this warp belong to?
    const int q_local = warp_id / NUM_WARPS_PER_Q;  // [0, n_q_per_kv)
    const int warp_in_q = warp_id % NUM_WARPS_PER_Q;  // [0, NUM_WARPS_PER_Q)
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
            for (int ti = warp_in_q + first_tok; ti < (tok_end - tok_start); ti += NUM_WARPS_PER_Q) {
                float dot = 0.0f;
                for (int i = 0; i < elems_per_thread; i++) {
                    int d = lane_id + i * WARP_SIZE;
                    if (d < head_dim) {
                        dot += q_reg[i] * __half2float(s_k_cur[ti * head_dim + d]);
                    }
                }
                dot = warp_reduce_sum(dot);
                dot *= scale;

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
        int base_w = q_local * NUM_WARPS_PER_Q;

        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS_PER_Q; w++) {
            global_max = fmaxf(global_max, red_max[base_w + w]);
        }

        float global_l = 0.0f;
        for (int w = 0; w < NUM_WARPS_PER_Q; w++) {
            global_l += expf(red_max[base_w + w] - global_max) * red_l[base_w + w];
        }

        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
                float o_val = 0.0f;
                for (int w = 0; w < NUM_WARPS_PER_Q; w++) {
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

// Fallback for non-GQA models (n_heads == n_kv_heads): original per-head kernel
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
    const int kv_head   = head_idx;  // MHA: kv_head == head_idx

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * head_dim
                          + (int64_t)head_idx  * head_dim;

    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    float q_reg[8];
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
                if (d < head_dim) dot += q_reg[i] * __half2float(K_tok[d]);
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;

            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

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
        if (d < head_dim) warp_o[warp_id * head_dim + d] = o_reg[i];
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
    int head_dim,
    int block_size,
    float scale,
    int max_num_blocks,
    int num_splits,
    int sliding_window)
{
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // ---- Contiguous thread-to-element mapping for vectorized loads ----
    // Each thread handles a contiguous chunk of head_dim elements.
    // Requires head_dim % WARP_SIZE == 0 (always true: 64 or 128).
    const int elems_per_thread = head_dim / WARP_SIZE;
    const int lane_offset = lane_id * elems_per_thread;

    // ---- Load Q vector into registers using half2 vectorized loads ----
    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * head_dim
                          + (int64_t)head_idx  * head_dim;

    float q_reg[8];
    {
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q_ptr + lane_offset);
        #pragma unroll
        for (int i = 0; i < elems_per_thread / 2; i++) {
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
        int partial_stride = 2 + head_dim;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;
        if (threadIdx.x == 0) {
            out[0] = -FLT_MAX;
            out[1] = 0.0f;
        }
        if (threadIdx.x < WARP_SIZE) {
            for (int i = 0; i < elems_per_thread; i++) {
                out[2 + lane_offset + i] = 0.0f;
            }
        }
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * head_dim;
    const int kv_slot_stride  = n_kv_heads * head_dim;

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[8];
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) o_reg[i] = 0.0f;

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
            const half* K_tok = K_block + t * kv_slot_stride + kv_head * head_dim;

            // Vectorized Q.K dot product
            float dot = 0.0f;
            {
#if __CUDA_ARCH__ >= 1200
                // Blackwell (sm_120+): 256-bit loads when elems_per_thread >= 16,
                // 128-bit loads when >= 8, otherwise half2 fallback.
                if constexpr (true) {
                    // Use float4 loads (128-bit, 8 halves) for elems_per_thread >= 8
                    const int K_vec8 = elems_per_thread / 8;
                    if (K_vec8 > 0) {
                        const float4* K_v = reinterpret_cast<const float4*>(K_tok + lane_offset);
                        #pragma unroll
                        for (int i = 0; i < K_vec8; i++) {
                            float4 k_raw = K_v[i];
                            const half2* k_h2 = reinterpret_cast<const half2*>(&k_raw);
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                dot += q_reg[i*8 + 2*j]   * __half2float(k_h2[j].x);
                                dot += q_reg[i*8 + 2*j+1] * __half2float(k_h2[j].y);
                            }
                        }
                        // Remainder via half2
                        const int done = K_vec8 * 8;
                        const half2* K_rem = reinterpret_cast<const half2*>(K_tok + lane_offset + done);
                        #pragma unroll
                        for (int i = 0; i < (elems_per_thread - done) / 2; i++) {
                            half2 k2 = K_rem[i];
                            dot += q_reg[done + 2*i]   * __half2float(k2.x);
                            dot += q_reg[done + 2*i+1] * __half2float(k2.y);
                        }
                    } else {
                        // Small head_dim: use half2 path
                        const half2* K_tok2 = reinterpret_cast<const half2*>(K_tok + lane_offset);
                        #pragma unroll
                        for (int i = 0; i < elems_per_thread / 2; i++) {
                            half2 k2 = K_tok2[i];
                            dot += q_reg[2*i]   * __half2float(k2.x);
                            dot += q_reg[2*i+1] * __half2float(k2.y);
                        }
                    }
                }
#else
                const half2* K_tok2 = reinterpret_cast<const half2*>(K_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < elems_per_thread / 2; i++) {
                    half2 k2 = K_tok2[i];
                    dot += q_reg[2*i]   * __half2float(k2.x);
                    dot += q_reg[2*i+1] * __half2float(k2.y);
                }
#endif
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;

            float m_new = fmaxf(m_w, dot);
            float exp_diff = expf(m_w - m_new);
            float p = expf(dot - m_new);
            float l_new = exp_diff * l_w + p;

            float rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float w_new   = (l_new > 0.0f) ? (p / l_new) : 0.0f;

            // Vectorized V accumulation
            const half* V_tok = V_block + t * kv_slot_stride + kv_head * head_dim;
            {
#if __CUDA_ARCH__ >= 1200
                if constexpr (true) {
                    const int V_vec8 = elems_per_thread / 8;
                    if (V_vec8 > 0) {
                        const float4* V_v = reinterpret_cast<const float4*>(V_tok + lane_offset);
                        #pragma unroll
                        for (int i = 0; i < V_vec8; i++) {
                            float4 v_raw = V_v[i];
                            const half2* v_h2 = reinterpret_cast<const half2*>(&v_raw);
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                o_reg[i*8 + 2*j]   = rescale * o_reg[i*8 + 2*j]   + w_new * __half2float(v_h2[j].x);
                                o_reg[i*8 + 2*j+1] = rescale * o_reg[i*8 + 2*j+1] + w_new * __half2float(v_h2[j].y);
                            }
                        }
                        const int done = V_vec8 * 8;
                        const half2* V_rem = reinterpret_cast<const half2*>(V_tok + lane_offset + done);
                        #pragma unroll
                        for (int i = 0; i < (elems_per_thread - done) / 2; i++) {
                            half2 v2 = V_rem[i];
                            o_reg[done + 2*i]   = rescale * o_reg[done + 2*i]   + w_new * __half2float(v2.x);
                            o_reg[done + 2*i+1] = rescale * o_reg[done + 2*i+1] + w_new * __half2float(v2.y);
                        }
                    } else {
                        const half2* V_tok2 = reinterpret_cast<const half2*>(V_tok + lane_offset);
                        #pragma unroll
                        for (int i = 0; i < elems_per_thread / 2; i++) {
                            half2 v2 = V_tok2[i];
                            o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * __half2float(v2.x);
                            o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * __half2float(v2.y);
                        }
                    }
                }
#else
                const half2* V_tok2 = reinterpret_cast<const half2*>(V_tok + lane_offset);
                #pragma unroll
                for (int i = 0; i < elems_per_thread / 2; i++) {
                    half2 v2 = V_tok2[i];
                    o_reg[2*i]   = rescale * o_reg[2*i]   + w_new * __half2float(v2.x);
                    o_reg[2*i+1] = rescale * o_reg[2*i+1] + w_new * __half2float(v2.y);
                }
#endif
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
    for (int i = 0; i < elems_per_thread; i++) {
        warp_o[warp_id * head_dim + lane_offset + i] = o_reg[i];
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
        int partial_stride = 2 + head_dim;
        float* out = partial_out + (int64_t)partial_idx * partial_stride;

        if (lane_id == 0) {
            out[0] = global_max;
            out[1] = global_l;
        }

        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_offset + i;
            float o_val = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight = expf(warp_max[w] - global_max) * warp_l[w];
                o_val += weight * warp_o[w * head_dim + d];
            }
            // Store unnormalized: sum_w(exp(m_w-gmax)*l_w * O_w)
            // The reduction kernel will divide by global_l across all splits.
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
// Host launcher
// ---------------------------------------------------------------------------
void paged_attention_decode(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const int* block_tables, const int* context_lens,
    int block_size, float scale, int max_context_len,
    int sliding_window, cudaStream_t stream)
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
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 128 && s_splitk_scratch != nullptr) {
        // Target: enough blocks to keep all SMs busy (aim for ~2 blocks/SM)
        // RTX 5090 = 170 SMs → target 340 blocks.
        int target_blocks = 340;
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

        paged_attention_splitk_kernel<<<grid1, block1, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(Q.data),
            reinterpret_cast<const half*>(K_cache.data),
            reinterpret_cast<const half*>(V_cache.data),
            partial,
            block_tables, context_lens,
            batch_size, n_heads, n_kv_heads, head_dim,
            block_size, scale, max_num_blocks, num_splits,
            sliding_window);

        // Phase 2: reduce partials
        dim3 grid2(batch_size, n_heads);
        dim3 block2(128);

        paged_attention_reduce_kernel<<<grid2, block2, 0, stream>>>(
            partial,
            reinterpret_cast<half*>(O.data),
            n_heads, head_dim, num_splits);
    } else if (n_q_per_kv > 1 && n_q_per_kv <= MAX_Q_PER_KV) {
        // GQA-aware kernel (short context fallback)
        int total_warps_gqa = n_q_per_kv * NUM_WARPS_PER_Q;
        int gqa_threads = total_warps_gqa * WARP_SIZE;

        // Double-buffered FP16: 4 tiles (2 buffers x {K,V}) of block_size * head_dim halfs
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
            n_q_per_kv, sliding_window);
    } else {
        // MHA fallback: simple per-head kernel
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

        paged_attention_decode_kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(Q.data),
            reinterpret_cast<const half*>(K_cache.data),
            reinterpret_cast<const half*>(V_cache.data),
            reinterpret_cast<half*>(O.data),
            block_tables, context_lens,
            batch_size, n_heads, n_kv_heads, head_dim,
            block_size, scale, max_context_len, max_num_blocks,
            sliding_window);
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
