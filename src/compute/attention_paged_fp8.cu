#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "compute/attention.h"
#include "core/logging.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>

namespace imp {

// ---------------------------------------------------------------------------
// FP8 E4M3 helper: convert a single FP8 byte to float
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 val;
    memcpy(&val, &bits, 1);
    return static_cast<float>(val);
}
// ---------------------------------------------------------------------------
// Pipelined Split-K: FP8 E4M3 variant
// ---------------------------------------------------------------------------

// Copy exactly ELEMS bytes per lane from global KV into the per-warp smem
// staging buffer. The valid cp.async transfer sizes are 4, 8 and 16 bytes, and
// each lane's address is `lane_id * ELEMS`, so the instruction size has to
// match ELEMS *and* the resulting alignment:
//
//   ELEMS 16 (hd=512) -> 16 B, lane_id*16 is 16 B aligned
//   ELEMS  8 (hd=256) ->  8 B, lane_id*8  is  8 B aligned
//   ELEMS  4 (hd=128) ->  4 B, lane_id*4  is  4 B aligned
//   ELEMS  2 (hd=64), 3 (hd=96) -> no cp.async size fits
//
// The old code had one branch for `ELEMS >= 8` (always 8 B — so hd=512 copied
// half its bytes and left the rest of K/V unwritten) and a hard-coded 4 B
// otherwise, which at hd=64 both over-copied (4 B for a 2 B slice, running past
// k_buf0 into k_buf1) and misaligned (odd lanes land on offset 2 mod 4). That
// misalignment is #1339: the kernel faulted and took the CUDA context with it.
//
// For the two head dims no cp.async size fits, the copy is synchronous. The
// pipeline loses its overlap there and nothing else changes — correct and
// slower beats fast and faulting, and hd=64/96 are not the shapes this kernel
// was tuned for.
template <int ELEMS>
__device__ __forceinline__ void fp8_kv_stage_lane(uint8_t* smem_dst, const uint8_t* glob_src) {
    if constexpr (ELEMS == 16) {
        cp_async_ca_16(smem_dst, glob_src);
    } else if constexpr (ELEMS == 8) {
        cp_async_ca_8(smem_dst, glob_src);
    } else if constexpr (ELEMS == 4) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(
                         static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),
                     "l"(glob_src));
    } else {
#pragma unroll
        for (int i = 0; i < ELEMS; i++)
            smem_dst[i] = glob_src[i];
    }
}

template <int HEAD_DIM>
__global__ void paged_attention_splitk_fp8_pipeline_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, float kv_scale, int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int FP8_VEC4 = ELEMS / 4;
    constexpr int FP8_REM = ELEMS % 4;

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
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;
    const float fused_scale = scale * kv_scale;

    // Per-warp smem: k_buf[2][ELEMS] + v_buf[ELEMS] bytes (FP8: 1 byte each)
    // Total per warp: 3 * HEAD_DIM bytes. 8 warps: 3 * 128 * 8 = 3 KiB for HD=128.
    extern __shared__ char smem_pipe_fp8[];
    constexpr int WARP_SMEM_BYTES = 3 * HEAD_DIM;
    uint8_t* my_smem = reinterpret_cast<uint8_t*>(smem_pipe_fp8) + warp_id * WARP_SMEM_BYTES;
    uint8_t* k_buf0 = my_smem;
    uint8_t* k_buf1 = my_smem + HEAD_DIM;
    uint8_t* v_buf = my_smem + 2 * HEAD_DIM;

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

        int tok_start = blk * block_size;
        int tok_end = tok_start + block_size;
        if (tok_end > ctx_len)
            tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;
        int n_toks = (tok_end - tok_start) - first_tok;
        if (n_toks <= 0)
            continue;

        // Prime: async load K[first_tok] into k_buf0
        // FP8: ELEMS bytes per thread (4 for HD=128, 8 for HD=256)
        {
            const uint8_t* K_tok = K_block + first_tok * kv_slot_stride + kv_head * HEAD_DIM;
            fp8_kv_stage_lane<ELEMS>(&k_buf0[lane_offset], &K_tok[lane_offset]);
            cp_async_commit();
        }

        int cur = 0;
        uint8_t* k_bufs[2] = {k_buf0, k_buf1};

        for (int ti = 0; ti < n_toks; ti++) {
            int t = first_tok + ti;
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // Start async V[t] + K[t+1] loads (branchless: clamp to last valid token)
            int t_next = min(t + 1, first_tok + n_toks - 1);
            const uint8_t* K_next = K_block + t_next * kv_slot_stride + kv_head * HEAD_DIM;
            fp8_kv_stage_lane<ELEMS>(&v_buf[lane_offset], &V_tok[lane_offset]);
            fp8_kv_stage_lane<ELEMS>(&k_bufs[1 - cur][lane_offset], &K_next[lane_offset]);
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
                        dot += q_reg[i * 4 + 0] * fp8_e4m3_to_float(packed & 0xFF);
                        dot += q_reg[i * 4 + 1] * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        dot += q_reg[i * 4 + 2] * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        dot += q_reg[i * 4 + 3] * fp8_e4m3_to_float((packed >> 24) & 0xFF);
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
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            cp_async_wait_group<0>();

            float w_new_scaled = w_new * kv_scale;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* V_v = reinterpret_cast<const uint32_t*>(v_buf + lane_offset);
#pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = V_v[i];
                        o_reg[i * 4 + 0] = rescale * o_reg[i * 4 + 0] +
                                           w_new_scaled * fp8_e4m3_to_float(packed & 0xFF);
                        o_reg[i * 4 + 1] = rescale * o_reg[i * 4 + 1] +
                                           w_new_scaled * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        o_reg[i * 4 + 2] = rescale * o_reg[i * 4 + 2] +
                                           w_new_scaled * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        o_reg[i * 4 + 3] = rescale * o_reg[i * 4 + 3] +
                                           w_new_scaled * fp8_e4m3_to_float((packed >> 24) & 0xFF);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
#pragma unroll
                    for (int i = 0; i < FP8_REM; i++)
                        o_reg[done + i] = rescale * o_reg[done + i] +
                                          w_new_scaled * fp8_e4m3_to_float(v_buf[lane_offset + done + i]);
                }
            }

            cur = 1 - cur;
        }
    }

    // ---- Cross-warp reduction ----
    __syncthreads();
    crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_pipe_fp8), m_w, l_w, o_reg, warp_id,
                                      lane_id, lane_offset, partial_out, batch_idx, n_heads, head_idx,
                                      num_splits, split_idx);
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

template <int HEAD_DIM>
__global__ void paged_attention_splitk_fp8_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    float* __restrict__ partial_out,  // [batch, n_heads, num_splits, (2 + HEAD_DIM)]
    const int* __restrict__ block_tables, const int* __restrict__ context_lens, int batch_size, int n_heads,
    int n_kv_heads, int block_size, float scale, float kv_scale, int max_num_blocks, int num_splits,
    int sliding_window, float softcap) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    // FP8 vectorization: 4 bytes per uint32_t load
    constexpr int FP8_VEC4 = ELEMS / 4;  // # of uint32_t loads per thread
    constexpr int FP8_REM = ELEMS % 4;   // remaining bytes

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // ---- Contiguous thread-to-element mapping ----
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q vector into registers using half2 vectorized loads ----
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
    int split_end = split_start + blocks_per_split;
    if (split_end > num_ctx_blocks)
        split_end = num_ctx_blocks;

    // Early exit if this split has no work
    if (split_start >= split_end) {
        write_empty_split_sentinel<HEAD_DIM>(partial_out, batch_idx, n_heads, head_idx, num_splits, split_idx,
                                             lane_offset);
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;

    // Fused scale: apply kv_scale together with softmax scale after dot product
    const float fused_scale = scale * kv_scale;

    // ---- Per-warp running softmax state ----
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    // ---- Iterate over assigned KV blocks ----
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;

        int tok_start = blk * block_size;
        int tok_end = tok_start + block_size;
        if (tok_end > ctx_len)
            tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // ---- Vectorized Q.K dot product with uint32_t FP8 loads ----
            float dot = 0.0f;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* K_v = reinterpret_cast<const uint32_t*>(K_tok + lane_offset);
#pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = __ldcs(&K_v[i]);
                        uint8_t b0 = packed & 0xFF;
                        uint8_t b1 = (packed >> 8) & 0xFF;
                        uint8_t b2 = (packed >> 16) & 0xFF;
                        uint8_t b3 = (packed >> 24) & 0xFF;
                        dot += q_reg[i * 4 + 0] * fp8_e4m3_to_float(b0);
                        dot += q_reg[i * 4 + 1] * fp8_e4m3_to_float(b1);
                        dot += q_reg[i * 4 + 2] * fp8_e4m3_to_float(b2);
                        dot += q_reg[i * 4 + 3] * fp8_e4m3_to_float(b3);
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
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            // ---- Vectorized V accumulation with scale folded into weight ----
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            float w_new_scaled = w_new * kv_scale;  // fuse kv_scale into weight
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* V_v = reinterpret_cast<const uint32_t*>(V_tok + lane_offset);
#pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = __ldcs(&V_v[i]);
                        uint8_t b0 = packed & 0xFF;
                        uint8_t b1 = (packed >> 8) & 0xFF;
                        uint8_t b2 = (packed >> 16) & 0xFF;
                        uint8_t b3 = (packed >> 24) & 0xFF;
                        o_reg[i * 4 + 0] = rescale * o_reg[i * 4 + 0] + w_new_scaled * fp8_e4m3_to_float(b0);
                        o_reg[i * 4 + 1] = rescale * o_reg[i * 4 + 1] + w_new_scaled * fp8_e4m3_to_float(b1);
                        o_reg[i * 4 + 2] = rescale * o_reg[i * 4 + 2] + w_new_scaled * fp8_e4m3_to_float(b2);
                        o_reg[i * 4 + 3] = rescale * o_reg[i * 4 + 3] + w_new_scaled * fp8_e4m3_to_float(b3);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    const uint8_t* V_rem = V_tok + lane_offset + done;
#pragma unroll
                    for (int i = 0; i < FP8_REM; i++) {
                        o_reg[done + i] = rescale * o_reg[done + i] +
                                          w_new_scaled * fp8_e4m3_to_float(V_rem[i]);
                    }
                }
            }
        }
    }

    // ---- Cross-warp reduction within this block ----
    extern __shared__ char smem_sk_fp8[];
    crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_sk_fp8), m_w, l_w, o_reg, warp_id,
                                      lane_id, lane_offset, partial_out, batch_idx, n_heads, head_idx,
                                      num_splits, split_idx);
}

template <int HEAD_DIM>
__global__ void paged_attention_decode_fp8_kernel(const half* __restrict__ Q,
                                                  const uint8_t* __restrict__ K_cache,  // FP8 E4M3 raw bytes
                                                  const uint8_t* __restrict__ V_cache,  // FP8 E4M3 raw bytes
                                                  half* __restrict__ O, const int* __restrict__ block_tables,
                                                  const int* __restrict__ context_lens, int batch_size,
                                                  int n_heads, int n_kv_heads, int block_size, float scale,
                                                  float kv_scale, int max_context_len, int max_num_blocks,
                                                  int sliding_window, float softcap,
                                                  const half* __restrict__ attn_sinks) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int FP8_VEC4 = ELEMS / 4;
    constexpr int FP8_REM = ELEMS % 4;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head = head_idx / (n_heads / n_kv_heads);  // GQA

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // ---- Load Q vector into registers using half2 vectorized loads ----
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
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;
    const float fused_scale = scale * kv_scale;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

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
        int tok_end = tok_start + block_size;
        if (tok_end > ctx_len)
            tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * HEAD_DIM;

            // Vectorized Q.K dot product with uint32_t FP8 loads
            float dot = 0.0f;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* K_v = reinterpret_cast<const uint32_t*>(K_tok + lane_offset);
#pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = __ldcs(&K_v[i]);
                        dot += q_reg[i * 4 + 0] * fp8_e4m3_to_float(packed & 0xFF);
                        dot += q_reg[i * 4 + 1] * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        dot += q_reg[i * 4 + 2] * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        dot += q_reg[i * 4 + 3] * fp8_e4m3_to_float((packed >> 24) & 0xFF);
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
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * HEAD_DIM;
            float w_new_scaled = w_new * kv_scale;
            {
                if constexpr (FP8_VEC4 > 0) {
                    const uint32_t* V_v = reinterpret_cast<const uint32_t*>(V_tok + lane_offset);
#pragma unroll
                    for (int i = 0; i < FP8_VEC4; i++) {
                        uint32_t packed = __ldcs(&V_v[i]);
                        o_reg[i * 4 + 0] = rescale * o_reg[i * 4 + 0] +
                                           w_new_scaled * fp8_e4m3_to_float(packed & 0xFF);
                        o_reg[i * 4 + 1] = rescale * o_reg[i * 4 + 1] +
                                           w_new_scaled * fp8_e4m3_to_float((packed >> 8) & 0xFF);
                        o_reg[i * 4 + 2] = rescale * o_reg[i * 4 + 2] +
                                           w_new_scaled * fp8_e4m3_to_float((packed >> 16) & 0xFF);
                        o_reg[i * 4 + 3] = rescale * o_reg[i * 4 + 3] +
                                           w_new_scaled * fp8_e4m3_to_float((packed >> 24) & 0xFF);
                    }
                }
                if constexpr (FP8_REM > 0) {
                    constexpr int done = FP8_VEC4 * 4;
                    const uint8_t* V_rem = V_tok + lane_offset + done;
#pragma unroll
                    for (int i = 0; i < FP8_REM; i++)
                        o_reg[done + i] = rescale * o_reg[done + i] +
                                          w_new_scaled * fp8_e4m3_to_float(V_rem[i]);
                }
            }
        }
    }

    // ---- Cross-warp reduction ----
    extern __shared__ char smem_fp8[];
    __syncthreads();
    crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_fp8), m_w, l_w, o_reg, warp_id,
                                         lane_id, lane_offset, O, batch_idx, n_heads, head_idx, attn_sinks);
}

// ---------------------------------------------------------------------------
// Host launcher -- FP8 E4M3 variant (with Split-K support)
// ---------------------------------------------------------------------------
void paged_attention_decode_fp8(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                const int* block_tables, const int* context_lens, int block_size, float scale,
                                float kv_scale, int max_context_len, int sliding_window, float softcap,
                                cudaStream_t stream, int max_blocks_per_seq, int n_sinks,
                                const void* attn_sinks) {
    // StreamingLLM (n_sinks > 0, evicted-token bookkeeping) is still not wired
    // into the FP8 kernels; classical sliding-window applies instead.
    //
    // LEARNED sinks (attn_sinks, gpt-oss) now are (#1345). They used to be
    // dropped here — the launcher had no pointer to take them through — so a
    // quantised KV cache served a softmax denominator missing the sink column,
    // and gpt-oss stopped answering at all rather than answering slightly worse.
    (void)n_sinks;
    const half* sinks_h = reinterpret_cast<const half*>(attn_sinks);
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                                                        : (max_context_len + block_size - 1) / block_size;

    size_t smem_bytes = NUM_WARPS * sizeof(float) + NUM_WARPS * sizeof(float) +
                        NUM_WARPS * head_dim * sizeof(float);

    // ---- Split-K decision ----
    void* scratch_ptr = nullptr;
    int num_splits = compute_splitk_splits(batch_size, n_heads, head_dim, max_context_len, block_size,
                                           &scratch_ptr);

    if (num_splits > 1) {
        // Split-K Phase 1: FP8 kernel
        float* partial = static_cast<float*>(scratch_ptr);

        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

        // Use pipelined cp.async kernel on sm_90+
        static int sm_ver_fp8 = get_device_sm_version();
        if (sm_ver_fp8 >= 90 &&
            paged_attention_splitk_fp8_tile_gqa_supported(head_dim, block_size, n_heads, n_kv_heads) &&
            process_diag_attention_fp8_tile() && process_diag_attention_fp8_tile_gqa()) {
            // GQA-batched tile variant: grid.y = n_kv_heads instead of n_heads
            // (each block computes all G Q heads from one shared smem tile ->
            // KV L2 traffic /G). The split count is re-derived for that
            // geometry next to the kernel.
            num_splits = paged_attention_splitk_fp8_tile_gqa_splits(batch_size, n_heads, n_kv_heads,
                                                                    head_dim, block_size, max_context_len);
            paged_attention_splitk_fp8_tile_gqa_launch(
                reinterpret_cast<const half*>(Q.data), reinterpret_cast<const uint8_t*>(K_cache.data),
                reinterpret_cast<const uint8_t*>(V_cache.data), partial, block_tables, context_lens,
                batch_size, n_heads, n_kv_heads, block_size, scale, kv_scale, max_num_blocks, num_splits,
                sliding_window, softcap, stream);
        } else if (sm_ver_fp8 >= 90 && paged_attention_splitk_fp8_tile_supported(head_dim, block_size) &&
                   process_diag_attention_fp8_tile()) {
            // Token-tiled variant (attention_paged_fp8_tile.cu): bulk-staged KV
            // pages instead of the per-token latency chain. hd=128, bs multiple of 16.
            //
            // Wave-aware split count: the tile kernel is smem-capped at 1
            // block/SM, so wall time quantizes to ceil(batch*heads*splits/SMs)
            // waves. The shared heuristic (targets 2*SMs blocks for multi-
            // block/SM kernels) lands mid-wave here — e.g. 32 heads * 11
            // splits = 2.07 waves, a nearly idle third wave. Pick the split
            // count <= the heuristic's (scratch is sized for that) minimizing
            // waves per unit of split work; ties -> fewer splits (fewer
            // per-warp pipeline prologues + smaller reduce).
            {
                const int sms = kpar_n_sms();
                const int bh = batch_size * n_heads;
                int best_s = 1;
                for (int s = 1; s <= num_splits; s++) {
                    const int waves_s = (bh * s + sms - 1) / sms;
                    const int waves_b = (bh * best_s + sms - 1) / sms;
                    if ((int64_t)waves_s * best_s < (int64_t)waves_b * s)
                        best_s = s;
                }
                num_splits = best_s;
            }
            paged_attention_splitk_fp8_tile_launch(
                reinterpret_cast<const half*>(Q.data), reinterpret_cast<const uint8_t*>(K_cache.data),
                reinterpret_cast<const uint8_t*>(V_cache.data), partial, block_tables, context_lens,
                batch_size, n_heads, n_kv_heads, block_size, scale, kv_scale, max_num_blocks, num_splits,
                sliding_window, softcap, stream);
        } else if (sm_ver_fp8 >= 90) {
            size_t pipe_smem = NUM_WARPS * 3 * head_dim;
            size_t launch_smem = (pipe_smem > smem_bytes) ? pipe_smem : smem_bytes;

#define LAUNCH_SPLITK_FP8_PIPE(HD)                                                                        \
    paged_attention_splitk_fp8_pipeline_kernel<HD>                                                        \
        <<<grid1, block1, launch_smem, stream>>>(reinterpret_cast<const half*>(Q.data),                   \
                                                 reinterpret_cast<const uint8_t*>(K_cache.data),          \
                                                 reinterpret_cast<const uint8_t*>(V_cache.data), partial, \
                                                 block_tables, context_lens, batch_size, n_heads,         \
                                                 n_kv_heads, block_size, scale, kv_scale, max_num_blocks, \
                                                 num_splits, sliding_window, softcap);                    \
    IMP_CUDA_CHECK_LAUNCH()

            switch (head_dim) {
                case 64:
                    LAUNCH_SPLITK_FP8_PIPE(64);
                    break;
                case 96:
                    LAUNCH_SPLITK_FP8_PIPE(96);
                    break;
                case 128:
                    LAUNCH_SPLITK_FP8_PIPE(128);
                    break;
                case 256:
                    LAUNCH_SPLITK_FP8_PIPE(256);
                    break;
                case 512:
                    LAUNCH_SPLITK_FP8_PIPE(512);
                    break;  // Gemma 4 global
                default:
                    paged_attention_unsupported_head_dim("paged_attention_splitk_fp8_pipeline", head_dim);
            }
#undef LAUNCH_SPLITK_FP8_PIPE
        } else {
#define LAUNCH_SPLITK_FP8(HD)                                                                                \
    paged_attention_splitk_fp8_kernel<HD>                                                                    \
        <<<grid1, block1, smem_bytes, stream>>>(reinterpret_cast<const half*>(Q.data),                       \
                                                reinterpret_cast<const uint8_t*>(K_cache.data),              \
                                                reinterpret_cast<const uint8_t*>(V_cache.data), partial,     \
                                                block_tables, context_lens, batch_size, n_heads, n_kv_heads, \
                                                block_size, scale, kv_scale, max_num_blocks, num_splits,     \
                                                sliding_window, softcap);                                    \
    IMP_CUDA_CHECK_LAUNCH()

            switch (head_dim) {
                case 64:
                    LAUNCH_SPLITK_FP8(64);
                    break;
                case 96:
                    LAUNCH_SPLITK_FP8(96);
                    break;
                case 128:
                    LAUNCH_SPLITK_FP8(128);
                    break;
                case 256:
                    LAUNCH_SPLITK_FP8(256);
                    break;
                case 512:
                    LAUNCH_SPLITK_FP8(512);
                    break;  // Gemma 4 global
                default:
                    paged_attention_unsupported_head_dim("paged_attention_splitk_fp8", head_dim);
            }
#undef LAUNCH_SPLITK_FP8
        }

        // Split-K Phase 2: reuse shared reduce launcher
        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data), batch_size, n_heads, head_dim,
                                      num_splits, stream, sinks_h);
    } else {
        // Fallback: non-Split-K FP8 kernel (templated + vectorized)
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

#define LAUNCH_FP8_FALLBACK(HD)                                                                             \
    paged_attention_decode_fp8_kernel<HD>                                                                   \
        <<<grid, block, smem_bytes, stream>>>(reinterpret_cast<const half*>(Q.data),                        \
                                              reinterpret_cast<const uint8_t*>(K_cache.data),               \
                                              reinterpret_cast<const uint8_t*>(V_cache.data),               \
                                              reinterpret_cast<half*>(O.data), block_tables, context_lens,  \
                                              batch_size, n_heads, n_kv_heads, block_size, scale, kv_scale, \
                                              max_context_len, max_num_blocks, sliding_window, softcap, \
                                              sinks_h);                                                        \
    IMP_CUDA_CHECK_LAUNCH()

        switch (head_dim) {
            case 64:
                LAUNCH_FP8_FALLBACK(64);
                break;
            case 96:
                LAUNCH_FP8_FALLBACK(96);
                break;
            case 128:
                LAUNCH_FP8_FALLBACK(128);
                break;
            case 256:
                LAUNCH_FP8_FALLBACK(256);
                break;
            case 512:
                LAUNCH_FP8_FALLBACK(512);
                break;  // Gemma 4 global attention
            default:
                paged_attention_unsupported_head_dim("paged_attention_decode_fp8", head_dim);
        }
#undef LAUNCH_FP8_FALLBACK
    }
}

}  // namespace imp
