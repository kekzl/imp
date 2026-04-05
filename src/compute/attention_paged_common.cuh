#pragma once

#include "compute/warp_reduce.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace imp {

// Shared constants for paged attention kernels
static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_THREADS = 256;
static constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;  // 8

// cp.async helpers for pipelined Split-K attention
__device__ __forceinline__ void cp_async_ca_8(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(glob));
}

// 16-byte async copy: loads 8 halves (128 bits) in one instruction.
// 2× bandwidth per instruction vs 8-byte variant. Requires 16-byte aligned addresses.
__device__ __forceinline__ void cp_async_ca_16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(glob));
}

// Streaming variant: cache at global level only (skip L1), evict-first from L2.
// Used for KV cache loads that have no intra-step reuse across kernels.
__device__ __forceinline__ void cp_async_cg_8(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(glob));
}

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(glob));
}

// ---------------------------------------------------------------------------
// L2 streaming load/store hints for paged attention decode.
// KV cache data is read once per decode step with no inter-kernel reuse.
// Streaming loads (__ldcs = .cs) hint L2 to evict these lines first,
// preserving L2 space for weight data used by subsequent FFN GEMV kernels.
// ---------------------------------------------------------------------------
__device__ __forceinline__ half ldcs_half(const half* p) {
    return __ushort_as_half(__ldcs(reinterpret_cast<const unsigned short*>(p)));
}

__device__ __forceinline__ half2 ldcs_half2(const half2* p) {
    unsigned int v = __ldcs(reinterpret_cast<const unsigned int*>(p));
    half2 r; memcpy(&r, &v, 4); return r;
}

__device__ __forceinline__ void stcs_half(half* p, half v) {
    __stcs(reinterpret_cast<unsigned short*>(p), __half_as_ushort(v));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Detect GPU SM count for split-K occupancy decisions. Cached after first call.
static inline int kpar_n_sms() {
    static int n_sms = 0;
    if (__builtin_expect(n_sms == 0, 0)) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        n_sms = prop.multiProcessorCount;
    }
    return n_sms;
}

// ---------------------------------------------------------------------------
// Online softmax step: update running max (m_w), sum-of-exp (l_w), and
// compute rescale factor for accumulated output and new attention weight.
// Used by all paged attention kernel variants (FP16, FP8, INT8, INT4).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void online_softmax_step(
    float dot, float& m_w, float& l_w, float& rescale, float& w_new) {
    float m_new = fmaxf(m_w, dot);
    float exp_diff = expf(m_w - m_new);
    float p = expf(dot - m_new);
    float l_new = exp_diff * l_w + p;
    rescale = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
    w_new = (l_new > 0.0f) ? (p / l_new) : 0.0f;
    m_w = m_new;
    l_w = l_new;
}

// ---------------------------------------------------------------------------
// Apply attention logit softcapping: tanh-based clamping used by some models.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float apply_softcap(float dot, float softcap) {
    return (softcap > 0.0f) ? (softcap * tanhf(dot / softcap)) : dot;
}

// ---------------------------------------------------------------------------
// Write sentinel partial result for an empty split-K split (max=-inf, sum=0,
// O=0). Must be called with all threads in the block active; only
// threadIdx.x==0 writes the scalar fields and the first warp zeroes O.
// ---------------------------------------------------------------------------
template<int HEAD_DIM>
__device__ __forceinline__ void write_empty_split_sentinel(
    float* partial_out, int batch_idx, int n_heads, int head_idx,
    int num_splits, int split_idx, int lane_offset) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
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
}

// ---------------------------------------------------------------------------
// Split-K decision logic shared across all paged attention host launchers.
// Returns the number of splits (1 = no split-K). Checks scratch buffer size.
//
// Forward-declare the scratch accessor from attention_paged.h so this header
// remains self-contained for .cu files that already include it.
// ---------------------------------------------------------------------------
void paged_attention_get_splitk_scratch(void** out_ptr, size_t* out_size);

static inline int compute_splitk_splits(
    int batch_size, int n_heads, int head_dim,
    int max_context_len, int block_size,
    void** out_scratch_ptr) {
    int total_blocks_nosplit = batch_size * n_heads;
    int num_ctx_blocks = (max_context_len + block_size - 1) / block_size;

    void* scratch_ptr = nullptr;
    size_t scratch_size = 0;
    paged_attention_get_splitk_scratch(&scratch_ptr, &scratch_size);

    int num_splits = 1;
    static int num_sms_cached = kpar_n_sms();
    if (num_ctx_blocks >= 4 && total_blocks_nosplit < 2 * num_sms_cached && scratch_ptr != nullptr) {
        int target_blocks = 2 * num_sms_cached;
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

    if (out_scratch_ptr) *out_scratch_ptr = scratch_ptr;
    return num_splits;
}

// ---------------------------------------------------------------------------
// WMMA tile dimensions shared by attention_tc.cu and attention_blackwell.cu.
// ---------------------------------------------------------------------------
static constexpr int kWmmaTileM = 16;
static constexpr int kWmmaTileN = 16;
static constexpr int kWmmaTileK = 16;

// ---------------------------------------------------------------------------
// Compute KV tile loop bounds for causal + sliding_window masking.
// Shared by all WMMA prefill attention kernels.
//
// On entry:  first_kv_tile = 0, num_kv_tiles = ceil(seq_kv / Bc).
// On return: both are narrowed to the range that can produce non-masked scores.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_kv_tile_bounds(
    int q_start, int Br, int Bc, int seq_q, int seq_kv,
    bool causal, int sliding_window,
    int& first_kv_tile, int& num_kv_tiles) {
    num_kv_tiles = (seq_kv + Bc - 1) / Bc;
    first_kv_tile = 0;
    if (causal) {
        int max_q = q_start + Br - 1;
        if (max_q >= seq_q) max_q = seq_q - 1;
        int furthest_kv_tile = (max_q + Bc) / Bc;
        if (furthest_kv_tile < num_kv_tiles) num_kv_tiles = furthest_kv_tile;
    }
    if (sliding_window > 0) {
        int earliest_kv = q_start - sliding_window + 1;
        if (earliest_kv > 0) {
            first_kv_tile = earliest_kv / Bc;
        }
    }
}

// ---------------------------------------------------------------------------
// Apply scale, softcap, and causal/sliding_window mask to a score tile.
// S_tile is [Br x Bc] row-major.  Called by all threads in the block with a
// strided loop.  Used by both Hopper and Blackwell WMMA prefill kernels.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void apply_score_masks(
    float* S_tile, int Br, int Bc, int block_threads,
    int tid, int q_start, int kv_start,
    int seq_q, int seq_kv, float scale,
    float softcap, bool causal, int sliding_window) {
    const int total = Br * Bc;
    for (int i = tid; i < total; i += block_threads) {
        int r = i / Bc;
        int c = i % Bc;
        int gq = q_start  + r;
        int gk = kv_start + c;

        if (gq < seq_q && gk < seq_kv) {
            float val = S_tile[i] * scale;
            if (softcap > 0.0f) val = apply_softcap(val, softcap);
            if (causal && gq < gk) val = -FLT_MAX;
            if (sliding_window > 0 && (gq - gk) >= sliding_window) val = -FLT_MAX;
            S_tile[i] = val;
        } else {
            S_tile[i] = -FLT_MAX;
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-warp reduction: merge per-warp softmax states (m_w, l_w, o_reg)
// into final output. Shared by all paged attention decode kernels.
//
// Non-split variant: writes normalized FP16 output directly to O[].
// Requires shared memory: float[NUM_WARPS + NUM_WARPS + NUM_WARPS * HEAD_DIM].
// Only the first warp (warp_id==0) writes to global memory.
// ---------------------------------------------------------------------------
template<int HEAD_DIM>
__device__ __forceinline__ void crosswarp_reduce_and_write(
    float* smem_base,      // shared memory region (warp_max | warp_l | warp_o)
    float m_w, float l_w,  // per-warp softmax state
    const float* o_reg,    // per-thread O accumulator [ELEMS]
    int warp_id, int lane_id, int lane_offset,
    half* O, int batch_idx, int n_heads, int head_idx) {

    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    float* warp_max = smem_base;
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
            stcs_half(&O[out_idx], __float2half(o_val));
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-warp reduction for Split-K: writes partial result
// (global_max, global_l, O_unnormalized) to partial_out buffer.
// The split-K reduction kernel merges partials into final output.
// ---------------------------------------------------------------------------
template<int HEAD_DIM>
__device__ __forceinline__ void crosswarp_reduce_splitk(
    float* smem_base,
    float m_w, float l_w,
    const float* o_reg,
    int warp_id, int lane_id, int lane_offset,
    float* partial_out, int batch_idx, int n_heads, int head_idx,
    int num_splits, int split_idx) {

    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    float* warp_max = smem_base;
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

} // namespace imp
