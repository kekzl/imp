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

// Streaming variant: cache at global level only (skip L1), evict-first from L2.
// Used for KV cache loads that have no intra-step reuse across kernels.
__device__ __forceinline__ void cp_async_cg_8(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(glob));
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

} // namespace imp
