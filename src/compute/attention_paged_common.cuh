#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace imp {

// Shared constants for paged attention kernels
static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_THREADS = 256;
static constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;  // 8

// Warp-level reductions (used by FP16, FP8, and INT8 kernels)
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

} // namespace imp
