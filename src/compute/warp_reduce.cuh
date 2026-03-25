#pragma once

#include <cuda_runtime.h>
#include <float.h>

namespace imp {

static constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reductions using warp shuffle + shared memory.
// Requires `float s_buf[32]` in shared memory (declared by the caller).
// All threads in the block must participate.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float val, float* s_buf) {
    val = warp_reduce_sum(val);
    int warp_id = threadIdx.x / kWarpSize;
    int lane    = threadIdx.x % kWarpSize;
    int n_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    if (lane == 0) s_buf[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_buf[w];
        s_buf[0] = total;
    }
    __syncthreads();
    return s_buf[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* s_buf) {
    val = warp_reduce_max(val);
    int warp_id = threadIdx.x / kWarpSize;
    int lane    = threadIdx.x % kWarpSize;
    int n_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    if (lane == 0) s_buf[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int w = 0; w < n_warps; w++) m = fmaxf(m, s_buf[w]);
        s_buf[0] = m;
    }
    __syncthreads();
    return s_buf[0];
}

} // namespace imp
