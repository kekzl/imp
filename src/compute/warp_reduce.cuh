#pragma once

#include <cuda_runtime.h>
#include <float.h>

namespace imp {

static constexpr int kWarpSize = 32;

// Order-preserving atomic max on a float stored as raw bits (#1305).
//
// `atomicMax((int*)addr, __float_as_int(v))` is only correct when every
// candidate is non-negative. IEEE-754 negatives have the sign bit set, so
// larger magnitude means larger unsigned pattern and MORE negative as int32:
// a signed atomicMax over negatives selects the float MINIMUM, and the usual
// -FLT_MAX sentinel (0xFF7FFFFF -> -9 437 185 as int32) beats every ordinary
// negative logit (-20.7f -> -1 046 326 805) and survives the reduction. The
// sampler's softmax then computed expf(x - (-FLT_MAX)) = inf for every entry.
//
// The standard fix: signed atomicMax for non-negative candidates, unsigned
// atomicMin for negative ones. The two are safe against each other because a
// negative float is >= 0x80000000 unsigned while a positive float is below it,
// so a negative can never displace a positive and vice versa.
__device__ __forceinline__ void atomic_max_float(float* addr, float value) {
    if (value >= 0.0f)
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
    else
        atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(value));
}

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
    int lane = threadIdx.x % kWarpSize;
    int n_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    if (lane == 0)
        s_buf[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++)
            total += s_buf[w];
        s_buf[0] = total;
    }
    __syncthreads();
    return s_buf[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* s_buf) {
    val = warp_reduce_max(val);
    int warp_id = threadIdx.x / kWarpSize;
    int lane = threadIdx.x % kWarpSize;
    int n_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    if (lane == 0)
        s_buf[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int w = 0; w < n_warps; w++)
            m = fmaxf(m, s_buf[w]);
        s_buf[0] = m;
    }
    __syncthreads();
    return s_buf[0];
}

}  // namespace imp
