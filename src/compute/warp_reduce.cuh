#pragma once

#include <cuda_runtime.h>
#include <float.h>

namespace imp {

static constexpr int kWarpSize = 32;

// Order-preserving atomic max on a float via raw bits (#1305). Signed atomicMax over
// negatives selects the float MINIMUM (larger magnitude = more negative as int32), so the
// -FLT_MAX sentinel (0xFF7FFFFF -> -9437185) beat every ordinary negative logit and
// survived reduction, making the softmax's expf(x-(-FLT_MAX)) = inf everywhere.
// Fix: signed atomicMax for non-negative candidates, unsigned atomicMin for negative ones -
// safe against each other since a negative float is >= 0x80000000 unsigned and a positive
// one is below it.
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

// Block-level reductions via warp shuffle + shared memory. Requires `float s_buf[32]` in
// shared memory (caller-declared); all threads in the block must participate.
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
