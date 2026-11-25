#pragma once
// Block reductions shared by the Inkling2-Loop kernels (inkling_ops.cu, inkling_fused.cu).

#include <cuda_runtime.h>

#include <cfloat>

namespace imp::inkling::detail {

constexpr int kThreads = 256;

// Cache row of key k: k % ring in a ring, else blocks[k / block] * block + k % block (blocks null: k).
// block 16 (the engine's kKVBlockSize) shifts: one int load per key, the same for 16 keys in a row.
__device__ __forceinline__ int kv_slot(int k, int ring, const int* blocks, int block) {
    if (ring > 0)
        return k % ring;
    if (blocks == nullptr)
        return k;
    if (block == 16)
        return (__ldg(blocks + (k >> 4)) << 4) | (k & 15);
    return __ldg(blocks + k / block) * block + k % block;
}

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum; sh holds one float per warp.
__device__ inline float block_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i)
        t += sh[i];
    __syncthreads();
    return t;
}

// K block-wide sums in one barrier round: acc[k] in, the block totals out. red holds K floats per warp.
__device__ inline void block_sum_many(float* acc, int K, float* red) {
    const int warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
    for (int k = 0; k < K; ++k) {
        const float v = warp_sum(acc[k]);
        if ((threadIdx.x & 31) == 0)
            red[k * n_warps + warp] = v;
    }
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float t = 0.0f;
        for (int w = 0; w < n_warps; ++w)
            t += red[k * n_warps + w];
        acc[k] = t;
    }
    __syncthreads();
}

__device__ inline float block_max(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = -FLT_MAX;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i)
        t = fmaxf(t, sh[i]);
    __syncthreads();
    return t;
}

}  // namespace imp::inkling::detail
