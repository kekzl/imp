#pragma once

// MoE kernel declarations + launchers for executor_forward_moe*.cu; kept out of executor_forward_moe_internal.h
// so host-only MoE TUs compile as C++ (#2209).

#include "exec/executor_kernels.cuh"  // executor kernels the MoE paths also launch

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {

__global__ void sanitize_fp16_kernel(__half* __restrict__ data, int64_t n);
__global__ void moe_apply_per_expert_scale_kernel(
    float* __restrict__ weights, const int32_t* __restrict__ indices,
    const __half* __restrict__ scales, int n_weights);

inline void sanitize_fp16(__half* data, int64_t n, cudaStream_t stream) {
    if (n <= 0)
        return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    sanitize_fp16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

}  // namespace imp
