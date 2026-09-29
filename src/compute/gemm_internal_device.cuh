#pragma once

// Device half of gemm_internal.cuh (#2209): gemm_internal.cuh stays host-parseable.

namespace imp {

// Warp-level sum reduction via __shfl_down_sync. Result valid in lane 0 only.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

}  // namespace imp
