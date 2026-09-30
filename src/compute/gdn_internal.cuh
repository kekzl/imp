#pragma once

// Shared preamble for the GDN kernel TUs, split (file-size gate) into gdn.cu (fused scan,
// RMSNorm-gated-SiLU, V-head reorder, reference scan, legacy decode), gdn_scan.cu (chunkwise
// scalar variants), gdn_scan_tc.cu (WY-rep TC/WMMA variants). Host launchers in compute/gdn.h.
// Shared __device__ helper: gdn_block_sum (K/Q L2-norm reduction); other math is inlined per kernel.

#include "compute/gdn.h"
#include <cmath>
#include <type_traits>

namespace imp {

// Sum of v over nt threads (nt a power of 2, tid < nt), returned to every thread; same add order
// as the inline tree it replaces. The trailing barrier keeps s_reduce[0] live until every warp has
// read it: without it thread 0's next write races the slow warps' read (#1750, #2254).
__device__ __forceinline__ float gdn_block_sum(float* s_reduce, float v, int tid, int nt) {
    s_reduce[tid] = v;
    __syncthreads();
    for (int stride = nt / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }
    const float sum = s_reduce[0];
    __syncthreads();
    return sum;
}

}  // namespace imp
