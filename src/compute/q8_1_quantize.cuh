#pragma once

#include <cuda_fp16.h>

#include "compute/gemm.h"  // block_q8_1

namespace imp {

// One warp quantizes the 32-element Q8_1 block holding `idx` (lane = idx & 31): same arithmetic as
// fused_act_quantize_q8_1_kernel<IdentityAct> (gemm_dp4a.cu); writes qs, d and d8, not s.
// All 32 lanes must call it.
__device__ __forceinline__ void q8_1_quantize_warp(float val, block_q8_1* q8_out, float* d8_out, int idx) {
    float amax = fabsf(val);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    const int blk = idx >> 5;
    q8_out[blk].qs[idx & 31] = static_cast<int8_t>(__float2int_rn(val * id));
    if ((idx & 31) == 0) {
        q8_out[blk].d = __float2half(d);
        d8_out[blk] = d;
    }
}

}  // namespace imp
