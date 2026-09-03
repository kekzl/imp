// Shared device helpers of the NVFP4 multitok decode kernels
// (attention_paged_nvfp4_multitok.cu, attention_paged_nvfp4_multitok_gqa.cu).
#pragma once

#include "compute/attention_paged_common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstring>

namespace imp {
namespace nvfp4_mt {

__device__ __forceinline__ float ue4m3_scale_to_float(uint8_t bits) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &bits, 1);
    return static_cast<float>(v);
}

// One packed FP4 byte (low nibble = .x, high nibble = .y) -> half2 via
// cvt.rn.f16x2.e2m1x2 (sm_120, CUDA 13.2+).
__device__ __forceinline__ half2 fp4_pair_to_half2(uint32_t byte_val) {
    uint32_t fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(fp16x2) : "r"(byte_val));
    return *reinterpret_cast<half2*>(&fp16x2);
}

// The lane's PACK packed bytes as one load (PACK = 4: one word, PACK = 2: one
// ushort). __ldg, not .cs: cross-head re-reads are L2 hits worth keeping (#1785).
template <int PACK>
__device__ __forceinline__ uint32_t load_packed(const uint8_t* __restrict__ p) {
    if constexpr (PACK == 4)
        return __ldg(reinterpret_cast<const uint32_t*>(p));
    else
        return static_cast<uint32_t>(__ldg(reinterpret_cast<const unsigned short*>(p)));
}

template <int HEAD_DIM>
__device__ __forceinline__ void load_q_half2(const half* __restrict__ Q, int batch_idx, int head_idx,
                                             int n_heads, int lane_offset, half2* q_h2) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q + (int64_t)batch_idx * n_heads * HEAD_DIM +
                                                         (int64_t)head_idx * HEAD_DIM + lane_offset);
#pragma unroll
    for (int i = 0; i < ELEMS / 2; i++)
        q_h2[i] = Q_ptr2[i];
}

}  // namespace nvfp4_mt
}  // namespace imp
