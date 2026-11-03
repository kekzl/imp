#pragma once

// SfAtom NVFP4 activation quantize of one 16-value block and the dense SwiGLU in front of it, shared by
// src/compute/nvfp4_sfatom_quant.cu and the tile GEMM epilogue (quant/nvfp4_gemm_tile_epilogue.cuh): one copy
// of the bytes the prompt down GEMM reads. Moved verbatim from nvfp4_sfatom_quant.cu.

#include "quant/nvfp4_pack.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// SfAtom, K-major, 16 values per scale: 128 rows x 4 K-groups per 512 B atom, row r at (r%32)*16 + (r/32)*4,
// atoms K-major per 128-row tile; n_k_tiles = ceil(K / 64).
__device__ __host__ __forceinline__ int sfatom_offset(int row, int k_group, int n_k_tiles) {
    int tile_row = row / 128;
    int tile_k = k_group / 4;
    int row_local = row % 128;
    int k_local = k_group % 4;

    int n0 = row_local % 32;  // within 32-row sub-block
    int n1 = row_local / 32;  // which of 4 sub-blocks

    int atom_offset = (n0 * 16) + (n1 * 4) + k_local;
    int tile_base = ((tile_row * n_k_tiles) + tile_k) * 512;
    return tile_base + atom_offset;
}

// One 16-value block: UE4M3 scale into *ue (nvfp4_encode_micro_scale: clamp [2^-9, 448]), FP4 byte i = values
// 2i (low), 2i + 1 into q[i]. act_scale: static activation tensor scale, 1.0 = none. True = scale clipped
// (#1544).
__device__ __forceinline__ bool nvfp4_quant_block16(const float* vals, float local_absmax, float act_scale,
                                                    uint8_t* ue, uint8_t (&q)[8]) {
    const bool clipped = local_absmax / (act_scale * 6.0f) > 448.0f;
    const float actual_scale = nvfp4_encode_micro_scale(local_absmax, act_scale, ue);
    float inv_scale = 1.0f / (act_scale * actual_scale);
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        float s0 = vals[i] * inv_scale;
        float s1 = vals[i + 1] * inv_scale;
        q[i / 2] = nvfp4_pack_pair_hw(s0, s1);
    }
    return clipped;
}

// Dense SwiGLU of 16 fp16 gate / up values (two 16 B vectors each): fp32 math, fp16 rounding into o, the
// rounded values into vals; returns their absmax.
__device__ __forceinline__ float swiglu_block16(const float4 (&gv)[2], const float4 (&uv)[2],
                                                float (&vals)[16], float4 (&ov)[2]) {
    float amax = 0.0f;
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const half2* g2 = reinterpret_cast<const half2*>(&gv[v]);
        const half2* u2 = reinterpret_cast<const half2*>(&uv[v]);
        half2* o2 = reinterpret_cast<half2*>(&ov[v]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 gf = __half22float2(g2[k]);
            const float2 uf = __half22float2(u2[k]);
            o2[k] = __float22half2_rn(
                make_float2(gf.x / (1.0f + __expf(-gf.x)) * uf.x, gf.y / (1.0f + __expf(-gf.y)) * uf.y));
            const float2 rf = __half22float2(o2[k]);
            vals[(v * 8) + (k * 2)] = rf.x;
            vals[(v * 8) + (k * 2) + 1] = rf.y;
            amax = fmaxf(amax, fmaxf(fabsf(rf.x), fabsf(rf.y)));
        }
    }
    return amax;
}

}  // namespace imp
