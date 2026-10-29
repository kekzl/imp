#pragma once

#include <cuda_runtime.h>

namespace imp {

// Own NVFP4 x NVFP4 prefill GEMM on the CUTLASS operand bytes: D[M,N] fp16 = alpha * A W^T.
// A [M,K/2] / W [N,K/2] packed E2M1 row-major, scales UE4M3 in SfAtom layout
// (cutlass_nvfp4_sf_size, rows padded to 128). K % 128 == 0, N % 8 == 0; false = refused. variant -1 = auto;
// 0 128x128, 1 128x256, 2 sync (bench).
[[nodiscard]] bool gemm_nvfp4_tile(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf,
                                   float alpha, void* d_fp16, int M, int N, int K, cudaStream_t stream,
                                   int variant = -1);

}  // namespace imp
