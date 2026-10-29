#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace imp {

// Own NVFP4 x NVFP4 prefill GEMM on the CUTLASS operand bytes: D[M,N] fp16 = alpha * A W^T.
// A [M,K/2] / W [N,K/2] packed E2M1 row-major, scales UE4M3 in SfAtom layout
// (cutlass_nvfp4_sf_size, rows padded to 128). K % 128 == 0, N % 8 == 0; false = refused. variant -1 = auto;
// 0 128x128, 1 128x256, 2 sync, 3 stream-K 128x128 (bench). workspace: stream-K partials
// (SMs x 64 KiB); without it auto never picks stream-K.
[[nodiscard]] bool gemm_nvfp4_tile(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf,
                                   float alpha, void* d_fp16, int M, int N, int K, cudaStream_t stream,
                                   int variant = -1, void* workspace = nullptr, size_t workspace_bytes = 0);

// Same kernel on MXFP8 operands (gemm_cutlass_mxfp8_sm120.h bytes): A [M,K] / W [N,K] E4M3, UE8M0 per 32 K
// in SfAtom layout (cutlass_mxfp8_sf_size). K % 64 == 0, N % 8 == 0. col_scale (may be null): D column n
// times col_scale[n], the FP8 row-scale view. Variants and workspace as above.
[[nodiscard]] bool gemm_mxfp8_tile(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf,
                                   const float* col_scale, void* d_fp16, int M, int N, int K,
                                   cudaStream_t stream, int variant = -1, void* workspace = nullptr,
                                   size_t workspace_bytes = 0);

}  // namespace imp
