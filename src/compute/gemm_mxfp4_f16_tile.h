#pragma once

#include <cuda_runtime.h>

namespace imp {

// Fused MXFP4 W4A16 GEMM: D[M,N] fp16 = alpha * A W^T + beta * D; A [M,K] fp16 row-major (16 B aligned),
// W the unpacked MXFP4 weight (unpack_mxfp4_gguf): E2M1 [N, K/2] (even element low nibble, 16 B aligned)
// + UE8M0 linear scales [N, K/32]. Equals dequant_mxfp4_linear_to_fp16 + gemm_f16_tile bit for bit for
// scales 2^-38 .. 2^1. K % 64 == 0, N % 8 == 0; false = refused. variant (fp16 acc): -1 auto, 1 128x128,
// 2 256x128.
[[nodiscard]] bool gemm_mxfp4_f16_tile(const void* a, const void* w_e2m1, const void* scales_ue8m0, void* d,
                                       int M, int N, int K, float alpha, float beta, bool acc_fp16,
                                       cudaStream_t stream, int variant = -1);

}  // namespace imp
