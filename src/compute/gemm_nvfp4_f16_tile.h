#pragma once

#include <cuda_runtime.h>

namespace imp {

// Fused NVFP4 W4A16 GEMM: D[M,N] fp16 = alpha * A W^T + beta * D; A [M,K] fp16 row-major (16 B aligned),
// W the plain NVFP4 layout (NvFP4QuantResult): E2M1 [N, K/2] (even element low nibble, 16 B aligned) + E4M3
// micro-scales [N, K/16]. W decodes to e2m1 * e4m3 * 2^-8, exact in fp16: pass alpha = tensor_scale * 256.
// Bit-equal to gemm_f16_tile on that W. K % 64 == 0, N % 8 == 0; false = refused. variant as gemm_f16_tile.
[[nodiscard]] bool gemm_nvfp4_f16_tile(const void* a, const void* w_e2m1, const void* scales_e4m3, void* d,
                                       int M, int N, int K, float alpha, float beta, bool acc_fp16,
                                       cudaStream_t stream, int variant = -1);

}  // namespace imp
