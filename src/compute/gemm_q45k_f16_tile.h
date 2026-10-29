#pragma once

#include <cuda_runtime.h>

namespace imp {

// Fused Q4_K / Q5_K GEMM: D[M,N] fp16 = alpha * A W^T + beta * D; A [M,K] fp16 row-major (16 B aligned),
// W the raw GGUF blocks [N, K/256 x 144 B (Q4_K) / 176 B (Q5_K)] (16 B aligned), D [M,N] fp16 (16 B
// aligned). Equals dequant_gpu + gemm_f16_tile bit for bit. K % 256 == 0, N % 8 == 0; false = refused.
// variant (fp16 acc): -1 auto, 1 128x128, 2 256x128.
[[nodiscard]] bool gemm_q45k_f16_tile(bool q5k, const void* a, const void* w, void* d, int M, int N, int K,
                                      float alpha, float beta, bool acc_fp16, cudaStream_t stream,
                                      int variant = -1);

}  // namespace imp
