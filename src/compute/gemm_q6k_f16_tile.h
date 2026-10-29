#pragma once

#include <cuda_runtime.h>

namespace imp {

// Fused Q6_K GEMM: D[M,N] fp16 = alpha * A W^T + beta * D; A [M,K] fp16 row-major (16 B aligned),
// W the raw GGUF Q6_K blocks [N, K/256 x 210 B], D [M,N] fp16 (16 B aligned). W is decoded per CTA
// stage in fp32 (dequant_q6k_v2 bits), so the output equals dequant + gemm_f16_tile bit for bit.
// K % 256 == 0, N % 8 == 0; false = refused. variant (fp16 acc): -1 auto, 1 128x128, 2 256x128.
[[nodiscard]] bool gemm_q6k_f16_tile(const void* a, const void* w_q6k, void* d, int M, int N, int K,
                                     float alpha, float beta, bool acc_fp16, cudaStream_t stream,
                                     int variant = -1);

// Route rule: deterministic, or M <= 1536 with >= 128 128x128 tiles.
[[nodiscard]] bool gemm_q6k_f16_tile_wins(int M, int N, bool deterministic);

}  // namespace imp
