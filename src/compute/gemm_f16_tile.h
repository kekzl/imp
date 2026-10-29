#pragma once

#include <cuda_runtime.h>

namespace imp {

struct Tensor;

// Own FP16 dense GEMM: D[M,N] fp16 = alpha * A W^T + beta * D; A [M,K], W [N,K], D [M,N] row-major
// fp16, 16 B aligned. acc_fp16: fp16 accumulator (cuBLASLt COMPUTE_16F), else fp32.
// K % 64 == 0, N % 8 == 0; false = refused (caller keeps its library path).
// variant (fp16 accumulate only): 0 128x256, 1 128x128 (bench).
[[nodiscard]] bool gemm_f16_tile(const void* a, const void* w, void* d, int M, int N, int K, float alpha,
                                 float beta, bool acc_fp16, cudaStream_t stream, int variant = 0);

// gemm() hook: runs the own kernel for an fp16-accumulate F16 GEMM when it beats cuBLASLt in situ
// (gemm.f16_tile, K >= N, 128x256 grid >= one wave). False = not taken.
[[nodiscard]] bool gemm_f16_tile_route(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                                       cudaStream_t stream);

}  // namespace imp
