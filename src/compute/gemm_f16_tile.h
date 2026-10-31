#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace imp {

struct Tensor;

// Own FP16 dense GEMM: D[M,N] fp16 = alpha * A W^T + beta * D; A [M,K], W [N,K], D [M,N] row-major
// fp16, 16 B aligned. acc_fp16: fp16 accumulator (cuBLASLt COMPUTE_16F), else fp32.
// K % 64 == 0, N % 8 == 0; false = refused (caller keeps its library path).
// variant (fp16 accumulate only): 0 128x256, 1 128x128 (bench).
[[nodiscard]] bool gemm_f16_tile(const void* a, const void* w, void* d, int M, int N, int K, float alpha,
                                 float beta, bool acc_fp16, cudaStream_t stream, int variant = 0);

// Narrow grids (prefill N <= a few tiles, e.g. the MoE router): D = alpha * A W^T, 128x128 fp32-accumulate
// tiles, K split over a cluster of up to 8 CTAs; fp32 partials in ws (16 B aligned), summed in rank order in
// the same launch (deterministic per shape). The split shrinks to fit ws_bytes. K % 64 == 0, N % 8 == 0;
// false = refused.
[[nodiscard]] bool gemm_f16_splitk(const void* a, const void* w, void* d, int M, int N, int K, float alpha,
                                   void* ws, size_t ws_bytes, cudaStream_t stream);
// ws bytes for the full split (8) of an M x N output.
[[nodiscard]] size_t gemm_f16_splitk_workspace_bytes(int M, int N);

// gemm() hook: runs the own kernel for an fp16-accumulate F16 GEMM when it beats cuBLASLt in situ
// (gemm.f16_tile, K >= N, 128x256 grid >= one wave). False = not taken.
[[nodiscard]] bool gemm_f16_tile_route(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                                       cudaStream_t stream);

}  // namespace imp
