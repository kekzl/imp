#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Dense Q8_0 (GGUF 34-B blocks, read in place) x Q8_1 dp4a GEMM for small M: out[M,N] = x[M,K] W^T
// (beta 0) or out += (beta 1). Quantizes x into q8_scratch [M*K/32 block_q8_1] / d8_scratch
// [M*K/32 float]. False (nothing launched) on K % 32, beta not 0/1, or K past the smem tile.
[[nodiscard]] bool gemm_q8_0_dp4a_dense(const void* w_q8_0, const half* x, half* out, void* q8_scratch,
                                        float* d8_scratch, int M, int N, int K, float beta,
                                        cudaStream_t stream);

// Rows per weight pass for this K (16, 8, 4; 0 = activation tile does not fit 99 KiB smem).
[[nodiscard]] int gemm_q8_0_dp4a_dense_tile_m(int K);

}  // namespace imp
