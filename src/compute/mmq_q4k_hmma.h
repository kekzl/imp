#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Q4_K x FP16 tiled GEMM: in-SMEM nibble decode + FP16 HMMA (m16n8k16), no
// dequant-to-FP16 materialization in global memory.
// A:[M,K] FP16 row-major; B: Q4_K packed (144B super-blocks, N x K/256); C:[M,N] FP16.
// Constraints: M,N % 16 == 0; K % 256 == 0. Returns false if shape unsupported.
bool mmq_q4k_hmma_gemm(const void* A_fp16, const void* B_q4k, void* C_fp16,
                       int M, int N, int K, cudaStream_t stream);

}  // namespace imp
