#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Custom Q4_K x FP16 tiled GEMM using in-SMEM nibble decode + FP16 HMMA (m16n8k16).
// Operates directly on packed Q4_K weights without full dequant-to-FP16 materialization
// in global memory.
//
// A (activations): [M, K] FP16 row-major
// B (weights):     Q4_K packed format (array of 144-byte super-blocks, N rows x K/256 blocks)
// C (output):      [M, N] FP16 row-major
//
// Phase 0 stub: dequantizes Q4_K blocks to FP16 in shared memory, then runs
// WMMA HMMA m16n8k16 on the dequantized data. Serves as the correctness baseline.
// Optimized in-SMEM interleaved decode comes in a later phase.
//
// Constraints:
//   - M >= 16 and M % 16 == 0
//   - N >= 16 and N % 16 == 0
//   - K % 256 == 0  (Q4_K super-block granularity)
//
// Returns true on success, false if shape is unsupported.
bool mmq_q4k_hmma_gemm(const void* A_fp16, const void* B_q4k, void* C_fp16,
                       int M, int N, int K, cudaStream_t stream);

}  // namespace imp
