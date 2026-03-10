#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// Block-diagonal Walsh-Hadamard transform on FP16 vectors.
//
// Applies the normalized WHT (1/sqrt(block_size)) independently to each
// contiguous block of `block_size` elements along the K dimension.
// This is the "online rotation" step needed for MR-GPTQ / QuTLASS:
//   X_rotated = X @ H_k  (block-diagonal Hadamard)
//
// The WHT uses the butterfly (Cooley-Tukey) decomposition:
//   log2(block_size) stages of paired add/sub operations.
// For block_size <= 32: pure warp-shuffle implementation (no shared memory).
// For block_size 64/128: warp shuffle + shared memory for cross-warp stages.
//
// input/output: [M, K] FP16 on device. K must be divisible by block_size.
// In-place: output == input is supported.

void hadamard_transform_fp16(const half* input, half* output,
                              int M, int K, int block_size,
                              cudaStream_t stream = nullptr);

// Check if a block size is supported (must be power of 2, 16..128).
inline bool hadamard_block_size_valid(int block_size) {
    return block_size == 16 || block_size == 32 ||
           block_size == 64 || block_size == 128;
}

} // namespace imp
