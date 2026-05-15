#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// Tiled MMQ (matrix-mul-quantized) kernel for Q4_K weights.
//
// Computes y[M, N] = x[M, K] @ W[N, K]^T directly from Q4_K-packed weights
// without a separate dequantization pass. Activations are quantized to Q8_1
// on the fly into the supplied scratch buffer.
//
// Layout matches ggml exactly so numerics are equivalent to ggml_mmvq_q4k
// (modulo dp4a accumulation order).
//
// Constraints (Phase A):
//   - K % 256 == 0 (one Q4_K super-block per K-step)
//   - M is rounded up to TILE_M = 32 (caller pads or zeros tail outputs)
//   - N is rounded up to TILE_N = 64
//   - scratch must be >= M * (K / 32) * sizeof(ggml_block_q8_1) bytes (= 36)
//
// Designed for M >= 32 (prefill / chunked prefill). For M <= 16 use
// ggml_mmvq_q4k — it wins at very low batch (see q4k_mmvq_crossover_2026_05_15
// memo).

void mmq_q4k(const void* W,   // [N, K/256 * 144] raw Q4_K bytes
             const half* x,   // [M, K] FP16 input
             half* y,         // [M, N] FP16 output
             int M, int N, int K, void* scratch, size_t scratch_size,
             cudaStream_t stream);

// Minimum scratch size for an (M, K) call.
inline size_t mmq_q4k_scratch_bytes(int M, int K) {
    // ggml_block_q8_1 == 36 bytes; one block per 32 elements
    return static_cast<size_t>(M) * (K / 32) * 36u;
}

}  // namespace imp
