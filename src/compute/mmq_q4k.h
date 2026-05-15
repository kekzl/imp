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
// Constraints:
//   - K % 256 == 0 (one Q4_K super-block per K-step)
//   - M, N are bounds-checked (caller may pass any shape; tail tiles get
//     masked store)
//   - scratch must be >= M * (K / 32) * sizeof(ggml_block_q8_1) bytes (= 36)
//
// Effective win zone: **M = 2..16**. Internal default tile is
// <16, 32, 1, 1> (512 threads, 1 output per thread). Beats both mmvq and
// dequant+cuBLAS at low batch. At M >= 32, FP16-TC cuBLAS wins (dp4a
// peak ~50 TFLOPS vs TC peak ~838 TFLOPS) — dispatch via gemm_dispatch
// bypasses this kernel above the threshold. See mmq_q4k_phase_a_2026_05_15
// memo for the full sweep and the cap rationale.

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
