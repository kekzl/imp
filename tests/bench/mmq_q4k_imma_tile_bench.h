#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// Phase 2B INT8 IMMA tile-GEMM kernel for the Q4_K_M direct-GEMM project.
// Companion to design memo `docs/plans/q4k_imma_design_2026_05_17.md` §4.
//
// Inputs (already in INT8 / FP16 form — Phase 2A's reorder produces W_s8 / α / β):
//   X_s8       [M, K]     int8  activation (row-major, per-row-per-sub-block s8 quant)
//   x_scale    [M, K/32]  FP16  activation per-sub-block scale (so x_fp16 = x_scale·X_s8)
//   x_rowsum   [M, K/32]  float Σ_{k in sub} X_s8[m, k]
//   W_s8       [N, K]     int8  weight, symmetric (q_sym = q - 8 ∈ [-8, 7])
//   eff_alpha  [N, K/32]  FP16  α[n, sub] = d_super · sc[n, sub]
//   eff_beta   [N, K/32]  FP16  β[n, sub] = 8·d_super·sc[n, sub] − dmin_super·m[n, sub]
//
// Output (FP16, row-major):
//   out        [M, N]
//
// Per-(m, n) the kernel computes:
//   out[m, n] = Σ_sub x_scale[m, sub] · ( α[n, sub] · Σ_{k in sub} X_s8·W_s8
//                                       + β[n, sub] · x_rowsum[m, sub] )
//
// Algebraically equivalent (proof in mmq_q4k_imma_tile_bench.cu header) to the
// full Q4_K dequant + FP16 GEMM:
//   out_ref[m, n] = Σ_k x_fp16[m, k] · w_fp16[n, k]
// modulo INT8 / FP16 quantisation noise.
//
// Phase 2B-minimum (this version):
//   - One warp per CTA. BLOCK_M = 16, BLOCK_N = 8, BLOCK_K = 32 = one MMA tile.
//   - Synchronous SMEM staging: load A (16×32), B (8×32), MMA, accumulate.
//   - Grid: (N / BLOCK_N, M / BLOCK_M, 1). M and N must be multiples of
//     BLOCK_M / BLOCK_N. K must be a multiple of 32.
//   - No cp.async pipelining, no ldmatrix.x4 (manual SMEM loads). Phase 2B.1
//     will add those for performance.
//
// Phase 2A's reorder is the canonical W producer. Phase 2C will be the
// production dispatch wiring.

void mmq_q4k_imma_tile(const int8_t* X_s8, const __half* x_scale, const float* x_rowsum,
                       const int8_t* W_s8, const __half* eff_alpha, const __half* eff_beta,
                       __half* out, int M, int N, int K, cudaStream_t stream);

}  // namespace imp
