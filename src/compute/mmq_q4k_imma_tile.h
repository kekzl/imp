#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// Phase 2B INT8 IMMA tile-GEMM kernel for the Q4_K_M direct-GEMM project.
// Phase 2B of the INT8 IMMA direct-GEMM experiment (outcome:
// docs/plans/2026-05-28-q4k-mmq-kernel-design.md).
//
// Inputs (Phase 2A's mmq_q4k_imma_reorder produces W_s8 / α / β; for activations
// use quantize_fp16_to_int8_subblock below or mmq_q4k_imma_gemm wrapper):
//
//   X_s8       [M, K]     int8  activation (row-major, per-row-per-sub-block s8 quant)
//   x_scale    [M, K/32]  FP16  activation per-sub-block scale (so x_fp16 = x_scale·X_s8)
//   x_rowsum   [M, K/32]  float Σ_{k in sub} X_s8[m, k]
//   W_s8       [N, K]     int8  weight, symmetric (q_sym = q - 8 ∈ [-8, 7])
//   eff_alpha  [N, K/32]  FP16  α[n, sub] = d_super · sc[n, sub]
//   eff_beta   [N, K/32]  FP16  β[n, sub] = 8·d_super·sc[n, sub] − dmin_super·m[n, sub]
//
// Output (FP16, row-major):  out [M, N]
//
// Per-(m, n) the kernel computes:
//   out[m, n] = Σ_sub x_scale[m, sub] · ( α[n, sub] · Σ_{k in sub} X_s8·W_s8
//                                       + β[n, sub] · x_rowsum[m, sub] )
//
// Algebraically equivalent to the full Q4_K dequant + FP16 GEMM:
//   out_ref[m, n] = Σ_k x_fp16[m, k] · w_fp16[n, k]
// modulo INT8 / FP16 quantisation noise.
//
// Architecture: BLOCK_M=64 BLOCK_N=32 BLOCK_K=32; 4 warps per CTA in 2×2 spatial
// with WRM·WRN=2·2 per warp (16 MMAs per CTA per K-block); 2-stage cp.async
// pipeline. Throughput plateaus at ~40 TOPS on sm_120a (4.3 % of the 931 TOPS
// raw MMA peak — the same figure is in docs/plans/2026-05-28-q4k-mmq-kernel-design.md).
//
// M and N must be multiples of 64 / 32 respectively; K must be a multiple of 32.
void mmq_q4k_imma_tile(const int8_t* X_s8, const __half* x_scale, const float* x_rowsum,
                       const int8_t* W_s8, const __half* eff_alpha, const __half* eff_beta,
                       __half* out, int M, int N, int K, cudaStream_t stream);

// Activation quantizer: FP16 [M, K] → (int8 [M, K], FP16 scale [M, K/32],
// FP32 rowsum [M, K/32]). Per-(row, K/32-sub-block) symmetric s8 quant
// with scale = amax / 127; rowsum is the int32 sum of the s8 values per
// sub-block (cast to float).
//
// K must be a multiple of 32. One CUDA block per (m, sub); 32 threads per
// block. Throughput is HBM-bound at ~0.5 TB/s on sm_120.
void quantize_fp16_to_int8_subblock(const __half* X_fp16, int M, int K, int8_t* X_s8,
                                    __half* x_scale, float* x_rowsum, cudaStream_t stream);

// High-level entry: full Q4_K_M dense GEMM via INT8 IMMA. Allocates scratch
// for the symmetric-s8 weight + α/β cache and the activation INT8 buffers on
// first call, reuses them per-(N, K, M) shape. Caller owns nothing internal.
//
// The path: (1) reorder Q4_K weight to symmetric s8 + α/β (if not cached),
// (2) quantize activation to s8 with per-sub-block scale + rowsum, (3) dispatch
// the IMMA tile kernel.
//
// Eligibility (caller checks):
//   - Weight qtype is Q4_K_M
//   - M ≥ 64 (BLOCK_M); N ≥ 32 (BLOCK_N); K % 32 == 0
//   - Production-recommended: M ≥ 1024 for the ~40 TOPS plateau to amortise
//     the activation-quant cost.
//
// Returns true if dispatch succeeded; false if shape is ineligible (no-op).
bool mmq_q4k_imma_gemm(const void* W_q4k_blocks, const __half* X_fp16, __half* Y_fp16,
                       int M, int N, int K, cudaStream_t stream);

}  // namespace imp
