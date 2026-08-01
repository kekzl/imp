#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Q4_K_M → symmetric-s8 reorder for the future INT8 IMMA direct-GEMM kernel.
// Phase 2A of the INT8 IMMA direct-GEMM experiment (outcome:
// docs/plans/2026-05-28-q4k-mmq-kernel-design.md).
//
// Splits each Q4_K weight super-block into three outputs:
//
//   w_sym_s8[N, K]      — int8 weight, symmetric (q_sym = q - 8 ∈ [-8, 7]).
//                          K-major, contiguous; row r occupies bytes [r*K, (r+1)*K).
//   eff_alpha[N, K/32]  — FP16 per-(row, sub-block) multiplicative factor
//                          α[j] = d_super * sc[j].
//   eff_beta[N, K/32]   — FP16 per-(row, sub-block) additive factor
//                          β[j] = 8 * d_super * sc[j] - dmin_super * m[j].
//                          The "8 *" term collapses the (q_sym + 8) shift into the
//                          GEMM epilogue; β couples to the activation row-sum.
//
// Decode equivalence (per element):
//   fp16_value = d_super * sc[j] * q
//              - dmin_super * m[j]
//              = α[j] * q_sym + β[j]
//
// `K` must be a multiple of `kSuperBlockSize = 256`. `N` is unconstrained.
//
// Launch: one CTA per super-block (N * K/256 CTAs), 32 threads per CTA.
//
// Phase 2B will add a layout permutation suited to `ldmatrix.x4` fragment loads.
// Phase 2A keeps the s8 tensor in plain row-major K-major to make the unit test
// trivial — the permutation only matters when the consuming tile kernel exists.

constexpr int kQ4kSuperBlockSize = 256;
constexpr int kQ4kSubBlocksPerSuper = 8;  // 8 sub-blocks × 32 elements = 256
constexpr int kQ4kSubBlockSize = 32;

void mmq_q4k_imma_reorder(const void* q4k_blocks, int N, int K, int8_t* w_sym_s8,
                          __half* eff_alpha, __half* eff_beta, cudaStream_t stream);

}  // namespace imp
