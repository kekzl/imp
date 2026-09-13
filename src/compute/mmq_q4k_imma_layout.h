#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Q4_K_M -> symmetric-s8 reorder (Phase 2A of the INT8 IMMA direct-GEMM experiment).
// See docs/plans/2026-05-28-q4k-mmq-kernel-design.md.
// w_sym_s8[N,K] int8 (q_sym=q-8 in [-8,7], K-major); eff_alpha[N,K/32] FP16 = d_super*sc[j];
// eff_beta[N,K/32] FP16 = 8*d_super*sc[j] - dmin_super*m[j] (folds q_sym+8 into the GEMM
// epilogue; beta couples to the activation rowsum).
// K % kSuperBlockSize(256) == 0, N unconstrained. Launch: 1 CTA/super-block, 32 threads/CTA.

constexpr int kQ4kSuperBlockSize = 256;
constexpr int kQ4kSubBlocksPerSuper = 8;  // 8 sub-blocks × 32 elements = 256
constexpr int kQ4kSubBlockSize = 32;

void mmq_q4k_imma_reorder(const void* q4k_blocks, int N, int K, int8_t* w_sym_s8,
                          __half* eff_alpha, __half* eff_beta, cudaStream_t stream);

}  // namespace imp
