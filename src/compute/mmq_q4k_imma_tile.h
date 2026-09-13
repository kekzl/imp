#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// Phase 2B INT8 IMMA tile-GEMM for Q4_K_M direct-GEMM.
// See docs/plans/2026-05-28-q4k-mmq-kernel-design.md.
// X_s8[M,K] int8, x_scale[M,K/32] FP16, x_rowsum[M,K/32] float; W_s8[N,K] int8 (q_sym=q-8);
// eff_alpha[N,K/32] FP16 = d_super*sc; eff_beta[N,K/32] FP16 = 8*d_super*sc - dmin_super*m.
// out[m,n] = sum_sub x_scale[m,sub]*(alpha[n,sub]*sum(X_s8*W_s8) + beta[n,sub]*x_rowsum[m,sub]);
// equals dequant+FP16 GEMM modulo int8/FP16 quant noise.
// Architecture: BLOCK_M=64 BLOCK_N=32 BLOCK_K=32, 4 warps/CTA 2x2 spatial, 2-stage cp.async.
// M % 64 == 0, N % 32 == 0, K % 32 == 0.
void mmq_q4k_imma_tile(const int8_t* X_s8, const __half* x_scale, const float* x_rowsum,
                       const int8_t* W_s8, const __half* eff_alpha, const __half* eff_beta,
                       __half* out, int M, int N, int K, cudaStream_t stream);

// Activation quantizer: FP16[M,K] -> int8[M,K] + FP16 scale[M,K/32] + FP32 rowsum[M,K/32].
// Per (row, K/32-subblock) symmetric s8: scale = amax/127; rowsum = int32 sum of s8 values.
// K % 32 == 0. One CUDA block per (m, sub); 32 threads/block; HBM-bound ~0.5 TB/s on sm_120.
void quantize_fp16_to_int8_subblock(const __half* X_fp16, int M, int K, int8_t* X_s8,
                                    __half* x_scale, float* x_rowsum, cudaStream_t stream);

// Full Q4_K_M dense GEMM via INT8 IMMA: allocates/caches s8 weight+alpha/beta and activation
// int8 buffers on first call, reused per (N,K,M) shape.
// Path: (1) reorder Q4_K weight if uncached, (2) quantize activation to s8, (3) dispatch IMMA tile.
// Eligibility: qtype Q4_K_M; M>=64, N>=32, K%32==0; M>=1024 recommended to amortize quant cost.
// Returns true if dispatched, false if shape ineligible (no-op).
bool mmq_q4k_imma_gemm(const void* W_q4k_blocks, const __half* X_fp16, __half* Y_fp16,
                       int M, int N, int K, cudaStream_t stream);

}  // namespace imp
