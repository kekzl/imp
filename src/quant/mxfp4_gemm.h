#pragma once

#include "quant/cutlass_mxfp4_weight.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// MXFP4 GEMV kernels for decode (M=1) dispatch.
//
// MXFP4 format: FP4 E2M1 packed nibbles + UE8M0 scale per 32 elements.
// Same prmt LUT nibble decoding as NVFP4, different scale handling:
//   NVFP4: FP8 E4M3 scale per 16 elements + FP32 tensor_scale
//   MXFP4: UE8M0 scale per 32 elements (tensor_scale absorbed)
//
// Architecture: 128 threads (4 warps), 1 row/block, N blocks.
// Uses CutlassMxFP4Weight.linear_scales (not SfAtom) for sequential access.
// ---------------------------------------------------------------------------

// Basic GEMV: y[N] = W_mxfp4[N,K] @ x[K]
void gemv_mxfp4_kpar(const CutlassMxFP4Weight& W, const half* x, half* y, int N, int K, cudaStream_t stream);

// FP32 output for LM head
void gemv_mxfp4_kpar_fp32(const CutlassMxFP4Weight& W, const half* x, float* y, int N, int K,
                          cudaStream_t stream);

// Fused QKV: 3 weight matrices, shared input, separate outputs
void gemv_mxfp4_qkv_fused(const CutlassMxFP4Weight& wq, const CutlassMxFP4Weight& wk,
                          const CutlassMxFP4Weight& wv, const half* x, half* yq, half* yk, half* yv,
                          int q_rows, int k_rows, int v_rows, int K, cudaStream_t stream);

// Fused Gate+Up: 2 weight matrices, shared input, separate outputs
void gemv_mxfp4_gate_up_fused(const CutlassMxFP4Weight& wg, const CutlassMxFP4Weight& wu, const half* x,
                              half* yg, half* yu, int rows, int K, cudaStream_t stream);

// GEMV with residual add: y[N] = W_mxfp4[N,K] @ x[K] + residual[N]
void gemv_mxfp4_residual(const CutlassMxFP4Weight& W, const half* x, half* y, const half* residual, int N,
                         int K, cudaStream_t stream);

// Fused SwiGLU + GEMV + residual
void gemv_mxfp4_swiglu_residual(const CutlassMxFP4Weight& W, const half* gate, const half* up, half* y,
                                const half* residual, int N, int K, cudaStream_t stream);

// Fused GeGLU + GEMV + residual
void gemv_mxfp4_geglu_residual(const CutlassMxFP4Weight& W, const half* gate, const half* up, half* y,
                               const half* residual, int N, int K, cudaStream_t stream);

// One-time L1 cache carveout setup for MXFP4 GEMV kernels.
void mxfp4_gemv_set_l1_carveout();

}  // namespace imp
