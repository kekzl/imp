#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// NVFP4 GEMV: y = A_nvfp4 @ x
// A is stored in NVFP4 format (packed_data + micro_scales + tensor_scale)
// x: [K] or [K,1] FP16 on device
// y: [M] or [M,1] FP16 on device
void gemv_nvfp4(const NvFP4QuantResult& A, const Tensor& x, Tensor& y,
                cudaStream_t stream = nullptr);

// NVFP4 GEMM via cuBLASLt (for M > 1, e.g., prefill).
// Falls back to dequant + standard GEMM if cuBLASLt NVFP4 is unavailable.
void gemm_nvfp4(const NvFP4QuantResult& A, const Tensor& B, Tensor& C,
                cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// K-parallel NVFP4 GEMV host launchers for decode (M=1) dispatch.
// These take raw NvFP4QuantResult + FP16 pointers — no Tensor overhead.
// Architecture: 128 threads (4 warps), 1 row/block, M blocks.
// ---------------------------------------------------------------------------

// Basic GEMV: y[M] = A_nvfp4[M,K] @ x[K]
void gemv_nvfp4_kpar(const NvFP4QuantResult& A, const half* x, half* y,
                     int M, int K, cudaStream_t stream);

// Fused QKV: 3 weight matrices, shared input, separate outputs
void gemv_nvfp4_qkv_fused(const NvFP4QuantResult& wq, const NvFP4QuantResult& wk,
                           const NvFP4QuantResult& wv, const half* x,
                           half* yq, half* yk, half* yv,
                           int q_rows, int k_rows, int v_rows, int K,
                           cudaStream_t stream);

// Fused Gate+Up: 2 weight matrices, shared input, separate outputs
void gemv_nvfp4_gate_up_fused(const NvFP4QuantResult& wg, const NvFP4QuantResult& wu,
                               const half* x, half* yg, half* yu,
                               int rows, int K, cudaStream_t stream);

// GEMV with residual add: y[M] = A_nvfp4[M,K] @ x[K] + residual[M]
void gemv_nvfp4_residual(const NvFP4QuantResult& A, const half* x, half* y,
                          const half* residual, int M, int K, cudaStream_t stream);

// Fused SwiGLU + GEMV + residual: y[M] = A_nvfp4[M,K] @ swiglu(gate,up) + residual[M]
// Eliminates separate SwiGLU kernel launch. gate, up: [K] FP16 on device.
void gemv_nvfp4_swiglu_residual(const NvFP4QuantResult& A,
                                 const half* gate, const half* up,
                                 half* y, const half* residual,
                                 int M, int K, cudaStream_t stream);

// Fused GeGLU + GEMV + residual: y[M] = A_nvfp4[M,K] @ geglu(gate,up) + residual[M]
// For Gemma-3 and similar models using GELU-tanh activation.
void gemv_nvfp4_geglu_residual(const NvFP4QuantResult& A,
                                const half* gate, const half* up,
                                half* y, const half* residual,
                                int M, int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// MoE NVFP4 GEMV: per-expert decode projections.
// FP16 input (no Q8_1 pre-quantization needed).
// ---------------------------------------------------------------------------

// MoE decode GEMV: y[expert_slot, rows] = W[expert_id, :, :] @ x[expert_slot, :]
// x_stride: 0 = shared input across experts, K = per-expert input.
void gemv_nvfp4_moe_decode(const NvFP4MoEQuantResult& w,
                            const int32_t* expert_indices, const half* x, half* y,
                            int rows, int K, int x_stride, int top_k,
                            cudaStream_t stream);

// Fused SwiGLU + MoE GEMV for down projection: computes swiglu(gate,up) inline.
// Eliminates the separate swiglu() kernel launch.
void gemv_nvfp4_moe_swiglu_decode(const NvFP4MoEQuantResult& w,
                                    const int32_t* expert_indices,
                                    const half* gate, const half* up, half* y,
                                    int rows, int K, int x_stride, int top_k,
                                    cudaStream_t stream);

// Fused gate+up MoE GEMV: two weight matrices, shared input, separate outputs.
void gemv_nvfp4_moe_gate_up_fused(const NvFP4MoEQuantResult& gate,
                                    const NvFP4MoEQuantResult& up,
                                    const int32_t* expert_indices, const half* x,
                                    half* y_gate, half* y_up,
                                    int rows, int K, int top_k,
                                    cudaStream_t stream);

// PDL registration for all NVFP4 GEMV kernels (called at init when PDL enabled).
void nvfp4_gemv_pdl_register();

} // namespace imp
