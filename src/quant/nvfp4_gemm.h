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
void gemv_nvfp4(const NvFP4QuantResult& A, const Tensor& x, Tensor& y, cudaStream_t stream = nullptr);

// NVFP4 GEMM via cuBLASLt (for M > 1, e.g., prefill).
// Falls back to dequant + standard GEMM if cuBLASLt NVFP4 is unavailable.
// beta: output accumulation factor (default 0 = overwrite). beta=1 enables
// residual-fused GEMM: y = dequant(A) @ B + y.
void gemm_nvfp4(const NvFP4QuantResult& A, const Tensor& B, Tensor& C, cudaStream_t stream = nullptr,
                float beta = 0.0f);

// ---------------------------------------------------------------------------
// K-parallel NVFP4 GEMV host launchers for decode (M=1) dispatch.
// These take raw NvFP4QuantResult + FP16 pointers — no Tensor overhead.
// Architecture: 128 threads (4 warps), 1 row/block, M blocks.
// ---------------------------------------------------------------------------

// Basic GEMV: y[M] = A_nvfp4[M,K] @ x[K]
void gemv_nvfp4_kpar(const NvFP4QuantResult& A, const half* x, half* y, int M, int K, cudaStream_t stream);

// FP32 output GEMV for LM head: y[M] = A_nvfp4[M,K] @ x[K] (float output)
void gemv_nvfp4_kpar_fp32(const NvFP4QuantResult& A, const half* x, float* y, int M, int K,
                          cudaStream_t stream);

// Batched-M FP32 GEMV (LM head at batch>1): y[n_act, N_out] computed in one weight
// pass per launch. x is [n_act, K] row-major, y is [n_act, N_out] row-major.
void gemv_nvfp4_kpar_batched_fp32(const NvFP4QuantResult& A, const half* x, float* y, int N_out, int K,
                                  int n_act, cudaStream_t stream);

// Batched-M FP16 GEMM for small-M chunk forwards (spec-verify, #998):
// y[n_act, N_out] = x[n_act, K] @ A^T, reading each NVFP4 weight row once per
// MR<=4 activation tile instead of dequantizing the source (the M>1 prefill
// fallback costs a full FP16 materialization of the weight per call — 52% of
// the decode window on Qwen3-14B Q6_K verify chunks). Same tiling as the
// FP32 LM-head variant above.
// Small-M W4A16 GEMM (plain NVFP4 layout, Marlin recipe: dequant-to-FP16 in
// smem + FP16 tensor-core MMA). M <= 32, K % 128 == 0. Built for the batched
// decode projections; see nvfp4_gemm_smallm.cu for the measured history.
size_t gemm_nvfp4_smallm_workspace_bytes(int N_out);
bool gemm_nvfp4_smallm(const NvFP4QuantResult& W, const half* x, half* y, int M, int N_out, int K,
                       void* d_workspace, cudaStream_t stream, bool accumulate = false);
// A4 variant: both sides packed NVFP4 (activations quantized by the caller).
bool gemm_nvfp4_smallm_a4(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M,
                          int N_out, int K, void* d_workspace, cudaStream_t stream,
                          bool accumulate = false);
// v2: native block-scaled mxf4nvf4 MMA on both plain-NVFP4 sides, fed by a
// producer/consumer cp.async pipeline (no dequant, no FP16 staging). M <= 32,
// K % 256 == 0, N % 64 == 0; see nvfp4_gemm_smallm_v2.cu for the design and
// gemm.h for the v1 postmortem that motivates it.
int gemm_nvfp4_smallm_v2_stripes(int N_out, int K);
size_t gemm_nvfp4_smallm_v2_workspace_bytes(int N_out, int K);
bool gemm_nvfp4_smallm_v2_a4(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M,
                             int N_out, int K, void* d_workspace, cudaStream_t stream,
                             bool accumulate = false);
// Tuning hook (tests only): explicit stage depth {2,3,4,6} and stripe count.
bool gemm_nvfp4_smallm_v2_a4_tuned(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M,
                                   int N_out, int K, void* d_workspace, cudaStream_t stream,
                                   bool accumulate, int stages, int stripes);
// Sibling-pair variant: two weights with the same K sharing one quantized
// activation (FFN gate|up, GDN in|z) in ONE launch. stripes==1 shapes only
// (both Ns >= 5120), fresh outputs (no accumulate); returns false otherwise
// and the caller falls back to two single calls. Bit-identical per tensor to
// the single-tensor kernel (same CTA body, same tile order).
bool gemm_nvfp4_smallm_v2_pair_a4(const NvFP4QuantResult& W1, const NvFP4QuantResult& W2,
                                  const NvFP4QuantResult& Xq, half* y1, half* y2, int M, int N1, int N2,
                                  int K, cudaStream_t stream);

void gemm_nvfp4_batched(const NvFP4QuantResult& A, const half* x, half* y, int N_out, int K,
                        int n_act, cudaStream_t stream);
// beta=1 twin: y += A @ x (o/down residual-add verify GEMMs, #1055).
void gemm_nvfp4_batched_acc(const NvFP4QuantResult& A, const half* x, half* y, int N_out, int K,
                            int n_act, cudaStream_t stream);

// Fused QKV: 3 weight matrices, shared input, separate outputs
void gemv_nvfp4_qkv_fused(const NvFP4QuantResult& wq, const NvFP4QuantResult& wk, const NvFP4QuantResult& wv,
                          const half* x, half* yq, half* yk, half* yv, int q_rows, int k_rows, int v_rows,
                          int K, cudaStream_t stream);

// Fused Gate+Up: 2 weight matrices, shared input, separate outputs
void gemv_nvfp4_gate_up_fused(const NvFP4QuantResult& wg, const NvFP4QuantResult& wu, const half* x, half* yg,
                              half* yu, int rows, int K, cudaStream_t stream);

// GEMV with residual add: y[M] = A_nvfp4[M,K] @ x[K] + residual[M]
void gemv_nvfp4_residual(const NvFP4QuantResult& A, const half* x, half* y, const half* residual, int M,
                         int K, cudaStream_t stream);

// Fused SwiGLU + GEMV + residual: y[M] = A_nvfp4[M,K] @ swiglu(gate,up) + residual[M]
// Eliminates separate SwiGLU kernel launch. gate, up: [K] FP16 on device.
void gemv_nvfp4_swiglu_residual(const NvFP4QuantResult& A, const half* gate, const half* up, half* y,
                                const half* residual, int M, int K, cudaStream_t stream);

// Fused GeGLU + GEMV + residual: y[M] = A_nvfp4[M,K] @ geglu(gate,up) + residual[M]
// For Gemma-3 and similar models using GELU-tanh activation.
void gemv_nvfp4_geglu_residual(const NvFP4QuantResult& A, const half* gate, const half* up, half* y,
                               const half* residual, int M, int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// MoE NVFP4 GEMV: per-expert decode projections.
// FP16 input (no Q8_1 pre-quantization needed).
// ---------------------------------------------------------------------------

// MoE decode GEMV: y[expert_slot, rows] = W[expert_id, :, :] @ x[expert_slot, :]
// x_stride: 0 = shared input across experts, K = per-expert input.
void gemv_nvfp4_moe_decode(const NvFP4MoEQuantResult& w, const int32_t* expert_indices, const half* x,
                           half* y, int rows, int K, int x_stride, int top_k, cudaStream_t stream);

// Fused gate+up MoE GEMV: two weight matrices, shared input, separate outputs.
void gemv_nvfp4_moe_gate_up_fused(const NvFP4MoEQuantResult& gate, const NvFP4MoEQuantResult& up,
                                  const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                  int rows, int K, int top_k, cudaStream_t stream);

// PDL registration for all NVFP4 GEMV kernels (called at init when PDL enabled).
void nvfp4_gemv_pdl_register();

// Set the pre-allocated dequant scratch buffer for the gemm_nvfp4 fallback
// path (M>1 dequant→FP16→cuBLAS). When set and large enough, gemm_nvfp4
// reuses this buffer instead of attempting a cudaMalloc — which would fail
// inside CUDA stream capture. Pass nullptr/0 to clear (e.g., on engine
// destroy).
//
// The fallback path only fires for M>1, so the buffer is sized for the
// FP16 dequant of the LARGEST NVFP4 weight matrix in the model (N×K×2
// bytes). The caller is responsible for the buffer's lifetime — this
// function only stores the pointer and size for later use by ensure_dequant_buffer().
void set_nvfp4_dequant_workspace(void* buf, size_t size_bytes);

// Test probe: returns the current size of the lazy cudaMalloc'd dequant
// buffer (the legacy non-graph-safe path). When a workspace is set via
// set_nvfp4_dequant_workspace(), subsequent gemm_nvfp4 calls should not
// grow this buffer. Exposed for tests to assert workspace-vs-lazy choice.
size_t nvfp4_lazy_dequant_buf_size_for_testing();

}  // namespace imp
