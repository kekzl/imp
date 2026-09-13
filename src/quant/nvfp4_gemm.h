#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// NVFP4 GEMV: y = A_nvfp4 @ x. A is packed_data+micro_scales+tensor_scale; x [K] or [K,1]
// FP16; y [M] or [M,1] FP16.
void gemv_nvfp4(const NvFP4QuantResult& A, const Tensor& x, Tensor& y, cudaStream_t stream = nullptr);

// NVFP4 GEMM via cuBLASLt (M>1, e.g. prefill); falls back to dequant+standard GEMM if
// cuBLASLt NVFP4 is unavailable. beta: output accumulation factor (default 0=overwrite);
// beta=1 enables residual-fused GEMM: y = dequant(A)@B + y.
void gemm_nvfp4(const NvFP4QuantResult& A, const Tensor& B, Tensor& C, cudaStream_t stream = nullptr,
                float beta = 0.0f);

// K-parallel NVFP4 GEMV host launchers for decode (M=1): raw NvFP4QuantResult+FP16
// pointers, no Tensor overhead. 128 threads (4 warps), 1 row/block, M blocks.

// Basic GEMV: y[M] = A_nvfp4[M,K] @ x[K]
void gemv_nvfp4_kpar(const NvFP4QuantResult& A, const half* x, half* y, int M, int K, cudaStream_t stream);

// FP32 output GEMV for LM head: y[M] = A_nvfp4[M,K] @ x[K] (float output)
void gemv_nvfp4_kpar_fp32(const NvFP4QuantResult& A, const half* x, float* y, int M, int K,
                          cudaStream_t stream);

// Batched-M FP32 GEMV (LM head at batch>1): y[n_act, N_out] computed in one weight
// pass per launch. x is [n_act, K] row-major, y is [n_act, N_out] row-major.
void gemv_nvfp4_kpar_batched_fp32(const NvFP4QuantResult& A, const half* x, float* y, int N_out, int K,
                                  int n_act, cudaStream_t stream);

// Batched-M FP16 GEMM for small-M chunk forwards (spec-verify, #998): reads each NVFP4
// weight row once per MR<=4 activation tile instead of dequantizing the source (the M>1
// prefill fallback fully materializes the weight per call). Marlin recipe: dequant-to-FP16
// in smem + FP16 tensor-core MMA. M<=32, K%128==0.
size_t gemm_nvfp4_smallm_workspace_bytes(int N_out);
bool gemm_nvfp4_smallm(const NvFP4QuantResult& W, const half* x, half* y, int M, int N_out, int K,
                       void* d_workspace, cudaStream_t stream, bool accumulate = false);
// A4 variant: both sides packed NVFP4 (activations quantized by the caller).
bool gemm_nvfp4_smallm_a4(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M,
                          int N_out, int K, void* d_workspace, cudaStream_t stream,
                          bool accumulate = false);
// v2: native block-scaled mxf4nvf4 MMA on both plain-NVFP4 sides, fed by a producer/
// consumer cp.async pipeline (no dequant, no FP16 staging). M<=32, K%256==0, N%64==0; see
// nvfp4_gemm_smallm_v2.cu for design, gemm.h for the v1 postmortem motivating it.
int gemm_nvfp4_smallm_v2_stripes(int N_out, int K);
size_t gemm_nvfp4_smallm_v2_workspace_bytes(int N_out, int K);
bool gemm_nvfp4_smallm_v2_a4(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M,
                             int N_out, int K, void* d_workspace, cudaStream_t stream,
                             bool accumulate = false);
// FP32-output twin for the batched LM head (samplers read float logits): single-stripe
// shapes only (a vocab-sized N tiles the card many times over), fresh output, no
// workspace; false otherwise. Same accumulators as the FP16 kernel, written before the
// FP16 rounding.
bool gemm_nvfp4_smallm_v2_a4_f32(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, float* y, int M,
                                 int N_out, int K, cudaStream_t stream);
// Tuning hook (tests only): explicit stage depth {2,3,4,6} and stripe count.
bool gemm_nvfp4_smallm_v2_a4_tuned(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M,
                                   int N_out, int K, void* d_workspace, cudaStream_t stream,
                                   bool accumulate, int stages, int stripes);
// Sibling variant: two or three weights with the same K sharing one quantized activation
// (FFN gate|up, GDN in|z, attention q|k|v) in ONE launch. Single-stripe policy applies to
// the COMBINED n-tile count (>=80 tiles), fresh outputs; returns false otherwise (caller
// falls back to single calls). Bit-identical per tensor to the single-tensor kernel at
// stripes=1.
constexpr int kSmallMV2MaxSiblings = 3;
struct SmallMV2Sibling {
    const NvFP4QuantResult* w;
    half* y;
    int N;
};
bool gemm_nvfp4_smallm_v2_multi_a4(const SmallMV2Sibling* t, int count, const NvFP4QuantResult& Xq, int M,
                                   int K, cudaStream_t stream);
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

// M=1 GDN input projections in one launch: in_proj+gate (NVFP4) and alpha+beta (FP16
// [ab_rows,K]) on one x. Returns false when K%16!=0, K>8192, or a weight's K differs; the
// caller keeps its four-call path. Registered for PDL by nvfp4_gemv_pdl_register().
bool gemv_nvfp4_gdn_input_fused(const NvFP4QuantResult& w_in, const NvFP4QuantResult& w_gate, const half* w_alpha,
                                const half* w_beta, int ab_rows, const half* x, half* y_in, half* y_gate,
                                half* y_alpha, half* y_beta, int K, cudaStream_t stream);
void nvfp4_gdn_input_pdl_register();

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

// MoE NVFP4 GEMV: per-expert decode projections. FP16 input (no Q8_1 pre-quantization
// needed).

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
// The batched LM-head GEMV instantiations (nvfp4_gemv_batched.cu); called by
// nvfp4_gemv_pdl_register().
void nvfp4_gemv_batched_pdl_register();

// Sets the pre-allocated dequant scratch buffer for gemm_nvfp4's M>1 fallback
// (dequant->FP16->cuBLAS): when set and large enough, reused instead of a cudaMalloc, which
// would fail inside CUDA stream capture. nullptr/0 clears it (e.g. engine destroy). Sized
// for the FP16 dequant of the LARGEST NVFP4 weight in the model (N*K*2 bytes); caller owns
// the buffer's lifetime, this only stores pointer+size.
void set_nvfp4_dequant_workspace(void* buf, size_t size_bytes);

// Test probe: current size of the lazy cudaMalloc'd dequant buffer (the legacy
// non-graph-safe path). After set_nvfp4_dequant_workspace() is set, subsequent gemm_nvfp4
// calls should not grow this. Exposed for tests to assert workspace-vs-lazy choice.
size_t nvfp4_lazy_dequant_buf_size_for_testing();

}  // namespace imp
