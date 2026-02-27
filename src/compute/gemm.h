#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Pre-initialize cuBLAS handle and workspace. Call early (before weight upload)
// to ensure workspace is allocated while GPU memory is available.
void gemm_init();

// Destroy cached cuBLASLt descriptors. Call at shutdown (e.g. Engine destructor).
void gemm_cleanup();

// cuBLAS GEMM wrapper: C = alpha * A @ B^T + beta * C
// A [M, K]  B [N, K]  C [M, N]   -- all row-major
void gemm(const Tensor& A, const Tensor& B, Tensor& C,
          float alpha = 1.0f, float beta = 0.0f,
          cudaStream_t stream = nullptr);

// cuBLASLt GEMM with explicit algorithm selection and FP8 scale support.
// aScale/bScale are optional per-tensor FP32 scales for FP8 operands.
void gemm_cublaslt(const Tensor& A, const Tensor& B, Tensor& C,
                   float alpha = 1.0f, float beta = 0.0f,
                   const float* aScale = nullptr,
                   const float* bScale = nullptr,
                   cudaStream_t stream = nullptr);

// Small batch GEMV for batch_size 1-4
void gemv(const Tensor& A, const Tensor& x, Tensor& y,
          cudaStream_t stream = nullptr);

// FP8 E4M3 GEMV: y = A_fp8 @ x_fp16 (with per-tensor scale)
// A: [M, K] FP8_E4M3, x: [K] FP16, y: [M] FP16
void gemv_fp8(const Tensor& A, const Tensor& x, Tensor& y,
              float scale, cudaStream_t stream = nullptr);

// Fused quantized GEMV: dequant + dot product in one pass (no intermediate FP16 buffer).
// W: raw quantized bytes [M rows, K cols], x: [K] FP16, y: [M] FP16.
void gemv_q6k(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream = nullptr);
void gemv_q8_0(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream = nullptr);

// FP16 GEMV with FP32 output: y = W @ x. W: [M, K] FP16, x: [K] FP16, y: [M] FP32.
// Designed for MoE gate logits (small M, large K). Replaces cuBLAS + FP16→FP32 cast.
void gemv_gate_fp32(const half* W, const half* x, float* y,
                    int M, int K, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// MMVQ (Mixed-precision Matrix-Vector Quantized) — dp4a-accelerated GEMV.
// Quantizes the FP16 input vector to Q8_1 format, then uses dp4a (INT8x4 dot
// product) for the accumulation. ~2x faster than the FP16 dequant path above.
// ---------------------------------------------------------------------------

// Q8_1 block: 32 int8 quantized values + FP16 scale (d) + FP16 sum (s).
// The sum field enables the dp4a bias subtraction trick.
struct block_q8_1 {
    half d;          // delta (scale): val = d * qs[i]
    half s;          // sum of qs[0..31] (unused for Q6_K path, used for Q4 variants)
    int8_t qs[32];   // quantized values
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// Quantize FP16 input vector to Q8_1 blocks.
// x: [K] FP16, q8_1_out: [K/32] block_q8_1, d8_out: [K/32] float (block scales for fast access).
void quantize_fp16_to_q8_1(const half* x, block_q8_1* q8_1_out, float* d8_out,
                            int K, cudaStream_t stream = nullptr);

// Fused SwiGLU + Q8_1 quantization: computes silu(gate) * up and quantizes
// the result directly to Q8_1 format, eliminating the intermediate FP16 buffer.
// gate/up: [total_elements] FP16, q8_out: [total_elements/32] Q8_1 blocks.
void swiglu_quantize_q8_1(const half* gate, const half* up,
                            block_q8_1* q8_out, float* d8_out,
                            int total_elements, cudaStream_t stream = nullptr);

// Fused relu² + Q8_1 quantization: applies relu²(x) = max(0,x)² and quantizes
// directly to Q8_1 format. Used by non-gated MoE experts (Nemotron).
// input: [total_elements] FP16, q8_out: [total_elements/32] Q8_1 blocks.
void relu_sqr_quantize_q8_1(const half* input,
                              block_q8_1* q8_out, float* d8_out,
                              int total_elements, cudaStream_t stream = nullptr);

// Fused RMSNorm + Q8_1 quantization: applies RMSNorm to input, then quantizes
// the normalized result directly to Q8_1 format. Eliminates the intermediate
// FP16 norm_out buffer write+read. Single-row only (n=1 decode).
// If norm_out is non-null, also writes the FP16 normalized output.
void rmsnorm_quantize_q8_1(const half* x, const half* weight,
                             block_q8_1* q8_out, float* d8_out,
                             half* norm_out,
                             int d_model, float eps,
                             cudaStream_t stream = nullptr,
                             float weight_offset = 0.0f);

// dp4a-accelerated GEMV: W_quant @ x_q8_1 using native INT8 SIMD.
// W: raw quantized bytes, q8_1: pre-quantized input, d8: pre-extracted scales, y: [M] FP16.
void gemv_q6k_q8_1(const void* W, const block_q8_1* q8_1, const float* d8,
                    half* y, int M, int K, cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1(const void* W, const block_q8_1* q8_1, const float* d8,
                     half* y, int M, int K, cudaStream_t stream = nullptr);

// Residual-fused variants: y[i] = dot(W[i], x) + residual[i]
void gemv_q6k_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8,
                              half* y, const half* residual,
                              int M, int K, cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8,
                               half* y, const half* residual,
                               int M, int K, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Fused QKV GEMV: reads input once, computes Q, K, V projections in one kernel.
// All three weight matrices must be the same quant type and inner dim K.
// q_rows/k_rows/v_rows are the output dimensions of each projection.
// ---------------------------------------------------------------------------
void gemv_qkv_fused_q6k_q8_1(const void* W_q, const void* W_k, const void* W_v,
                               const block_q8_1* q8_1, const float* d8,
                               half* y_q, half* y_k, half* y_v,
                               int q_rows, int k_rows, int v_rows, int K,
                               cudaStream_t stream = nullptr);
void gemv_qkv_fused_q8_0_q8_1(const void* W_q, const void* W_k, const void* W_v,
                                const block_q8_1* q8_1, const float* d8,
                                half* y_q, half* y_k, half* y_v,
                                int q_rows, int k_rows, int v_rows, int K,
                                cudaStream_t stream = nullptr);

// Batched K/V projection: input @ [wk; wv]^T → k_out, v_out in a single cuBLAS call.
// weight_kv: [2 * nkv_hd, d_model] — wk at rows [0..nkv_hd), wv at [nkv_hd..2*nkv_hd)
// k_out, v_out: [n_tokens, nkv_hd] — must be contiguous (v_out = k_out + stride)
void gemm_kv_batched(const Tensor& input, const Tensor& weight_kv,
                     Tensor& k_out, Tensor& v_out,
                     cudaStream_t stream = nullptr);

// MoE decode GEMV: processes all top_k experts in a single kernel launch.
// packed_weights: base pointer to packed expert tensor (all experts contiguous).
// expert_indices: [top_k] int32 on device — selects which expert's weights to use.
// x: input vector(s). x_stride: 0 = shared input (gate/up), K = per-expert input (down).
// y: output [top_k, rows] FP16.
// expert_stride_bytes: byte offset between experts in packed_weights.
void gemv_q6k_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                          const half* x, half* y, int rows, int K,
                          size_t expert_stride_bytes, int x_stride, int top_k,
                          cudaStream_t stream = nullptr);
void gemv_q8_0_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                           const half* x, half* y, int rows, int K,
                           size_t expert_stride_bytes, int x_stride, int top_k,
                           cudaStream_t stream = nullptr);

// dp4a-accelerated MoE decode GEMV variants.
// Same interface as above but uses pre-quantized Q8_1 input for dp4a acceleration.
// q8_1: pre-quantized input blocks, d8: block scales.
// q8_1_stride: 0 = shared input for all experts (gate/up), K/32 = per-expert (down).
// d8_stride: 0 = shared, K/32 = per-expert.
void gemv_q6k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                                const block_q8_1* q8_1, const float* d8,
                                half* y, int rows, int K,
                                size_t expert_stride_bytes,
                                int q8_1_stride, int d8_stride, int top_k,
                                cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                                 const block_q8_1* q8_1, const float* d8,
                                 half* y, int rows, int K,
                                 size_t expert_stride_bytes,
                                 int q8_1_stride, int d8_stride, int top_k,
                                 cudaStream_t stream = nullptr);

// Fused gate+up MoE GEMV: both projections in a single kernel launch.
// Uses blockIdx.y to select gate(0) or up(1). Saves one launch per MoE layer.
void gemv_q6k_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices, const half* x,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int x_stride, int top_k, cudaStream_t stream = nullptr);
void gemv_q8_0_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices, const half* x,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int x_stride, int top_k, cudaStream_t stream = nullptr);

// dp4a-accelerated fused gate+up MoE GEMV variants.
void gemv_q6k_q8_1_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices,
        const block_q8_1* q8_1, const float* d8,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int q8_1_stride, int d8_stride, int top_k,
        cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices,
        const block_q8_1* q8_1, const float* d8,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int q8_1_stride, int d8_stride, int top_k,
        cudaStream_t stream = nullptr);

// dp4a-accelerated GEMV with FP32 output: W_quant @ x_q8_1 → y[M] float.
// Designed for the LM head projection where FP32 logits are needed for sampling.
void gemv_q6k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8,
                          float* y, int M, int K, cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8,
                           float* y, int M, int K, cudaStream_t stream = nullptr);

} // namespace imp
