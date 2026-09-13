#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// cuBLASLt workspace + algo-selection bench scratch, taken from the engine-persistent
// (T2) arena by gemm_init() (A7 step 8), charged in exec_t2_demand as `cublas_workspace`
// so Engine::init reserves them before anything else spends the free VRAM they need.
inline constexpr size_t kGemmCublasWorkspaceBytes = 64ull << 20;  // 64 MiB
inline constexpr size_t kGemmBenchScratchBytes = 32ull << 20;     // 32 MiB

// Pre-initialize cuBLAS handle and workspace. Call early (before weight upload)
// to ensure workspace is allocated while GPU memory is available.
void gemm_init();

// Graph-captured verify (#847): allows cuBLASLt calls to record into an active stream
// capture. Default off: cold-shape Lt calls fail with status 14 under capture on
// sm_120; the verify capturer warms every shape eagerly first and toggles this around
// its capture.
void gemm_set_lt_capture_allowed(bool allowed);
bool gemm_lt_capture_allowed();

// Destroy cached cuBLASLt descriptors. Call at shutdown (e.g. Engine destructor).
void gemm_cleanup();

// cuBLAS GEMM wrapper: C = alpha * A @ B^T + beta * C
// A [M, K]  B [N, K]  C [M, N]   -- all row-major
void gemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f, float beta = 0.0f,
          cudaStream_t stream = nullptr);

// cuBLASLt GEMM with explicit algorithm selection and FP8 scale support.
// aScale/bScale are optional per-tensor FP32 scales for FP8 operands.
void gemm_cublaslt(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f, float beta = 0.0f,
                   const float* aScale = nullptr, const float* bScale = nullptr,
                   cudaStream_t stream = nullptr);

// Probe whether cuBLASLt supports FP8 E4M3 GEMM on this GPU/driver.
// Runs a tiny 8×64×8 FP8 matmul and returns true if cublasLtMatmul succeeds.
bool gemm_cublaslt_fp8_probe();

// Small batch GEMV for batch_size 1-4
void gemv(const Tensor& A, const Tensor& x, Tensor& y, cudaStream_t stream = nullptr);

// FP8 E4M3 GEMV: y = A_fp8 @ x_fp16 (with per-tensor scale)
// A: [M, K] FP8_E4M3, x: [K] FP16, y: [M] FP16
void gemv_fp8(const Tensor& A, const Tensor& x, Tensor& y, float scale, cudaStream_t stream = nullptr);
// Per-row-scale variant (scale[row] = row_absmax/448, e.g. the fp8_ssm_proj sidecar).
void gemv_fp8_rowscale(const Tensor& A, const Tensor& x, Tensor& y, const float* d_row_scales,
                       cudaStream_t stream = nullptr);

// Fused quantized GEMV: dequant + dot product in one pass (no intermediate FP16 buffer).
// W: raw quantized bytes [M rows, K cols], x: [K] FP16, y: [M] FP16.
void gemv_q6k(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream = nullptr);
void gemv_q8_0(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream = nullptr);

// FP16 GEMV with FP32 output: y = W @ x. W: [M, K] FP16, x: [K] FP16, y: [M] FP32.
// Designed for MoE gate logits (small M, large K). Replaces cuBLAS + FP16→FP32 cast.
void gemv_gate_fp32(const half* W, const half* x, float* y, int M, int K, cudaStream_t stream = nullptr);

// FP32-input variant: avoids FP16 truncation of router input (Gemma-4 MoE precision).
void gemv_gate_fp32_fp32input(const half* W, const float* x, float* y, int M, int K,
                              cudaStream_t stream = nullptr);

// MMVQ (Mixed-precision Matrix-Vector Quantized) GEMV: quantizes the FP16 input to
// Q8_1, then uses dp4a (INT8x4 dot product) for the accumulation.

// Q8_1 block: 32 int8 quantized values + FP16 scale (d) + FP16 sum (s) (dp4a
// bias-subtraction trick). Layout: qs first + alignas(16) + 48-B stride (#598) so the
// 32-B activation reads compile to 2x LDG.128 instead of 8x LDG.32 (ggml's offset-4 qs
// was never 16-B aligned). mmvq keeps its own ggml_block_q8_1 (imported ggml vec_dot
// kernels expect that layout).
struct alignas(16) block_q8_1 {
    int8_t qs[32];  // quantized values (16-B aligned: offset 0, stride 48)
    half d;         // delta (scale): val = d * qs[i]
    half s;         // sum of qs[0..31] (unused for Q6_K path, used for Q4 variants)
    int8_t pad[12];
};
static_assert(sizeof(block_q8_1) == 48, "block_q8_1: 48-B stride (16-B-aligned qs plane)");

// Quantize FP16 input vector to Q8_1 blocks.
// x: [K] FP16, q8_1_out: [K/32] block_q8_1, d8_out: [K/32] float (block scales for fast access).
void quantize_fp16_to_q8_1(const half* x, block_q8_1* q8_1_out, float* d8_out, int K,
                           cudaStream_t stream = nullptr);

// Fused SwiGLU + Q8_1 quantization: computes silu(gate) * up and quantizes
// the result directly to Q8_1 format, eliminating the intermediate FP16 buffer.
// gate/up: [total_elements] FP16, q8_out: [total_elements/32] Q8_1 blocks.
void swiglu_quantize_q8_1(const half* gate, const half* up, block_q8_1* q8_out, float* d8_out,
                          int total_elements, cudaStream_t stream = nullptr);

// Fused GEGLU + Q8_1 quantization: computes gelu_tanh(gate) * up and quantizes
// directly to Q8_1 format, eliminating the intermediate FP16 buffer for Gemma-3.
void geglu_quantize_q8_1(const half* gate, const half* up, block_q8_1* q8_out, float* d8_out,
                         int total_elements, cudaStream_t stream = nullptr);

// Fused relu² + Q8_1 quantization: applies relu²(x) = max(0,x)² and quantizes
// directly to Q8_1 format. Used by non-gated MoE experts (Nemotron).
// input: [total_elements] FP16, q8_out: [total_elements/32] Q8_1 blocks.
void relu_sqr_quantize_q8_1(const half* input, block_q8_1* q8_out, float* d8_out, int total_elements,
                            cudaStream_t stream = nullptr);

// Fused RMSNorm + Q8_1 quantization: normalises input then quantizes directly to Q8_1,
// eliminating the intermediate FP16 norm_out write+read. Single-row only (n=1 decode).
// If norm_out is non-null, also writes the FP16 normalized output.
void rmsnorm_quantize_q8_1(const half* x, const half* weight, block_q8_1* q8_out, float* d8_out,
                           half* norm_out, int d_model, float eps, cudaStream_t stream = nullptr,
                           float weight_offset = 0.0f);

// dp4a-accelerated GEMV: W_quant @ x_q8_1 using native INT8 SIMD.
// W: raw quantized bytes, q8_1: pre-quantized input, d8: pre-extracted scales, y: [M] FP16.
void gemv_q6k_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,
                   cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,
                    cudaStream_t stream = nullptr);
void gemv_q4_0_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,
                    cudaStream_t stream = nullptr);
void gemv_q4_k_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,
                    cudaStream_t stream = nullptr);
void gemv_q5_k_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,
                    cudaStream_t stream = nullptr);
void gemv_q2_k_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,
                    cudaStream_t stream = nullptr);
void gemv_q3_k_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,
                    cudaStream_t stream = nullptr);

// Residual-fused variants: y[i] = dot(W[i], x) + residual[i]
void gemv_q6k_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,
                            const half* residual, int M, int K, cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,
                             const half* residual, int M, int K, cudaStream_t stream = nullptr);
void gemv_q4_0_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,
                             const half* residual, int M, int K, cudaStream_t stream = nullptr);
void gemv_q4_k_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,
                             const half* residual, int M, int K, cudaStream_t stream = nullptr);
void gemv_q5_k_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,
                             const half* residual, int M, int K, cudaStream_t stream = nullptr);
void gemv_q2_k_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,
                             const half* residual, int M, int K, cudaStream_t stream = nullptr);
void gemv_q3_k_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,
                             const half* residual, int M, int K, cudaStream_t stream = nullptr);

// Fused gate+up dense GEMV: both projections in one kernel launch, dispatched
// internally by quant type. M = output rows, K = inner dim.
void gemv_gate_up_fused(const void* gate_weights, const void* up_weights, const block_q8_1* q8_1,
                        const float* d8, half* y_gate, half* y_up, int M, int K, QType qtype,
                        cudaStream_t stream = nullptr);

// Fused QKV GEMV: reads input once, computes Q/K/V projections in one kernel. All
// three weight matrices must share quant type and inner dim K. q_rows/k_rows/v_rows
// are each projection's output dim.
void gemv_qkv_fused_q6k_q8_1(const void* W_q, const void* W_k, const void* W_v, const block_q8_1* q8_1,
                             const float* d8, half* y_q, half* y_k, half* y_v, int q_rows, int k_rows,
                             int v_rows, int K, cudaStream_t stream = nullptr);
void gemv_qkv_fused_q8_0_q8_1(const void* W_q, const void* W_k, const void* W_v, const block_q8_1* q8_1,
                              const float* d8, half* y_q, half* y_k, half* y_v, int q_rows, int k_rows,
                              int v_rows, int K, cudaStream_t stream = nullptr);
void gemv_qkv_fused_q4_0_q8_1(const void* W_q, const void* W_k, const void* W_v, const block_q8_1* q8_1,
                              const float* d8, half* y_q, half* y_k, half* y_v, int q_rows, int k_rows,
                              int v_rows, int K, cudaStream_t stream = nullptr);
void gemv_qkv_fused_q4_k_q8_1(const void* W_q, const void* W_k, const void* W_v, const block_q8_1* q8_1,
                              const float* d8, half* y_q, half* y_k, half* y_v, int q_rows, int k_rows,
                              int v_rows, int K, cudaStream_t stream = nullptr);
void gemv_qkv_fused_q5_k_q8_1(const void* W_q, const void* W_k, const void* W_v, const block_q8_1* q8_1,
                              const float* d8, half* y_q, half* y_k, half* y_v, int q_rows, int k_rows,
                              int v_rows, int K, cudaStream_t stream = nullptr);
void gemv_qkv_fused_q2_k_q8_1(const void* W_q, const void* W_k, const void* W_v, const block_q8_1* q8_1,
                              const float* d8, half* y_q, half* y_k, half* y_v, int q_rows, int k_rows,
                              int v_rows, int K, cudaStream_t stream = nullptr);
void gemv_qkv_fused_q3_k_q8_1(const void* W_q, const void* W_k, const void* W_v, const block_q8_1* q8_1,
                              const float* d8, half* y_q, half* y_k, half* y_v, int q_rows, int k_rows,
                              int v_rows, int K, cudaStream_t stream = nullptr);

// Batched K/V projection: input @ [wk; wv]^T → k_out, v_out in a single cuBLAS call.
// weight_kv: [2 * nkv_hd, d_model] — wk at rows [0..nkv_hd), wv at [nkv_hd..2*nkv_hd)
// k_out, v_out: [n_tokens, nkv_hd] — must be contiguous (v_out = k_out + stride)
void gemm_kv_batched(const Tensor& input, const Tensor& weight_kv, Tensor& k_out, Tensor& v_out,
                     cudaStream_t stream = nullptr);

// Batched two-way GEMM: input @ [w1; w2]^T → out1, out2 in a single cuBLAS call.
// weight_fused: [2 * N, K] — w1 at rows [0..N), w2 at [N..2*N)
// out1, out2: [M, N] — may have arbitrary memory stride between them.
void gemm_pair_batched(const Tensor& input, const Tensor& weight_fused, Tensor& out1, Tensor& out2,
                       cudaStream_t stream = nullptr);

// MoE decode GEMV: processes all top_k experts in one kernel launch. packed_weights:
// base pointer to contiguous packed expert tensors. expert_indices: [top_k] int32 on
// device. x_stride: 0 = shared input (gate/up), K = per-expert input (down). y:
// [top_k,rows] FP16.
void gemv_q6k_moe_decode(const void* packed_weights, const int32_t* expert_indices, const half* x, half* y,
                         int rows, int K, size_t expert_stride_bytes, int x_stride, int top_k,
                         cudaStream_t stream = nullptr);
void gemv_q8_0_moe_decode(const void* packed_weights, const int32_t* expert_indices, const half* x, half* y,
                          int rows, int K, size_t expert_stride_bytes, int x_stride, int top_k,
                          cudaStream_t stream = nullptr);
// FP16 experts (MTP draft head only; main model experts are always quantized). Stride
// is in ELEMENTS not bytes, unlike the quantized variants: no block packing here, so
// elements are the natural unit (a byte stride would invite an off-by-factor-of-two).
void gemv_f16_moe_decode(const void* packed_weights, const int32_t* expert_indices, const half* x, half* y,
                         int rows, int K, size_t expert_stride_elems, int x_stride, int top_k,
                         cudaStream_t stream = nullptr);

// dp4a-accelerated MoE decode GEMV: same interface as above but with pre-quantized Q8_1
// input. q8_1_stride: 0 = shared input for all experts (gate/up), K/32 = per-expert
// (down). d8_stride matches (0 shared, K/32 per-expert).
void gemv_q6k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                              const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                              size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                              cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream = nullptr);
void gemv_q4_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream = nullptr);
void gemv_q5_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream = nullptr);
void gemv_q4_0_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream = nullptr);
void gemv_q2_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream = nullptr);
void gemv_q3_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream = nullptr);

void gemv_q5_1_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream = nullptr);

// Fused gate+up MoE GEMV: both projections in a single kernel launch.
// Uses blockIdx.y to select gate(0) or up(1). Saves one launch per MoE layer.
void gemv_q6k_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                int rows, int K, size_t gate_stride_bytes, size_t up_stride_bytes,
                                int x_stride, int top_k, cudaStream_t stream = nullptr);
void gemv_q8_0_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                 const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                 int rows, int K, size_t gate_stride_bytes, size_t up_stride_bytes,
                                 int x_stride, int top_k, cudaStream_t stream = nullptr);

// dp4a-accelerated fused gate+up MoE GEMV variants.
void gemv_q6k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                     const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                     half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                     size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                     cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream = nullptr);
void gemv_q4_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream = nullptr);
void gemv_q5_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream = nullptr);
void gemv_q4_0_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream = nullptr);
void gemv_q2_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream = nullptr);
void gemv_q3_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream = nullptr);

// Max-L1 SMEM carveout on every dp4a GEMV kernel template instantiation.
// Called from GraphExecutor::init(), independent of PDL: the kernels carry no
// pdl_wait() and are not registered (AUDIT_arch_2026 A1-3 / A2-1).
void gemv_dp4a_set_l1_carveout();

// dp4a-accelerated GEMV with FP32 output: W_quant @ x_q8_1 → y[M] float.
// Designed for the LM head projection where FP32 logits are needed for sampling.
void gemv_q6k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                        cudaStream_t stream = nullptr);
void gemv_q8_0_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream = nullptr);
void gemv_q4_0_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream = nullptr);
void gemv_q4_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream = nullptr);
void gemv_q5_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream = nullptr);
void gemv_q2_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream = nullptr);
void gemv_q3_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream = nullptr);

// Batched-activation LM head GEMV for spec-verify (#847 lever 2): n_act pre-quantized
// rows (row r at q8_1/d8 + r*act_stride_blocks) share one pass over W; logits land at
// y + r*M. Falls back to per-row GEMV when the quant type has no dp4a traits or rows
// don't fit smem.
void gemv_dp4a_fp32_batched(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8,
                            float* y, int M, int K, int n_act, int act_stride_blocks,
                            cudaStream_t stream = nullptr);

}  // namespace imp
