#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// Fused multi-token GDN scan.
// Processes all tokens in a SINGLE kernel launch with register-cached state.
// conv_f32: [n_tokens, conv_channels] FP32 — full conv+SiLU output per token
//           layout per token: [Q(BC_size), K(BC_size), V(inner)]
// ---------------------------------------------------------------------------
// grouped_layout: 0 = GGUF/tiled (kernel uses g = h % n_groups; default).
//                 1 = HF SafeTensors/grouped (g = h / (n_heads / n_groups);
//                     used by Qwen3.5/3.6 NVFP4 SafeTensors).
void gdn_scan_fused_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                        const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                        int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                        int grouped_layout = 0);

// EXPERIMENTAL — Phase 1b scaffolding for chunkwise SSD scan.
// Same signature as `gdn_scan_fused_f32`. Will eventually implement the
// parallel-within-chunk Mamba2 SSD algorithm (cf. Yang et al. 2024 "Parallel
// Linear Attention With The Delta Rule" for the delta-rule adaptation; the
// standard Mamba2 SSD assumes scalar decay and doesn't cover GDN's rank-1
// `(I - β k k^T)` multiplicative update).
//
// Currently implemented as a chunk-iterating sequential wrapper around
// `gdn_scan_fused_f32` — the FUNCTIONAL behavior is identical to the
// monolithic call, validated by `tests/test_gdn.cu::ChunkBoundaryHandoff`.
// Phase 1b.1 replaces the within-chunk sequential scan with the SSD matmul
// algorithm; this scaffolding establishes the API + regression-gate plumbing.
//
// Design doc: docs/plans/gdn_chunkwise_scan_design_2026_05_23.md
// Phase 0 verdict: docs/plans/gdn_chunkwise_scan_design_2026_05_23.md →
// PROCEED (latency-bound, 5.47 % mem + 5.47 % compute at peak).
void gdn_scan_chunkwise_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int chunk_size = 64, int grouped_layout = 0);

// FP32-output chunkwise variant — matches `gdn_scan_fused_fp32out`'s contract.
// Same chunked shared-memory layout as `gdn_scan_chunkwise_f32`, output kept
// in FP32 for the gdn.fp32_scan path (RMSNorm+Gate+SiLU pipeline).
void gdn_scan_chunkwise_fp32out(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                                const float* A_log, const float* dt_bias, float* h_state, float* y_fp32,
                                int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                                cudaStream_t stream, int chunk_size = 64, int grouped_layout = 0);

// Phase 2a — WY-representation parallel delta-rule scan prototype.
// Numerically equivalent to the sequential delta rule but factors the
// chunk-internal dependency into a forward triangular solve + matrix-matrix
// products (steps 2/3/5/6 of the algorithm in
// docs/plans/gdn_chunkwise_scan_design_2026_05_23.md §"Phase 2a"). Naive
// shared-memory matmuls; Tensor Core MMA is Phase 2b.
//
// Currently instantiated for HD=SS=128, CHUNK=32 (chunk_size=32 dispatch);
// other shapes fall back to gdn_scan_fused_f32 to avoid silent precision
// loss before validation lands.
void gdn_scan_chunkwise_wy_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                               const float* A_log, const float* dt_bias, float* h_state, half* y,
                               int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                               cudaStream_t stream, int grouped_layout = 0);

// Phase 2b — WY-rep delta-rule scan with Tensor Core MMA on the four
// chunk-internal matmuls (KK, QK, KH, QH). The H_L update remains scalar
// but with hoisted cumulative-decay caching. CHUNK=16 to fit the FP16
// K̃/Q̃/H_0 + FP32 KH/QH outputs within the 99 KiB sm_120 opt-in cap.
//
// Numerically near-equivalent to the sequential reference (FP16 storage
// of K̃/Q̃/H_0 introduces a small precision drop, but FP32 accumulation
// throughout the WMMA + downstream ops keeps the output within Phase 1a's
// tolerance budget).
//
// HD=SS=128 only; other shapes fall back to gdn_scan_fused_f32.
void gdn_scan_chunkwise_wy_tc_f32(const float* conv_f32, int conv_channels, const half* alpha,
                                  const half* beta, const float* A_log, const float* dt_bias,
                                  float* h_state, half* y, int n_tokens, int n_heads, int head_dim_ssm,
                                  int state_size, int n_groups, cudaStream_t stream, int grouped_layout = 0);

// Fused RMSNormGated + SiLU: y = rmsnorm(y) * silu(gate)
// Processes all tokens × heads in one launch.
void gdn_rmsnorm_gated_silu(half* y, const half* gate, const half* weight, float eps, int n_tokens,
                            int n_heads, int head_dim, cudaStream_t stream);

// FP32-input variant. Reads scan output from FP32 buffer (preserves precision
// when scan values are subnormal in FP16, ~6e-5). Writes FP16 result to `y`.
// Use together with FP32 scan output to match llama.cpp numerics.
void gdn_rmsnorm_gated_silu_fp32in(half* y_fp16_out, const float* y_fp32_in, const half* gate,
                                   const half* weight, float eps, int n_tokens, int n_heads, int head_dim,
                                   cudaStream_t stream);

// FP32-in, FP32-out variant. Keeps precision all the way through the normalized
// gated activation so the downstream ssm_out GEMM can accumulate over 4096
// terms without 6% FP16 drift (the Qwen 3.6 L0 sign-flip root cause).
void gdn_rmsnorm_gated_silu_fp32inout(float* y_fp32_out, const float* y_fp32_in, const half* gate,
                                      const half* weight, float eps, int n_tokens, int n_heads, int head_dim,
                                      cudaStream_t stream);

// FP32-output scan. Same math as `gdn_scan_fused_f32` but keeps result in FP32
// for feeding into `gdn_rmsnorm_gated_silu_fp32in`.
void gdn_scan_fused_fp32out(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, float* y_fp32,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int grouped_layout = 0);

// ---------------------------------------------------------------------------
// Reference multi-token GDN scan.
// Deliberately unfused: state lives in global memory, per-token loop serial,
// L2-norm of Q/K applied in-kernel but via shared-memory reductions (no
// register-cached state). Same delta-rule math as `gdn_scan_fused_f32` but
// trivially inspectable for validation. Enable via `IMP_GDN_REF=1`.
// ---------------------------------------------------------------------------
void gdn_scan_reference_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                            int grouped_layout = 0);

// ---------------------------------------------------------------------------
// Legacy per-token interfaces (kept for fallback / testing)
// ---------------------------------------------------------------------------
void gdn_scan_decode_f32(const float* x, const float* B, const float* C, const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias, float* h_state, half* y, const half* z,
                         int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                         int grouped_layout = 0);

void gdn_scan_prefill_f32(const float* x, const float* B, const float* C, const half* alpha, const half* beta,
                          const float* A_log, const float* dt_bias, float* h_state, half* y, const half* z,
                          int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                          cudaStream_t stream, int grouped_layout = 0);

// V-head reorder: tiled → grouped (FP16 variant, for post-scan y_buf).
void vhead_tiled_to_grouped(const half* src, half* dst, int n_tokens, int n_heads, int head_dim, int n_groups,
                            cudaStream_t stream);

// V-head reorder: tiled → grouped (FP32 variant, for conv1d-SiLU output).
// Asymmetric heads (n_groups != n_heads): the GGUF converter may store V in
// tiled order (h0_g0, h1_g1, ..., h15_g15, h0_g0_r1, ...) while the scan
// kernel reads V[h * HD + d] assuming grouped order. Qwen 3.6 uses 16K/32V
// and triggers this mismatch. Skips when n_heads == n_groups.
void vhead_tiled_to_grouped_f32(const float* src, float* dst, int n_tokens, int n_heads, int head_dim,
                                int n_groups, cudaStream_t stream);

// Legacy stubs
void gdn_scan_decode(const half*, const half*, const half*, const half*, const half*, const float*,
                     const float*, float*, half*, const half*, int, int, int, int, cudaStream_t);
void gdn_scan_prefill(const half*, const half*, const half*, const half*, const half*, const float*,
                      const float*, float*, half*, const half*, int, int, int, int, int, cudaStream_t);
void gdn_decode(const half*, const half*, const half*, const half*, const half*, float*, half*, const half*,
                int, int, int, int, cudaStream_t);
void gdn_prefill(const half*, const half*, const half*, const half*, const half*, float*, half*, const half*,
                 int, int, int, int, int, cudaStream_t);

}  // namespace imp
