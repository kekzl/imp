#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace imp {

// Fused multi-token GDN scan: all tokens in one launch, register-cached state. conv_f32:
// [n_tokens,conv_channels] FP32, per-token layout [Q(BC_size),K(BC_size),V(inner)].
// grouped_layout: 0=GGUF/tiled (g=h%n_groups), 1=HF SafeTensors/grouped (g=h/(n_heads/n_groups),
// Qwen3.5/3.6 NVFP4). d_real_n: real chunk length when padded (#847 captured verify); y is
// written for every row but h_state stops at the real last row.
// Batched scan over N independent sequences: tokens stay sequential per sequence, sequences
// parallelise across blockIdx.y (see gdn.cu launcher comment for the layout contract).
// seq_row_offsets (ragged batch, roadmap 0(d)): prefix sums [n_seq+1] into concatenated rows,
// sequence i owns [off[i],off[i+1]); n_tokens ignored per-seq. d_real_n must be nullptr if ragged.
// Uniform batches with n_tokens>1 are a multi-candidate verify chunk (W candidates x n_tokens
// rows, one slot each); d_real_n bounds each sequence's committed row, h_snap/d_snap_n written
// from sequence 0 only. out_slots/snap_slots (batched speculative verify, uniform only):
// per-sequence slot ids into h_state_pool; out_slots[i] at d_real_n rows (nullptr=in place),
// snap_slots[i] at d_snap_n rows for every sequence (nullptr=h_snap contract). In-place snapshot
// (snap_slots[i]==seq_slots[i]) is allowed.
void gdn_scan_fused_f32_batched(const float* conv_f32, int conv_channels, const half* alpha,
                                const half* beta, const float* A_log, const float* dt_bias,
                                float* h_state_pool, const int* seq_slots, int64_t h_state_seq_stride,
                                half* y, int n_seq, int n_tokens, int n_heads, int head_dim_ssm,
                                int state_size, int n_groups, cudaStream_t stream,
                                int grouped_layout = 0, const int* d_real_n = nullptr,
                                const int* seq_row_offsets = nullptr, float* h_snap = nullptr,
                                const int* d_snap_n = nullptr, const int* out_slots = nullptr,
                                const int* snap_slots = nullptr,
                                // Factored spare (compute/gdn_factor.cuh): fac_out gets the rank-1 row for
                                // real_n instead
                                // of a full state copy (out_slots then unused); fac_in is a pending row
                                // applied after the
                                // state load. Both [n_seq, n_heads, fac_stride] floats.
                                float* fac_out = nullptr, const float* fac_in = nullptr,
                                int fac_stride = 0);

void gdn_scan_fused_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                        const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                        int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                        int grouped_layout = 0, const int* d_real_n = nullptr);

// BF16-state twins (gdn.state_bf16): h_state as __nv_bfloat16, all arithmetic FP32 in
// registers, halves the state traffic dominating batched decode. HD=SS=128 only; init
// resolver refuses BF16 state elsewhere and for ref/chunkwise routes. Stride in BF16 elements.
void gdn_scan_fused_bf16_batched(const float* conv_f32, int conv_channels, const half* alpha,
                                 const half* beta, const float* A_log, const float* dt_bias,
                                 __nv_bfloat16* h_state_pool, const int* seq_slots,
                                 int64_t h_state_seq_stride, half* y, int n_seq, int n_tokens, int n_heads,
                                 int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                                 int grouped_layout = 0, const int* d_real_n = nullptr,
                                 const int* seq_row_offsets = nullptr, __nv_bfloat16* h_snap = nullptr,
                                 const int* d_snap_n = nullptr, const int* out_slots = nullptr,
                                 const int* snap_slots = nullptr, float* fac_out = nullptr,
                                 const float* fac_in = nullptr, int fac_stride = 0);
void gdn_scan_fused_bf16(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias, __nv_bfloat16* h_state, half* y,
                         int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                         cudaStream_t stream, int grouped_layout = 0, const int* d_real_n = nullptr);
void gdn_scan_fused_fp32out_bf16(const float* conv_f32, int conv_channels, const half* alpha,
                                 const half* beta, const float* A_log, const float* dt_bias,
                                 __nv_bfloat16* h_state, float* y_fp32, int n_tokens, int n_heads,
                                 int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                                 int grouped_layout = 0, const int* d_real_n = nullptr,
                                 __nv_bfloat16* h_snap = nullptr, const int* d_snap_n = nullptr);

// EXPERIMENTAL scaffolding for chunkwise SSD scan (Mamba2 SSD adapted to GDN's rank-1
// (I - beta k k^T) update, unlike standard scalar-decay SSD; cf. Yang et al. 2024 delta rule).
// Same signature as gdn_scan_fused_f32, functionally identical; validated by ChunkBoundaryHandoff.
void gdn_scan_chunkwise_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int chunk_size = 64, int grouped_layout = 0,
                            const int* d_real_n = nullptr);

// FP32-output chunkwise variant — matches `gdn_scan_fused_fp32out`'s contract.
// Same chunked shared-memory layout as `gdn_scan_chunkwise_f32`, output kept
// in FP32 for the gdn.fp32_scan path (RMSNorm+Gate+SiLU pipeline).
void gdn_scan_chunkwise_fp32out(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                                const float* A_log, const float* dt_bias, float* h_state, float* y_fp32,
                                int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                                cudaStream_t stream, int chunk_size = 64, int grouped_layout = 0,
                                const int* d_real_n = nullptr, float* h_snap = nullptr,
                                const int* d_snap_n = nullptr);

// Phase 2a WY-rep parallel delta-rule scan: numerically equivalent to the sequential delta
// rule, factors the chunk-internal dependency into a triangular solve + matmuls (naive
// shared-memory; TC MMA is Phase 2b). HD=SS=128 CHUNK=32 only, else falls back to gdn_scan_fused_f32.
void gdn_scan_chunkwise_wy_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                               const float* A_log, const float* dt_bias, float* h_state, half* y,
                               int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                               cudaStream_t stream, int grouped_layout = 0);

// Phase 2b: WY-rep delta-rule scan with Tensor Core MMA on KK/QK/KH/QH; H_L stays scalar.
// CHUNK=16 to fit FP16 K~/Q~/H_0 + FP32 KH/QH in the 99 KiB sm_120 opt-in cap. HD=SS=128
// only, else falls back to gdn_scan_fused_f32. FP16 storage error stays in tolerance budget.
void gdn_scan_chunkwise_wy_tc_f32(const float* conv_f32, int conv_channels, const half* alpha,
                                  const half* beta, const float* A_log, const float* dt_bias,
                                  float* h_state, half* y, int n_tokens, int n_heads, int head_dim_ssm,
                                  int state_size, int n_groups, cudaStream_t stream, int grouped_layout = 0);

// Phase 2c: WY-rep TC MMA on all 5 chunk-internal matmuls (KK, QK, KH, QH-inlined, H_L).
// CHUNK=32 by dropping s_qh and reusing s_kh for the H_L strip output. Parallel L2 norm
// across 4 warps (one token per warp, warp-shuffle reduction). HD=SS=128 only, else fallback.
void gdn_scan_chunkwise_wy_tc2_f32(const float* conv_f32, int conv_channels, const half* alpha,
                                   const half* beta, const float* A_log, const float* dt_bias,
                                   float* h_state, half* y, int n_tokens, int n_heads, int head_dim_ssm,
                                   int state_size, int n_groups, cudaStream_t stream,
                                   int grouped_layout = 0);

// Chunk-parallel prefill scan (gdn.chunkpar_scan): same numerics as the WY kernels; grid
// (chunks x heads), only a short matmul chain stays sequential. HD=SS=128 only; needs
// gdn_scan_chunkpar_workspace_bytes(n_heads) bytes, 256B aligned; no d_real_n (padded verify
// chunks use the fused kernel). StateT float or BF16, rounds to BF16 once at commit.
// strip_chunks (gdn.chunkpar_strip): 0 = auto, else clamped to [1, 16].
size_t gdn_scan_chunkpar_workspace_bytes(int n_heads);
void gdn_scan_chunkpar_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                           const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                           int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                           int grouped_layout, float* ws, size_t ws_bytes, int strip_chunks = 0);
void gdn_scan_chunkpar_bf16(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, __nv_bfloat16* h_state, half* y,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int grouped_layout, float* ws, size_t ws_bytes,
                            int strip_chunks = 0);

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
                            cudaStream_t stream, int grouped_layout = 0, const int* d_real_n = nullptr,
                            float* h_snap = nullptr, const int* d_snap_n = nullptr);

// Reference multi-token GDN scan: deliberately unfused, state in global memory, per-token
// serial loop; L2-norm via shared-memory reductions (no register-cached state). Same
// delta-rule math as gdn_scan_fused_f32. Selected by gdn.ref_kernel / runtime.debug_raw.
void gdn_scan_reference_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                            int grouped_layout = 0, const int* d_real_n = nullptr);

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

// V-head reorder tiled -> grouped (FP32, conv1d-SiLU output). GGUF may store V tiled
// (h0_g0,h1_g1,...,h0_g0_r1,...) while the scan kernel expects grouped V[h*HD+d]; Qwen 3.6
// (16K/32V) triggers this. Skipped when n_heads == n_groups.
void vhead_tiled_to_grouped_f32(const float* src, float* dst, int n_tokens, int n_heads, int head_dim,
                                int n_groups, cudaStream_t stream);

}  // namespace imp
