#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

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
// Arguments of the batched fused scans (audit D-11): built with designated initializers, so a
// swapped n_heads / head_dim_ssm / state_size / n_groups reads wrong at the call site.
// StateT float (gdn_scan_fused_f32_batched) or __nv_bfloat16 (the BF16-state twin below).
template <typename StateT>
struct GdnScanBatchedArgs {
    const float* conv_f32 = nullptr;
    int conv_channels = 0;
    const half* alpha = nullptr;
    const half* beta = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    StateT* h_state_pool = nullptr;
    const int* seq_slots = nullptr;
    int64_t h_state_seq_stride = 0;  // in StateT elements
    half* y = nullptr;
    int n_seq = 0;
    int n_tokens = 0;
    int n_heads = 0;
    int head_dim_ssm = 0;
    int state_size = 0;
    int n_groups = 0;
    cudaStream_t stream = nullptr;
    int grouped_layout = 0;
    const int* d_real_n = nullptr;
    const int* seq_row_offsets = nullptr;
    StateT* h_snap = nullptr;
    const int* d_snap_n = nullptr;
    const int* out_slots = nullptr;
    const int* snap_slots = nullptr;
    // Factor rows (compute/gdn_factor.cuh), with out_slots only: rows d_snap_n + 1 .. real_n - 1.
    float* fac_out = nullptr;
    int fac_stride = 0;
};

void gdn_scan_fused_f32_batched(const GdnScanBatchedArgs<float>& a);

// Arguments of the single-sequence scans (fused f32 / bf16, chunkwise f32; audit D-11), in the old
// parameter order. StateT float or __nv_bfloat16.
template <typename StateT>
struct GdnScanArgs {
    const float* conv_f32 = nullptr;
    int conv_channels = 0;
    const half* alpha = nullptr;
    const half* beta = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    StateT* h_state = nullptr;
    half* y = nullptr;
    int n_tokens = 0;
    int n_heads = 0;
    int head_dim_ssm = 0;
    int state_size = 0;
    int n_groups = 0;
    cudaStream_t stream = nullptr;
    int grouped_layout = 0;
    const int* d_real_n = nullptr;
    StateT* h_snap = nullptr;
    const int* d_snap_n = nullptr;
    int64_t snap2_off = 0;  // bytes from h_snap to a second slab: state at d_snap_n + 1 rows; 0 = none
};

void gdn_scan_fused_f32(const GdnScanArgs<float>& a);

// BF16-state twins (gdn.state_bf16): h_state as __nv_bfloat16, all arithmetic FP32 in
// registers, halves the state traffic dominating batched decode. HD=SS=128 only; init
// resolver refuses BF16 state elsewhere and for ref/chunkwise routes. Stride in BF16 elements.
void gdn_scan_fused_bf16_batched(const GdnScanBatchedArgs<__nv_bfloat16>& a);
void gdn_scan_fused_bf16(const GdnScanArgs<__nv_bfloat16>& a);

// EXPERIMENTAL scaffolding for chunkwise SSD scan (Mamba2 SSD adapted to GDN's rank-1
// (I - beta k k^T) update, unlike standard scalar-decay SSD; cf. Yang et al. 2024 delta rule).
// Same arguments as gdn_scan_fused_f32, functionally identical; validated by ChunkBoundaryHandoff.
void gdn_scan_chunkwise_f32(const GdnScanArgs<float>& a, int chunk_size = 64);

// Chunk-parallel prefill scan (gdn.chunkpar_scan): WY-rep delta rule (Yang et al. 2024); grid
// (chunks x heads), only a short matmul chain stays sequential. HD=SS=128 only; needs
// gdn_scan_chunkpar_workspace_bytes(n_heads) bytes, 256B aligned; no d_real_n (padded verify
// chunks use the fused kernel). StateT float or BF16, rounds to BF16 once at commit.
// strip_chunks (kGdnChunkparStrip): 0 = auto, else clamped to [1, 16].
size_t gdn_scan_chunkpar_workspace_bytes(int n_heads);
// Arguments of the chunk-parallel scans (audit D-11), in the old parameter order.
template <typename StateT>
struct GdnChunkparArgs {
    const float* conv_f32 = nullptr;
    int conv_channels = 0;
    const half* alpha = nullptr;
    const half* beta = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    StateT* h_state = nullptr;
    half* y = nullptr;
    int n_tokens = 0;
    int n_heads = 0;
    int head_dim_ssm = 0;
    int state_size = 0;
    int n_groups = 0;
    cudaStream_t stream = nullptr;
    int grouped_layout = 0;
    float* ws = nullptr;
    size_t ws_bytes = 0;
    int strip_chunks = 0;
};
void gdn_scan_chunkpar_f32(const GdnChunkparArgs<float>& a);
void gdn_scan_chunkpar_bf16(const GdnChunkparArgs<__nv_bfloat16>& a);

// Fused RMSNormGated + SiLU: y = rmsnorm(y) * silu(gate)
// Processes all tokens × heads in one launch.
// sigmoid_gate: Qwen4Exp output_gate_type=sigmoid; false = SiLU (Qwen3.5/3.6 hidden_act).
void gdn_rmsnorm_gated_silu(half* y, const half* gate, const half* weight, float eps, int n_tokens,
                            int n_heads, int head_dim, cudaStream_t stream, bool sigmoid_gate = false);

// Reference multi-token GDN scan: deliberately unfused, state in global memory, per-token
// serial loop; L2-norm via shared-memory reductions (no register-cached state). Same
// delta-rule math as gdn_scan_fused_f32. Selected by gdn.ref_kernel.
// Arguments in the old parameter order (audit D-11); grouped_layout 0 = tiled, d_real_n nullptr = n_tokens.
struct GdnScanReferenceArgs {
    const float* conv_f32 = nullptr;
    int conv_channels = 0;
    const half* alpha = nullptr;
    const half* beta = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    float* h_state = nullptr;
    half* y = nullptr;
    int n_tokens = 0;
    int n_heads = 0;
    int head_dim_ssm = 0;
    int state_size = 0;
    int n_groups = 0;
    cudaStream_t stream = nullptr;
    int grouped_layout = 0;
    const int* d_real_n = nullptr;
};
void gdn_scan_reference_f32(const GdnScanReferenceArgs& a);

// ---------------------------------------------------------------------------
// Legacy per-token interfaces (kept for fallback / testing)
// ---------------------------------------------------------------------------
// Arguments in the old parameter order (audit D-11); z nullptr = no gate, grouped_layout 0 = tiled.
struct GdnScanDecodeArgs {
    const float* x = nullptr;
    const float* B = nullptr;
    const float* C = nullptr;
    const half* alpha = nullptr;
    const half* beta = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    float* h_state = nullptr;
    half* y = nullptr;
    const half* z = nullptr;
    int n_heads = 0;
    int head_dim_ssm = 0;
    int state_size = 0;
    int n_groups = 0;
    cudaStream_t stream = nullptr;
    int grouped_layout = 0;
};
void gdn_scan_decode_f32(const GdnScanDecodeArgs& a);

}  // namespace imp
