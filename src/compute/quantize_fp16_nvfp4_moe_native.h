#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Per-expert FP16 -> NVFP4 quantize, native row-major UE4M3 scale layout.
// Input:  [expanded, K] FP16 activations, expert_offsets[ne+1] partitions rows.
// Output: per-expert packed NVFP4 (FP4 nibbles, K/2 bytes per row) +
//         per-expert UE4M3 scales (K/16 bytes per row), both row-major dense.
//
// d_packed_ptrs[e] points to a [M_e, K/2] tightly-packed FP4 buffer.
// d_sf_ptrs[e]     points to a [M_e, K/16] UE4M3 row-major buffer.
// expert_offsets[e..e+1] gives the row range in src_fp16.
//
// Two-level scaling matches quantize_fp16_to_nvfp4 (nvfp4_quant.cu):
//   tensor_scale   = per-expert absmax / 6.0  (FP32)
//   micro_scale    = local_absmax / (tensor_scale * 6.0), encoded FP8 UE4M3
//   quantized FP4  = val / (tensor_scale * micro_scale_actual), E2M1 HW sat
//
// Layout convention (native, no SfAtom padding):
//   packed[m][k/2] = packed nibbles (low=even, high=odd element index)
//   sf[m][kb]      = UE4M3 byte for elements [kb*16, kb*16+15]
//   where kb = k_block = k/16
//
// This layout is read directly by gemm_grouped_nvfp4_smallM (cache_moe_native_nvfp4 /
// nvfp4_moe_ms_native buffers). See bench/sm120_smallM_audit.md.
void quantize_fp16_to_nvfp4_moe_native(
    const __half* src_fp16,              // [expanded, K]
    void* const* d_packed_ptrs,          // [n_experts] per-expert packed FP4
    void* const* d_sf_ptrs,              // [n_experts] per-expert UE4M3
    const int* d_expert_offsets,         // [n_experts + 1] device pointer
    int expanded,
    int K,
    int n_experts,
    cudaStream_t stream);

// Same as above, but additionally writes per-expert FP32 tensor scale into
// `d_tensor_scales` (device pointer to [n_experts] FP32). The convention is
// `tensor_scale_e = absmax_e / 6.0` (1.0 if absmax_e==0), matching the
// internal scale used during quantization. Required for the smallM grouped
// GEMM dispatch which folds (a_tensor_scale * b_tensor_scale) into alpha.
void quantize_fp16_to_nvfp4_moe_native_with_scales(
    const __half* src_fp16,
    void* const* d_packed_ptrs,
    void* const* d_sf_ptrs,
    float* d_tensor_scales,              // [n_experts] FP32, written by callee
    const int* d_expert_offsets,
    int expanded,
    int K,
    int n_experts,
    cudaStream_t stream);

// Compute per-expert alpha = activation_tensor_scale * weight_tensor_scale
// on device, given both scale arrays on device. Result is written to
// d_alpha_out. All three pointers are device pointers.
//
// d_act_scales    : [n_experts] device floats — per-expert activation tensor_scale
//                   (output of quantize_fp16_to_nvfp4_moe_native_with_scales)
// d_weight_scales : [n_experts] device floats — per-expert weight tensor_scale
//                   (W->tensor_scales from NvFP4MoEQuantResult, already on device)
// d_alpha_out     : [n_experts] device floats — output (act * weight per expert)
//
// Launches a tiny 1-block kernel; no host/device synchronization required.
void compute_moe_alpha_device(
    const float* d_act_scales,
    const float* d_weight_scales,
    float* d_alpha_out,
    int n_experts,
    cudaStream_t stream);

// Compute per-expert M_per values from device-resident expert_offsets array.
// M_per[e] = expert_offsets[e+1] - expert_offsets[e]   (token count routed to expert e)
//
// Replaces the host-side  `cudaMemcpyAsync(h_offsets, d_offsets, ...) +
// cudaStreamSynchronize + for(e) M_per[e]=h_offsets[e+1]-h_offsets[e]`  pattern
// used in MoE prefill dispatch (executor_forward_moe.cu). Eliminating that D2H
// sync is the prerequisite for CUDA-graph capture of MoE prefill — the decode
// fast-path already does no D2H but the prefill path falls back to host-driven
// dispatch.
//
// d_expert_offsets : [n_experts + 1] device int32 — exclusive scan of token counts
// d_M_per_out      : [n_experts]     device int32 — written by callee
//
// Launches a single tiny block; safe inside a captured CUDA graph.
void compute_M_per_from_offsets_device(
    const int32_t* d_expert_offsets,
    int32_t* d_M_per_out,
    int n_experts,
    cudaStream_t stream);

}  // namespace imp
