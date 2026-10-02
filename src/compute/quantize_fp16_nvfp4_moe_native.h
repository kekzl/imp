#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// Per-expert FP16 -> NVFP4 quantize, native row-major UE4M3 scale layout.
// Input: [expanded,K] FP16, expert_offsets[ne+1] partitions rows.
// Output per expert: packed NVFP4 [M_e,K/2] (d_packed_ptrs[e]) + UE4M3 scales
// [M_e,K/16] (d_sf_ptrs[e]), row-major dense.
// Scaling (matches quantize_fp16_to_nvfp4_with_scale(1.0), nvfp4_quant.cu): fixed activation
// tensor_scale=1 (a row never depends on its expert mates, #2167); micro_scale=local_absmax/6 UE4M3;
// fp4 = val/(tensor_scale*micro_scale_actual), E2M1 HW sat.
// Read by gemm_grouped_nvfp4_smallM (cache_moe_native_nvfp4 / nvfp4_moe_ms_native).
// Returns false, nothing launched, when the pointer-array upload fails (#2446).
[[nodiscard]] bool quantize_fp16_to_nvfp4_moe_native(
    const __half* src_fp16,       // [expanded, K]
    void* const* d_packed_ptrs,   // [n_experts] per-expert packed FP4
    void* const* d_sf_ptrs,       // [n_experts] per-expert UE4M3
    const int* d_expert_offsets,  // [n_experts + 1] device pointer
    int expanded, int K, int n_experts, cudaStream_t stream);

// Same as above, plus writes per-expert FP32 tensor scale to d_tensor_scales
// ([n_experts]): the fixed activation tensor_scale 1.0, matching the
// internal quant scale. Needed by the smallM grouped GEMM, which folds
// (a_tensor_scale * b_tensor_scale) into alpha.
[[nodiscard]] bool quantize_fp16_to_nvfp4_moe_native_with_scales(
    const __half* src_fp16, void* const* d_packed_ptrs, void* const* d_sf_ptrs,
    float* d_tensor_scales,  // [n_experts] FP32, written by callee
    const int* d_expert_offsets, int expanded, int K, int n_experts,
    void** d_ptr_scratch,  // [2*n_experts] device scratch; nullptr: allocated per call
    cudaStream_t stream);

// Per-expert alpha = activation_tensor_scale * weight_tensor_scale, on device.
// d_act_scales/d_weight_scales/d_alpha_out: [n_experts] device floats.
// Tiny 1-block kernel; no host/device sync.
void compute_moe_alpha_device(
    const float* d_act_scales,
    const float* d_weight_scales,
    float* d_alpha_out,
    int n_experts,
    cudaStream_t stream);

// Per-expert M_per[e] = expert_offsets[e+1] - expert_offsets[e], computed device-side.
// Replaces the host D2H + sync + subtract loop in MoE prefill dispatch
// (executor_forward_moe.cu); eliminating that sync is a CUDA-graph-capture
// prerequisite for MoE prefill. Single tiny block; safe inside a captured graph.
void compute_M_per_from_offsets_device(
    const int32_t* d_expert_offsets,
    int32_t* d_M_per_out,
    int n_experts,
    cudaStream_t stream);

// Compacts per-expert alpha to active experts (M_per[e] > 0), ascending index order.
// Replaces the host D2H+sync+compact+H2D pattern at executor_forward_moe.cu:1492-1514;
// the second graph-capture prerequisite for MoE prefill (moe_prefill_graphs_plan_2026_05_10
// Phase 2). Single-block 256-thread launch; safe in a captured graph. n_experts <= 256.
void compact_alpha_active(
    const float* d_alpha,
    const int32_t* d_M_per,
    float* d_alpha_compact,
    int32_t* d_na_out,
    int n_experts,
    cudaStream_t stream);

// Exclusive prefix sum of cutlass_nvfp4_sf_size(M_per[e],K) into device SFA offsets, so
// ptr_SFA[e] = base_SFA + d_sfa_offsets_out[e] matches the CUTLASS 3.x grouped NVFP4
// wrapper's host-side staging (gemm_cutlass_grouped_3x.cu). Phase 3 of
// moe_prefill_graphs_plan_2026_05_10.
// Padding: n_row_tiles=ceil(M_e/128), n_k_tiles=ceil(K/64), bytes=n_row_tiles*n_k_tiles*512
// (see cutlass_nvfp4_sf_size, gemm_cutlass_sm120.cu). Single-block 256-thread; n_experts<=256.
// d_sfa_bases_out (nullable): also writes d_sfa_bases_out[e] = base_sf + offset[e] in the
// same launch. Capture-safe.
void compute_sfa_offsets_device(const int32_t* d_M_per, int64_t* d_sfa_offsets_out, int n_experts, int K,
                                cudaStream_t stream, uint8_t** d_sfa_bases_out = nullptr,
                                void* base_sf = nullptr);

// Zeros the active prefix of an SFA staging buffer, reading the true byte count from
// d_sfa_offsets[n_experts] instead of a worst-case cudaMemsetAsync(buf,0,max_bytes).
// Grid-strided, guards against max_bytes for OOB safety. Capture-safe.
void bzero_sfa_active(
    void* dst,
    const int64_t* d_sfa_offsets,
    int n_experts,
    size_t max_bytes,
    cudaStream_t stream);

}  // namespace imp
