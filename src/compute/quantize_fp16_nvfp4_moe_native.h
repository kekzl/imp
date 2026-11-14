#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

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
// ptr_SFA[e] = base_SFA + d_sfa_offsets_out[e] matches the grouped NVFP4 GEMM's
// GroupedNvfp4DeviceArgs (gemm_grouped_nvfp4_tile.h). Phase 3 of
// moe_prefill_graphs_plan_2026_05_10.
// Padding: n_row_tiles=ceil(M_e/128), n_k_tiles=ceil(K/64), bytes=n_row_tiles*n_k_tiles*512
// (see cutlass_nvfp4_sf_size, nvfp4_sfatom_quant.cu). Single-block 256-thread; n_experts<=256.
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
