#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight

namespace imp {

// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM for MoE (SM120).
// Zero D2H sync — problem shapes built on GPU from device-side expert offsets.
// Replaces CUTLASS 2.x GemmGrouped for NVFP4-quantized MoE expert weights.
bool cutlass_grouped_3x_nvfp4_available();

// NVFP4 × NVFP4 → FP16 grouped GEMM.
// A: [expanded, K] NVFP4 quantized activations (packed + SfAtom scales)
// B: [n_experts] × CutlassNvFP4Weight (per-expert NVFP4 weights)
// D: [expanded, N] FP16 output
// d_offsets: [n_experts+1] device expert offsets into expanded dimension
bool gemm_grouped_cutlass_3x_nvfp4(
    const void* a_packed,          // [expanded, K/2] NVFP4 packed activations
    const void* a_sf,              // SfAtom UE4M3 activation scales
    void* d_fp16,                  // [expanded, N] FP16 output
    const int32_t* d_offsets,      // [n_experts+1] device expert offsets
    const CutlassNvFP4Weight* const* d_weight_ptrs,  // [n_experts] device weight struct pointers
    int K, int N,
    int n_experts,
    float tensor_scale,            // global tensor scale (applied as alpha)
    cudaStream_t stream);

void gemm_grouped_3x_nvfp4_cleanup();

} // namespace imp
