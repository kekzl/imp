#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight

namespace imp {

// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM for MoE (SM120).
// Per-expert M varies, shared N and K across all experts.
// Replaces CUTLASS 2.x GemmGrouped for NVFP4-quantized MoE expert weights.
bool cutlass_grouped_3x_nvfp4_available();

// Per-expert inputs for grouped NVFP4×NVFP4 → FP16 GEMM.
// All pointer fields below are HOST arrays of DEVICE pointers (length n_experts).
// The dispatch copies these to device internally and builds per-expert layouts.
//
//   A_i : [M_i,   K] packed NVFP4 (K-contiguous RowMajor, K/2 bytes per row)
//   SFA_i: SfAtom UE4M3 layout (size = cutlass_nvfp4_sf_size(M_i, K))
//   B_i : [N,     K] packed NVFP4 (from CutlassNvFP4Weight::data, per-expert)
//   SFB_i: SfAtom UE4M3 layout (from CutlassNvFP4Weight::scale_factors, per-expert)
//   D_i : [M_i,   N] FP16 output (RowMajor)
//   alpha_i: per-expert tensor_scale (applied as GEMM alpha)
//
// K and N must be identical across all experts.  M_i varies.
bool gemm_grouped_cutlass_3x_nvfp4(
    int n_experts,
    const int* host_M,  // [n_experts] M_i per expert
    int N, int K,
    const void* const* host_ptr_A,    // [n_experts] device pointers to packed A
    const void* const* host_ptr_SFA,  // [n_experts] device pointers to SFA
    const void* const* host_ptr_B,    // [n_experts] device pointers to packed B weight
    const void* const* host_ptr_SFB,  // [n_experts] device pointers to SFB
    void* const* host_ptr_D,          // [n_experts] device pointers to FP16 output
    const float* host_alpha,          // [n_experts] per-expert tensor_scale (alpha)
    cudaStream_t stream);

void gemm_grouped_3x_nvfp4_cleanup();

}  // namespace imp
