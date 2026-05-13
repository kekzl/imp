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

// Phase 3b: graph-capturable variant. All per-expert state lives on the
// device — the staging buffer is built by an in-stream device kernel
// (no host iteration, no D2H/H2D sync). Designed to replace the host-args
// wrapper above inside the MoE prefill dispatch once Phase 3c wires it.
//
// Activation buffer layout assumptions (matching executor_forward_moe.cu's
// CUTLASS 3.x quantize_once lambda):
//   - A packed FP4: contiguous, K/2 bytes per row.
//                   ptr_A[e] = base_A_packed + d_expert_offsets[e] * (K/2)
//   - SFA UE4M3:    SfAtom-padded slab, byte offsets from d_sfa_offsets.
//                   ptr_SFA[e] = base_A_sf + d_sfa_offsets[e]
//   - B packed FP4: per-expert, fixed byte stride.
//                   ptr_B[e] = base_B_packed + e * b_expert_stride_packed
//   - SFB UE4M3:    per-expert, fixed byte stride.
//                   ptr_SFB[e] = base_B_sf + e * b_expert_stride_sf
//   - D FP16:       contiguous output (alias for C, beta=0).
//                   ptr_D[e] = base_D + d_expert_offsets[e] * N * sizeof(half)
struct GroupedNvfp4DeviceArgs {
    const int32_t* d_M_per;           // [n_experts]   per-expert token count
    const int32_t* d_expert_offsets;  // [n_experts+1] exclusive prefix sum of M_per
    const int64_t* d_sfa_offsets;     // [n_experts+1] exclusive prefix sum of cutlass_nvfp4_sf_size
    const float*   d_alpha;           // [n_experts]   per-expert alpha (act_ts * weight_ts)

    const void* base_A_packed;        // contiguous activation packed FP4 base
    const void* base_A_sf;            // SfAtom-padded SFA base
    const void* base_B_packed;        // expert weight packed FP4 base
    int64_t     b_expert_stride_packed;
    const void* base_B_sf;            // expert SFB base
    int64_t     b_expert_stride_sf;
    void*       base_D;               // FP16 output base
};

bool gemm_grouped_cutlass_3x_nvfp4_device_args(
    int n_experts,
    int N, int K,
    const GroupedNvfp4DeviceArgs& args,
    cudaStream_t stream);

void gemm_grouped_3x_nvfp4_cleanup();

}  // namespace imp
