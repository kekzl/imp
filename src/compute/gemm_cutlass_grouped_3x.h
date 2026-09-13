#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight

namespace imp {

// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM for MoE (SM120).
// Per-expert M varies, shared N and K across all experts.
// Replaces CUTLASS 2.x GemmGrouped for NVFP4-quantized MoE expert weights.
bool cutlass_grouped_3x_nvfp4_available();

// Per-expert inputs for grouped NVFP4xNVFP4 -> FP16 GEMM. Pointer fields are HOST
// arrays of DEVICE pointers (length n_experts); dispatch copies them to device and
// builds per-expert layouts.
//   A_i [M_i,K] packed NVFP4 (K-contiguous RowMajor, K/2 bytes/row); SFA_i SfAtom UE4M3
//   B_i [N,K] packed NVFP4 (per-expert); SFB_i SfAtom UE4M3 (per-expert)
//   D_i [M_i,N] FP16 output RowMajor; alpha_i per-expert tensor_scale as GEMM alpha
// K and N must be identical across all experts; M_i varies.
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

// Phase 3b graph-capturable variant: all per-expert state lives on device, the staging
// buffer is built by an in-stream kernel (no host iteration, no D2H/H2D sync). Replaces
// the host-args wrapper once Phase 3c wires it into the MoE prefill dispatch.
// Activation buffer layout (matches executor_forward_moe.cu's CUTLASS 3.x
// quantize_once lambda):
//   A packed FP4: ptr_A[e] = base_A_packed + d_expert_offsets[e]*(K/2), K/2 bytes/row
//   SFA UE4M3: ptr_SFA[e] = base_A_sf + d_sfa_offsets[e], SfAtom-padded slab
//   B packed FP4 / SFB UE4M3: per-expert, fixed byte stride from base_B_packed/base_B_sf
//   D FP16: ptr_D[e] = base_D + d_expert_offsets[e]*N*sizeof(half), contiguous (alias
//   for C, beta=0)
struct GroupedNvfp4DeviceArgs {
    const int32_t* d_M_per;           // [n_experts]   per-expert token count
    const int32_t* d_expert_offsets;  // [n_experts+1] exclusive prefix sum of M_per
    const int64_t* d_sfa_offsets;     // [n_experts+1] exclusive prefix sum of cutlass_nvfp4_sf_size
    const float*   d_alpha;           // [n_experts]   per-expert alpha (act_ts * weight_ts)

    const void* base_A_packed;        // contiguous activation packed FP4 base
    const void* base_A_sf;            // SfAtom-padded SFA base

    // B/SFB storage, two mutually exclusive modes (Phase 3c-MVP):
    //   (a) contiguous slab + per-expert stride: set base_B_packed/b_expert_stride_packed
    //       (+ SFB equivalents), d_B_ptrs = d_SFB_ptrs = nullptr. Used by
    //       NvFP4MoEQuantResult (smallM path).
    //   (b) per-expert device pointer array: set d_B_ptrs/d_SFB_ptrs to [n_experts]
    //       device-resident arrays; base_B_*/b_expert_stride_* ignored when non-null. Used
    //       by registry-handle weights (CUTLASS 3.x prefill).
    const void* base_B_packed;        // mode (a) only
    int64_t     b_expert_stride_packed;
    const void* base_B_sf;            // mode (a) only
    int64_t     b_expert_stride_sf;
    const void* const* d_B_ptrs;      // mode (b): [n_experts] device array, or nullptr
    const void* const* d_SFB_ptrs;    // mode (b): [n_experts] device array, or nullptr

    void* base_D;                     // FP16 output base
};

bool gemm_grouped_cutlass_3x_nvfp4_device_args(
    int n_experts,
    int N, int K,
    const GroupedNvfp4DeviceArgs& args,
    cudaStream_t stream);

void gemm_grouped_3x_nvfp4_cleanup();

// The engine-persistent (T2) charge of the grouped path, taken by the prewarm
// below. The workspace figure is measured — see the prewarm's comment — not a
// conservative cap; exec_t2_demand replicates both for MoE models.
inline constexpr size_t kGrouped3xStagingBytes = 1ull << 20;    // 1 MiB
inline constexpr size_t kGrouped3xWorkspaceBytes = 1ull << 20;  // 1 MiB

// Pre-take the staging + workspace slices so every subsequent call finds them
// sufficient. Call once at engine init, after the T2 arena is open.
void gemm_grouped_3x_nvfp4_prewarm();

}  // namespace imp
