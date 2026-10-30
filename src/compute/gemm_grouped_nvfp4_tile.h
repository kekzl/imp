#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp {

// Grouped NVFP4 x NVFP4 -> FP16 MoE GEMM operands, all per-expert state on the device:
//   A packed FP4: base_A_packed + d_expert_offsets[e]*(K/2), K/2 bytes/row
//   SFA UE4M3: base_A_sf + d_sfa_offsets[e], SfAtom-padded slab
//   B packed FP4 / SFB UE4M3: per-expert, (a) fixed byte stride from base_B_packed/base_B_sf or
//   (b) [n_experts] device pointer arrays d_B_ptrs/d_SFB_ptrs (used when non-null)
//   D FP16: base_D + d_expert_offsets[e]*N*sizeof(half), alpha_e = d_alpha[e]
struct GroupedNvfp4DeviceArgs {
    const int32_t* d_M_per;           // [n_experts]   per-expert token count
    const int32_t* d_expert_offsets;  // [n_experts+1] exclusive prefix sum of M_per
    const int64_t* d_sfa_offsets;  // [n_experts] exclusive prefix sum of cutlass_nvfp4_sf_size ([ne] unread)
    const float* d_alpha;          // [n_experts]   per-expert alpha (act_ts * weight_ts)

    const void* base_A_packed;  // contiguous activation packed FP4 base
    const void* base_A_sf;      // SfAtom-padded SFA base

    const void* base_B_packed;  // mode (a) only
    int64_t b_expert_stride_packed;
    const void* base_B_sf;  // mode (a) only
    int64_t b_expert_stride_sf;
    const void* const* d_B_ptrs;    // mode (b): [n_experts] device array, or nullptr
    const void* const* d_SFB_ptrs;  // mode (b): [n_experts] device array, or nullptr

    void* base_D;  // FP16 output base
};

// Same operands with the per-expert arrays on the HOST (legacy host-args MoE prefill). Copied into the
// T2 staging slice of gemm_grouped_nvfp4_prewarm in one stream-ordered H2D copy; never under capture, one
// stream (the slice is shared by every call).
struct GroupedNvfp4HostArgs {
    const int32_t* offsets;     // [n_experts+1] exclusive prefix sum of M_per
    const size_t* sfa_offsets;  // [n_experts] byte offset of expert e's SFA slab in base_A_sf
    const void* const* B;       // [n_experts] packed B
    const void* const* SFB;     // [n_experts] SFB
    const float* alpha;         // [n_experts]
    const void* base_A_packed;
    const void* base_A_sf;
    void* base_D;
};

// T2 staging for the host-args form: 36 B/expert + section alignment, 512 experts at most.
inline constexpr size_t kGroupedNvfp4StagingBytes = 64ull << 10;

// True when the own kernel serves the shape: 0 < n_experts <= 512, N % 8 == 0, K % 64 == 0.
[[nodiscard]] bool gemm_grouped_nvfp4_supports(int n_experts, int N, int K);

// D_e[M_e, N] = alpha_e A_e B_e^T for every expert. expanded: rows of the expert-sorted activation buffer
// (bound of its TMA map). false = refused (gemm_grouped_nvfp4_supports). Four tensormap slots per CTA live
// in a static device array (one launch at a time).
[[nodiscard]] bool gemm_grouped_nvfp4_tile(int n_experts, int N, int K, int expanded,
                                           const GroupedNvfp4DeviceArgs& args, cudaStream_t stream);

// MoE prefill entry: gemm_grouped_nvfp4_tile, ACTIVE logged once.
[[nodiscard]] bool gemm_grouped_nvfp4_dispatch(int n_experts, int N, int K, int expanded,
                                               const GroupedNvfp4DeviceArgs& args, cudaStream_t stream);

// Host-args form; false when refused or the staging slice is missing.
[[nodiscard]] bool gemm_grouped_nvfp4_host_args(int n_experts, int N, int K, int expanded,
                                                const GroupedNvfp4HostArgs& args, cudaStream_t stream);

// Both MoE GEMM shapes of a layer: gate/up N = eff, K = d; down N = d, K = eff.
[[nodiscard]] inline bool gemm_grouped_nvfp4_moe_supports(int n_experts, int d, int eff) {
    return gemm_grouped_nvfp4_supports(n_experts, eff, d) && gemm_grouped_nvfp4_supports(n_experts, d, eff);
}

// Takes the staging slice from the T2 arena (once per arena generation). MoE models, at engine init.
void gemm_grouped_nvfp4_prewarm();

}  // namespace imp
