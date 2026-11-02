#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp {

// Epilogue of the grouped NVFP4 MoE decode GEMM.
//   kF16:    D = alpha_e A B_e^T as FP16 [expanded, N] (the down projection).
//   kSwiglu, kGeglu: gate and up in one pass (tile = 64 gate + 64 up rows of the same columns),
//            act(fp16(alpha_g A G^T), fp16(alpha_u A U^T)) quantized to NVFP4 in the epilogue: packed
//            [expanded, N/2] + M16 scales, bit-equal to tile GEMM x 2 + fused_act_quantize.
//   kRelu2:  non-gated experts, relu(u)^2 of one projection, same NVFP4 output.
enum class MoeDecodeEpi { kF16, kSwiglu, kGeglu, kRelu2 };

// Per-expert state on the device; B operands as [n_experts] pointer arrays (SfAtom scales).
//   A packed FP4: base_A_packed + d_expert_offsets[e]*(K/2).
//   SFA: SfAtom at base_A_sf + d_sfa_offsets[e], or with d_sfa_offsets == nullptr the M16 layout an act
//   epilogue writes: per expert, per 16-row M tile, per 64-K atom 64 B (row i at i*4), experts packed by
//   prefix of ceil(M_e/16) tiles. Act epilogues write their scales in that M16 layout at base_D_sf.
struct GroupedNvfp4DecodeArgs {
    const int32_t* d_M_per = nullptr;           // [n_experts]
    const int32_t* d_expert_offsets = nullptr;  // [n_experts+1]
    const int64_t* d_sfa_offsets = nullptr;     // [n_experts] SfAtom offsets, or nullptr (M16 layout)
    const void* base_A_packed = nullptr;
    const void* base_A_sf = nullptr;
    const void* const* d_B_ptrs = nullptr;  // up (gated) or the only projection
    const void* const* d_SFB_ptrs = nullptr;
    const float* d_alpha = nullptr;
    const void* const* d_G_ptrs = nullptr;  // gate, gated epilogues only
    const void* const* d_SFG_ptrs = nullptr;
    const float* d_galpha = nullptr;
    void* base_D = nullptr;     // kF16: FP16 [expanded, N]; else packed FP4 [expanded, N/2]
    void* base_D_sf = nullptr;  // act epilogues: M16 scales
};

// True when the kernel serves the shape: 0 < n_experts <= 512, K % 64 == 0, K >= 320 (map slot reuse),
// N % 8 == 0 (kF16), N % 64 == 0 (gated), N % 16 == 0 (kRelu2); false before the prewarm.
[[nodiscard]] bool gemm_grouped_nvfp4_decode_supports(int n_experts, int N, int K, MoeDecodeEpi epi);

// Bytes an act epilogue writes for `expanded` rows: packed rows (128 B aligned) + the M16 scale bound,
// <= 1.5625 x expanded x N (fits the FP16 [expanded, N] buffer). The SF base is base_D + the aligned packed
// bytes (act_sf_offset).
[[nodiscard]] size_t gemm_grouped_nvfp4_decode_act_bytes(int n_experts, int expanded, int N);
[[nodiscard]] size_t gemm_grouped_nvfp4_decode_act_sf_offset(int expanded, int N);

// M16 scale byte of (expert-local row r, 16-element group kg) inside expert e's region (tests).
inline size_t gemm_grouped_nvfp4_decode_m16_sf(int r, int kg, int N) {
    return (((static_cast<size_t>(r / 16) * ((N + 63) / 64)) + (kg / 4)) * 64) +
           (static_cast<size_t>(r % 16) * 4) + (kg % 4);
}

// One launch; 16-row M tiles, persistent CTAs over (expert, m tile, n tile). Per-row bits equal
// gemm_grouped_nvfp4_tile (same K order, same rounding). false = refused (supports) or bad args.
[[nodiscard]] bool gemm_grouped_nvfp4_decode(int n_experts, int N, int K, int expanded, MoeDecodeEpi epi,
                                             const GroupedNvfp4DecodeArgs& args, cudaStream_t stream);

// Opt-in shared memory of every instance, outside any stream capture (gemm_grouped_nvfp4_prewarm calls it);
// without it supports() refuses.
void gemm_grouped_nvfp4_decode_prewarm();

// The act epilogues' scale clipping flag (absmax > 2688): gemm_grouped_nvfp4_decode_scale_clipped(),
// declared in gemm_cutlass_sm120.h next to nvfp4_report_scale_clipping, which reports it.

}  // namespace imp
