#pragma once

#include "model/model_config.h"  // FFNActivation

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

struct NvFP4QuantResult;  // forward

// Converted weight data for CUTLASS sm_120 block-scaled GEMM.
// Packed FP4 data pointer is borrowed from the NVFP4 cache (RowMajor K-contiguous).
// Scale factors hold micro_scale only (UE4M3 SfAtom layout); tensor_scale is
// deferred to the GEMM epilogue alpha for better precision (avoids UE4M3
// denormalized range).
struct CutlassNvFP4Weight {
    const void* data = nullptr;     // borrowed from NvFP4QuantResult::packed_data (not owned)
    void* scale_factors = nullptr;  // SfAtom layout UE4M3 scale factor bytes
    float tensor_scale = 1.0f;      // deferred global scale (applied as GEMM alpha)
    int64_t N = 0;
    int64_t K = 0;
    size_t sf_bytes = 0;  // total bytes for scale_factors buffer
    // When true, `scale_factors` points into a shared buffer owned elsewhere
    // (e.g. MoE per-projection SfAtom buffer in the VRAM allocator). Used so
    // 128 experts of one projection can share one allocation; cleanup must
    // skip cudaFree on these entries.
    bool sf_borrowed = false;
};

// Convert imp NvFP4QuantResult to CUTLASS block-scaled format.
// Borrows packed_data pointer (RowMajor). tensor_scale is stored for GEMM alpha.
void convert_nvfp4_to_cutlass(const NvFP4QuantResult& src, CutlassNvFP4Weight& dst, cudaStream_t stream);

// Same conversion, but writes the SfAtom scale factors into a caller-provided,
// pre-zeroed buffer `sf_dst` (a sub-region of a shared slab) instead of doing a
// per-tensor cudaMalloc+cudaMemset. Sets dst.sf_borrowed=true — the caller owns
// the backing slab and frees it once. `sf_dst` must have at least
// cutlass_nvfp4_sf_size(src.N, src.K) bytes and be pre-zeroed (padding rows).
void convert_nvfp4_to_cutlass_borrowed(const NvFP4QuantResult& src, CutlassNvFP4Weight& dst, void* sf_dst,
                                       cudaStream_t stream);

void free_cutlass_nvfp4_weight(CutlassNvFP4Weight& w);

// Compute SfAtom buffer size for given dimensions (rows x K).
// Returns number of bytes (one UE4M3 per scale factor, plus alignment padding).
size_t cutlass_nvfp4_sf_size(int rows, int K);

// MoE-fused scale conversion: native row-major UE4M3 [ne, N, K/16] →
// SfAtom layout UE4M3 [ne, cutlass_nvfp4_sf_size(N, K)]. Per-expert strides
// computed from N, K. Single launch (grid.y = ne) so 128-expert layers do not
// pay 128× kernel-launch overhead. Caller pre-allocates dst sized
// `ne * cutlass_nvfp4_sf_size(N, K)` bytes.
void convert_nvfp4_moe_scales_to_sfatom(const void* src_native_ms, void* dst_sfatom_sf, int ne, int N,
                                        int K, cudaStream_t stream);

// Quantize FP16 activation [M,K] to NVFP4 in CUTLASS block-scaled format.
// dst_data: pre-allocated [M, K/2] RowMajor packed FP4 bytes
// dst_sf:   pre-allocated SfAtom layout UE4M3 scales (cutlass_nvfp4_sf_size bytes)
void quantize_fp16_to_nvfp4_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream);

// MoE fused variant: single kernel quantizes all [expanded, K] rows into
//   dst_packed: [expanded, K/2] contiguous FP4 bytes (row-major, same as input layout)
//   d_sfa_bases[e]: pointer to expert e's SFA slab (device array of ne pointers)
//   d_offsets[ne+1]: cumulative row offsets (device int array)
// Inactive experts (offsets[e+1] == offsets[e]) contribute no threads and their
// sfa_bases entry is unused; set to nullptr for defensive no-op.
void quantize_fp16_to_nvfp4_cutlass_moe(const void* src_fp16, void* dst_packed, uint8_t* const* d_sfa_bases,
                                        const int* d_offsets, int expanded, int K, int ne,
                                        cudaStream_t stream);

// MoE fused gather + NVFP4 CUTLASS quantize. Same packing + scale layout as
// quantize_fp16_to_nvfp4_cutlass_moe, but the input is read in *token order*
// (i.e. the pre-permute MoE input `norm_out[n_tokens, K]`) and each output row
// `r` reads from `src_fp16[sorted_token_ids[r] * K + ...]`. Designed to enable
// a future skip-gather optimisation (drop the upstream moe_gather + write
// straight from norm_out to NVFP4) by giving the dispatcher one entry point
// that consumes the permutation directly. Bit-identical to the
// (moe_gather → quantize_fp16_to_nvfp4_cutlass_moe) pair when the gather is
// idempotent (it is — `sorted_token_ids` is a permutation).
//
// Today (2026-05-23) the upstream `moe_gather` still runs, so this saves only
// the gathered_base HBM read which is an L2 hit on a 96 MB-L2 RTX 5090 — net
// perf delta ~0 on Qwen3-Coder-30B-A3B-NVFP4 pp512 (measured). The real win
// (~+1.3 % from skipping the moe_gather HBM write drain) is gated on a
// lazy-gather addition in the legacy fallback path (Phase 2 of the multi-week
// plan in docs/plans/moe_prefill_cudagraph_via_cutlass_moe_scheduler_*.md).
// Ship the kernel + dispatch wiring now so the future fix is one branch + one
// `if (!gather_done) moe_gather(...);` instead of also a new kernel.
void quantize_fp16_to_nvfp4_cutlass_moe_gather(const void* src_fp16,
                                               const int32_t* sorted_token_ids,
                                               void* dst_packed,
                                               uint8_t* const* d_sfa_bases,
                                               const int* d_offsets, int expanded, int K, int ne,
                                               cudaStream_t stream);

// Fused activation + NVFP4 CUTLASS quantize for the MoE down-projection input.
// Replaces apply_expert_activation(gate, up -> swiglu) + quantize_fp16_to_nvfp4_cutlass_moe(swiglu).
// Reads gate + up directly from HBM, computes the activation in registers, and
// writes only the packed FP4 + SFA — saving one full HBM round-trip of the
// swiglu intermediate per MoE layer prefill call (~188 MiB on Qwen3-Coder
// NVFP4 at expanded=32k, eff=2880; ~+5-10% pp512 per review/phase5_synthesis
// §2.2 M1).
//
//   gate        : [expanded, K] FP16, OR nullptr when non_gated_experts=true
//                 (RELU_SQR reads up only).
//   up          : [expanded, K] FP16
//   dst_packed  : [expanded, K/2] packed FP4 nibbles
//   d_sfa_bases : [ne] per-expert SFA base pointers
//   d_offsets   : [ne+1] cumulative row offsets
//   act_type    : FFNActivation::SWIGLU / GEGLU / RELU_SQR
//
// Behavior is bit-identical to (apply_expert_activation + quantize_..._moe)
// on the SWIGLU/GEGLU paths because both compute the activation in float
// before quantization. RELU_SQR is also fused (gate ignored).
void fused_act_quantize_fp16_to_nvfp4_cutlass_moe(const void* gate_fp16, const void* up_fp16,
                                                  void* dst_packed, uint8_t* const* d_sfa_bases,
                                                  const int* d_offsets, int expanded, int K, int ne,
                                                  FFNActivation act_type, cudaStream_t stream);

// Run CUTLASS sm_120 block-scaled NVFP4xNVFP4 GEMM: D = alpha * A x B^T
//   A (activation): [M, K] NVFP4 RowMajor + SFA scale factors
//   B (weight):     [N, K] NVFP4 RowMajor + SFB scale factors (micro_scale only)
//   D (output):     [M, N] FP16 RowMajor
//   alpha = b.tensor_scale (compensates for deferred tensor_scale)
// Returns false if CUTLASS kernel can't handle the dimensions.
bool gemm_nvfp4_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream);

// Get CUTLASS GEMM workspace size for given problem dimensions.
size_t gemm_nvfp4_cutlass_sm120_workspace(int M, int N, int K);

// Check if sm_120 CUTLASS NVFP4 GEMM is compiled and available.
bool cutlass_sm120_nvfp4_available();

}  // namespace imp
