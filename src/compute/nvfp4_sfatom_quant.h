#pragma once

namespace imp {
enum class FFNActivation;  // model/model_config.h; passed by value only
}  // namespace imp

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

struct NvFP4QuantResult;  // forward

// Weight for the block-scaled NVFP4 GEMMs (own tile, cuBLASLt on refusal; "Cutlass" names historical, #2938).
// Packed FP4 borrowed from the NVFP4 cache (RowMajor K-contiguous); scale_factors = micro_scale only (UE4M3
// SfAtom); tensor_scale is deferred to the GEMM epilogue alpha (avoids UE4M3 denormal range).
struct NvFP4SfAtomWeight {
    const void* data = nullptr;     // borrowed from NvFP4QuantResult::packed_data (not owned)
    void* scale_factors = nullptr;  // SfAtom layout UE4M3 scale factor bytes
    float tensor_scale = 1.0f;      // deferred global scale (applied as GEMM alpha)
    int64_t N = 0;
    int64_t K = 0;
    size_t sf_bytes = 0;  // total bytes for scale_factors buffer
};

struct NvFP4SfAtomOwned;  // compute/nvfp4_sfatom_owned.h: owns the SfAtom buffer (I5)

// Convert imp NvFP4QuantResult to CUTLASS block-scaled format.
// Borrows packed_data pointer (RowMajor). tensor_scale is stored for GEMM alpha.
void convert_nvfp4_to_sfatom(const NvFP4QuantResult& src, NvFP4SfAtomOwned& dst, cudaStream_t stream);

// Same conversion but writes SfAtom scales into a caller-provided, pre-zeroed buffer `sf_dst`
// (a sub-region of a shared slab) instead of a per-tensor cudaMalloc+cudaMemset. `dst` is a
// view; the caller owns the slab. sf_dst must be >=
// nvfp4_sfatom_size(src.N, src.K) bytes and pre-zeroed.
void convert_nvfp4_to_sfatom_borrowed(const NvFP4QuantResult& src, NvFP4SfAtomWeight& dst, void* sf_dst,
                                      cudaStream_t stream);

// Compute SfAtom buffer size for given dimensions (rows x K).
// Returns number of bytes (one UE4M3 per scale factor, plus alignment padding).
size_t nvfp4_sfatom_size(int rows, int K);

// MoE-fused scale conversion: native row-major UE4M3 [ne,N,K/16] -> SfAtom layout UE4M3
// [ne, nvfp4_sfatom_size(N,K)]. Single launch (grid.y=ne) so 128-expert layers don't pay
// 128x kernel-launch overhead. Caller pre-allocates dst sized ne*nvfp4_sfatom_size(N,K).
// expert_offsets (ne + 1 prefix sums): experts without rows keep their bytes unconverted.
void convert_nvfp4_moe_scales_to_sfatom(const void* src_native_ms, void* dst_sfatom_sf, int ne, int N, int K,
                                        cudaStream_t stream, const int32_t* expert_offsets = nullptr);

// Quantizes FP16 activation [M,K] to NVFP4 CUTLASS block-scaled format. dst_data: [M,K/2]
// packed FP4; dst_sf: SfAtom UE4M3 scales. Warns once at shutdown if any micro-block clipped
// the UE4M3 scale ceiling during the run (#1544); called from gemm_cleanup(). Includes the MoE
// decode act-quantize flag (gemm_grouped_nvfp4_decode_scale_clipped, gemm_grouped_nvfp4_decode.h).
void nvfp4_report_scale_clipping();
[[nodiscard]] bool gemm_grouped_nvfp4_decode_scale_clipped();

// act_scale (all SfAtom quantizers): static activation tensor scale, dequant form; the GEMM alpha must
// carry the same factor. 1.0 = block scale amax / 6 (bytes unchanged).
void quantize_fp16_to_nvfp4_sfatom(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                   cudaStream_t stream, float act_scale = 1.0f);

// MoE fused variant: single kernel quantizes all [expanded,K] rows into dst_packed
// [expanded,K/2] contiguous FP4 bytes (row-major); d_sfa_bases[e] is expert e's SFA slab
// pointer; d_offsets[ne+1] cumulative row offsets. Inactive experts (offsets[e+1]==offsets[e])
// contribute no threads; their sfa_bases entry is unused, set nullptr for defensive no-op.
void quantize_fp16_to_nvfp4_sfatom_moe(const void* src_fp16, void* dst_packed, uint8_t* const* d_sfa_bases,
                                       const int* d_offsets, int expanded, int K, int ne, cudaStream_t stream,
                                       float act_scale = 1.0f);

// MoE fused gather + NVFP4 CUTLASS quantize. Same packing/scale layout as
// quantize_fp16_to_nvfp4_sfatom_moe, but reads input in token order (pre-permute
// norm_out[n_tokens,K]); output row r reads src_fp16[sorted_token_ids[r]*K+...]. Bit-identical
// to (moe_gather -> quantize_fp16_to_nvfp4_sfatom_moe) since sorted_token_ids is a
// permutation. Enables a future skip-gather optimization gated on a lazy-gather addition in
// the legacy fallback (docs/plans/moe_prefill_cudagraph_via_cutlass_moe_scheduler_*.md).
void quantize_fp16_to_nvfp4_sfatom_moe_gather(const void* src_fp16, const int32_t* sorted_token_ids,
                                              void* dst_packed, uint8_t* const* d_sfa_bases,
                                              const int* d_offsets, int expanded, int K, int ne,
                                              cudaStream_t stream, float act_scale = 1.0f);

// Fused activation + NVFP4 CUTLASS quantize for the MoE down-projection input. Replaces
// apply_expert_activation(gate,up->swiglu) + quantize_fp16_to_nvfp4_sfatom_moe(swiglu): reads
// gate+up from HBM, computes the activation in registers, writes only packed FP4 + SFA, saving
// one HBM round-trip of the swiglu intermediate per MoE layer prefill call.
//   gate: [expanded,K] FP16, or nullptr when non_gated_experts=true (RELU_SQR reads up only)
//   up: [expanded,K] FP16; dst_packed: [expanded,K/2] packed FP4; d_sfa_bases: [ne] per-expert
//   SFA base pointers; d_offsets: [ne+1] cumulative row offsets; act_type: SWIGLU/GEGLU/RELU_SQR
// Bit-identical to (apply_expert_activation + quantize_..._moe) on SWIGLU/GEGLU (both compute
// the activation in float before quantization); RELU_SQR is also fused (gate ignored).
// GPT_OSS_GLU needs gate/up biases [ne,K] FP16, other acts ignore them. Bit-identical to
// bias_sorted x2 + gpt_oss_glu + quantize_..._moe (FP16 adds, FP16-rounded GLU output), #2466.
void fused_act_quantize_fp16_to_nvfp4_sfatom_moe(const void* gate_fp16, const void* up_fp16, void* dst_packed,
                                                 uint8_t* const* d_sfa_bases, const int* d_offsets,
                                                 int expanded, int K, int ne, FFNActivation act_type,
                                                 cudaStream_t stream, const void* gate_bias_fp16 = nullptr,
                                                 const void* up_bias_fp16 = nullptr, float act_scale = 1.0f);

// Dense prefill SwiGLU into out_fp16 [M,K] plus its CUTLASS NVFP4 quantize (dst_data [M,K/2],
// dst_sf SfAtom): bit-identical to swiglu() + quantize_fp16_to_nvfp4_sfatom(out_fp16), #2470.
void swiglu_quantize_fp16_to_nvfp4_sfatom(const void* gate_fp16, const void* up_fp16, void* out_fp16,
                                          void* dst_data, void* dst_sf, int M, int K, cudaStream_t stream,
                                          float act_scale = 1.0f);

}  // namespace imp
