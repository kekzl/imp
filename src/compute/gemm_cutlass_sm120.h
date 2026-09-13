#pragma once

namespace imp {
enum class FFNActivation;  // model/model_config.h; passed by value only
}

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

struct NvFP4QuantResult;  // forward

// Converted weight data for CUTLASS sm_120 block-scaled GEMM. Packed FP4 pointer is borrowed
// from the NVFP4 cache (RowMajor K-contiguous). Scale factors hold micro_scale only (UE4M3
// SfAtom); tensor_scale is deferred to the GEMM epilogue alpha (avoids UE4M3 denormal range).
struct CutlassNvFP4Weight {
    const void* data = nullptr;     // borrowed from NvFP4QuantResult::packed_data (not owned)
    void* scale_factors = nullptr;  // SfAtom layout UE4M3 scale factor bytes
    float tensor_scale = 1.0f;      // deferred global scale (applied as GEMM alpha)
    int64_t N = 0;
    int64_t K = 0;
    size_t sf_bytes = 0;  // total bytes for scale_factors buffer
    // When true, `scale_factors` points into a shared buffer owned elsewhere (e.g. MoE
    // per-projection SfAtom buffer): lets 128 experts of one projection share one allocation.
    // Cleanup must skip cudaFree on these entries.
    bool sf_borrowed = false;
};

// Convert imp NvFP4QuantResult to CUTLASS block-scaled format.
// Borrows packed_data pointer (RowMajor). tensor_scale is stored for GEMM alpha.
void convert_nvfp4_to_cutlass(const NvFP4QuantResult& src, CutlassNvFP4Weight& dst, cudaStream_t stream);

// Same conversion but writes SfAtom scales into a caller-provided, pre-zeroed buffer `sf_dst`
// (a sub-region of a shared slab) instead of a per-tensor cudaMalloc+cudaMemset. Sets
// dst.sf_borrowed=true; caller owns and frees the slab. sf_dst must be >=
// cutlass_nvfp4_sf_size(src.N, src.K) bytes and pre-zeroed.
void convert_nvfp4_to_cutlass_borrowed(const NvFP4QuantResult& src, CutlassNvFP4Weight& dst, void* sf_dst,
                                       cudaStream_t stream);

void free_cutlass_nvfp4_weight(CutlassNvFP4Weight& w);

// Compute SfAtom buffer size for given dimensions (rows x K).
// Returns number of bytes (one UE4M3 per scale factor, plus alignment padding).
size_t cutlass_nvfp4_sf_size(int rows, int K);

// MoE-fused scale conversion: native row-major UE4M3 [ne,N,K/16] -> SfAtom layout UE4M3
// [ne, cutlass_nvfp4_sf_size(N,K)]. Single launch (grid.y=ne) so 128-expert layers don't pay
// 128x kernel-launch overhead. Caller pre-allocates dst sized ne*cutlass_nvfp4_sf_size(N,K).
void convert_nvfp4_moe_scales_to_sfatom(const void* src_native_ms, void* dst_sfatom_sf, int ne, int N,
                                        int K, cudaStream_t stream);

// Quantizes FP16 activation [M,K] to NVFP4 CUTLASS block-scaled format. dst_data: [M,K/2]
// packed FP4; dst_sf: SfAtom UE4M3 scales. Warns once at shutdown if any micro-block clipped
// the UE4M3 scale ceiling during the run (#1544); called from gemm_cleanup().
void nvfp4_report_scale_clipping();

void quantize_fp16_to_nvfp4_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream);

// MoE fused variant: single kernel quantizes all [expanded,K] rows into dst_packed
// [expanded,K/2] contiguous FP4 bytes (row-major); d_sfa_bases[e] is expert e's SFA slab
// pointer; d_offsets[ne+1] cumulative row offsets. Inactive experts (offsets[e+1]==offsets[e])
// contribute no threads; their sfa_bases entry is unused, set nullptr for defensive no-op.
void quantize_fp16_to_nvfp4_cutlass_moe(const void* src_fp16, void* dst_packed, uint8_t* const* d_sfa_bases,
                                        const int* d_offsets, int expanded, int K, int ne,
                                        cudaStream_t stream);

// MoE fused gather + NVFP4 CUTLASS quantize. Same packing/scale layout as
// quantize_fp16_to_nvfp4_cutlass_moe, but reads input in token order (pre-permute
// norm_out[n_tokens,K]); output row r reads src_fp16[sorted_token_ids[r]*K+...]. Bit-identical
// to (moe_gather -> quantize_fp16_to_nvfp4_cutlass_moe) since sorted_token_ids is a
// permutation. Enables a future skip-gather optimization gated on a lazy-gather addition in
// the legacy fallback (docs/plans/moe_prefill_cudagraph_via_cutlass_moe_scheduler_*.md).
void quantize_fp16_to_nvfp4_cutlass_moe_gather(const void* src_fp16,
                                               const int32_t* sorted_token_ids,
                                               void* dst_packed,
                                               uint8_t* const* d_sfa_bases,
                                               const int* d_offsets, int expanded, int K, int ne,
                                               cudaStream_t stream);

// Fused activation + NVFP4 CUTLASS quantize for the MoE down-projection input. Replaces
// apply_expert_activation(gate,up->swiglu) + quantize_fp16_to_nvfp4_cutlass_moe(swiglu): reads
// gate+up from HBM, computes the activation in registers, writes only packed FP4 + SFA, saving
// one HBM round-trip of the swiglu intermediate per MoE layer prefill call.
//   gate: [expanded,K] FP16, or nullptr when non_gated_experts=true (RELU_SQR reads up only)
//   up: [expanded,K] FP16; dst_packed: [expanded,K/2] packed FP4; d_sfa_bases: [ne] per-expert
//   SFA base pointers; d_offsets: [ne+1] cumulative row offsets; act_type: SWIGLU/GEGLU/RELU_SQR
// Bit-identical to (apply_expert_activation + quantize_..._moe) on SWIGLU/GEGLU (both compute
// the activation in float before quantization); RELU_SQR is also fused (gate ignored).
void fused_act_quantize_fp16_to_nvfp4_cutlass_moe(const void* gate_fp16, const void* up_fp16,
                                                  void* dst_packed, uint8_t* const* d_sfa_bases,
                                                  const int* d_offsets, int expanded, int K, int ne,
                                                  FFNActivation act_type, cudaStream_t stream);

// Runs CUTLASS sm_120 block-scaled NVFP4xNVFP4 GEMM: D = alpha*A@B^T. A (activation) [M,K]
// NVFP4 RowMajor + SFA; B (weight) [N,K] NVFP4 RowMajor + SFB (micro_scale only); D [M,N] FP16
// RowMajor; alpha = b.tensor_scale (compensates the deferred tensor_scale). Returns false if
// CUTLASS can't handle the dimensions.
bool gemm_nvfp4_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream);

// Get CUTLASS GEMM workspace size for given problem dimensions.
size_t gemm_nvfp4_cutlass_sm120_workspace(int M, int N, int K);

// Stream-K variant of the cooperative tile, callable directly (the dispatch above takes it via
// gemm.nvfp4_cutlass_streamk and grid size). force=true pins the stream-K decomposition; false
// lets the scheduler's heuristic choose data-parallel vs stream-K.
bool gemm_nvfp4_cutlass_sm120_streamk(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                      void* d_fp16, int M, int N, int K, void* workspace,
                                      size_t workspace_size, cudaStream_t stream, bool force);
size_t gemm_nvfp4_cutlass_sm120_streamk_workspace(int M, int N, int K);
// A/B probe: the pingpong 128x64 tile regardless of N.
bool gemm_nvfp4_cutlass_sm120_smalln(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                     void* d_fp16, int M, int N, int K, void* workspace,
                                     size_t workspace_size, cudaStream_t stream);
// Stream-K units the scheduler would launch for the shape (0 = data-parallel).
int gemm_nvfp4_cutlass_sm120_streamk_units(int M, int N, int K, bool force);

// FP32-output NVFP4 GEMM (large-N cooperative tile). Used for the batched-decode
// LM head, which needs float logits. d_fp32 is [M, N] row-major float.
bool gemm_nvfp4_cutlass_sm120_fp32(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                   void* d_fp32, int M, int N, int K, void* workspace,
                                   size_t workspace_size, cudaStream_t stream);
size_t gemm_nvfp4_cutlass_sm120_fp32_workspace(int M, int N, int K);

// Check if sm_120 CUTLASS NVFP4 GEMM is compiled and available.
bool cutlass_sm120_nvfp4_available();

}  // namespace imp
