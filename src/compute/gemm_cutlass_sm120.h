#pragma once

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
    const void* data = nullptr;    // borrowed from NvFP4QuantResult::packed_data (not owned)
    void* scale_factors = nullptr; // SfAtom layout UE4M3 scale factor bytes (owned)
    float tensor_scale = 1.0f;     // deferred global scale (applied as GEMM alpha)
    int64_t N = 0;
    int64_t K = 0;
    size_t sf_bytes = 0;           // total bytes for scale_factors buffer
};

// Convert imp NvFP4QuantResult to CUTLASS block-scaled format.
// Borrows packed_data pointer (RowMajor). tensor_scale is stored for GEMM alpha.
void convert_nvfp4_to_cutlass(const NvFP4QuantResult& src,
                               CutlassNvFP4Weight& dst,
                               cudaStream_t stream);

void free_cutlass_nvfp4_weight(CutlassNvFP4Weight& w);

// Compute SfAtom buffer size for given dimensions (rows x K).
// Returns number of bytes (one UE4M3 per scale factor, plus alignment padding).
size_t cutlass_nvfp4_sf_size(int rows, int K);

// Quantize FP16 activation [M,K] to NVFP4 in CUTLASS block-scaled format.
// dst_data: pre-allocated [M, K/2] RowMajor packed FP4 bytes
// dst_sf:   pre-allocated SfAtom layout UE4M3 scales (cutlass_nvfp4_sf_size bytes)
void quantize_fp16_to_nvfp4_cutlass(const void* src_fp16, void* dst_data,
                                     void* dst_sf, int M, int K,
                                     cudaStream_t stream);

// Run CUTLASS sm_120 block-scaled NVFP4xNVFP4 GEMM: D = alpha * A x B^T
//   A (activation): [M, K] NVFP4 RowMajor + SFA scale factors
//   B (weight):     [N, K] NVFP4 RowMajor + SFB scale factors (micro_scale only)
//   D (output):     [M, N] FP16 RowMajor
//   alpha = b.tensor_scale (compensates for deferred tensor_scale)
// Returns false if CUTLASS kernel can't handle the dimensions.
bool gemm_nvfp4_cutlass_sm120(const void* a_data, const void* a_sf,
                               const CutlassNvFP4Weight& b,
                               void* d_fp16, int M, int N, int K,
                               void* workspace, size_t workspace_size,
                               cudaStream_t stream);

// Get CUTLASS GEMM workspace size for given problem dimensions.
size_t gemm_nvfp4_cutlass_sm120_workspace(int M, int N, int K);

// Check if sm_120 CUTLASS NVFP4 GEMM is compiled and available.
bool cutlass_sm120_nvfp4_available();

} // namespace imp
