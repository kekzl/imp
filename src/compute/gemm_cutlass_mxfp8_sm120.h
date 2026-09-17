#pragma once

// CUTLASS sm_120 block-scaled MXFP8 x MXFP8 GEMM (E4M3 elements, one UE8M0 scale per 32 K
// elements, SfAtom layout as the MXFP4 twin). Built for the F16 GDN projections at prefill
// (gemm.mxfp8_gdn_proj_prefill): E4M3 keeps 3 mantissa bits per element where NVFP4's E2M1
// keeps 1, so the W8A8 copy is meant to take the tensor-core prefill without the NVFP4 PPL
// price (roadmap Open 12). Weight and activation share one quantizer: row-major [rows, K] FP16
// -> [rows, K] E4M3 + SfAtom UE8M0, scale = the power of two that puts the block absmax in
// (224, 448].

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp {

// Buffers are the caller's (VRAMAllocator in src/exec, cudaMalloc in tests): this TU never
// allocates (docs/internals/MEMORY.md A3).
struct CutlassMxFP8Weight {
    void* data = nullptr;           // [N, K] E4M3
    void* scale_factors = nullptr;  // SfAtom UE8M0
    int64_t N = 0;
    int64_t K = 0;
    size_t data_bytes = 0;
    size_t sf_bytes = 0;
};

// SfAtom buffer size for (rows x K) at SFVecSize 32 (128-row x 4-group atoms).
size_t cutlass_mxfp8_sf_size(int rows, int K);

// FP16 [M,K] -> E4M3 [M,K] + SfAtom UE8M0. K % 32 == 0; dst_sf is cleared first (atom padding).
void quantize_fp16_to_mxfp8_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream);

// D[M,N] FP16 = A[M,K] (MXFP8 RowMajor + SFA) x B[N,K]^T (MXFP8 + SFB). False when CUTLASS
// declines the shape or the workspace is too small (the caller falls back).
bool gemm_mxfp8_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassMxFP8Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream);
size_t gemm_mxfp8_cutlass_sm120_workspace(int M, int N, int K);

bool cutlass_sm120_mxfp8_available();

}  // namespace imp
