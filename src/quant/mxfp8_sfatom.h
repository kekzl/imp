#pragma once

// MXFP8 operands for the own tile GEMM (gemm_mxfp8_tile): E4M3 elements, one UE8M0 scale per 32 K
// elements in the SfAtom layout of the MXFP4 twin (quant/mxfp4_sfatom.h). Built for the F16 GDN projections
// at prefill (gemm.mxfp8_gdn_proj_prefill). Weight and activation share one quantizer: row-major [rows, K]
// FP16 -> [rows, K] E4M3 + SfAtom UE8M0, scale = the power of two that puts the block absmax in (224, 448].

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
    // Row-scaled view of an FP8 row-scale sidecar: data borrowed (never freed here), SFB holds
    // each row's power of two, col_scale[n] the residual factor in [1, 2) applied to D column n.
    const float* col_scale = nullptr;
    bool borrowed_data = false;
};

// SfAtom UE8M0 [rows, K] + col_scale[rows] for E4M3 rows quantized with one FP32 scale per row:
// SFB = 2^floor(log2 row_scale), col_scale = row_scale / SFB. dst_sf holds cutlass_mxfp8_sf_size.
void mxfp8_sf_from_row_scales(const float* row_scales, void* dst_sf, float* dst_col_scale, int rows, int K,
                              cudaStream_t stream);

// SfAtom buffer size for (rows x K) at SFVecSize 32 (128-row x 4-group atoms).
size_t cutlass_mxfp8_sf_size(int rows, int K);

// FP16 [M,K] -> E4M3 [M,K] + SfAtom UE8M0. K % 32 == 0; dst_sf is cleared first (atom padding).
void quantize_fp16_to_mxfp8_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream);
// Inverse: the [N, K] copy back to FP16 rows (serves small-M GEMMs once the F16 source is freed).
void dequantize_mxfp8_cutlass_to_fp16(const CutlassMxFP8Weight& w, void* dst_fp16, cudaStream_t stream);

}  // namespace imp
