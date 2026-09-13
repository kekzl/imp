#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#include "quant/cutlass_mxfp4_weight.h"  // CutlassMxFP4Weight (moved down, see there)

namespace imp {

// MXFP4 weight data for CUTLASS sm_120 block-scaled GEMM. Differs from NVFP4 in scale
// format: UE4M3/16 elements (NVFP4) vs UE8M0/32 elements (MXFP4, pure exponent, wider
// range 2^-127..2^127, coarser granularity). Existing NVFP4 weights are converted
// UE4M3/16 -> UE8M0/32 to use the MXFP4 tensor core path.

// Compute SfAtom buffer size for MXFP4 (UE8M0, SFVecSize=32).
size_t cutlass_mxfp4_sf_size(int rows, int K);

// Convert NVFP4 weights (UE4M3 per 16) to MXFP4 scale format (UE8M0 per 32).
// Merges pairs of UE4M3 micro-scales into one UE8M0 scale per 32 elements.
// Packed data pointer is borrowed (same E2M1 nibbles, same layout).
struct NvFP4QuantResult;  // forward
void convert_nvfp4_to_mxfp4_cutlass(const NvFP4QuantResult& src, CutlassMxFP4Weight& dst,
                                    cudaStream_t stream);

void free_cutlass_mxfp4_weight(CutlassMxFP4Weight& w);

// Unpack native MXFP4 GGUF blocks (17 bytes each: 16 data + 1 scale)
// into separate data and SfAtom scale arrays for CUTLASS GEMM.
bool unpack_mxfp4_gguf(const void* raw_gpu, int64_t N, int64_t K, CutlassMxFP4Weight& dst,
                       cudaStream_t stream);

// Quantizes FP16 activation [M,K] to MXFP4 CUTLASS block-scaled format: absmax per 32
// elements -> UE8M0 scale. Optional Walsh-Hadamard rotation before quantization
// (hadamard_size>0). dst_data: pre-allocated [M,K/2] packed FP4 bytes; dst_sf:
// pre-allocated SfAtom UE8M0 scales.
void quantize_fp16_to_mxfp4_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream);

// Runs CUTLASS sm_120 block-scaled MXFP4xMXFP4 GEMM: D = alpha*A@B^T. A (activation)
// [M,K] MXFP4 RowMajor + SFA UE8M0; B (weight) [N,K] MXFP4 RowMajor + SFB UE8M0; D
// [M,N] FP16 RowMajor.
bool gemm_mxfp4_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassMxFP4Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream);

size_t gemm_mxfp4_cutlass_sm120_workspace(int M, int N, int K);

// Dequantize raw MXFP4 GGUF blocks to FP16.
// Input: raw interleaved blocks (17 bytes each: 16 E2M1 + 1 UE8M0).
// Output: [N, K] FP16 on device (caller allocates).
void dequant_mxfp4_to_fp16(const void* raw_mxfp4_data, int64_t N, int64_t K, void* dst_fp16,
                           cudaStream_t stream);

bool cutlass_sm120_mxfp4_available();

}  // namespace imp
