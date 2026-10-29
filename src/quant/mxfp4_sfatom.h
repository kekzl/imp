#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#include "quant/cutlass_mxfp4_weight.h"  // CutlassMxFP4Weight

namespace imp {

// MXFP4 operand bytes in the SfAtom layout the own tile GEMM reads (gemm_mxfp4_tile): E2M1 nibbles
// [rows, K/2] plus UE8M0 per 32 elements, 128 rows x 4 scales = 512 B per atom, atoms K-major.

// SfAtom buffer size for MXFP4 (UE8M0, one per 32 K, rows padded to 128).
size_t cutlass_mxfp4_sf_size(int rows, int K);

void free_cutlass_mxfp4_weight(CutlassMxFP4Weight& w);

// Unpack native MXFP4 GGUF blocks (17 bytes each: 16 data + 1 scale) in place into
// separate data and SfAtom scale arrays.
[[nodiscard]] bool unpack_mxfp4_gguf(const void* raw_gpu, int64_t N, int64_t K, CutlassMxFP4Weight& dst,
                                     cudaStream_t stream);

// Quantizes FP16 activation [M,K] to MXFP4: absmax per 32 elements -> UE8M0 scale.
// dst_data: pre-allocated [M,K/2] packed FP4 bytes; dst_sf: pre-allocated SfAtom UE8M0 scales.
void quantize_fp16_to_mxfp4_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream);

// Dequantize raw MXFP4 GGUF blocks to FP16.
// Input: raw interleaved blocks (17 bytes each: 16 E2M1 + 1 UE8M0).
// Output: [N, K] FP16 on device (caller allocates).
// Unpacked weight (unpack_mxfp4_gguf output: data + linear_scales) to FP16 [N, K].
void dequant_mxfp4_linear_to_fp16(const void* data, const void* linear_scales, int64_t N, int64_t K,
                                  void* dst_fp16, cudaStream_t stream);

void dequant_mxfp4_to_fp16(const void* raw_mxfp4_data, int64_t N, int64_t K, void* dst_fp16,
                           cudaStream_t stream);

}  // namespace imp
