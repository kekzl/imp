#pragma once

#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight, quantize/convert functions
#include <cuda_runtime.h>

namespace imp {

// cuBLASLt block-scaled NVFP4 GEMM: D = tensor_scale * A_fp4 × B_fp4^T
//
// Uses CUDA 12.8+ cublasLtMatmul with VEC16_UE4M3 block-scale mode for native
// hardware-accelerated FP4 GEMM on Blackwell (sm_120).
//
// Same data format as gemm_nvfp4_cutlass_sm120 — CutlassNvFP4Weight and SfAtom
// scale factors are directly compatible.
//
//   A (activation): [M, K] NVFP4 RowMajor packed + SfAtom UE4M3 scales
//   B (weight):     CutlassNvFP4Weight with [N, K] packed + SfAtom scales
//   D (output):     [M, N] FP16 RowMajor
//
// Returns false if cuBLASLt NVFP4 is not available or the kernel fails.
bool gemm_nvfp4_cublaslt(const void* a_data, const void* a_sf,
                          const CutlassNvFP4Weight& b,
                          void* d_fp16, int M, int N, int K,
                          cudaStream_t stream);

// Check if cuBLASLt NVFP4 GEMM is available at runtime.
bool cublaslt_nvfp4_available();

} // namespace imp
