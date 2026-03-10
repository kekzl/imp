#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

// MXFP4 weight data for CUTLASS sm_120 block-scaled GEMM.
//
// MXFP4 differs from NVFP4 in scale format:
//   NVFP4: UE4M3 scale per 16 elements (SFVecSize=16)
//   MXFP4: UE8M0 scale per 32 elements (SFVecSize=32)
//
// UE8M0 = pure exponent (no mantissa), representing powers of 2.
// This gives a wider dynamic range (2^-127 to 2^127) but coarser
// granularity than UE4M3 (which has 3 mantissa bits).
//
// For our use case: we convert existing NVFP4 weights (UE4M3 per 16)
// to MXFP4 format (UE8M0 per 32) to use the MXFP4 tensor core path.
// This may trade some precision for potentially better hardware scheduling.

struct CutlassMxFP4Weight {
    const void* data = nullptr;       // [N, K/2] packed E2M1 nibbles
    void* scale_factors = nullptr;    // SfAtom layout UE8M0 (owned)
    float tensor_scale = 1.0f;        // deferred global scale
    int64_t N = 0;
    int64_t K = 0;
    size_t sf_bytes = 0;
    bool owns_data = false;           // true if data was allocated (Hadamard path)
};

// Compute SfAtom buffer size for MXFP4 (UE8M0, SFVecSize=32).
size_t cutlass_mxfp4_sf_size(int rows, int K);

// Convert NVFP4 weights (UE4M3 per 16) to MXFP4 scale format (UE8M0 per 32).
// Merges pairs of UE4M3 micro-scales into one UE8M0 scale per 32 elements.
// Packed data pointer is borrowed (same E2M1 nibbles, same layout).
struct NvFP4QuantResult;  // forward
void convert_nvfp4_to_mxfp4_cutlass(const NvFP4QuantResult& src,
                                     CutlassMxFP4Weight& dst,
                                     cudaStream_t stream);

// Convert NVFP4 weights to MXFP4 with Hadamard rotation:
//   1. Dequant NVFP4 → FP16 (into scratch)
//   2. Apply block-diagonal Hadamard along K dimension
//   3. Requant FP16 → MXFP4 (E2M1 + UE8M0 scales, new allocation)
// This changes the packed data (not borrowed) — weights are re-quantized.
// scratch_fp16: pre-allocated [N, K] FP16 buffer on device.
void convert_nvfp4_to_mxfp4_hadamard(const NvFP4QuantResult& src,
                                      CutlassMxFP4Weight& dst,
                                      void* scratch_fp16,
                                      int hadamard_block_size,
                                      cudaStream_t stream);

void free_cutlass_mxfp4_weight(CutlassMxFP4Weight& w);

// Quantize FP16 activation [M,K] to MXFP4 in CUTLASS block-scaled format.
// Uses absmax per 32 elements → UE8M0 scale.
// Optionally applies Walsh-Hadamard rotation before quantization (hadamard_size > 0).
// dst_data: pre-allocated [M, K/2] packed FP4 bytes
// dst_sf:   pre-allocated SfAtom layout UE8M0 scales
void quantize_fp16_to_mxfp4_cutlass(const void* src_fp16, void* dst_data,
                                     void* dst_sf, int M, int K,
                                     cudaStream_t stream);

// Run CUTLASS sm_120 block-scaled MXFP4×MXFP4 GEMM: D = alpha * A × B^T
//   A (activation): [M, K] MXFP4 RowMajor + SFA UE8M0 scale factors
//   B (weight):     [N, K] MXFP4 RowMajor + SFB UE8M0 scale factors
//   D (output):     [M, N] FP16 RowMajor
bool gemm_mxfp4_cutlass_sm120(const void* a_data, const void* a_sf,
                               const CutlassMxFP4Weight& b,
                               void* d_fp16, int M, int N, int K,
                               void* workspace, size_t workspace_size,
                               cudaStream_t stream);

size_t gemm_mxfp4_cutlass_sm120_workspace(int M, int N, int K);

bool cutlass_sm120_mxfp4_available();

} // namespace imp
