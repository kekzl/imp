#pragma once

// Describes a quantised weight's layout: packed E2M1 nibbles plus the two scale
// representations prefill and decode read. Lives here rather than in
// compute/gemm_cutlass_mxfp4_sm120.h to avoid forcing src/quant to include src/compute (one
// of the two backward edges in an otherwise forward compute->quant relationship).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

struct CutlassMxFP4Weight {
    const void* data = nullptr;     // [N, K/2] packed E2M1 nibbles
    void* scale_factors = nullptr;  // SfAtom layout UE8M0 (for CUTLASS prefill)
    void* linear_scales = nullptr;  // [N, K/32] row-major UE8M0 (for GEMV decode)
    float tensor_scale = 1.0f;      // deferred global scale
    int64_t N = 0;
    int64_t K = 0;
    size_t sf_bytes = 0;
    bool owns_data = false;  // true if data was allocated (Hadamard path)
    int hadamard_bs = 0;     // Hadamard block size for online rotation (0=disabled)
};

}  // namespace imp
