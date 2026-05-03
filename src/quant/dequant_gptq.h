#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Dequantize GPTQ 4-bit packed weights to FP16.
// qweight: [K/8, N] packed INT32 (8 x 4-bit values per INT32)
// qzeros:  [num_groups/8, N] packed INT32 zero points
// scales:  [num_groups, N] FP16 per-group scales
// g_idx:   [K] INT32 group index (optional, nullptr for sequential groups)
// out:     [N, K] FP16 output
void dequant_gptq4(half* out, const int32_t* qweight, const int32_t* qzeros, const half* scales,
                   const int32_t* g_idx, int N, int K, int group_size, cudaStream_t stream = nullptr);

}  // namespace imp
