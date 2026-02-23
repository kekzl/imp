#pragma once

#include <cuda_runtime.h>

namespace imp {

// Dequantize INT4 data to FP16 with per-group scales.
void dequant_int4_fp16(const void* input, void* output,
                       const void* scales, int n, int group_size,
                       cudaStream_t stream);

} // namespace imp
