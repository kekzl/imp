#pragma once

#include <cuda_runtime.h>

namespace imp {

// Dequantize INT8 data to FP16 with per-channel scales.
void dequant_int8_fp16(const void* input, void* output,
                       const void* scales, int n,
                       cudaStream_t stream);

} // namespace imp
