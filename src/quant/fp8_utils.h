#pragma once

#include <cuda_runtime.h>

namespace imp {

// Cast FP16 tensor data to FP8 (E4M3 format).
void cast_fp16_to_fp8(const void* input, void* output, int n,
                      cudaStream_t stream);

// Cast FP8 (E4M3 format) tensor data to FP16.
void cast_fp8_to_fp16(const void* input, void* output, int n,
                      cudaStream_t stream);

} // namespace imp
