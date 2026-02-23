#pragma once

#include "core/tensor.h"
#include "quant/quant_types.h"
#include <cuda_runtime.h>

namespace imp {

// Quantize FP16 weights to FP8 E4M3 with per-tensor scale.
// input: [N, K] FP16 on device
// output: [N, K] FP8_E4M3 on device (caller pre-allocates)
// scale_out: scalar FP32 (written to device pointer)
void quantize_fp16_to_fp8_e4m3(const Tensor& input, Tensor& output,
                                float* d_scale_out,
                                cudaStream_t stream = nullptr);

// Calibrate optimal FP8 scale by finding absmax of input tensor.
// Returns the scale = absmax / 448.0 (E4M3 max representable).
float calibrate_fp8_scale(const Tensor& input, cudaStream_t stream = nullptr);

// Quantize with a known scale (for weight pre-processing).
void quantize_fp16_to_fp8_e4m3_scaled(const void* input_fp16, void* output_fp8,
                                       int n_elements, float scale,
                                       cudaStream_t stream = nullptr);

// Dequantize FP8 to FP16 with scale.
void dequantize_fp8_e4m3_to_fp16(const void* input_fp8, void* output_fp16,
                                  int n_elements, float scale,
                                  cudaStream_t stream = nullptr);

} // namespace imp
