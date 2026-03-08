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

// Fully async calibrate+quantize: no host sync, all on device.
// Caller provides reusable temp buffers d_block_maxes[max_grid] and d_absmax[1].
// Scale is written to d_scale_out on device.
void calibrate_and_quantize_fp8_async(
    const void* input_fp16, void* output_fp8, int n_elements,
    float* d_block_maxes, int max_grid,
    float* d_absmax, float* d_scale_out,
    cudaStream_t stream = nullptr);

// Dequantize FP8 to FP16 with scale.
void dequantize_fp8_e4m3_to_fp16(const void* input_fp8, void* output_fp16,
                                  int n_elements, float scale,
                                  cudaStream_t stream = nullptr);

// Per-expert FP8 scale calibration for MoE.
// Computes scale = absmax / 448.0 for each expert's gathered activations.
// input_fp16: [total_tokens, K] FP16 on device
// offsets:    device array [n_experts+1], expert e has tokens [offsets[e], offsets[e+1])
// d_scales_out: device array [n_experts] receives per-expert scales
void calibrate_fp8_scales_per_expert(const void* input_fp16, int K,
                                      const int32_t* d_offsets, int n_experts,
                                      float* d_scales_out,
                                      cudaStream_t stream = nullptr);

// Per-expert FP8 quantization for MoE.
// Quantizes each expert's activation slice with its own scale.
// input_fp16: [total_tokens, K] FP16 on device
// output_fp8: [total_tokens, K] FP8_E4M3 on device (caller pre-allocates)
// d_offsets:  device array [n_experts+1]
// d_scales:   device array [n_experts] per-expert scales
void quantize_fp16_to_fp8_e4m3_per_expert(const void* input_fp16, void* output_fp8,
                                            int K, const int32_t* d_offsets,
                                            int n_experts, const float* d_scales,
                                            cudaStream_t stream = nullptr);

} // namespace imp
