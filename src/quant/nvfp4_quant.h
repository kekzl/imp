#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// NVFP4 (FP4 E2M1) quantization with two-level scaling:
//   Level 1: FP8 E4M3 micro-scale per 16 values (micro-block)
//   Level 2: FP32 tensor-scale (global)
//
// Packed format: 2 FP4 values per byte (low nibble = even index, high = odd)
// Micro-scales: [N, K/16] as uint8_t (FP8 E4M3)
// Tensor-scale: single FP32

struct NvFP4QuantResult {
    void* packed_data = nullptr;     // [N, K/2] packed nibbles on device
    void* micro_scales = nullptr;    // [N, K/16] FP8 E4M3 micro-scales on device
    float tensor_scale = 1.0f;       // global tensor scale
    int64_t N = 0;
    int64_t K = 0;
};

// Quantize FP16 tensor to NVFP4 with two-level scaling.
// input: [N, K] FP16 on device. K must be multiple of 16.
void quantize_fp16_to_nvfp4(const Tensor& input, NvFP4QuantResult& result,
                             cudaStream_t stream = nullptr);

// Calibrate optimal tensor scale and micro-scales.
// Returns tensor_scale = global_absmax / 6.0 (FP4 E2M1 max = 6.0)
float calibrate_nvfp4_scales(const Tensor& input, cudaStream_t stream = nullptr);

// Dequantize NVFP4 back to FP16 (for verification).
void dequantize_nvfp4_to_fp16(const NvFP4QuantResult& quant,
                               void* output_fp16,
                               cudaStream_t stream = nullptr);

// Free NVFP4 result device memory.
void free_nvfp4_result(NvFP4QuantResult& result);

} // namespace imp
