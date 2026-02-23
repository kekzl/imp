#pragma once

#include "core/tensor.h"

namespace imp {

struct QuantConfig {
    DType quant_dtype = DType::FP16;   // quantized storage type
    DType compute_dtype = DType::FP16; // compute type
    int group_size = 128;
    bool has_zero_point = false;
    Tensor scales;      // per-channel or per-group scales
    Tensor zero_points; // optional

    // FP8 per-tensor scale (for FP8_E4M3 quantization)
    float per_tensor_scale = 1.0f;

    // NVFP4 two-level scaling
    Tensor micro_scales;    // [N, K/16] FP8 E4M3 micro-scales
    float tensor_scale = 1.0f;  // global FP32 tensor scale
};

} // namespace imp
