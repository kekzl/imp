#pragma once

#include "model/model.h"  // GGMLQuantType
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

// Compute raw byte count for one row of quantized GGML data with given column count.
// For Q6_K: (cols / 256) * 210
// For Q8_0: (cols / 32) * 34
// For Q4_0: (cols / 32) * 18
// For F16:  cols * 2
// For F32:  cols * 4
size_t ggml_quant_row_bytes(GGMLQuantType qtype, int64_t cols);

// Returns true if the quant type supports on-GPU dequant to FP16.
bool dequant_gpu_supported(GGMLQuantType qtype);

// Dequantize one expert's weight matrix from raw GGML block format to FP16 on GPU.
//
// src:  raw quantized bytes on GPU (one expert matrix: rows * ggml_quant_row_bytes(qtype, cols))
// dst:  output FP16 buffer on GPU (must hold rows * cols * sizeof(half))
// rows: number of rows in the weight matrix
// cols: number of columns (must be divisible by the quant block size)
void dequant_gpu(const void* src, void* dst, GGMLQuantType qtype,
                 int rows, int cols, cudaStream_t stream);

// Dequantize raw GGML quantized data to FP8 E4M3 on GPU.
// Same interface as dequant_gpu() but writes FP8 E4M3 (1 byte/element).
// Currently supports Q6_K only. Q6_K values are within FP8 E4M3 range (±448).
void dequant_gpu_fp8(const void* src, void* dst, GGMLQuantType qtype,
                     int rows, int cols, cudaStream_t stream);

} // namespace imp
