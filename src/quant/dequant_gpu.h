#pragma once

#include "core/qtype.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

// qtype_row_bytes lives in core/qtype.h — included above.

// Returns true if the quant type supports on-GPU dequant to FP16.
bool dequant_gpu_supported(QType qtype);

// Dequantizes one expert's weight matrix from raw GGML block format to FP16 on GPU.
// src: raw quantized bytes (rows*qtype_row_bytes(qtype,cols)). dst: FP16 [rows,cols].
// cols must be divisible by the quant block size.
void dequant_gpu(const void* src, void* dst, QType qtype, int rows, int cols, cudaStream_t stream);

// Dequantize raw GGML quantized data to FP8 E4M3 on GPU.
// Same interface as dequant_gpu() but writes FP8 E4M3 (1 byte/element).
// Currently supports Q6_K only. Q6_K values are within FP8 E4M3 range (±448).
void dequant_gpu_fp8(const void* src, void* dst, QType qtype, int rows, int cols, cudaStream_t stream);

}  // namespace imp
