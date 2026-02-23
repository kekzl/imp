#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Quantized GEMM: C = A @ dequant(B_quant, scales)
// A is in compute dtype, B_quant is INT4-packed, scales are per-group.
void quant_gemm_int4(const Tensor& A, const Tensor& B_quant,
                     const Tensor& scales, Tensor& C,
                     cudaStream_t stream);

} // namespace imp
