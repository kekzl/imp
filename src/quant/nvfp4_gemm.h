#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>

namespace imp {

// NVFP4 GEMV: y = A_nvfp4 @ x
// A is stored in NVFP4 format (packed_data + micro_scales + tensor_scale)
// x: [K] or [K,1] FP16 on device
// y: [M] or [M,1] FP16 on device
void gemv_nvfp4(const NvFP4QuantResult& A, const Tensor& x, Tensor& y,
                cudaStream_t stream = nullptr);

// NVFP4 GEMM via cuBLASLt (for M > 1, e.g., prefill).
// Falls back to dequant + standard GEMM if cuBLASLt NVFP4 is unavailable.
void gemm_nvfp4(const NvFP4QuantResult& A, const Tensor& B, Tensor& C,
                cudaStream_t stream = nullptr);

} // namespace imp
