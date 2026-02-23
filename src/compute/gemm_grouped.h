#pragma once

#include "core/tensor.h"
#include <vector>
#include <cuda_runtime.h>

namespace imp {

// cuBLASLt Grouped GEMM for MoE expert parallelism
void gemm_grouped(const std::vector<Tensor>& A,
                  const std::vector<Tensor>& B,
                  std::vector<Tensor>& C,
                  cudaStream_t stream = nullptr);

} // namespace imp
