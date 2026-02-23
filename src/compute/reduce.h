#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

void reduce_sum(const Tensor& input, Tensor& output, int dim,
                cudaStream_t stream = nullptr);

void reduce_max(const Tensor& input, Tensor& output, int dim,
                cudaStream_t stream = nullptr);

} // namespace imp
