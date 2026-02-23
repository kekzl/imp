#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

void softmax(const Tensor& input, Tensor& output,
             cudaStream_t stream = nullptr);

} // namespace imp
