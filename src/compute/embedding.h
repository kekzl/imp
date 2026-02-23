#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

void embedding_lookup(const Tensor& table, const int32_t* token_ids,
                      int n_tokens, Tensor& out,
                      cudaStream_t stream = nullptr);

} // namespace imp
