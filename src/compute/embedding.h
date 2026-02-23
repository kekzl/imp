#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

enum class GGMLQuantType : uint32_t;  // forward declaration

void embedding_lookup(const Tensor& table, const int32_t* token_ids,
                      int n_tokens, Tensor& out,
                      cudaStream_t stream = nullptr);

// Overload for raw quantized embedding tables (Q8_0/Q6_K).
// Dequantizes only the needed rows on the fly.
void embedding_lookup(const Tensor& table, const int32_t* token_ids,
                      int n_tokens, Tensor& out,
                      GGMLQuantType qtype,
                      cudaStream_t stream);

} // namespace imp
