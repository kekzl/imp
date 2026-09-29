#pragma once

#include "core/tensor.h"
#include "exec/gemm_context.h"

namespace imp {

// executor_gemm_dispatch.cu: the route for a weight with no cache entry (F16 source, dequantable
// quant, dropped source); also the tail of released_source_gemm_ (executor_gemm_released.cpp).
void gemm_dispatch_uncached_fallback(const Tensor& input, const Tensor& weight, Tensor& output,
                                     const GemmContext& ctx);

}  // namespace imp
