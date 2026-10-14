#pragma once

#include "core/tensor.h"
#include "core/weight_handle.h"
#include "exec/gemm_context.h"

namespace imp {

// executor_gemm_dispatch.cu: the route for a weight with no cache entry (F16 source, dequantable
// quant, dropped source); also the tail of released_source_gemm_ (executor_gemm_released.cpp).
// decode_rows: M>1 decode/verify rows (Executor::cur_decode_rows_); never a prompt row (#2152).
void gemm_dispatch_uncached_fallback(const Tensor& input, const Tensor& weight, Tensor& output,
                                     const GemmContext& ctx, bool decode_rows = false);

// A GGUF-source GEMM no cache or IMMA took: the source, or for a Phase 4c weight (in
// ctx.wcache->q8_planes_only, source freed) the FP16 rebuild of its pinned planes.
void gguf_source_fallback(const WeightHandle& h, const Tensor& input, Tensor& output, const GemmContext& ctx,
                          bool decode_rows);

}  // namespace imp
