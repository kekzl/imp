#pragma once

#include "core/process_diag.h"
#include "core/tensor.h"
#include "core/weight_handle.h"
#include "exec/gemm_context.h"

namespace imp {

// Q4_K IMMA K-split at M <= 32: decode/verify rows always; prompt rows (a whole prompt under 33 tokens,
// chunks keep >= 33 rows) unless runtime.deterministic, since the split changes their k order (#2152).
[[nodiscard]] inline bool q4k_split_ok(bool decode_rows) {
    return decode_rows || !process_diag_deterministic();
}

// gemm_dispatch_uncached.cpp: the GGUF blocks read in place (dp4a / IMMA by qtype and rows); false =
// no in-place kernel took the call, nothing launched.
[[nodiscard]] bool gemm_uncached_in_place(const Tensor& input, const Tensor& weight, Tensor& output,
                                          const GemmContext& ctx, bool decode_rows);

// gemm_dispatch_uncached.cpp: Q4_K / Q5_K prompt rows on the fused decode tile (gemm.q45k_tile, fp16
// accumulate, M > 32 or deterministic); false = declined, nothing launched.
[[nodiscard]] bool gemm_q45k_tile_rows(const Tensor& input, const Tensor& weight, Tensor& output,
                                       const GemmContext& ctx, bool decode_rows);

// gemm_dispatch_uncached.cpp: the route for a weight with no cache entry (F16 source, dequantable
// quant, dropped source); also the tail of released_source_gemm_ (executor_gemm_released.cpp).
// decode_rows: M>1 decode/verify rows (Executor::cur_decode_rows_); never a prompt row (#2152).
void gemm_dispatch_uncached_fallback(const Tensor& input, const Tensor& weight, Tensor& output,
                                     const GemmContext& ctx, bool decode_rows = false);

// A GGUF-source GEMM no cache or IMMA took: the source, or for a Phase 4c weight (in
// ctx.wcache->q8_planes_only, source freed) the FP16 rebuild of its pinned planes.
void gguf_source_fallback(const WeightHandle& h, const Tensor& input, Tensor& output, const GemmContext& ctx,
                          bool decode_rows);

}  // namespace imp
