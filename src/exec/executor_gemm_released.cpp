// released_source_gemm_: GEMMs on a weight whose F16 source Phase 4b freed (GDN in/gate/out with
// host-resident experts).

#include "compute/gemm.h"
#include "quant/mxfp8_sfatom.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/gemm_dispatch_fallback.h"
#include "quant/fp8_quant.h"

#include <cuda_fp16.h>

namespace imp {

// M <= 32 F16 reads the FP8 row-scale sidecar; everything else rebuilds the F16 weight in the scratch
// from the MXFP8 prefill copy (gate, out) or the pack's FP8 rows (in).
void GraphExecutor::released_source_gemm_(const WeightHandle& h, const Tensor& input, Tensor& output,
                                          const GemmContext& ctx) {
    const int M = static_cast<int>(input.shape[0]);
    if (M <= 32 && ctx.beta == 0.0f && output.qtype == QType::F16 && input.qtype == QType::F16) {
        const FP8CacheEntry* e = h.entries.fp8 != nullptr ? h.entries.fp8 : h.entries.released_fp8;
        const int64_t K = input.shape[1], N = output.shape[1];
        // Decode and spec verify (M <= 8): the M=1 GEMV per row, same bits as decode. Batch decode and
        // short prefill: one weight pass for all M rows.
        if (e && e->d_row_scales && M > 1 && !(ctx.mode.spec_verify && M <= 8) &&
            gemv_fp8_rowscale_f16(e->weight.data, e->d_row_scales, static_cast<const half*>(input.data),
                                  static_cast<half*>(output.data), static_cast<int>(N), static_cast<int>(K),
                                  M, ctx.stream))
            return;
        if (e && e->d_row_scales && (M <= 4 || (ctx.mode.spec_verify && M <= 8))) {
            for (int r = 0; r < M; ++r) {
                const int64_t xs[2] = {1, K}, ys[2] = {1, N};
                const Tensor x(static_cast<char*>(input.data) + (r * K * sizeof(half)), QType::F16, 2, xs,
                               true);
                Tensor y(static_cast<char*>(output.data) + (r * N * sizeof(half)), QType::F16, 2, ys, true);
                gemv_fp8_rowscale(e->weight, x, y, e->d_row_scales, ctx.stream);
            }
            return;
        }
    }
    const size_t bytes = static_cast<size_t>(h.shape[0]) * h.shape[1] * sizeof(half);
    IMP_CHECK(wcache_.released_f16_scratch && bytes <= wcache_.released_f16_scratch.bytes(),
              "released_source_gemm_: no rebuild scratch for a %lld x %lld weight",
              static_cast<long long>(h.shape[0]), static_cast<long long>(h.shape[1]));
    void* w16 = wcache_.released_f16_scratch.get();
    if (const CutlassMxFP8CacheEntry* mx = h.entries.mxfp8_prefill; mx != nullptr) {
        dequantize_mxfp8_cutlass_to_fp16(*mx, w16, ctx.stream);
    } else {
        const FP8CacheEntry* v = h.entries.released_fp8;
        IMP_CHECK(v != nullptr, "released_source_gemm_: no copy for a released %lld x %lld weight",
                  static_cast<long long>(h.shape[0]), static_cast<long long>(h.shape[1]));
        dequantize_fp8_rows_async(v->weight.data, v->d_row_scales, w16, static_cast<int>(h.shape[0]),
                                  static_cast<int>(h.shape[1]), ctx.stream);
    }
    Tensor weight(w16, QType::F16, 2, h.shape, true);
    weight.kind = h.kind;
    gemm_dispatch_uncached_fallback(input, weight, output, ctx);
}

}  // namespace imp
