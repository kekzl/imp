// released_source_gemm_: GEMMs on a weight whose F16 source Phase 4b freed (GDN in/gate/out with
// host-resident experts).

#include "compute/gemm.h"
#include "compute/gemm_cutlass_mxfp8_sm120.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/gemm_dispatch_fallback.h"
#include "quant/fp8_quant.h"

#include <cuda_fp16.h>

namespace imp {

// M=1 F16 keeps its FP8 row-scale sidecar GEMV (ssm_out decode); everything else rebuilds the F16
// weight in the scratch from the MXFP8 prefill copy (gate, out) or the pack's FP8 rows (in).
void GraphExecutor::released_source_gemm_(const WeightHandle& h, const Tensor& input, Tensor& output,
                                          const GemmContext& ctx) {
    const int M = static_cast<int>(input.shape[0]);
    // M <= 4 (verify chunk): one FP8 weight pass for all M rows (#2272), each row bit-equal to the
    // decode GEMV; 1 B/elem vs the rebuild's ~5 B/elem (FP8 read + F16 write + F16 read).
    if (M <= 4 && ctx.beta == 0.0f && output.qtype == QType::F16 && input.qtype == QType::F16) {
        const FP8CacheEntry* e = nullptr;
        if (const auto fp8 = wcache_.fp8.find(h.source_data); fp8 != wcache_.fp8.end())
            e = &fp8->second;
        else if (const auto v = wcache_.released_fp8_view.find(h.source_data);
                 v != wcache_.released_fp8_view.end())
            e = &v->second;
        if (e && e->d_row_scales && e->weight.shape[0] == output.shape[1] && e->weight.shape[1] == input.shape[1] &&
            gemv_fp8_rowscale_rows(e->weight.data, e->d_row_scales, static_cast<const half*>(input.data),
                                   static_cast<half*>(output.data), static_cast<int>(e->weight.shape[0]),
                                   static_cast<int>(e->weight.shape[1]), M, ctx.stream))
            return;
    }
    const size_t bytes = static_cast<size_t>(h.shape[0]) * h.shape[1] * sizeof(half);
    IMP_CHECK(wcache_.released_f16_scratch && bytes <= wcache_.released_f16_scratch_bytes,
              "released_source_gemm_: no rebuild scratch for a %lld x %lld weight",
              static_cast<long long>(h.shape[0]), static_cast<long long>(h.shape[1]));
    void* w16 = wcache_.released_f16_scratch;
    if (const auto mx = wcache_.cutlass_mxfp8_prefill.find(h.source_data); mx != wcache_.cutlass_mxfp8_prefill.end()) {
        dequantize_mxfp8_cutlass_to_fp16(mx->second, w16, ctx.stream);
    } else {
        const auto v = wcache_.released_fp8_view.find(h.source_data);
        IMP_CHECK(v != wcache_.released_fp8_view.end(), "released_source_gemm_: no copy for a released %lld x %lld weight",
                  static_cast<long long>(h.shape[0]), static_cast<long long>(h.shape[1]));
        dequantize_fp8_rows_async(v->second.weight.data, v->second.d_row_scales, w16, static_cast<int>(h.shape[0]),
                                  static_cast<int>(h.shape[1]), ctx.stream);
    }
    Tensor weight(w16, QType::F16, 2, h.shape, true);
    weight.kind = h.kind;
    gemm_dispatch_uncached_fallback(input, weight, output, ctx);
}

}  // namespace imp
