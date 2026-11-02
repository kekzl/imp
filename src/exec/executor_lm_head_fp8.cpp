// LM-head dispatch for the per-row FP8 E4M3 head (gemm.nvfp4_lm_head=fp8, #2156). One kernel
// family for n==1 decode, batched decode, ragged prefill and --perplexity: row bits never depend on n.

#include "exec/executor.h"
#include "compute/gemm.h"
#include "compute/layernorm.h"
#include "core/logging.h"
#include "core/qtype.h"
#include "quant/fp8_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

bool GraphExecutor::lm_head_fp8_(const Tensor& h, Tensor& lg, cudaStream_t stream) {
    const FP8CacheEntry& e = wcache_.lm_head_fp8;
    if (e.weight.data == nullptr)
        return false;
    // Phase 4b may have freed the source head: no fallback exists past this point.
    IMP_CHECK(h.qtype == QType::F16 && lg.qtype == QType::F32,
              "FP8 LM head: hidden %s / logits %s, need F16 / F32", qtype_name(h.qtype),
              qtype_name(lg.qtype));
    const auto& cfg = model_->config();
    const int n = static_cast<int>(h.shape[0]);
    Tensor no = view_tokens(norm_out_, n);
    rmsnorm(h, model_->output_norm(), no, cfg.rms_norm_eps, stream, norm_w_off_);
    return gemv_fp8_rowscale_fp32(e.weight.data, e.d_row_scales, static_cast<const half*>(no.data),
                                  static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, n, stream);
}

bool GraphExecutor::lm_head_fp8_view(const void*& weight, const float*& row_scales) const {
    if (wcache_.lm_head_fp8.weight.data == nullptr)
        return false;
    weight = wcache_.lm_head_fp8.weight.data;
    row_scales = wcache_.lm_head_fp8.d_row_scales;
    return true;
}

// FP8-tier head (registry tier, not the #2156 cache): per-token activation scales, FP32 logits scaled per row
// after the GEMM (beta 0), so a row never depends on the other rows (was a batch-wide absmax).
bool GraphExecutor::lm_head_fp8_tier_(const WeightHandle& lm_h, const Tensor& no, Tensor& lg,
                                      cudaStream_t stream) {
    const int n = static_cast<int>(no.shape[0]);
    const int d = static_cast<int>(no.shape[no.ndim - 1]);
    if (!qscratch_.fp8_act || !qscratch_.d_act_row_scales || n > qscratch_.act_row_scales_n ||
        static_cast<size_t>(no.numel()) > qscratch_.fp8_act_size ||
        !quantize_fp8_act_rows(no.data, qscratch_.fp8_act, n, d, qscratch_.d_act_row_scales, stream))
        return false;
    int64_t wshape[2] = {lm_h.shape[0], lm_h.shape[1]};
    Tensor w8(lm_h.payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
    Tensor a8(qscratch_.fp8_act, QType::FP8_E4M3, no.ndim, no.shape, true);
    gemm_cublaslt(a8, w8, lg, 1.0f, 0.0f, nullptr, lm_h.payload.fp8.d_scale, stream);
    scale_rows_f32(static_cast<float*>(lg.data), qscratch_.d_act_row_scales, n,
                   static_cast<int>(lg.shape[lg.ndim - 1]), stream);
    return true;
}

}  // namespace imp
