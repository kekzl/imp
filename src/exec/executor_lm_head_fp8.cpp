// LM-head dispatch for the per-row FP8 E4M3 head (gemm.nvfp4_lm_head=fp8, #2156). One kernel
// family for n==1 decode, batched decode, ragged prefill and --perplexity: row bits never depend on n.

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/lm_head_route.h"
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
    if (inkling_take_logits_(lg, stream))  // a row of an Inkling2 lookahead call: its logits exist
        return true;
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
    if (!qscratch_.fp8_act.get() || !qscratch_.d_act_row_scales.data() || n > qscratch_.act_row_scales_n ||
        static_cast<size_t>(no.numel()) > qscratch_.fp8_act_size ||
        !quantize_fp8_act_rows(no.data, qscratch_.fp8_act.get(), n, d, qscratch_.d_act_row_scales.data(),
                               stream))
        return false;
    int64_t wshape[2] = {lm_h.shape[0], lm_h.shape[1]};
    Tensor w8(lm_h.payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
    Tensor a8(qscratch_.fp8_act.get(), QType::FP8_E4M3, no.ndim, no.shape, true);
    gemm_cublaslt(a8, w8, lg, 1.0f, 0.0f, nullptr, lm_h.payload.fp8.d_scale, stream);
    scale_rows_f32(static_cast<float*>(lg.data), qscratch_.d_act_row_scales.data(), n,
                   static_cast<int>(lg.shape[lg.ndim - 1]), stream);
    return true;
}

LmHeadFacts GraphExecutor::lm_head_facts_() const {
    // No handle = no head data, so no cache entry either (Phase 4 registers every non-null tensor).
    const WeightHandle* lm_h = (model_->out_proj_id != kInvalidTensorID)
                                   ? &registry_.handle(model_->out_proj_id)
                                   : nullptr;
    LmHeadFacts f;
    f.tier = lm_h ? lm_h->primary_tier : StorageTier::Undefined;
    f.fp8_cache = lm_h != nullptr && lm_h->entries.fp8 != nullptr;
    f.nvfp4_cache = lm_h != nullptr && lm_h->entries.nvfp4 != nullptr;
    f.mxfp4_linear = f.tier == StorageTier::MXFP4 && lm_h->payload.mxfp4.linear_scales != nullptr;
    f.dp4a_ready = qscratch_.q8_1_buf.data() != nullptr && compute_dtype_ == QType::F16 &&
                   is_dp4a_qtype(model_->out_proj_.qtype);
    return f;
}

NvFP4QuantResult GraphExecutor::lm_head_nvfp4_() const {
    const WeightHandle& h = registry_.handle(model_->out_proj_id);
    if (h.entries.nvfp4 != nullptr)
        // NOLINTNEXTLINE(cppcoreguidelines-slicing): the base is the view (weight_cache_entries.h)
        return *h.entries.nvfp4;
    const auto& cfg = model_->config();
    NvFP4QuantResult r;
    r.packed_data = h.payload.nvfp4.data;
    r.micro_scales = h.payload.nvfp4.block_scales;
    r.tensor_scale = (h.payload.nvfp4.tensor_scale != nullptr) ? *h.payload.nvfp4.tensor_scale : 1.0f;
    r.N = cfg.vocab_size;
    r.K = cfg.d_model;
    return r;
}

}  // namespace imp
