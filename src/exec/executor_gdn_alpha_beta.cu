// Batched-decode GDN alpha/beta projections: one launch for both.
//
// At 2 <= M <= 32 the two [n_heads, d_model] FP16 weights went through
// gemm() as two cuBLAS GEMMs, nvjet + splitKreduce each: 4 launches per GDN
// layer, none PDL-registered. gemm_f16_narrow_smallm runs both in one
// split-K tensor-core launch. Output layout is the two-call one (alpha at
// ssm_dt_buf_, beta at the 256-byte-aligned offset), so the scan is untouched.

#include "compute/gemm_f16_narrow_smallm.h"
#include "core/dispatch_policy.h"
#include "core/logging.h"
#include "exec/executor.h"

namespace imp {

namespace {

// FP16 device pointer of an FP16-resident weight: the dequant cache entry when
// the source was BF16/quantized, else the FP16 source itself (tier Undefined:
// the Qwen3.8 alpha/beta are dequantized NVFP4 -> F16 at load and sit in no
// cache, so gemm_via_handle_ ran them through the uncached fallback = gemm()).
// nullptr = not resident as FP16, or a different shape than expected.
const half* fp16_weight_ptr(const WeightCaches& wcache, const WeightHandle& h, int64_t N, int64_t K) {
    if ((h.primary_tier != StorageTier::FP16 && h.primary_tier != StorageTier::Undefined) ||
        h.shape[0] != N || h.shape[1] != K)
        return nullptr;
    auto it = wcache.fp16.find(h.source_data);
    if (it != wcache.fp16.end() && it->second.qtype == QType::F16 && it->second.data)
        return static_cast<const half*>(it->second.data);
    if (h.source_qtype == QType::F16 && h.source_data)
        return static_cast<const half*>(h.source_data);
    return nullptr;
}

}  // namespace

bool GraphExecutor::try_gdn_alpha_beta_narrow_(TensorID alpha_id, TensorID beta_id, const Tensor& input,
                                               Tensor& alpha_out, Tensor& beta_out, cudaStream_t stream) {
    if (!dispatch_policy().gdn.alpha_beta_smallm || gdn_ab_ws_ == nullptr)
        return false;
    if (alpha_id == kInvalidTensorID || beta_id == kInvalidTensorID)
        return false;
    if (cur_spec_verify_ || lora_ != nullptr || calib_)
        return false;
    const int M = static_cast<int>(input.shape[0]);
    const int64_t K = input.shape[1];
    if (M < 2 || M > 32 || input.qtype != QType::F16 || alpha_out.qtype != QType::F16 ||
        beta_out.qtype != QType::F16 || input.stride[0] != K)
        return false;
    const int64_t N = alpha_out.shape[1];
    if (beta_out.shape[1] != N)
        return false;
    const half* wa = fp16_weight_ptr(wcache_, registry_.handle(alpha_id), N, K);
    const half* wb = fp16_weight_ptr(wcache_, registry_.handle(beta_id), N, K);
    if (!wa || !wb) {
        if (!gdn_ab_narrow_logged_) {
            gdn_ab_narrow_logged_ = true;
            const auto& h = registry_.handle(alpha_id);
            IMP_LOG_INFO(
                "gdn alpha/beta narrow GEMM declined: alpha tier=%d source_qtype=%d shape=[%lld,%lld] "
                "expected [%lld,%lld] fp16_cache=%d (two-call route)",
                static_cast<int>(h.primary_tier), static_cast<int>(h.source_qtype), (long long)h.shape[0],
                (long long)h.shape[1], (long long)N, (long long)K, wcache_.fp16.count(h.source_data) ? 1 : 0);
        }
        return false;
    }
    const bool ok = gemm_f16_narrow_smallm(static_cast<const half*>(input.data), M, static_cast<int>(K), wa,
                                           static_cast<half*>(alpha_out.data), static_cast<int>(N), wb,
                                           static_cast<half*>(beta_out.data), static_cast<int>(N), gdn_ab_ws_,
                                           gdn_ab_ws_bytes_, stream);
    if (ok && !gdn_ab_narrow_logged_) {
        gdn_ab_narrow_logged_ = true;
        IMP_LOG_INFO("gdn alpha/beta narrow GEMM ACTIVE (M=%d, K=%lld, N=%lld x 2)", M, (long long)K,
                     (long long)N);
    }
    return ok;
}

}  // namespace imp
