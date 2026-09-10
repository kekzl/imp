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
#include "quant/nvfp4_gemm.h"
#include <stdexcept>

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

// Mirrors the M=1, beta=0 NVFP4 cases of gemm_via_handle_ (executor_gemm_dispatch.cu).
bool GraphExecutor::nvfp4_decode_weight_(TensorID id, NvFP4QuantResult& out) const {
    if (id == kInvalidTensorID)
        return false;
    const WeightHandle& h = registry_.handle(id);
    const StorageTier decode = (h.decode_tier == StorageTier::Undefined) ? h.primary_tier : h.decode_tier;
    if (decode == StorageTier::NVFP4) {
        auto it = wcache_.nvfp4.find(h.source_data);
        if (it == wcache_.nvfp4.end())
            return false;
        out = it->second;
        out.owned = false;
        return true;
    }
    if (decode == StorageTier::CUTLASS_NVFP4 && h.source_data && h.source_scales) {
        out.packed_data = const_cast<void*>(h.source_data);
        out.micro_scales = const_cast<void*>(h.source_scales);
        out.tensor_scale = h.source_tensor_scale;
        out.N = h.shape[0];
        out.K = h.shape[1] * 2;
        out.owned = false;
        return true;
    }
    return false;
}

bool GraphExecutor::try_gdn_input_fused_m1_(const TransformerLayer& ly, const Tensor& input, Tensor& proj,
                                            Tensor& gate_out, int n_heads, cudaStream_t stream) {
    if (!dispatch_policy().gdn.m1_fused || cur_spec_verify_ || calib_ || n_heads <= 0)
        return false;
    if (input.shape[0] != 1 || input.qtype != QType::F16 || proj.qtype != QType::F16 ||
        gate_out.qtype != QType::F16 || compute_dtype_ != QType::F16 || ssm_dt_buf_.data == nullptr)
        return false;
    // The packed alpha|beta weight (F16 checkpoints) owns that layout.
    if (ly.gdn_alpha_id == kInvalidTensorID || ly.gdn_beta_id == kInvalidTensorID ||
        ly.gdn_alpha_beta_packed.data != nullptr)
        return false;
    const int64_t K = input.shape[1];
    NvFP4QuantResult w_in, w_gate;
    if (!nvfp4_decode_weight_(ly.ssm_in_id, w_in) || !nvfp4_decode_weight_(ly.gdn_gate_id, w_gate))
        return false;
    if (w_in.K != K || w_gate.K != K || w_in.N != proj.shape[1] || w_gate.N != gate_out.shape[1])
        return false;
    const int64_t N = n_heads;
    const half* wa = fp16_weight_ptr(wcache_, registry_.handle(ly.gdn_alpha_id), N, K);
    const half* wb = fp16_weight_ptr(wcache_, registry_.handle(ly.gdn_beta_id), N, K);
    if (!wa || !wb)
        return false;
    half* alpha_out = static_cast<half*>(ssm_dt_buf_.data);
    half* beta_out = reinterpret_cast<half*>(static_cast<char*>(ssm_dt_buf_.data) +
                                             ((static_cast<size_t>(N) * sizeof(half) + 255) & ~size_t(255)));
    const bool ok = gemv_nvfp4_gdn_input_fused(w_in, w_gate, wa, wb, static_cast<int>(N),
                                               static_cast<const half*>(input.data), static_cast<half*>(proj.data),
                                               static_cast<half*>(gate_out.data), alpha_out, beta_out,
                                               static_cast<int>(K), stream);
    if (ok && !gdn_m1_input_logged_) {
        gdn_m1_input_logged_ = true;
        IMP_LOG_INFO("gdn M=1 fused input GEMV ACTIVE (K=%lld, rows %lld + %lld + 2 x %lld)", (long long)K,
                     (long long)w_in.N, (long long)w_gate.N, (long long)N);
    }
    return ok;
}

bool GraphExecutor::gdn_out_residual_m1_ok_(const TransformerLayer& ly, int n, const Tensor& h) const {
    if (!dispatch_policy().gdn.m1_fused || n != 1 || h.qtype != QType::F16)
        return false;
    if (cur_spec_verify_ || calib_ || lora_ != nullptr)
        return false;
    NvFP4QuantResult w;
    return nvfp4_decode_weight_(ly.ssm_out_id, w) && w.N == h.shape[1];
}

void GraphExecutor::gdn_out_residual_m1_(const TransformerLayer& ly, const Tensor& y, Tensor& h,
                                         cudaStream_t stream) {
    NvFP4QuantResult w;
    if (!nvfp4_decode_weight_(ly.ssm_out_id, w) || w.K != y.shape[1] || w.N != h.shape[1] ||
        y.qtype != QType::F16)
        throw std::runtime_error("gdn_out_residual_m1_: out-projection shape/tier changed after the gate");
    // One thread owns each output row: it reads residual[row] before it
    // writes y[row], so h may be both.
    gemv_nvfp4_residual(w, static_cast<const half*>(y.data), static_cast<half*>(h.data),
                        static_cast<const half*>(h.data), static_cast<int>(w.N), static_cast<int>(w.K), stream);
    if (!gdn_m1_out_logged_) {
        gdn_m1_out_logged_ = true;
        IMP_LOG_INFO("gdn M=1 out-projection with residual epilogue ACTIVE (N=%lld, K=%lld)", (long long)w.N,
                     (long long)w.K);
    }
}

bool GraphExecutor::try_attn_qkv_fused_m1_(const TransformerLayer& ly, const Tensor& input, Tensor& q_out,
                                           Tensor& k_out, Tensor& v_out, cudaStream_t stream) {
    if (!dispatch_policy().gdn.m1_fused || cur_spec_verify_ || calib_)
        return false;
    if (input.shape[0] != 1 || input.qtype != QType::F16 || q_out.qtype != QType::F16 ||
        k_out.qtype != QType::F16 || v_out.qtype != QType::F16)
        return false;
    const int64_t K = input.shape[1];
    NvFP4QuantResult wq, wk, wv;
    if (!nvfp4_decode_weight_(ly.wq_id, wq) || !nvfp4_decode_weight_(ly.wk_id, wk) ||
        !nvfp4_decode_weight_(ly.wv_id, wv))
        return false;
    if (wq.K != K || wk.K != K || wv.K != K || wq.N != q_out.shape[1] || wk.N != k_out.shape[1] ||
        wv.N != v_out.shape[1])
        return false;
    gemv_nvfp4_qkv_fused(wq, wk, wv, static_cast<const half*>(input.data), static_cast<half*>(q_out.data),
                         static_cast<half*>(k_out.data), static_cast<half*>(v_out.data), static_cast<int>(wq.N),
                         static_cast<int>(wk.N), static_cast<int>(wv.N), static_cast<int>(K), stream);
    if (!attn_m1_qkv_logged_) {
        attn_m1_qkv_logged_ = true;
        IMP_LOG_INFO("attention M=1 fused q|k|v NVFP4 GEMV ACTIVE (K=%lld, rows %lld + %lld + %lld)",
                     (long long)K, (long long)wq.N, (long long)wk.N, (long long)wv.N);
    }
    return true;
}

}  // namespace imp
