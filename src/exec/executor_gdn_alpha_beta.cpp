// GDN alpha/beta at M >= 2: one launch for both instead of two cuBLAS
// GEMMs (nvjet+splitKreduce each, 4 launches/layer, none PDL-registered):
// gemm_f16_narrow_smallm (M<=32, batched decode) or gemm_f16_narrow_prefill
// (M>32), both split-K tensor-core launches. Output layout matches the
// two-call form (alpha at ssm_dt_buf_, beta at the 256-byte-aligned
// offset), so the scan is untouched.

#include "compute/gemm_f16_narrow_prefill.h"
#include "compute/gemm_f16_narrow_smallm.h"
#include "compute/gemm.h"
#include "compute/gemv_dp4a_multi.h"
#include "compute/layernorm.h"
#include "core/dispatch_policy.h"
#include "core/logging.h"
#include "core/process_diag.h"
#include "compute/mmq_q8_imma.h"
#include "exec/executor.h"
#include "exec/gemm_context.h"
#include "exec/workspace_sizes.h"
#include "quant/nvfp4_gemm.h"
#include <stdexcept>

namespace imp {

namespace {

// FP16 device pointer of an FP16-resident weight: the dequant cache entry
// when the source was BF16/quantized, else the FP16 source itself (tier
// Undefined: Qwen3.8 alpha/beta are dequantized NVFP4->F16 at load and sit
// in no cache, so gemm_via_handle_ runs them through the uncached
// fallback). nullptr = not FP16-resident, or a different shape than expected.
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

// FP16 row-major input of >= 2 rows, FP16 alpha/beta outputs of one width.
bool narrow_operands_ok(const Tensor& input, const Tensor& alpha_out, const Tensor& beta_out) {
    return input.shape[0] >= 2 && input.qtype == QType::F16 && alpha_out.qtype == QType::F16 &&
           beta_out.qtype == QType::F16 && input.stride[0] == input.shape[1] &&
           beta_out.shape[1] == alpha_out.shape[1];
}

}  // namespace

bool GraphExecutor::try_gdn_alpha_beta_narrow_(TensorID alpha_id, TensorID beta_id, const Tensor& input,
                                               Tensor& alpha_out, Tensor& beta_out, cudaStream_t stream) {
    const int M = static_cast<int>(input.shape[0]);
    const bool prefill = M > 32;
    // runtime.deterministic: prefill at split 1 (no workspace), a row's bits independent of M.
    // cuBLAS picks its algorithm per M bucket (#2556).
    const bool det_prefill = prefill && process_diag_deterministic();
    if (prefill
            ? (!det_prefill && (!dispatch_policy().gdn.alpha_beta_prefill || gdn_ws_.ab_prefill == nullptr))
            : (!dispatch_policy().gdn.alpha_beta_smallm || gdn_ws_.ab == nullptr))
        return false;
    if (alpha_id == kInvalidTensorID || beta_id == kInvalidTensorID)
        return false;
    if (lora_ != nullptr || calib_)
        return false;
    if (!narrow_operands_ok(input, alpha_out, beta_out))
        return false;
    const int64_t K = input.shape[1];
    const int64_t N = alpha_out.shape[1];
    const half* wa = fp16_weight_ptr(wcache_, registry_.handle(alpha_id), N, K);
    const half* wb = fp16_weight_ptr(wcache_, registry_.handle(beta_id), N, K);
    if (!wa || !wb) {
        if (!gdn_ws_.ab_narrow_logged) {
            gdn_ws_.ab_narrow_logged = true;
            const auto& h = registry_.handle(alpha_id);
            IMP_LOG_INFO(
                "gdn alpha/beta narrow GEMM declined: alpha tier=%d source_qtype=%d shape=[%lld,%lld] "
                "expected [%lld,%lld] fp16_cache=%d (two-call route)",
                static_cast<int>(h.primary_tier), static_cast<int>(h.source_qtype), (long long)h.shape[0],
                (long long)h.shape[1], (long long)N, (long long)K,
                wcache_.fp16.contains(h.source_data) ? 1 : 0);
        }
        return false;
    }
    if (prefill) {
        const bool ok = gemm_f16_narrow_prefill(static_cast<const half*>(input.data), M, static_cast<int>(K),
                                                wa, static_cast<half*>(alpha_out.data), static_cast<int>(N),
                                                wb, static_cast<half*>(beta_out.data), static_cast<int>(N),
                                                det_prefill ? nullptr : gdn_ws_.ab_prefill,
                                                det_prefill ? 0 : gdn_ws_.ab_prefill_bytes, stream);
        if (ok && !gdn_ws_.ab_prefill_logged) {
            gdn_ws_.ab_prefill_logged = true;
            IMP_LOG_INFO("gdn alpha/beta prefill GEMM ACTIVE (M=%d, K=%lld, N=%lld x 2)", M, (long long)K,
                         (long long)N);
        }
        return ok;
    }
    const bool ok = gemm_f16_narrow_smallm(static_cast<const half*>(input.data), M, static_cast<int>(K), wa,
                                           static_cast<half*>(alpha_out.data), static_cast<int>(N), wb,
                                           static_cast<half*>(beta_out.data), static_cast<int>(N), gdn_ws_.ab,
                                           gdn_ws_.ab_bytes, stream);
    if (ok && !gdn_ws_.ab_narrow_logged) {
        gdn_ws_.ab_narrow_logged = true;
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
                                            Tensor& gate_out, int n_heads, cudaStream_t stream,
                                            const NvFP4NormFoldIn& fold) {
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
                                             exec_gdn_beta_offset(1, static_cast<int>(N), sizeof(half)));
    const bool ok = gemv_nvfp4_gdn_input_fused(w_in, w_gate, wa, wb, static_cast<int>(N),
                                               static_cast<const half*>(input.data), static_cast<half*>(proj.data),
                                               static_cast<half*>(gate_out.data), alpha_out, beta_out,
                                               static_cast<int>(K), stream, fold);
    if (ok && !gdn_ws_.m1_input_logged) {
        gdn_ws_.m1_input_logged = true;
        IMP_LOG_INFO("gdn M=1 fused input GEMV ACTIVE (K=%lld, rows %lld + %lld + 2 x %lld)", (long long)K,
                     (long long)w_in.N, (long long)w_gate.N, (long long)N);
    }
    return ok;
}

int GraphExecutor::gdn_unpacked_input_(const TransformerLayer& ly, const Tensor& h, Tensor& no, Tensor& proj,
                                       int inner, int n_heads, float eps, const NvFP4NormFoldIn& in_fold,
                                       const GemmContext& ctx, cudaStream_t stream) {
    const int n = static_cast<int>(no.shape[0]);
    // Batched-decode sibling pair: in-projection and gate both consume `no`,
    // and the gate output buffer (ssm_z_buf_) is untouched until
    // RMSNormGated after the scan, so computing the gate here is a pure
    // reorder; one smallm v2 launch for both when both route there.
    int64_t gate_shape_early[2] = {static_cast<int64_t>(n), static_cast<int64_t>(inner)};
    Tensor gate_early(ssm_z_buf_.data, compute_dtype_, 2, gate_shape_early, true);
    // M=1 (gdn.m1_fused): in_proj, gate, alpha and beta in one GEMV
    // launch; alpha/beta land in the 4-call layout of step 4 of run_gdn.
    if (try_gdn_input_fused_m1_(ly, no, proj, gate_early, n_heads, stream, in_fold))
        return kGdnM1Fused | kGdnGateDone;
    if (in_fold.ssq)
        rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
    // M=1 GGUF (gdn.m1_dp4a): the dp4a-routed projections of mixed qtypes in one launch.
    int d = try_gdn_input_dp4a_m1_(ly, no, proj, gate_early, n_heads, ctx, stream);
    bool in_done = (d & kGdnDp4aIn) != 0;
    if ((d & kGdnDp4aGate) != 0)
        d |= kGdnGateDone;
    if (!in_done && (d & kGdnGateDone) == 0 &&
        try_smallm_pair_dispatch_(ly.ssm_in_id, ly.gdn_gate_id, no, proj, gate_early, ctx)) {
        in_done = true;
        d |= kGdnGateDone;
    }
    if (!in_done)
        gemm_via_handle_(ly.ssm_in_id, no, proj, ctx);
    return d;
}

const void* GraphExecutor::dp4a_m1_source_(TensorID id, int64_t N, int64_t K, QType& qtype) const {
    if (id < 0 || id >= static_cast<TensorID>(registry_.size()))
        return nullptr;
    const WeightHandle& h = registry_.handle(id);
    const StorageTier decode = (h.decode_tier == StorageTier::Undefined) ? h.primary_tier : h.decode_tier;
    if ((decode != StorageTier::FP16 && decode != StorageTier::Undefined) || h.source_released ||
        h.source_data == nullptr || h.shape[0] != N || h.shape[1] != K ||
        !gemv_dp4a_multi_supports(h.source_qtype, static_cast<int>(K)) ||
        wcache_.nvfp4.contains(h.source_data) || wcache_.fp8.contains(h.source_data))
        return nullptr;
    qtype = h.source_qtype;
    return h.source_data;
}

int GraphExecutor::try_gdn_input_dp4a_m1_(const TransformerLayer& ly, const Tensor& input, Tensor& proj,
                                          Tensor& gate_out, int n_heads, const GemmContext& ctx,
                                          cudaStream_t stream) {
    if (!dispatch_policy().gdn.m1_dp4a || cur_spec_verify_ || calib_ || lora_ != nullptr || n_heads <= 0 ||
        ctx.gemm_no_dp4a_gemv || ctx.force_mmvq || input.shape[0] != 1 || input.qtype != QType::F16 ||
        proj.qtype != QType::F16 || gate_out.qtype != QType::F16 || ssm_dt_buf_.data == nullptr ||
        ly.gdn_alpha_beta_packed.data != nullptr || qscratch_.q8_1_buf == nullptr ||
        qscratch_.d8_buf == nullptr)
        return 0;
    const int64_t K = input.shape[1];
    if (K % 32 != 0 || K / 32 > qscratch_.q8_1_max_blocks)
        return 0;
    Dp4aGemvSeg segs[kDp4aMultiMaxSegs];
    int n = 0;
    int done = 0;
    auto add = [&](TensorID id, int64_t N, half* y, int bit) {
        QType qt = QType::NONE;
        if (const void* w = dp4a_m1_source_(id, N, K, qt)) {
            segs[n++] = {w, qt, y, static_cast<int>(N)};
            done |= bit;
        }
    };
    add(ly.ssm_in_id, proj.shape[1], static_cast<half*>(proj.data), kGdnDp4aIn);
    add(ly.gdn_gate_id, gate_out.shape[1], static_cast<half*>(gate_out.data), kGdnDp4aGate);
    // alpha and beta join together (the scan reads both from one layout) or not at all.
    QType qa = QType::NONE;
    QType qb = QType::NONE;
    if (dp4a_m1_source_(ly.gdn_alpha_id, n_heads, K, qa) != nullptr &&
        dp4a_m1_source_(ly.gdn_beta_id, n_heads, K, qb) != nullptr) {
        half* alpha_out = static_cast<half*>(ssm_dt_buf_.data);
        half* beta_out = reinterpret_cast<half*>(static_cast<char*>(ssm_dt_buf_.data) +
                                                 exec_gdn_beta_offset(1, n_heads, sizeof(half)));
        add(ly.gdn_alpha_id, n_heads, alpha_out, kGdnDp4aAb);
        add(ly.gdn_beta_id, n_heads, beta_out, kGdnDp4aAb);
    }
    if (n < 2)
        return 0;  // nothing to merge: the per-projection route is the same launch count
    auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
    quantize_fp16_to_q8_1(static_cast<const half*>(input.data), q8, qscratch_.d8_buf, static_cast<int>(K),
                          stream);
    if (!gemv_dp4a_multi(segs, n, q8, qscratch_.d8_buf, static_cast<int>(K), stream))
        throw std::runtime_error("try_gdn_input_dp4a_m1_: gemv_dp4a_multi refused vetted segments");
    static bool logged = false;  // activity proof, once
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("gdn M=1 dp4a multi input GEMV ACTIVE (K=%lld, %d weights, mask %d)", (long long)K, n,
                     done);
    }
    return done;
}

bool GraphExecutor::dp4a_multi_m1_(const TensorID* ids, Tensor* const* outs, int n, const Tensor& input,
                                   const GemmContext& ctx, cudaStream_t stream) {
    if (!dispatch_policy().gdn.m1_dp4a || cur_spec_verify_ || calib_ || lora_ != nullptr || n < 2 ||
        n > kDp4aMultiMaxSegs || ctx.gemm_no_dp4a_gemv || ctx.force_mmvq || input.shape[0] != 1 ||
        input.qtype != QType::F16 || qscratch_.q8_1_buf == nullptr || qscratch_.d8_buf == nullptr)
        return false;
    const int64_t K = input.shape[1];
    if (K % 32 != 0 || K / 32 > qscratch_.q8_1_max_blocks)
        return false;
    Dp4aGemvSeg segs[kDp4aMultiMaxSegs];
    for (int i = 0; i < n; ++i) {
        QType qt = QType::NONE;
        const void* w = outs[i]->qtype == QType::F16 ? dp4a_m1_source_(ids[i], outs[i]->shape[1], K, qt)
                                                     : nullptr;
        if (w == nullptr)
            return false;
        segs[i] = {w, qt, static_cast<half*>(outs[i]->data), static_cast<int>(outs[i]->shape[1])};
    }
    auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
    quantize_fp16_to_q8_1(static_cast<const half*>(input.data), q8, qscratch_.d8_buf, static_cast<int>(K),
                          stream);
    if (!gemv_dp4a_multi(segs, n, q8, qscratch_.d8_buf, static_cast<int>(K), stream))
        throw std::runtime_error("dp4a_multi_m1_: gemv_dp4a_multi refused vetted segments");
    static bool logged = false;  // activity proof, once
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("M=1 dp4a multi GEMV ACTIVE (K=%lld, %d weights, rows %d + %d)", (long long)K, n,
                     segs[0].rows, segs[1].rows);
    }
    return true;
}

bool GraphExecutor::gdn_out_residual_m1_ok_(const TransformerLayer& ly, int n, const Tensor& h) const {
    if (!dispatch_policy().gdn.m1_fused || n != 1 || h.qtype != QType::F16)
        return false;
    if (cur_spec_verify_ || calib_ || lora_ != nullptr)
        return false;
    NvFP4QuantResult w;
    return nvfp4_decode_weight_(ly.ssm_out_id, w) && w.N == h.shape[1];
}

void GraphExecutor::gdn_out_residual_m1_(int layer, const TransformerLayer& ly, const Tensor& y, Tensor& h,
                                         cudaStream_t stream) {
    NvFP4QuantResult w;
    if (!nvfp4_decode_weight_(ly.ssm_out_id, w) || w.K != y.shape[1] || w.N != h.shape[1] ||
        y.qtype != QType::F16)
        throw std::runtime_error("gdn_out_residual_m1_: out-projection shape/tier changed after the gate");
    // One thread owns each output row: it reads residual[row] before it
    // writes y[row], so h may be both.
    gemv_nvfp4_residual(w, static_cast<const half*>(y.data), static_cast<half*>(h.data),
                        static_cast<const half*>(h.data), static_cast<int>(w.N), static_cast<int>(w.K), stream,
                        norm_fold_arm_(layer, /*after_ffn=*/false, view_tokens(norm_out_, 1)));
    if (!gdn_ws_.m1_out_logged) {
        gdn_ws_.m1_out_logged = true;
        IMP_LOG_INFO("gdn M=1 out-projection with residual epilogue ACTIVE (N=%lld, K=%lld)", (long long)w.N,
                     (long long)w.K);
    }
}

bool GraphExecutor::try_attn_qkv_fused_m1_(const TransformerLayer& ly, const Tensor& input, Tensor& q_out,
                                           Tensor& k_out, Tensor& v_out, cudaStream_t stream,
                                           const NvFP4NormFoldIn& fold) {
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
                         static_cast<int>(wk.N), static_cast<int>(wv.N), static_cast<int>(K), stream, fold);
    if (!attn_m1_qkv_logged_) {
        attn_m1_qkv_logged_ = true;
        IMP_LOG_INFO("attention M=1 fused q|k|v NVFP4 GEMV ACTIVE (K=%lld, rows %lld + %lld + %lld)",
                     (long long)K, (long long)wq.N, (long long)wk.N, (long long)wv.N);
    }
    return true;
}

// Uncached Q8_0 gate/alpha/beta at rows 2..32 reuse the IMMA quantize of `no` that ssm_in (any IMMA
// kind) or the gate wrote. The tag is cleared first, so only this layer's quantize counts (#2416).
GemmContext GraphExecutor::gdn_q8_reuse_ctx_(int layer, const Tensor& no, const GemmContext& ctx) const {
    const int n = static_cast<int>(no.shape[0]);
    const auto& ly = model_->layer(layer);
    auto raw_q8 = [&](TensorID id) {
        return id >= 0 && id < static_cast<TensorID>(registry_.size()) &&
               registry_.handle(id).source_qtype == QType::Q8_0 && !prefill_routes_cutlass_nvfp4_(id, n);
    };
    if (n < 2 || n > 32 || !raw_q8(ly.gdn_gate_id) || !raw_q8(ly.gdn_alpha_id) || !raw_q8(ly.gdn_beta_id))
        return ctx;
    mmq_imma_act_forget();
    return ctx.with_act_quant_hint(no.data, n, static_cast<int>(no.shape[1]));
}

}  // namespace imp
