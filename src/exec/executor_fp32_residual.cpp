// FP32-residual sandwich-norm fusion (Gemma-4 decode, gemm.fp32_residual_fuse). Per layer the unfused
// chain is: attn_norm (+ the Q GEMM's quantize), post_attn accumulate, residual copy, MoE pre-norm,
// MoE-branch sanitize + post-norm, dense-branch sanitize + post-norm, add, post_ffn accumulate,
// layer_out_scale on FP16 and FP32. Here: post-attn launch, tail launch (carries the next layer's
// attn_norm + quantize), and the shared-MLP norm + quantize. Bits equal (tests/test_fp32_residual_norm.cu).

#include "compute/fp32_residual_norm.h"
#include "compute/gemm.h"
#include "compute/layernorm.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_debug.h"
#include "exec/executor_helpers.h"
#include "exec/executor_kernels.h"

#include <cmath>
#include <stdexcept>

namespace imp {

namespace {
bool f16_weight(const Tensor& t) { return t.data != nullptr && t.qtype == QType::F16; }
}  // namespace

bool GraphExecutor::fp32_resid_fuse_ok_(int n) const {
    // Decode rows only (n == 1 or batched decode / verify); prefill keeps the unfused chain.
    return n >= 1 && (n == 1 || cur_decode_rows_) && dispatch_policy().gemm.fp32_residual_fuse &&
           fp32_accum_buf_ != nullptr && model_->profile().fp32_residual_norms &&
           compute_dtype_ == QType::F16 && !calib_ && lora_ == nullptr && cvec_ == nullptr &&
           sae_rows_.empty() && !debug_forward_enabled() && !dispatch_policy().moe.force_fp16_sync &&
           !model_->config().parallel_block && !model_->profile().gated_residual;
}

void GraphExecutor::fp32_attn_norm_(int layer, Tensor& no, int n, float eps, cudaStream_t stream) {
    const auto& ly = model_->layer(layer);
    // The previous layer's tail already wrote no (and the Q GEMM's quantize) for this norm.
    const bool pre = prenorm_matches_(ly.attn_norm, no, n);
    prenorm_out_ = nullptr;
    if (pre)
        return;
    Tensor fp32_h = view_tokens(fp32_hidden_, n);
    const int d = model_->config().d_model;
    if (fp32_resid_fuse_ok_(n) && f16_weight(ly.attn_norm) &&
        fp32_residual_norm_fits(Fp32ResidMode::kNorm, d)) {
        uint8_t* xs = nullptr;
        float as = 1.0f;
        uint8_t* xp = smallm_producer_xq_(ly.wq_id, n, d, stream, &xs, &as);
        Fp32ResidArgs a;
        a.d_model = d;
        a.eps = eps;
        a.w_off = norm_w_off_;
        a.fp32 = static_cast<float*>(fp32_h.data);
        a.w_norm = static_cast<const half*>(ly.attn_norm.data);
        a.no_out = static_cast<half*>(no.data);
        a.xq_packed = xp;
        a.xq_scales = xs;
        a.act_scale = as;
        if (fp32_residual_norm(Fp32ResidMode::kNorm, a, n, stream)) {
            if (xp != nullptr)
                smallm_producer_tag_(no.data, n, d, as);
            else
                smallm_xq_forget_(no.data);
            return;
        }
    }
    rmsnorm_fp32_to_fp16(fp32_h, ly.attn_norm, no, eps, stream, norm_w_off_);
    smallm_xq_forget_(no.data);
}

bool GraphExecutor::fp32_post_attn_fused_(int layer, const Tensor& po, int n, cudaStream_t stream) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    // The MoE phase-2 norm this launch takes over: sandwich models read ffn_pre_norm_2 (Gemma-4).
    if (!fp32_resid_fuse_ok_(n) || !layer_has_moe(layer) || dispatch_policy().moe.skip ||
        !model_->profile().sandwich_norms || !f16_weight(ly.ffn_pre_norm_2) ||
        !f16_weight(ly.post_attn_norm) || po.qtype != QType::F16 ||
        !fp32_residual_norm_fits(Fp32ResidMode::kPostAttn, cfg.d_model))
        return false;
    Tensor h = view_tokens(hidden_, n);
    Tensor r = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    Fp32ResidArgs a;
    a.d_model = cfg.d_model;
    a.eps = cfg.rms_norm_eps;
    a.w_off = norm_w_off_;
    a.in_a = static_cast<const half*>(po.data);
    a.w_accum = static_cast<const half*>(ly.post_attn_norm.data);
    a.fp32 = static_cast<float*>(view_tokens(fp32_hidden_, n).data);
    a.h_out = static_cast<half*>(h.data);
    a.r_out = static_cast<half*>(r.data);
    a.w_norm = static_cast<const half*>(ly.ffn_pre_norm_2.data);
    a.no_out = static_cast<half*>(no.data);
    // Gemma-4 scaled router, rows >= 2 (n == 1 keeps its FP32 router input): moe_router_in_fp16_'s
    // norm reads the same FP32 row, so it shares this launch's inv_rms.
    const bool router = n >= 2 && model_->profile().scaled_router_norm && f16_weight(ly.ffn_gate_inp_scale) &&
                        moe_.scatter_out.data != nullptr && compute_dtype_ == QType::F16;
    if (router) {
        a.w_norm2 = static_cast<const half*>(ly.ffn_gate_inp_scale.data);
        a.w_off2 = 0.0f;
        a.no2_out = static_cast<half*>(moe_.scatter_out.data);
        a.no2_scale = 1.0f / std::sqrt(static_cast<float>(cfg.d_model));
    }
    if (!fp32_residual_norm(Fp32ResidMode::kPostAttn, a, n, stream))
        return false;
    router_prenorm_ = router ? moe_.scatter_out.data : nullptr;
    router_prenorm_n_ = n;
    smallm_xq_forget_(no.data);
    // moe_ffn_phase2_state_and_norm_ skips its residual copy and norm on this match.
    prenorm_out_ = no.data;
    prenorm_w_ = ly.ffn_pre_norm_2.data;
    prenorm_n_ = n;
    return true;
}

void GraphExecutor::fp32_post_attn_(int layer, const Tensor& po, Tensor& h, Tensor& fp32_h, int n, float eps,
                                    cudaStream_t stream) {
    if (fp32_post_attn_fused_(layer, po, n, stream))
        return;
    // FP32 residual += post_attn_norm(po), h = FP16 view of it.
    rmsnorm_fp32_accum_to_fp16(static_cast<const half*>(po.data),
                               static_cast<const half*>(model_->layer(layer).post_attn_norm.data),
                               static_cast<float*>(fp32_h.data), static_cast<half*>(h.data), n,
                               model_->config().d_model, eps, norm_w_off_, stream);
}

bool GraphExecutor::moe_take_prenorm_(const Tensor& norm_w, const MoeFfnContext& ctx) {
    if (!model_->profile().fp32_residual_norms || fp32_accum_buf_ == nullptr ||
        !prenorm_matches_(norm_w, ctx.no, ctx.n))
        return false;
    prenorm_out_ = nullptr;
    return true;
}

void GraphExecutor::moe_fp32_prenorm_(const Tensor& norm_w, MoeFfnContext& ctx, bool prenormed,
                                      cudaStream_t stream) {
    // Gemma-4 FP32 accum: read the FP32 residual directly, no FP16 round-trip.
    if (!prenormed) {
        Tensor fp32_h = view_tokens(fp32_hidden_, ctx.n);
        rmsnorm_fp32_to_fp16(fp32_h, norm_w, ctx.no, ctx.eps, stream, norm_w_off_);
        smallm_xq_forget_(ctx.no.data);
    }
    // Populate Q8_1 buffer from FP16 norm_out for dp4a decode path
    if (ctx.n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr) {
        quantize_fp16_to_q8_1(static_cast<const half*>(ctx.no.data),
                              static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, ctx.d, stream);
    }
}

bool GraphExecutor::fp32_tail_ok_(int layer, int n) const {
    const auto& prof = model_->profile();
    const auto& ly = model_->layer(layer);
    return fp32_resid_fuse_ok_(n) && prof.sandwich_norms && prof.sanitize_ffn_fp16 &&
           f16_weight(ly.ffn_post_norm_2) && f16_weight(ly.ffn_post_norm_1) && f16_weight(ly.post_ffn_norm) &&
           f16_weight(ly.ffn_pre_norm_2) && f16_weight(ly.ffn_norm) && ly.w_up_shared.data != nullptr &&
           ly.shared_expert_gate_inp.data == nullptr && !dispatch_policy().moe.no_shared_mlp &&
           !shexp_overlap_.pending && (gpli_.inputs.data == nullptr || ly.pli_gate.data == nullptr) &&
           fp32_residual_norm_fits(Fp32ResidMode::kTail, model_->config().d_model);
}

void GraphExecutor::fp32_tail_fused_(int layer, Tensor& h, const Tensor& sd, int n, cudaStream_t stream) {
    const auto& cfg = model_->config();
    const auto& prof = model_->profile();
    const auto& ly = model_->layer(layer);
    const int d = cfg.d_model;
    Fp32ResidArgs a;
    a.d_model = d;
    a.eps = cfg.rms_norm_eps;
    a.w_off = norm_w_off_;
    a.in_a = static_cast<const half*>(h.data);
    a.w_a = static_cast<const half*>(ly.ffn_post_norm_2.data);
    a.in_b = static_cast<const half*>(sd.data);
    a.w_b = static_cast<const half*>(ly.ffn_post_norm_1.data);
    a.w_accum = static_cast<const half*>(ly.post_ffn_norm.data);
    a.fp32 = static_cast<float*>(view_tokens(fp32_hidden_, n).data);
    a.h_out = static_cast<half*>(h.data);
    // The forward's layer_out_scale block scales only a device-resident scalar.
    if (ly.layer_out_scale.data != nullptr && ly.layer_out_scale.on_device)
        a.layer_scale = static_cast<const half*>(ly.layer_out_scale.data);
    // Next layer's attention norm: only where its Q/K/V input comes from fp32_attn_norm_ (rows >= 2:
    // n == 1 takes the fused NVFP4 QKV GEMV on the FP16 view).
    const int nl = layer + 1;
    const bool next = n >= 2 && nl < cfg.n_layers && offload_mgr_ == nullptr && layer_has_attention(nl) &&
                      !layer_has_gdn(nl) && !layer_has_ssm(nl) &&
                      prof.attn_variant != ModelProfile::AttnVariant::MLA &&
                      attn_post_norm_(model_->layer(nl)) && f16_weight(model_->layer(nl).attn_norm);
    Tensor no = view_tokens(norm_out_, n);
    uint8_t* xp = nullptr;
    uint8_t* xs = nullptr;
    if (next) {
        a.w_norm = static_cast<const half*>(model_->layer(nl).attn_norm.data);
        a.no_out = static_cast<half*>(no.data);
        float as = 1.0f;
        xp = smallm_producer_xq_(model_->layer(nl).wq_id, n, d, stream, &xs, &as);
        a.xq_packed = xp;
        a.xq_scales = xs;
        a.act_scale = as;
    }
    if (!fp32_residual_norm(Fp32ResidMode::kTail, a, n, stream))
        throw std::runtime_error("fp32_tail_fused_: launch refused inside fp32_tail_ok_'s envelope");
    fp32_tail_layer_ = layer;
    if (!next)
        return;
    if (xp != nullptr)
        smallm_producer_tag_(no.data, n, d, a.act_scale);
    else
        smallm_xq_forget_(no.data);
    prenorm_out_ = no.data;
    prenorm_w_ = model_->layer(nl).attn_norm.data;
    prenorm_n_ = n;
}

}  // namespace imp
