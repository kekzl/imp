// Inkling2-Loop text forward (inkling_forward.h). Order of operations follows the transformers
// reference layer classes; fp16 at the linear layers, fp32 between them.

#include "exec/inkling_forward.h"

#include "compute/gemm.h"
#include "compute/inkling_ops.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "memory/vram_allocator.h"
#include "quant/nvfp4_gemm.h"

#include <algorithm>

namespace imp {

using namespace inkling;

InklingForward::~InklingForward() {
    for (void* p : owned_)
        va_ ? va_->free(p) : static_cast<void>(cudaFree(p));
}

void* InklingForward::alloc_(size_t bytes) {
    void* p = va_ ? va_->allocate(bytes, "inkling_state") : nullptr;
    if (!p && (va_ || cudaMalloc(&p, bytes) != cudaSuccess)) {
        IMP_LOG_ERROR("inkling: workspace allocation of %.1f MiB failed", bytes / 1048576.0);
        ok_ = false;
        return nullptr;
    }
    owned_.push_back(p);
    return p;
}

bool InklingForward::init(const InklingWeights& w, int max_tokens, int max_ctx, VRAMAllocator* va) {
    w_ = &w;
    va_ = va;
    const InklingConfig& c = w.cfg;
    max_tokens_ = max_tokens;
    max_ctx_ = max_ctx;
    ring_ = c.window + max_tokens;
    const size_t Tm = static_cast<size_t>(max_tokens), H = static_cast<size_t>(c.hidden);
    const size_t QD = static_cast<size_t>(c.n_heads) * c.head_dim, KD = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    const size_t slots = Tm * c.top_k;
    for (float** p : {&h_, &resid_, &bs_[0], &bs_[1], &bs_[2], &bs_[3], &pre_h_, &msrc_[0], &msrc_[1], &msrc_[2],
                      &msrc_[3], &mix_[0], &mix_[1], &mix_[2], &msg_[0], &msg_[1], &msg_[2], &msg_[3], &g_[0],
                      &g_[1], &g_[2], &a_, &res2_, &mm_, &xmf_, &mo_, &co_, &of_})
        *p = F_(Tm * H);
    qf_ = F_(Tm * QD);
    attn_ = F_(Tm * QD);
    rf_ = F_(Tm * c.n_heads * c.d_rel);
    kf_ = F_(Tm * KD);
    kc_ = F_(Tm * KD);
    wts_ = F_(slots);
    for (half** p : {&xq_, &xk_, &xv_, &o_h_, &gf_, &gg_, &xm_})
        *p = Hf_(Tm * H);
    q_h_ = Hf_(Tm * QD);
    gate_h_ = Hf_(Tm * QD);
    ao_h_ = Hf_(Tm * QD);
    r_h_ = Hf_(Tm * c.n_heads * c.d_rel);
    sink_h_ = Hf_(Tm * c.n_heads);
    k_h_ = Hf_(Tm * KD);
    dyn_ = Hf_(Tm * c.n_kv_heads * 4);
    cat_ = Hf_(Tm * 2 * H);
    gd_ = Hf_(Tm * c.gate_rank);
    gu_ = Hf_(slots * 2 * c.inter);
    act_ = Hf_(slots * c.inter);
    ey_ = Hf_(slots * H);
    logits_h_ = Hf_(static_cast<size_t>(c.vocab));
    idx_ = static_cast<int32_t*>(alloc_(slots * sizeof(int32_t)));
    conv_state_floats_ = 3 * (2 * KD + 2 * H);
    for (int a = 0; a < c.n_apps() && ok_; ++a) {
        // Application index -> weight layer.
        const int layer = a < c.n_layers ? a : c.loop_start + (a - c.n_layers) % (c.loop_end - c.loop_start);
        const size_t rows = c.global[static_cast<size_t>(layer)] ? static_cast<size_t>(max_ctx) : static_cast<size_t>(ring_);
        k_.push_back(Hf_(rows * KD));
        v_.push_back(Hf_(rows * KD));
        conv_state_.push_back(F_(conv_state_floats_));
    }
    if (ok_)
        reset(nullptr);
    return ok_;
}

void InklingForward::reset(cudaStream_t s) {
    pos_ = 0;
    for (float* p : conv_state_)
        cudaMemsetAsync(p, 0, conv_state_floats_ * sizeof(float), s);
}

void InklingForward::lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s) {
    const int64_t as[2] = {T, K}, bs[2] = {N, K}, cs[2] = {T, N};
    const Tensor A(const_cast<half*>(x), QType::F16, 2, as, true);
    const Tensor B(const_cast<half*>(w), QType::F16, 2, bs, true);
    Tensor C(y, QType::F16, 2, cs, true);
    gemm(A, B, C, 1.0f, 0.0f, s);
}

const float* InklingForward::gate_(const float* h, const float* msg, const InklingGate& g, float* out, int T,
                                   cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    gate_cat(h, msg, cat_, T, c.hidden, c.eps, s);
    lin_(cat_, T, 2 * c.hidden, g.down, c.gate_rank, gd_, s);
    lin_(gd_, T, c.gate_rank, g.forget, c.hidden, gf_, s);
    lin_(gd_, T, c.gate_rank, g.gain, c.hidden, gg_, s);
    gate_apply(h, msg, gf_, gg_, out, static_cast<size_t>(T) * c.hidden, s);
    return out;
}

// Key or value: conv over the projection x_h [T, KD], clip or norm, write into the application's cache.
void InklingForward::conv_(const half* x_h, const InklingConv& cv, const half* xin, int app, int site, bool is_key,
                           int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const int KD = c.n_kv_heads * c.head_dim;
    float* st = conv_state_[static_cast<size_t>(app)] + static_cast<size_t>(site) * 3 * KD;
    f16_to_f32(x_h, kf_, static_cast<size_t>(T) * KD, s);
    if (cv.static_w) {
        lin_(xin, T, c.hidden, cv.proj, c.n_kv_heads * 4, dyn_, s);
        conv_dynamic(kf_, cv.static_w, dyn_, st, kc_, T, KD, c.n_kv_heads, s);
    } else {
        conv_static(kf_, cv.w, st, kc_, T, KD, s);
    }
    conv_state_update(kf_, st, T, KD, s);
    const int layer_global = cv.static_w != nullptr;
    kv_store(kc_, (is_key ? k_ : v_)[static_cast<size_t>(app)], T, c.n_kv_heads, c.head_dim, is_key, c.qk_clip,
             c.eps, pos_, layer_global ? 0 : ring_, s);
}

void InklingForward::apply_(int layer, int pass, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const InklingLayer& ly = w_->layers[static_cast<size_t>(layer)];
    const int app = c.app_index(layer, pass);
    const bool global = c.global[static_cast<size_t>(layer)];
    const int H = c.hidden, QD = c.n_heads * c.head_dim, KD = c.n_kv_heads * c.head_dim;
    const size_t n = static_cast<size_t>(T) * H;
    add_f32(h_, nullptr, resid_, n, s);

    MixStates st;
    for (int j = 0; j < nb_; ++j)
        st.s[st.n++] = bs_[j];
    const float *hq = h_, *hk = h_, *hv = h_;
    if (layer > 0) {
        MixStates in = st;
        in.s[in.n++] = h_;
        MixOut out;
        out.o[0] = mix_[0], out.o[1] = mix_[1], out.o[2] = mix_[2];
        block_mix(in, ly.attn_query, global ? 3 : 1, out, T, H, s);
        hq = mix_[0];
        hk = global ? mix_[1] : mix_[0];
        hv = global ? mix_[2] : mix_[0];
    }
    if (pass > 0) {
        MixStates ms;
        for (int j = 0; j < n_msrc_; ++j)
            ms.s[ms.n++] = msrc_[j];
        MixOut out;
        for (int j = 0; j < 4; ++j)
            out.o[j] = msg_[j];
        block_mix(ms, w_->msg_query + static_cast<size_t>(layer - c.loop_start) * 4 * H, 4, out, T, H, s);
        const float *q0 = hq, *k0 = hk, *v0 = hv;
        hq = gate_(q0, msg_[0], ly.gates[0], g_[0], T, s);
        hk = gate_(k0, msg_[1], ly.gates[1], g_[1], T, s);
        hv = gate_(v0, msg_[2], ly.gates[2], g_[2], T, s);
    }
    rms_norm(hq, ly.norm_q, nullptr, xq_, T, H, c.eps, s);
    const half *xk = xq_, *xv = xq_;
    if (hk != hq || ly.norm_k != ly.norm_q) {
        rms_norm(hk, ly.norm_k, nullptr, xk_, T, H, c.eps, s);
        xk = xk_;
    }
    if (hv != hq || ly.norm_v != ly.norm_q) {
        rms_norm(hv, ly.norm_v, nullptr, xv_, T, H, c.eps, s);
        xv = xv_;
    }

    const float alpha = global ? c.log_alpha : 0.0f;
    lin_(xq_, T, H, ly.wq, QD, q_h_, s);
    scaled_l2_clip(q_h_, ly.q_scale, qf_, T, c.n_heads, c.head_dim, c.qk_clip, pos_, alpha, c.log_floor, s);
    lin_(xq_, T, H, ly.wr, c.n_heads * c.d_rel, r_h_, s);
    scaled_l2_clip(r_h_, ly.rel_scale, rf_, T, c.n_heads, c.d_rel, c.rel_clip, pos_, alpha, c.log_floor, s);
    lin_(xq_, T, H, ly.sink, c.n_heads, sink_h_, s);
    lin_(xk, T, H, ly.wk, KD, k_h_, s);
    conv_(k_h_, ly.k_conv, xk, app, 0, true, T, s);
    lin_(xv, T, H, ly.wv, KD, k_h_, s);
    conv_(k_h_, ly.v_conv, xv, app, 1, false, T, s);

    AttnArgs a;
    a.q = qf_, a.k = k_[static_cast<size_t>(app)], a.v = v_[static_cast<size_t>(app)];
    a.rel = rf_, a.proj = ly.rel_proj, a.sink = sink_h_, a.out = attn_;
    a.T = T, a.n_heads = c.n_heads, a.n_kv = c.n_kv_heads, a.hd = c.head_dim;
    a.d_rel = c.d_rel, a.rel_extent = global ? c.rel_extent : c.window;
    a.pos0 = pos_, a.window = global ? 0 : c.window, a.ring = global ? 0 : ring_;
    a.scale = 1.0f / sqrtf(static_cast<float>(c.head_dim));
    attention(a, s);
    lin_(xq_, T, H, ly.gate_attn, QD, gate_h_, s);
    gate_mul(attn_, gate_h_, ao_h_, static_cast<size_t>(T) * QD, s);
    lin_(ao_h_, T, QD, ly.wo, H, o_h_, s);
    f16_to_f32(o_h_, of_, n, s);
    float* cs = conv_state_[static_cast<size_t>(app)] + static_cast<size_t>(6) * KD;
    conv_static(of_, ly.attn_conv, cs, a_, T, H, s);
    conv_state_update(of_, cs, T, H, s);

    if (layer % c.block_layers == 0) {
        add_f32(resid_, nullptr, bs_[nb_], n, s);
        st.s[st.n++] = bs_[nb_++];
        add_f32(a_, nullptr, res2_, n, s);
    } else {
        add_f32(resid_, a_, res2_, n, s);
    }
    st.s[st.n++] = res2_;
    MixOut mo;
    mo.o[0] = mm_;
    block_mix(st, ly.mlp_query, 1, mo, T, H, s);
    const float* m = mm_;
    if (pass > 0)
        m = gate_(mm_, msg_[3], ly.gates[3], g_[0], T, s);
    rms_norm(m, ly.mlp_norm, nullptr, xm_, T, H, c.eps, s);
    f16_to_f32(xm_, xmf_, n, s);
    route_topk(xmf_, ly.router_w, ly.router_b, idx_, wts_, T, H, c.n_experts, c.top_k, c.route_scale, s);
    const int slots = T * c.top_k;
    if (ly.ex.fp8) {
        gemv_mxfp8_slots(ly.ex.w13_q, ly.ex.w13_s, idx_, xm_, gu_, slots, 2 * c.inter, H, c.top_k, s);
        expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
        gemv_mxfp8_slots(ly.ex.w2_q, ly.ex.w2_s, idx_, act_, ey_, slots, H, c.inter, 1, s);
    } else {
        gemv_nvfp4_moe_decode(ly.ex.w13, idx_, xm_, gu_, 2 * c.inter, H, H, slots, s, c.top_k);
        expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
        gemv_nvfp4_moe_decode(ly.ex.w2, idx_, act_, ey_, H, c.inter, c.inter, slots, s, 1);
    }
    expert_combine(ey_, wts_, mo_, T, c.top_k, H, s);
    float* ms2 = cs + static_cast<size_t>(3) * H;
    conv_static(mo_, ly.mlp_conv, ms2, co_, T, H, s);
    conv_state_update(mo_, ms2, T, H, s);
    add_f32(res2_, co_, h_, n, s);
    if (tap) {
        cudaStreamSynchronize(s);
        tap(app_seq_, h_, T);
    }
    ++app_seq_;
}

void InklingForward::layers_(int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const int H = c.hidden;
    const size_t n = static_cast<size_t>(T) * H;
    rms_norm(h_, w_->embed_norm, h_, nullptr, T, H, c.eps, s);
    nb_ = 0;
    app_seq_ = 0;
    if (tap) {
        cudaStreamSynchronize(s);
        tap(-1, h_, T);
    }
    for (int i = 0; i < c.loop_start; ++i)
        apply_(i, 0, T, s);
    add_f32(h_, nullptr, pre_h_, n, s);
    const int pre_nb = nb_;
    for (int pass = 0; pass < c.passes; ++pass) {
        if (pass > 0) {
            n_msrc_ = 0;
            for (int j = pre_nb + 1; j < nb_ && n_msrc_ < 3; ++j)
                add_f32(bs_[j], nullptr, msrc_[n_msrc_++], n, s);
            add_f32(h_, nullptr, msrc_[n_msrc_++], n, s);
            add_f32(pre_h_, nullptr, h_, n, s);
            nb_ = pre_nb;
        }
        for (int i = c.loop_start; i < c.loop_end; ++i)
            apply_(i, pass, T, s);
    }
    for (int i = c.loop_end; i < c.n_layers; ++i)
        apply_(i, 0, T, s);
    MixStates st;
    for (int j = 0; j < nb_; ++j)
        st.s[st.n++] = bs_[j];
    st.s[st.n++] = h_;
    MixOut out;
    out.o[0] = mix_[0];
    block_mix(st, w_->out_query, 1, out, T, H, s);
}

bool InklingForward::run(half* h, int T, int pos0, cudaStream_t s) {
    if (pos0 == 0)
        reset(s);
    if (pos0 != pos_ || T < 1 || T > max_tokens_ || pos_ + T > max_ctx_) {
        IMP_LOG_ERROR("inkling: %d rows at position %d do not continue the sequence at %d (%d per call, %d context)",
                      T, pos0, pos_, max_tokens_, max_ctx_);
        return false;
    }
    const size_t n = static_cast<size_t>(T) * w_->cfg.hidden;
    f16_to_f32(h, h_, n, s);
    layers_(T, s);
    f32_to_f16(mix_[0], h, n, s);
    pos_ += T;
    return ok_;
}

bool InklingForward::forward(const int32_t* d_ids, int T, float* d_logits, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    if (T < 1 || T > max_tokens_ || pos_ + T > max_ctx_ || !w_->embed) {
        IMP_LOG_ERROR("inkling: %d tokens at position %d exceed the workspace (%d per call, %d context)", T, pos_,
                      max_tokens_, max_ctx_);
        return false;
    }
    const int H = c.hidden;
    embed_rows(w_->embed, d_ids, h_, T, H, s);
    layers_(T, s);
    if (d_logits) {
        rms_norm(mix_[0], w_->norm, tap ? mix_[1] : nullptr, xq_, T, H, c.eps, s);
        if (tap) {
            cudaStreamSynchronize(s);
            tap(c.n_apps(), mix_[1], T);
        }
        lin_(xq_ + static_cast<size_t>(T - 1) * H, 1, H, w_->unembed, c.vocab, logits_h_, s);
        softcap_f32(logits_h_, d_logits, static_cast<size_t>(c.vocab), c.logit_cap, s);
    }
    pos_ += T;
    return ok_;
}

}  // namespace imp
