// Inkling2-Loop text forward (inkling_forward.h). Order of operations follows the transformers
// reference layer classes; fp16 at the linear layers, fp32 between them.

#include "exec/inkling_forward.h"

#include "compute/gemm.h"
#include "compute/inkling_ops.h"
#include "core/logging.h"
#include "core/cuda_raii.h"
#include "core/tensor.h"
#include "memory/vram_allocator.h"
#include "quant/nvfp4_gemm.h"

#include <algorithm>

namespace imp {

using namespace inkling;

InklingForward::~InklingForward() {
    for (void* p : owned_)
        va_->free(p);
}

void* InklingForward::alloc_(size_t bytes) {
    void* p = va_->allocate(bytes, "inkling_state");
    if (!p) {
        IMP_LOG_ERROR("inkling: workspace allocation of %.1f MiB failed", bytes / 1048576.0);
        ok_ = false;
        return nullptr;
    }
    owned_.push_back(p);
    return p;
}

bool InklingForward::init(const InklingWeights& w, VRAMAllocator& va, int max_tokens, int max_ctx) {
    w_ = &w;
    va_ = &va;
    const InklingConfig& c = w.cfg;
    max_tokens_ = max_tokens;
    max_ctx_ = max_ctx;
    ring_ = c.window + max_tokens;
    const size_t Tm = static_cast<size_t>(max_tokens), H = static_cast<size_t>(c.hidden);
    const size_t QD = static_cast<size_t>(c.n_heads) * c.head_dim,
                 KD = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    const size_t slots = Tm * c.top_k;
    for (float** p : {&h_,       &bs_[0],   &bs_[1],  &bs_[2],  &bs_[3],  &pre_h_,  &msrc_[0], &msrc_[1],
                      &msrc_[2], &msrc_[3], &mix_[0], &mix_[1], &mix_[2], &msg_[0], &msg_[1],  &msg_[2],
                      &msg_[3],  &g_[0],    &g_[1],   &g_[2],   &res2_,   &mm_})
        *p = F_(Tm * H);
    wts_ = F_(slots);
    rlog_ = F_(Tm * c.n_experts);
    gd_ = F_(Tm * 3 * (2 * static_cast<size_t>(c.gate_rank) + 2));
    mixp_ = F_(mix_partial_floats(max_tokens, c.hidden));
    part_ = F_(attention_part_floats(c.n_heads, c.head_dim, c.window, max_ctx));
    for (half** p : {&xq_, &xk_, &xv_, &o_h_, &xm_})
        *p = Hf_(Tm * H);
    const size_t taps = static_cast<size_t>(c.n_kv_heads) * 4;
    proj_h_ = Hf_(Tm * (2 * QD + static_cast<size_t>(c.n_heads) * c.d_rel + c.n_heads + 2 * (KD + taps)));
    pk_ = Hf_(Tm * (KD + taps));
    pv_ = Hf_(Tm * (KD + taps));
    ao_h_ = Hf_(Tm * QD);
    gu_ = Hf_(slots * 2 * c.inter);
    act_ = Hf_(slots * c.inter);
    ey_ = Hf_(slots * H);
    logits_h_ = Hf_(static_cast<size_t>(c.vocab));
    idx_ = static_cast<int32_t*>(alloc_(slots * sizeof(int32_t)));
    d_pos_ = static_cast<int*>(alloc_(sizeof(int)));
    conv_state_floats_ = 3 * (2 * KD + 2 * H);
    for (int a = 0; a < c.n_apps() && ok_; ++a) {
        // Application index -> weight layer.
        const int layer = a < c.n_layers ? a : c.loop_start + (a - c.n_layers) % (c.loop_end - c.loop_start);
        const size_t rows = c.global[static_cast<size_t>(layer)] ? static_cast<size_t>(max_ctx)
                                                                 : static_cast<size_t>(ring_);
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
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(p, 0, conv_state_floats_ * sizeof(float), s));
}

void InklingForward::lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s) {
    const int64_t as[2] = {T, K}, bs[2] = {N, K}, cs[2] = {T, N};
    const Tensor A(const_cast<half*>(x), QType::F16, 2, as, true);
    const Tensor B(const_cast<half*>(w), QType::F16, 2, bs, true);
    Tensor C(y, QType::F16, 2, cs, true);
    gemm(A, B, C, 1.0f, 0.0f, s);
}

void InklingForward::apply_(int layer, int pass, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const InklingLayer& ly = w_->layers[static_cast<size_t>(layer)];
    const size_t app = static_cast<size_t>(c.app_index(layer, pass));
    const bool global = c.global[static_cast<size_t>(layer)];
    const int H = c.hidden, QD = c.n_heads * c.head_dim, KD = c.n_kv_heads * c.head_dim;
    const int RD = c.n_heads * c.d_rel, TD = global ? c.n_kv_heads * 4 : 0;

    MixStates st;
    for (int j = 0; j < nb_; ++j)
        st.s[st.n++] = bs_[j];
    // The q, k and v inputs: one normed state on a sliding layer's first pass, three otherwise.
    const bool split = global || pass > 0;
    if (pass == 0) {
        MixArgs m;
        m.st = st;
        m.st.s[m.st.n++] = h_;
        m.mix = layer > 0;
        m.query = ly.attn_query;
        m.nq = global ? 3 : 1;
        m.eps = c.eps, m.partial = mixp_;
        m.norm_w[0] = ly.norm_q, m.norm_w[1] = ly.norm_k, m.norm_w[2] = ly.norm_v;
        m.out_h[0] = xq_, m.out_h[1] = xk_, m.out_h[2] = xv_;
        mix(m, T, H, s);
    } else {
        MixArgs in;
        in.st = st;
        in.st.s[in.st.n++] = h_;
        in.query = ly.attn_query, in.nq = global ? 3 : 1, in.partial = mixp_;
        in.out_f[0] = mix_[0], in.out_f[1] = mix_[1], in.out_f[2] = mix_[2];
        mix(in, T, H, s);
        MixArgs ms;
        for (int j = 0; j < n_msrc_; ++j)
            ms.st.s[ms.st.n++] = msrc_[j];
        ms.query = w_->msg_query + static_cast<size_t>(layer - c.loop_start) * 4 * H, ms.nq = 4,
        ms.partial = mixp_;
        for (int j = 0; j < 4; ++j)
            ms.out_f[j] = msg_[j];
        mix(ms, T, H, s);
        GateNormArgs gn;
        gn.n = 3, gn.rank = c.gate_rank, gn.eps = c.eps;
        gn.h[0] = mix_[0], gn.h[1] = global ? mix_[1] : mix_[0], gn.h[2] = global ? mix_[2] : mix_[0];
        gn.norm_w[0] = ly.norm_q, gn.norm_w[1] = ly.norm_k, gn.norm_w[2] = ly.norm_v;
        gn.out_h[0] = xq_, gn.out_h[1] = xk_, gn.out_h[2] = xv_;
        for (int j = 0; j < 3; ++j) {
            gn.msg[j] = msg_[j];
            gn.down[j] = ly.gates[j].down, gn.forget[j] = ly.gates[j].forget, gn.gain[j] = ly.gates[j].gain;
        }
        GateOut go;
        go.o[0] = g_[0], go.o[1] = g_[1], go.o[2] = g_[2];
        gate_norm(gn, go, gd_, T, H, s);
    }

    // Projections: rows [wq; wr; sink; gate; wk; k taps; wv; v taps] of one block.
    const int q_rows = 2 * QD + RD + c.n_heads, kv_rows = KD + TD;
    const half *kx = pk_, *vx = pv_;
    int q_stride = q_rows, kv_stride = kv_rows;
    if (split) {
        lin_(xq_, T, H, ly.wq, q_rows, proj_h_, s);
        lin_(xk_, T, H, ly.wk, kv_rows, pk_, s);
        lin_(xv_, T, H, ly.wv, kv_rows, pv_, s);
    } else {
        lin_(xq_, T, H, ly.wq, ly.proj_rows, proj_h_, s);
        q_stride = kv_stride = ly.proj_rows;
        kx = proj_h_ + q_rows;
        vx = kx + kv_rows;
    }
    float* cs = conv_state_[app];
    KvSiteArgs ka;
    ka.x = kx, ka.w = ly.k_conv.w, ka.static_w = ly.k_conv.static_w, ka.dyn = kx + KD;
    ka.state = cs, ka.cache = k_[app], ka.d_pos = d_pos_;
    ka.T = T, ka.stride = kv_stride, ka.n_heads = c.n_kv_heads, ka.hd = c.head_dim,
    ka.ring = global ? 0 : ring_;
    ka.is_key = true, ka.thr = c.qk_clip, ka.eps = c.eps;
    kv_site(ka, s);
    KvSiteArgs va = ka;
    va.x = vx, va.w = ly.v_conv.w, va.static_w = ly.v_conv.static_w, va.dyn = vx + KD;
    va.state = cs + static_cast<size_t>(3) * KD, va.cache = v_[app], va.is_key = false;
    kv_site(va, s);

    AttnArgs a;
    a.q = proj_h_, a.rel = proj_h_ + QD, a.sink = proj_h_ + QD + RD, a.gate = proj_h_ + QD + RD + c.n_heads;
    a.q_scale = ly.q_scale, a.rel_scale = ly.rel_scale, a.proj = ly.rel_proj;
    a.k = k_[app], a.v = v_[app], a.out = ao_h_, a.d_pos = d_pos_;
    a.part = part_, a.max_ctx = max_ctx_;
    a.T = T, a.stride = q_stride, a.n_heads = c.n_heads, a.n_kv = c.n_kv_heads, a.hd = c.head_dim;
    a.d_rel = c.d_rel, a.rel_extent = global ? c.rel_extent : c.window;
    a.window = global ? 0 : c.window, a.ring = global ? 0 : ring_;
    a.scale = 1.0f / sqrtf(static_cast<float>(c.head_dim)), a.q_thr = c.qk_clip, a.rel_thr = c.rel_clip;
    a.log_alpha = global ? c.log_alpha : 0.0f, a.log_floor = c.log_floor;
    attention(a, s);
    lin_(ao_h_, T, QD, ly.wo, H, o_h_, s);

    // Block boundary: the layer input joins the block states and the residual restarts at the conv output.
    float* as = cs + static_cast<size_t>(6) * KD;
    const bool boundary = layer % c.block_layers == 0;
    conv_residual(o_h_, ly.attn_conv, as, h_, !boundary, res2_, boundary ? bs_[nb_] : nullptr, T, H, s);
    if (boundary)
        st.s[st.n++] = bs_[nb_++];
    st.s[st.n++] = res2_;
    if (pass == 0) {
        MixArgs m;
        m.st = st;
        m.query = ly.mlp_query;
        m.eps = c.eps, m.partial = mixp_;
        m.norm_w[0] = ly.mlp_norm;
        m.out_h[0] = xm_;
        mix(m, T, H, s);
    } else {
        MixArgs mo;
        mo.st = st;
        mo.query = ly.mlp_query, mo.partial = mixp_;
        mo.out_f[0] = mm_;
        mix(mo, T, H, s);
        GateNormArgs gn;
        gn.n = 1, gn.rank = c.gate_rank, gn.eps = c.eps;
        gn.h[0] = mm_, gn.msg[0] = msg_[3], gn.norm_w[0] = ly.mlp_norm, gn.out_h[0] = xm_;
        gn.down[0] = ly.gates[3].down, gn.forget[0] = ly.gates[3].forget, gn.gain[0] = ly.gates[3].gain;
        GateOut go;
        go.o[0] = g_[0], go.o[1] = g_[1], go.o[2] = g_[2];
        gate_norm(gn, go, gd_, T, H, s);
    }
    route_topk(xm_, ly.router_w, ly.router_b, rlog_, idx_, wts_, T, H, c.n_experts, c.top_k, c.route_scale,
               s);
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
    moe_tail(ey_, wts_, ly.mlp_conv, as + static_cast<size_t>(3) * H, res2_, h_, mm_, T, c.top_k, H, s);
    if (tap) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(s));
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
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(s));
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
    MixArgs fin;
    for (int j = 0; j < nb_; ++j)
        fin.st.s[fin.st.n++] = bs_[j];
    fin.st.s[fin.st.n++] = h_;
    fin.query = w_->out_query, fin.partial = mixp_;
    fin.out_f[0] = mix_[0];
    mix(fin, T, H, s);
}

bool InklingForward::run(half* h, int T, int pos0, cudaStream_t s) {
    if (pos0 == 0)
        reset(s);
    if (pos0 != pos_ || T < 1 || T > max_tokens_ || pos_ + T > max_ctx_) {
        IMP_LOG_ERROR(
            "inkling: %d rows at position %d do not continue the sequence at %d (%d per call, %d context)", T,
            pos0, pos_, max_tokens_, max_ctx_);
        return false;
    }
    const size_t n = static_cast<size_t>(T) * w_->cfg.hidden;
    // The kernels read the position from the device: one captured graph serves every decode step.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_pos_, &pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    const bool graph = T == 1 && s != nullptr && !tap && decode_graph_ok_;
    if (graph && (!decode_exec_.get() || decode_h_ != h)) {
        // Capture the one-token step once: about 2000 kernel and cuBLAS launches become one replay.
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            decode_graph_ok_ = false;
        } else {
            f16_to_f32(h, h_, n, s);
            layers_(1, s);
            f32_to_f16(mix_[0], h, n, s);
            cudaGraphExec_t ex = nullptr;
            if (cudaStreamEndCapture(s, &g) != cudaSuccess ||
                cudaGraphInstantiate(&ex, g, 0) != cudaSuccess) {
                IMP_LOG_WARN("inkling: decode graph capture failed, decode stays eager");
                decode_graph_ok_ = false;
            }
            decode_graph_.reset(g);
            decode_exec_.reset(ex);
            decode_h_ = h;
        }
    }
    if (graph && decode_graph_ok_ && decode_exec_.get()) {
        if (cudaGraphLaunch(decode_exec_.get(), s) != cudaSuccess)
            return false;
    } else {
        f16_to_f32(h, h_, n, s);
        layers_(T, s);
        f32_to_f16(mix_[0], h, n, s);
    }
    pos_ += T;
    return ok_;
}

bool InklingForward::forward(const int32_t* d_ids, int T, float* d_logits, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    if (T < 1 || T > max_tokens_ || pos_ + T > max_ctx_ || !w_->embed) {
        IMP_LOG_ERROR("inkling: %d tokens at position %d exceed the workspace (%d per call, %d context)", T,
                      pos_, max_tokens_, max_ctx_);
        return false;
    }
    const int H = c.hidden;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_pos_, &pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    embed_rows(w_->embed, d_ids, h_, T, H, s);
    layers_(T, s);
    if (d_logits) {
        rms_norm(mix_[0], w_->norm, tap ? mix_[1] : nullptr, xq_, T, H, c.eps, s);
        if (tap) {
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(s));
            tap(c.n_apps(), mix_[1], T);
        }
        lin_(xq_ + static_cast<size_t>(T - 1) * H, 1, H, w_->unembed, c.vocab, logits_h_, s);
        softcap_f32(logits_h_, d_logits, static_cast<size_t>(c.vocab), c.logit_cap, s);
    }
    pos_ += T;
    return ok_;
}

}  // namespace imp
