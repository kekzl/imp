// Inkling2-Loop text forward (inkling_forward.h). Order of operations follows the transformers
// reference layer classes; fp16 at the linear layers, fp32 between them.

#include "exec/inkling_forward.h"

#include "compute/gemm.h"
#include "compute/gemm_grouped_nvfp4_f16_tile.h"
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

// Side stream i, started behind everything queued on s so far; s itself for the default stream (tests).
cudaStream_t InklingForward::fork_(cudaStream_t s, int i) {
    if (s == nullptr || cudaEventRecord(fork_ev_[i].get(), s) != cudaSuccess ||
        cudaStreamWaitEvent(side_[i].get(), fork_ev_[i].get(), 0) != cudaSuccess)
        return s;
    return side_[i].get();
}

// s goes on only after the side stream's work.
void InklingForward::join_(cudaStream_t side, cudaStream_t s, int i) {
    if (side == s)
        return;
    IMP_CUDA_CHECK_LOG(cudaEventRecord(join_ev_[i].get(), side));
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(s, join_ev_[i].get(), 0));
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
    // Calls are cut into segments that end on the checkpoint grid: the workspace holds one segment.
    max_tokens_ = max_tokens;
    max_ctx_ = max_ctx;
    seg_ = std::min(max_tokens, kGrid);
    ring_ = c.window + seg_ + kSlack;
    const size_t Tm = static_cast<size_t>(seg_), H = static_cast<size_t>(c.hidden);
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
    mixp2_ = F_(mix_partial_floats(max_tokens, c.hidden));
    for (int i = 0; i < 2 && ok_; ++i)
        ok_ = side_[i].create() && fork_ev_[i].create() && join_ev_[i].create();
    part_ = F_(attention_part_floats(c.n_heads, c.head_dim, c.window, max_ctx));
    if (!attn_work_) {
        attn_work_bytes_ = attention_rows_bytes(c.n_heads, c.head_dim, c.d_rel, c.window, max_ctx, seg_);
        attn_work_ = alloc_(attn_work_bytes_);
    }
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
    xs_ = Hf_(slots * H);
    perm_ = static_cast<int32_t*>(alloc_(slots * sizeof(int32_t)));
    off_ = static_cast<int32_t*>(alloc_((static_cast<size_t>(c.n_experts) + 1) * sizeof(int32_t)));
    d_pos_ = static_cast<int*>(alloc_(sizeof(int)));
    dec_h_ = Hf_(H * kGraphRows);
    conv_state_floats_ = 3 * (2 * KD + 2 * H);
    conv_all_ = F_(conv_state_floats_ * static_cast<size_t>(c.n_apps()));
    for (int a = 0; a < c.n_apps() && ok_; ++a) {
        // Application index -> weight layer.
        const int layer = a < c.n_layers ? a : c.loop_start + (a - c.n_layers) % (c.loop_end - c.loop_start);
        const size_t rows = c.global[static_cast<size_t>(layer)] ? static_cast<size_t>(max_ctx)
                                                                 : static_cast<size_t>(ring_);
        k_.push_back(Hf_(rows * KD));
        v_.push_back(Hf_(rows * KD));
        conv_state_.push_back(conv_all_ + conv_state_floats_ * static_cast<size_t>(a));
    }
    if (ok_)
        reset(nullptr);
    return ok_;
}

void InklingForward::reset(cudaStream_t s) {
    pos_ = hw_ = ckpt_valid_ = 0;
    look_pos0_ = -1;
    drafted_ = false;
    if (!armed_)
        hist_.clear();
    armed_ = false;
    if (conv_all_)
        IMP_CUDA_CHECK_LOG(
            cudaMemsetAsync(conv_all_, 0, conv_state_floats_ * conv_state_.size() * sizeof(float), s));
}

void InklingForward::checkpoint_(cudaStream_t s, const float* state) {
    const size_t g = static_cast<size_t>(pos_ / kGrid), n = conv_state_floats_ * conv_state_.size();
    if (ckpt_.size() < g)
        ckpt_.resize(g);
    ckpt_[g - 1].resize(n);
    const bool ok = cudaMemcpyAsync(ckpt_[g - 1].data(), state, n * sizeof(float), cudaMemcpyDeviceToHost,
                                    s) == cudaSuccess &&
                    cudaStreamSynchronize(s) == cudaSuccess;
    ckpt_valid_ = ok ? static_cast<int>(g) : std::min(ckpt_valid_, static_cast<int>(g) - 1);
}

int InklingForward::match(std::span<const int32_t> tokens) const {
    // The last row is always computed: its logits start the reply.
    const size_t lim = std::min(
        {hist_.size(), static_cast<size_t>(pos_), tokens.empty() ? 0 : tokens.size() - 1});
    size_t d = 0;
    while (d < lim && hist_[d] == tokens[d])
        ++d;
    const int g = std::min(static_cast<int>(d) / kGrid, ckpt_valid_);
    // Sliding K/V of [R - window, R) must not have been overwritten by positions up to hw_.
    return hw_ - (g * kGrid) > ring_ - w_->cfg.window ? 0 : g * kGrid;
}

int InklingForward::rewind(std::span<const int32_t> tokens, cudaStream_t s) {
    int g = match(tokens) / kGrid;
    const size_t n = conv_state_floats_ * conv_state_.size();
    if (g > 0 && cudaMemcpyAsync(conv_all_, ckpt_[static_cast<size_t>(g) - 1].data(), n * sizeof(float),
                                 cudaMemcpyHostToDevice, s) != cudaSuccess)
        g = 0;
    hist_.assign(tokens.begin(), tokens.end());
    armed_ = true;
    look_pos0_ = -1;  // the restored (or reset) conv state replaces a lookahead call's
    if (g == 0)
        return 0;  // run() at position 0 resets
    armed_ = false;
    pos_ = g * kGrid;
    ckpt_valid_ = g;
    return pos_;
}

void InklingForward::lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s) {
    const int64_t as[2] = {T, K}, bs[2] = {N, K}, cs[2] = {T, N};
    const Tensor A(const_cast<half*>(x), QType::F16, 2, as, true);
    const Tensor B(const_cast<half*>(w), QType::F16, 2, bs, true);
    Tensor C(y, QType::F16, 2, cs, true);
    gemm(A, B, C, 1.0f, 0.0f, s);
}

bool InklingForward::moe_rows_(const InklingLayer& ly, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const InklingExperts& ex = ly.ex;
    const int slots = T * c.top_k, H = c.hidden;
    if (H % 8 != 0 || (ex.fp8 && (!ex.w13_bias || !ex.w2_bias)) ||
        !moe_sort(idx_, slots, c.n_experts, perm_, off_, s))
        return false;
    moe_rows_copy(xm_, perm_, xs_, slots, H, c.top_k, true, s);
    const size_t n13 = static_cast<size_t>(2) * c.inter * H, n2 = static_cast<size_t>(H) * c.inter;
    GroupedNvfp4F16Args g;
    g.a = xs_, g.rows = slots, g.N = 2 * c.inter, g.K = H, g.ne = c.n_experts, g.d_offsets = off_;
    g.d = gu_;
    if (ex.fp8) {
        g.w = ex.w13_q, g.w_stride = n13, g.sc = ex.w13_s, g.sc_stride = n13 / 32;
        g.d_tensor_scales = ex.w13_ts;
    } else {
        g.w = ex.w13.packed_data, g.w_stride = ex.w13.expert_stride_packed;
        g.sc = ex.w13.micro_scales, g.sc_stride = ex.w13.expert_stride_ms;
        g.d_tensor_scales = ex.w13.tensor_scales;
    }
    if (!(ex.fp8 ? gemm_grouped_mxfp8_f16_tile(g, ex.w13_bias, s) : gemm_grouped_nvfp4_f16_tile(g, s)))
        return false;
    expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
    g.a = act_, g.N = H, g.K = c.inter, g.d = xs_;
    if (ex.fp8) {
        g.w = ex.w2_q, g.w_stride = n2, g.sc = ex.w2_s, g.sc_stride = n2 / 32;
        g.d_tensor_scales = ex.w2_ts;
    } else {
        g.w = ex.w2.packed_data, g.w_stride = ex.w2.expert_stride_packed;
        g.sc = ex.w2.micro_scales, g.sc_stride = ex.w2.expert_stride_ms;
        g.d_tensor_scales = ex.w2.tensor_scales;
    }
    if (!(ex.fp8 ? gemm_grouped_mxfp8_f16_tile(g, ex.w2_bias, s) : gemm_grouped_nvfp4_f16_tile(g, s))) {
        IMP_LOG_ERROR("inkling: grouped expert GEMM refused the down projection (%d x %d)", H, c.inter);
        ok_ = false;
    }
    moe_rows_copy(xs_, perm_, ey_, slots, H, 1, false, s);
    return true;
}

// xm_ -> ey_ for the slots idx_.
void InklingForward::experts_(const InklingLayer& ly, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const int slots = T * c.top_k, H = c.hidden;
    if (T > kGraphRows && grouped_ && moe_rows_(ly, T, s))
        return;
    if (ly.ex.fp8) {
        gemv_mxfp8_slots(ly.ex.w13_q, ly.ex.w13_s, idx_, xm_, gu_, slots, 2 * c.inter, H, c.top_k, s);
        expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
        gemv_mxfp8_slots(ly.ex.w2_q, ly.ex.w2_s, idx_, act_, ey_, slots, H, c.inter, 1, s);
        return;
    }
    gemv_nvfp4_moe_decode(ly.ex.w13, idx_, xm_, gu_, 2 * c.inter, H, H, slots, s, c.top_k);
    expert_act(gu_, act_, slots, c.inter, c.gate_cap, c.up_cap, s);
    gemv_nvfp4_moe_decode(ly.ex.w2, idx_, act_, ey_, H, c.inter, c.inter, slots, s, 1);
}

inkling::MixPre InklingForward::take_tail_() {
    const MixPre p = tail_;
    tail_ = MixPre();
    return p;
}

void InklingForward::flush_tail_(cudaStream_t s) {
    if (tail_.kind == 0)
        return;
    const MixPre p = take_tail_();
    moe_tail(p.x, p.weights, p.w, p.state, p.resid, p.out, mm_, 1, p.top_k, w_->cfg.hidden, s, p.log);
}

// A looped layer's later passes: the block mix and the message mix side by side, then the gated norms
// of the q, k and v inputs.
void InklingForward::mix_looped_(int layer, int T, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    const InklingLayer& ly = w_->layers[static_cast<size_t>(layer)];
    const bool global = c.global[static_cast<size_t>(layer)];
    const int H = c.hidden;
    MixStates st;
    for (int j = 0; j < nb_; ++j)
        st.s[st.n++] = bs_[j];
    MixArgs in;
    in.st = st;
    in.st.s[in.st.n++] = h_;
    in.query = ly.attn_query, in.nq = global ? 3 : 1, in.partial = mixp_;
    in.out_f[0] = mix_[0], in.out_f[1] = mix_[1], in.out_f[2] = mix_[2];
    in.pre = take_tail_();
    cudaStream_t sm = fork_(s, 0);
    mix(in, T, H, s);
    MixArgs ms;
    for (int j = 0; j < n_msrc_; ++j)
        ms.st.s[ms.st.n++] = msrc_[j];
    ms.query = w_->msg_query + static_cast<size_t>(layer - c.loop_start) * 4 * H, ms.nq = 4,
    ms.partial = sm == s ? mixp_ : mixp2_;
    for (int j = 0; j < 4; ++j)
        ms.out_f[j] = msg_[j];
    mix(ms, T, H, sm);
    join_(sm, s, 0);
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
        m.pre = take_tail_();
        mix(m, T, H, s);
    } else {
        mix_looped_(layer, T, s);
    }

    // Projections: rows [wq; wr; sink; gate; wk; k taps; wv; v taps] of one block.
    const int q_rows = 2 * QD + RD + c.n_heads, kv_rows = KD + TD;
    const half *kx = pk_, *vx = pv_;
    int q_stride = q_rows, kv_stride = kv_rows;
    // The k and v chains (projection, conv site) run beside the q projection on the side streams.
    // One row only: more rows project through cuBLAS, whose workspace three GEMMs at once would share
    // (lookahead text fell apart); there the side streams start behind the projections.
    cudaStream_t sk = s, sv = s;
    if (split) {
        if (T == 1)
            sk = fork_(s, 0), sv = fork_(s, 1);
        lin_(xq_, T, H, ly.wq, q_rows, proj_h_, s);
        lin_(xk_, T, H, ly.wk, kv_rows, pk_, sk);
        lin_(xv_, T, H, ly.wv, kv_rows, pv_, sv);
        if (T > 1)
            sk = fork_(s, 0), sv = fork_(s, 1);
    } else {
        lin_(xq_, T, H, ly.wq, ly.proj_rows, proj_h_, s);
        q_stride = kv_stride = ly.proj_rows;
        kx = proj_h_ + q_rows;
        vx = kx + kv_rows;
        sv = fork_(s, 1);
    }
    float* cs = conv_state_[app];
    KvSiteArgs ka;
    ka.x = kx, ka.w = ly.k_conv.w, ka.static_w = ly.k_conv.static_w, ka.dyn = kx + KD;
    ka.state = cs, ka.cache = k_[app], ka.d_pos = d_pos_;
    ka.T = T, ka.stride = kv_stride, ka.n_heads = c.n_kv_heads, ka.hd = c.head_dim,
    ka.ring = global ? 0 : ring_;
    ka.is_key = true, ka.thr = c.qk_clip, ka.eps = c.eps;
    const ConvLog lg = conv_log_at_(app, T);
    ka.log = lg.k;
    kv_site(ka, sk);
    KvSiteArgs va = ka;
    va.x = vx, va.w = ly.v_conv.w, va.static_w = ly.v_conv.static_w, va.dyn = vx + KD;
    va.state = cs + static_cast<size_t>(3) * KD, va.cache = v_[app], va.is_key = false;
    va.log = lg.v;
    kv_site(va, sv);
    join_(sk, s, 0);
    join_(sv, s, 1);

    AttnArgs a;
    a.q = proj_h_, a.rel = proj_h_ + QD, a.sink = proj_h_ + QD + RD, a.gate = proj_h_ + QD + RD + c.n_heads;
    a.q_scale = ly.q_scale, a.rel_scale = ly.rel_scale, a.proj = ly.rel_proj;
    a.k = k_[app], a.v = v_[app], a.out = ao_h_, a.d_pos = d_pos_;
    a.part = part_, a.max_ctx = max_ctx_;
    a.work = attn_work_, a.work_bytes = attn_work_bytes_, a.pos0 = T > kGraphRows ? pos_ : -1;
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
    // One row: the attention conv site runs inside the MLP mix, the MoE one inside the next mix over h_.
    const bool fused = fuse_ && T == 1;
    MixPre ap;
    ap.kind = 1, ap.x = o_h_, ap.w = ly.attn_conv, ap.state = as, ap.resid = h_, ap.add = !boundary;
    ap.out = res2_, ap.copy = boundary ? bs_[nb_] : nullptr, ap.log = lg.attn;
    if (!fused)
        conv_residual(o_h_, ly.attn_conv, as, h_, !boundary, res2_, ap.copy, T, H, s, lg.attn);
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
        if (fused)
            m.pre = ap;
        mix(m, T, H, s);
    } else {
        MixArgs mo;
        mo.st = st;
        mo.query = ly.mlp_query, mo.partial = mixp_;
        mo.out_f[0] = mm_;
        if (fused)
            mo.pre = ap;
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
    experts_(ly, T, s);
    if (fused) {
        tail_.kind = 2, tail_.x = ey_, tail_.weights = wts_, tail_.w = ly.mlp_conv;
        tail_.state = as + static_cast<size_t>(3) * H, tail_.resid = res2_, tail_.out = h_,
        tail_.log = lg.mlp;
        tail_.top_k = c.top_k;
    } else {
        moe_tail(ey_, wts_, ly.mlp_conv, as + static_cast<size_t>(3) * H, res2_, h_, mm_, T, c.top_k, H, s,
                 lg.mlp);
    }
    if (tap) {
        flush_tail_(s);
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(s));
        tap(app_seq_, h_, T);
    }
    ++app_seq_;
    tap_features_(T, s);
}

void InklingForward::layers_(int T, cudaStream_t s, const int32_t* ids, int32_t image_token) {
    const InklingConfig& c = w_->cfg;
    const int H = c.hidden;
    const size_t n = static_cast<size_t>(T) * H;
    // The embedding norm belongs to the token embedding: image rows arrive as tower features.
    for (int a = 0, b = 0; a < T; a = b) {
        const bool img = ids && image_token >= 0 && ids[a] == image_token;
        for (b = a + 1; b < T && ids && image_token >= 0 && (ids[b] == image_token) == img; ++b) {}
        b = (ids && image_token >= 0) ? b : T;
        if (!img)
            rms_norm(h_ + static_cast<size_t>(a) * H, w_->embed_norm, h_ + static_cast<size_t>(a) * H,
                     nullptr, b - a, H, c.eps, s);
    }
    nb_ = 0;
    app_seq_ = 0;
    if (tap) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(s));
        tap(-1, h_, T);
    }
    for (int i = 0; i < c.loop_start; ++i)
        apply_(i, 0, T, s);
    flush_tail_(s);
    add_f32(h_, nullptr, pre_h_, n, s);
    const int pre_nb = nb_;
    for (int pass = 0; pass < c.passes; ++pass) {
        if (pass > 0) {
            flush_tail_(s);
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
    fin.pre = take_tail_();
    mix(fin, T, H, s);
    if (feat_)
        rms_norm(mix_[0], w_->norm, feat_ + (taps_.size() - 1) * static_cast<size_t>(seg_) * H, nullptr, T, H,
                 c.eps, s);
}

// Captures the T-row step once: about 2000 kernel and cuBLAS launches become one replay.
void InklingForward::capture_decode_(int T, cudaStream_t s) {
    const size_t n = static_cast<size_t>(T) * static_cast<size_t>(w_->cfg.hidden);
    cudaGraph_t g = nullptr;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        decode_graph_ok_ = false;
        return;
    }
    f16_to_f32(dec_h_, h_, n, s);
    layers_(T, s);
    f32_to_f16(mix_[0], dec_h_, n, s);
    cudaGraphExec_t ex = nullptr;
    if (cudaStreamEndCapture(s, &g) != cudaSuccess || cudaGraphInstantiate(&ex, g, 0) != cudaSuccess) {
        IMP_LOG_WARN("inkling: graph capture of the %d-row step failed, small calls stay eager", T);
        decode_graph_ok_ = false;
    }
    decode_graph_[T - 1].reset(g);
    decode_exec_[T - 1].reset(ex);
}

bool InklingForward::run(half* h, int T, int pos0, cudaStream_t s, const int32_t* ids, int32_t image_token) {
    if (pos0 == 0)
        reset(s);
    armed_ = false;
    if (pos0 != pos_ || T < 1 || T > max_tokens_ || pos_ + T > max_ctx_) {
        IMP_LOG_ERROR(
            "inkling: %d rows at position %d do not continue the sequence at %d (%d per call, %d context)", T,
            pos0, pos_, max_tokens_, max_ctx_);
        return false;
    }
    // Tokens past the prompt rewind() recorded (the decode steps) join the history.
    for (int i = 0; i < T; ++i)
        if (static_cast<size_t>(pos0) + static_cast<size_t>(i) >= hist_.size())
            hist_.push_back(ids ? ids[i] : -1);
    const int Hd = w_->cfg.hidden;
    if (T == 1 && spec_.propose && ids && s != nullptr && !tap)
        return step_spec_(h, ids[0], s);
    settle_(s);
    for (int done = 0; done < T;) {
        const int seg = std::min(T - done, kGrid - pos_ % kGrid);
        if (!segment_(h + static_cast<size_t>(done) * Hd, seg, s, ids ? ids + done : nullptr, image_token))
            return false;
        done += seg;
    }
    return ok_;
}

// One segment of run(): rows [pos_, pos_ + seg), a checkpoint when it ends on the grid. The kernels read
// the position from the device, so one captured graph per row count serves every small call (decode
// step, paced prefill) that carries no image rows.
bool InklingForward::segment_(half* hs, int seg, cudaStream_t s, const int32_t* ids, int32_t image_token) {
    const size_t n = static_cast<size_t>(seg) * static_cast<size_t>(w_->cfg.hidden);
    bool graph = s != nullptr && !tap && seg <= kGraphRows && decode_graph_ok_;
    for (int i = 0; graph && ids && image_token >= 0 && i < seg; ++i)
        graph = ids[i] != image_token;
    if (graph && !decode_exec_[seg - 1].get())
        capture_decode_(seg, s);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_pos_, &pos_, sizeof(int), cudaMemcpyHostToDevice, s));
    if (graph && decode_graph_ok_ && decode_exec_[seg - 1].get()) {
        const size_t hb = n * sizeof(half);
        if (cudaMemcpyAsync(dec_h_, hs, hb, cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
            cudaGraphLaunch(decode_exec_[seg - 1].get(), s) != cudaSuccess ||
            cudaMemcpyAsync(hs, dec_h_, hb, cudaMemcpyDeviceToDevice, s) != cudaSuccess)
            return false;
    } else {
        f16_to_f32(hs, h_, n, s);
        layers_(seg, s, ids, image_token);
        f32_to_f16(mix_[0], hs, n, s);
    }
    pos_ += seg;
    hw_ = std::max(hw_, pos_);
    if (pos_ % kGrid == 0)
        checkpoint_(s, conv_all_);
    if (spec_.on_rows)
        spec_.on_rows(feat_, seg_, seg, pos_ - seg, s);
    return true;
}

InklingForward::ConvLog InklingForward::conv_log_at_(size_t app, int T) const {
    // Calls of 2..kGraphRows rows log their conv inputs: a lookahead call may be cut short later.
    if (!conv_log_ || T < 2 || T > kGraphRows)
        return {};
    const InklingConfig& c = w_->cfg;
    const size_t R = kGraphRows, KD = static_cast<size_t>(c.n_kv_heads) * c.head_dim,
                 H = static_cast<size_t>(c.hidden);
    float* base = conv_log_ + app * R * (2 * KD + 2 * H);
    return {base, base + R * KD, base + 2 * R * KD, base + 2 * R * KD + R * H};
}

// Feature plane k: the stream the next visit starts from.
void InklingForward::tap_features_(int T, cudaStream_t s) {
    const size_t H = static_cast<size_t>(w_->cfg.hidden);
    for (size_t k = 0; feat_ && k + 1 < taps_.size(); ++k)
        if (taps_[k] == app_seq_) {
            flush_tail_(s);
            add_f32(h_, nullptr, feat_ + k * static_cast<size_t>(seg_) * H, static_cast<size_t>(T) * H, s);
        }
}

bool InklingForward::forward(const int32_t* d_ids, int T, float* d_logits, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    if (T < 1 || T > seg_ || pos_ + T > max_ctx_ || !w_->embed) {
        IMP_LOG_ERROR("inkling: %d tokens at position %d exceed the workspace (%d per call, %d context)", T,
                      pos_, seg_, max_ctx_);
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
    hw_ = std::max(hw_, pos_);
    return ok_;
}

}  // namespace imp
