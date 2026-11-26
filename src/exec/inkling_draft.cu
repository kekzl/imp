// Inkling2 DSpark draft (inkling_draft.h): fp32 residual stream, fp16 at the linear layers.

#include "exec/inkling_draft.h"

#include "compute/gemv_i8b32.h"

#include "exec/inkling_draft_head.h"
#include "exec/inkling_draft_policy.h"

#include "compute/dspark_attention.h"
#include "compute/gemm.h"
#include "compute/inkling_ops.h"
#include "core/cuda_raii.h"
#include "core/graph_epoch.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "memory/vram_allocator.h"
#include "model/bf16_loader.h"
#include "model/json_util.h"
#include "model/safetensors_loader.h"

#include <sys/mman.h>

#include <algorithm>
#include <cmath>

namespace imp {

using namespace inkling;

namespace {

constexpr int kT = 256;
constexpr int kHd = 128;
constexpr float kLogitCap = 30.0f;  // the target's final_logit_softcapping

unsigned blocks(size_t n) { return static_cast<unsigned>((n + kT - 1) / kT); }

// One block per row: out = rms_norm(sum_t alpha[t] * feat[t][row0 + row]) * w.
// d_row (optional, device): added to row0, so a captured launch follows the sequence.
__global__ void draft_mix_k(const float* feat, size_t plane, int row0, const float* alpha, int taps,
                            const float* w, half* out, int H, float eps, const int* d_row) {
    __shared__ float sh[kT / 32];
    const size_t r = (size_t)(row0 + (d_row ? *d_row : 0) + blockIdx.x) * H;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        float m = 0.0f;
        for (int t = 0; t < taps; ++t)
            m += alpha[t] * feat[(size_t)t * plane + r + c];
        ss += m * m;
    }
    const float inv = rsqrtf(draft_block_sum(ss, sh) / H + eps);
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        float m = 0.0f;
        for (int t = 0; t < taps; ++t)
            m += alpha[t] * feat[(size_t)t * plane + r + c];
        out[(size_t)blockIdx.x * H + c] = __float2half(m * inv * w[c]);
    }
}

// Per (row, head) of the sub-block starting at column col0: RMSNorm over the head (w, or none) and
// NeoX RoPE at position pos0 + row, in place. 64 threads: thread i owns the pair (i, i + 64).
// d_pos (optional, device): added to pos0.
__global__ void draft_norm_rope_k(half* x, int stride, int col0, int heads, const float* w, int pos0,
                                  float log2_theta, float eps, const int* d_pos) {
    __shared__ float part[2];
    const int row = blockIdx.x / heads, head = blockIdx.x % heads;
    half* v = x + (size_t)row * stride + col0 + (size_t)head * kHd;
    const int i = threadIdx.x;
    float a = __half2float(v[i]), b = __half2float(v[i + kHd / 2]);
    if (w) {
        float ss = a * a + b * b;
        for (int o = 16; o > 0; o >>= 1)
            ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if ((i & 31) == 0)
            part[i >> 5] = ss;
        __syncthreads();
        const float inv = rsqrtf((part[0] + part[1]) / kHd + eps);
        a *= inv * w[i];
        b *= inv * w[i + kHd / 2];
    }
    const float freq = exp2f(-log2_theta * (2.0f * i) / kHd);
    float sn = 0.0f, cs = 0.0f;
    sincosf(static_cast<float>(pos0 + (d_pos ? *d_pos : 0) + row) * freq, &sn, &cs);
    v[i] = __float2half(a * cs - b * sn);
    v[i + kHd / 2] = __float2half(b * cs + a * sn);
}

// dst[row][c] = src[row][col0 + c] for n rows of width columns; dst row 0 is row row0 (+ *d_row, device,
// when set) of the array, through blocks (16 rows per entry) when set: a pool layer.
__global__ void draft_cols_k(const half* src, int stride, int col0, int width, half* dst, size_t n,
                             const int* d_row, int row0 = 0, const int* blocks = nullptr) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const int r = row0 + (d_row ? *d_row : 0) + static_cast<int>(i / width);
    const int slot = blocks ? (blocks[r >> 4] << 4) | (r & 15) : r;
    dst[static_cast<size_t>(slot) * width + i % width] = src[(i / width) * stride + col0 + i % width];
}

// draft_cols_k into a ring of M rows kept twice (rows r and r + M): row i goes to position pos0 + i, so
// any run of up to M positions reads as one contiguous range.
__global__ void draft_ring_k(const half* src, int stride, int col0, int width, half* dst, size_t n, int pos0,
                             int M) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const size_t row = i / width, c = i % width, r = (static_cast<size_t>(pos0) + row) % M;
    const half v = src[row * stride + col0 + c];
    dst[r * width + c] = v;
    dst[(r + M) * width + c] = v;
}

// Row 0 = rms_norm(anchor embedding) * w, rows 1.. = the mask embedding.
__global__ void draft_block_k(const half* emb, const float* w, const half* mask, float* h, int rows, int H,
                              float eps) {
    __shared__ float sh[kT / 32];
    float ss = 0.0f;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        const float v = __half2float(emb[c]);
        ss += v * v;
    }
    const float inv = rsqrtf(draft_block_sum(ss, sh) / H + eps);
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        h[c] = __half2float(emb[c]) * inv * w[c];
        for (int r = 1; r < rows; ++r)
            h[(size_t)r * H + c] = __half2float(mask[c]);
    }
}

__global__ void draft_add_k(float* h, const half* y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        h[i] += __half2float(y[i]);
}

// act[row][c] = silu(gu[row][c]) * gu[row][I + c].
__global__ void draft_silu_k(const half* gu, half* act, size_t n, int I) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const size_t w = I, row = i / w, c = i % w;
    const float g = __half2float(gu[row * 2 * w + c]), u = __half2float(gu[row * 2 * w + w + c]);
    act[i] = __float2half(g / (1.0f + expf(-g)) * u);
}

// x2[row] = [hid[row + 1], f_rnn].
__global__ void draft_cat_k(const half* hid, const half* frnn, half* x2, size_t n, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const size_t h = H, row = i / (2 * h), c = i % (2 * h);
    x2[i] = c < h ? hid[(row + 1) * h + c] : frnn[c - h];
}

}  // namespace

InklingDraft::~InklingDraft() {
    for (auto& b : owned_)
        b.reset();
}

void InklingDraft::lin8_(const half* x, int T, int K, const InklingDraftWeights::Q8& w8, const half* w, int N,
                         half* y, cudaStream_t s) {
    if (w8.q && gemv_i8b32_f16(w8.q, w8.scales, x, y, N, K, T, s))
        return;
    if (w)
        lin_(x, T, K, w, N, y, s);
    else
        ok_ = false;
}

void* InklingDraft::alloc_(size_t bytes) {
    VramOwned<std::byte> buf(*va_, bytes, "inkling_draft_ctx");
    void* p = buf.get();
    if (!p) {
        IMP_LOG_WARN("inkling draft: context allocation of %.1f MiB failed", bytes / 1048576.0);
        ok_ = false;
        return nullptr;
    }
    owned_.push_back(std::move(buf));
    return p;
}

void InklingDraft::lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s) {
    const int64_t as[2] = {T, K}, bs[2] = {N, K}, cs[2] = {T, N};
    const Tensor A(const_cast<half*>(x), QType::F16, 2, as, true);
    const Tensor B(const_cast<half*>(w), QType::F16, 2, bs, true);
    Tensor C(y, QType::F16, 2, cs, true);
    gemm(A, B, C, 1.0f, 0.0f, s);
}

bool InklingDraft::init(const InklingDraftWeights& w, VRAMAllocator& va, int max_ctx, int cap, int back,
                        bool own_kv) {
    w_ = &w;
    va_ = &va;
    max_ctx_ = max_ctx;
    cap_ = cap;
    // A local layer reads window + block rows: its K/V live in a ring that also keeps the back positions a
    // restart may return to and the cap rows of one call. Not worth it below twice the ring, except that
    // the pool holds one layer per position only.
    const int ring = w.cfg.window + w.cfg.block + cap + std::max(back, 0);
    ring_ = back >= 0 && (!own_kv || 2 * static_cast<long long>(ring) < max_ctx + w.cfg.block) ? ring : 0;
    const InklingDraftConfig& c = w.cfg;
    const size_t H = static_cast<size_t>(c.hidden), B = static_cast<size_t>(c.block), taps = c.taps.size();
    const size_t QD = static_cast<size_t>(c.heads) * kHd, KD = static_cast<size_t>(c.kv_heads) * kHd;
    const size_t V = static_cast<size_t>(w.head.vocab), I = static_cast<size_t>(c.inter),
                 C = static_cast<size_t>(cap);
    const auto F = [&](size_t n) { return static_cast<float*>(alloc_(n * sizeof(float))); };
    const auto Hf = [&](size_t n) { return static_cast<half*>(alloc_(n * sizeof(half))); };
    // Layers that keep one row per position: own rows here, or the pool's through bind_kv.
    for (int i = 0; i < c.layers; ++i) {
        const bool ringed = ring_ > 0 && c.local[static_cast<size_t>(i)];
        const size_t rows = ringed ? 2 * static_cast<size_t>(ring_) : static_cast<size_t>(max_ctx) + B;
        k_.push_back(ringed || own_kv ? Hf(rows * KD) : nullptr);
        v_.push_back(ringed || own_kv ? Hf(rows * KD) : nullptr);
        ok_ = ok_ && (!(ringed || own_kv) || (k_.back() && v_.back()));
    }
    bound_ = own_kv;
    pend_ = F(taps * static_cast<size_t>(cap) * H), h_ = F(B * H), state_ = F(static_cast<size_t>(c.rank));
    work_ = F(dspark_attention_work_floats(c.block, c.heads, max_ctx + c.block));
    xc_ = Hf(C * H), kvc_ = Hf(C * 2 * KD), xn_ = Hf(B * H), qkv_ = Hf(B * (QD + 2 * KD)), q_ = Hf(B * QD);
    o_ = Hf(B * QD), y_ = Hf(B * H), gu_ = Hf(B * 2 * I), act_ = Hf(B * I), hid_ = Hf(B * H);
    logits_ = Hf(B * V), x2_ = Hf(B * 2 * H), rc_ = Hf(B * 3 * static_cast<size_t>(c.rank));
    latent_ = Hf(static_cast<size_t>(c.rank)), bias_ = Hf(V), frnn_ = Hf(H);
    d_tok_ = static_cast<int32_t*>(alloc_((B + 1) * sizeof(int32_t)));
    d_par_ = static_cast<int*>(alloc_(sizeof(par_)));
    emb_ = Hf(H);
    d_conf_ = F(B);
    raw_ = F(3 * static_cast<size_t>(c.rank)), latent_f_ = F(static_cast<size_t>(c.rank));
    const size_t n_cand = draft_head_cands(V);
    val_ = F(n_cand), cand0_ = F(B * n_cand), mass_ = F(B * n_cand / 2);
    d_cand_ = static_cast<int32_t*>(alloc_(B * n_cand * sizeof(int32_t)));
    return ok_;
}

InklingSpecHooks InklingDraft::hooks() {
    InklingSpecHooks h;
    h.on_rows = [this](const float* feat, int cap, int n, int pos0, cudaStream_t s) {
        on_rows(feat, cap, n, pos0, s);
    };
    h.propose = [this](int32_t anchor, const half* d_emb, int pos, int32_t* out, cudaStream_t s) {
        return propose(anchor, d_emb, pos, out, s);
    };
    h.restart = [this](int keep) { restart(keep); };
    return h;
}

bool InklingDraft::bind_kv(half* k, half* v, const int* table) {
    const InklingDraftConfig& c = w_->cfg;
    int open = 0;  // layers without memory: the one the pool layer serves
    for (int i = 0; i < c.layers; ++i)
        open += k_[static_cast<size_t>(i)] == nullptr ? 1 : 0;
    if (bound_ || open != 1 || !k || !v || !table) {
        IMP_LOG_WARN("inkling draft: %d full-context layers, the pool holds one: lookahead off", open);
        return false;
    }
    for (int i = 0; i < c.layers; ++i)
        if (!k_[static_cast<size_t>(i)]) {
            k_[static_cast<size_t>(i)] = k;
            v_[static_cast<size_t>(i)] = v;
            global_ = i;
        }
    table_ = table;
    bound_ = true;
    return true;
}

void InklingDraft::restart(int keep) {
    // Kept rows before keep belong to the same tokens: they get their K/V at the next flush.
    if (keep <= kept_pos0_)
        kept_pos0_ = std::max(keep, 0), kept_ = 0;
    else if (keep < kept_pos0_ + kept_)
        kept_ = keep - kept_pos0_;
    done_ = std::min(done_, kept_);
    drafted_pos_ = -1;
}

// Context K/V of n feature rows (planes of plane floats, rows from row0) at positions pos0...
void InklingDraft::convert_(const float* feat, size_t plane, int row0, int n, int pos0, cudaStream_t s) {
    const InklingDraftConfig& c = w_->cfg;
    const int H = c.hidden, KD = c.kv_heads * kHd, taps = static_cast<int>(c.taps.size());
    const float log2_theta = std::log2(c.theta);
    for (int i = 0; i < c.layers; ++i) {
        const InklingDraftWeights::Layer& ly = w_->layers[static_cast<size_t>(i)];
        draft_mix_k<<<n, kT, 0, s>>>(feat, plane, row0, w_->alpha + static_cast<size_t>(i) * taps, taps,
                                     ly.dflare, xc_, H, c.eps, nullptr);
        IMP_CUDA_CHECK_LAUNCH();
        lin_(xc_, n, H, ly.kv_ctx, 2 * KD, kvc_, s);
        draft_norm_rope_k<<<n * c.kv_heads, kHd / 2, 0, s>>>(kvc_, 2 * KD, 0, c.kv_heads, ly.k_norm_ctx, pos0,
                                                             log2_theta, c.eps, nullptr);
        IMP_CUDA_CHECK_LAUNCH();
        const size_t nk = static_cast<size_t>(n) * KD;
        if (ring_ > 0 && c.local[static_cast<size_t>(i)]) {
            draft_ring_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, 0, KD, k_[static_cast<size_t>(i)], nk, pos0,
                                                   ring_);
            IMP_CUDA_CHECK_LAUNCH();
            draft_ring_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, KD, KD, v_[static_cast<size_t>(i)], nk, pos0,
                                                   ring_);
            IMP_CUDA_CHECK_LAUNCH();
            continue;
        }
        const int* tbl = i == global_ ? table_ : nullptr;
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, 0, KD, k_[static_cast<size_t>(i)], nk, nullptr,
                                               pos0, tbl);
        IMP_CUDA_CHECK_LAUNCH();
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, KD, KD, v_[static_cast<size_t>(i)], nk, nullptr,
                                               pos0, tbl);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// The kept rows that have no context K/V yet get it: one pass for all of them.
void InklingDraft::flush_(cudaStream_t s) {
    if (done_ < kept_)
        convert_(pend_, static_cast<size_t>(cap_) * w_->cfg.hidden, done_, kept_ - done_, kept_pos0_ + done_,
                 s);
    done_ = kept_;
}

// Feature rows are kept and turned into context K/V only when a draft needs them (propose) or the
// buffer is full: a resting draft costs four row copies per token instead of three projections, and
// rows a later call replaces are never projected.
void InklingDraft::on_rows(const float* feat, int cap, int n, int pos0, cudaStream_t s) {
    const InklingDraftConfig& c = w_->cfg;
    if (!ok_ || !bound_ || n < 1 || n > cap_ || pos0 < 0 || pos0 + n > max_ctx_)
        return;
    const size_t H = static_cast<size_t>(c.hidden), plane = static_cast<size_t>(cap) * H;
    const int end = kept_pos0_ + kept_;
    if (pos0 >= kept_pos0_ && pos0 <= end && pos0 - kept_pos0_ + n <= cap_) {
        kept_ = pos0 - kept_pos0_;  // rows from pos0 on are replaced
        done_ = std::min(done_, kept_);
    } else {
        flush_(s);
        kept_pos0_ = pos0, kept_ = 0, done_ = 0;
    }
    for (size_t t = 0; t < c.taps.size(); ++t)
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(pend_ + t * static_cast<size_t>(cap_) * H + static_cast<size_t>(kept_) * H,
                            feat + t * plane, static_cast<size_t>(n) * H * sizeof(float),
                            cudaMemcpyDeviceToDevice, s));
    kept_ += n;
    // The rows of a drafted step: the next step most likely drafts again, and queuing their projections
    // takes the host 0.19 to 0.31 ms. Here that time passes under the call that just went to the GPU.
    if (pos0 == drafted_pos_)
        flush_(s);
}

// The draft block and the Markov head for the anchor at d_par_[0]: every launch reads the position, the
// local window and the feature row from d_par_, so one captured graph serves every call.
bool InklingDraft::block_(cudaStream_t s) {
    const InklingDraftConfig& c = w_->cfg;
    const int B = c.block, H = c.hidden, QD = c.heads * kHd, KD = c.kv_heads * kHd, I = c.inter, r = c.rank;
    const int V = w_->head.vocab, taps = static_cast<int>(c.taps.size()), n_out = B - 1;
    const float log2_theta = std::log2(c.theta), scale = 1.0f / std::sqrt(static_cast<float>(kHd));
    const size_t bH = static_cast<size_t>(B) * H;
    const int *d_pos = d_par_, *d_ctx_local = d_par_ + 1, *d_first_local = d_par_ + 2, *d_frow = d_par_ + 3;
    const int* d_row_local = d_par_ + 4;
    draft_block_k<<<1, kT, 0, s>>>(emb_, w_->embed_norm, w_->mask, h_, B, H, c.eps);
    IMP_CUDA_CHECK_LAUNCH();
    for (int i = 0; i < c.layers; ++i) {
        const InklingDraftWeights::Layer& ly = w_->layers[static_cast<size_t>(i)];
        half *kc = k_[static_cast<size_t>(i)], *vc = v_[static_cast<size_t>(i)];
        rms_norm(h_, ly.in_norm, nullptr, xn_, B, H, c.eps, s);
        lin8_(xn_, B, H, ly.qkv8, ly.qkv, QD + 2 * KD, qkv_, s);
        draft_norm_rope_k<<<B * c.heads, kHd / 2, 0, s>>>(qkv_, QD + 2 * KD, 0, c.heads, ly.q_norm, 0,
                                                          log2_theta, c.eps, d_pos);
        IMP_CUDA_CHECK_LAUNCH();
        draft_norm_rope_k<<<B * c.kv_heads, kHd / 2, 0, s>>>(qkv_, QD + 2 * KD, QD, c.kv_heads, ly.k_norm, 0,
                                                             log2_theta, c.eps, d_pos);
        IMP_CUDA_CHECK_LAUNCH();
        // The block's own K/V rows go behind the context rows, at its positions.
        // A local layer attends the last window positions before the block and the block.
        const bool local = c.local[static_cast<size_t>(i)];
        const int* d_row = local ? d_row_local : d_pos;
        const size_t nq = static_cast<size_t>(B) * QD, nk = static_cast<size_t>(B) * KD;
        draft_cols_k<<<blocks(nq), kT, 0, s>>>(qkv_, QD + 2 * KD, 0, QD, q_, nq, nullptr);
        IMP_CUDA_CHECK_LAUNCH();
        const int* tbl = i == global_ ? table_ : nullptr;
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(qkv_, QD + 2 * KD, QD, KD, kc, nk, d_row, 0, tbl);
        IMP_CUDA_CHECK_LAUNCH();
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(qkv_, QD + 2 * KD, QD + KD, KD, vc, nk, d_row, 0, tbl);
        IMP_CUDA_CHECK_LAUNCH();
        if (dspark_attention_dev(q_, kc, vc, o_, work_, B, c.heads, c.kv_heads, local ? d_ctx_local : d_pos,
                                 (local ? std::min(c.window, max_ctx_) : max_ctx_) + B, scale, s,
                                 local ? d_first_local : nullptr, tbl) != cudaSuccess)
            return false;
        lin8_(o_, B, QD, ly.wo8, ly.wo, H, y_, s);
        draft_add_k<<<blocks(bH), kT, 0, s>>>(h_, y_, bH);
        IMP_CUDA_CHECK_LAUNCH();
        rms_norm(h_, ly.post_norm, nullptr, xn_, B, H, c.eps, s);
        lin8_(xn_, B, H, ly.gate_up8, ly.gate_up, 2 * I, gu_, s);
        const size_t na = static_cast<size_t>(B) * I;
        draft_silu_k<<<blocks(na), kT, 0, s>>>(gu_, act_, na, I);
        IMP_CUDA_CHECK_LAUNCH();
        lin8_(act_, B, I, ly.down8, ly.down, H, y_, s);
        draft_add_k<<<blocks(bH), kT, 0, s>>>(h_, y_, bH);
        IMP_CUDA_CHECK_LAUNCH();
    }
    rms_norm(h_, w_->norm, nullptr, hid_, B, H, c.eps, s);
    // Rows 1.. draft: base logits through the target's head, then the recurrent head row by row.
    if (!gemv_fp8_rowscale_f16(w_->head.w8, w_->head.row_scales, hid_ + H, logits_, V, H, n_out, s))
        return false;
    draft_mix_k<<<1, kT, 0, s>>>(pend_, static_cast<size_t>(cap_) * H, 0,
                                 w_->alpha + static_cast<size_t>(c.layers) * taps, taps, w_->markov_dflare,
                                 frnn_, H, c.eps, d_frow);
    IMP_CUDA_CHECK_LAUNCH();
    const size_t nx = static_cast<size_t>(n_out) * 2 * H;
    draft_cat_k<<<blocks(nx), kT, 0, s>>>(hid_, frnn_, x2_, nx, H);
    IMP_CUDA_CHECK_LAUNCH();
    lin_(x2_, n_out, 2 * H, w_->rnn_ctx_w, 3 * r, rc_, s);
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(state_, 0, static_cast<size_t>(r) * sizeof(float), s));
    DraftHeadArgs ha;
    ha.logits = logits_, ha.rc = rc_, ha.rnn_state_w = w_->rnn_state_w, ha.w1 = w_->w1,
    ha.fold_w = w_->fold_w;
    ha.rnn_b = w_->rnn_b, ha.fold_b = w_->fold_b;
    ha.state = state_, ha.raw = raw_, ha.latent = latent_f_, ha.val = val_, ha.cand0 = cand0_,
    ha.mass = mass_;
    ha.conf = d_conf_, ha.cand = d_cand_, ha.tok = d_tok_;
    ha.V = V, ha.rank = r, ha.rows = n_out, ha.cap = kLogitCap;
    draft_head_pick(ha, s);
    return ok_;
}

// The block is 84 launches for 0.73 ms of kernel time (1.83 ms per call launched one by one): captured
// once, replayed as one graph (1.19 ms).
void InklingDraft::capture_(cudaStream_t s) {
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ex = nullptr;
    exec_epoch_ = graph_epoch();
    graph_ok_ = cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) == cudaSuccess;
    if (graph_ok_) {
        const bool body = block_(s);
        graph_ok_ = cudaStreamEndCapture(s, &g) == cudaSuccess && body &&
                    cudaGraphInstantiate(&ex, g, 0) == cudaSuccess;
    }
    graph_.reset(g);
    exec_.reset(graph_ok_ ? ex : nullptr);
    if (!graph_ok_)
        IMP_LOG_WARN("inkling draft: graph capture of the block failed, it runs launch by launch");
}

int InklingDraft::propose(int32_t anchor, const half* d_emb, int pos, int32_t* out, cudaStream_t s) {
    const InklingDraftConfig& c = w_->cfg;
    const int B = c.block, V = w_->head.vocab, n_out = B - 1;
    // Context for every position below pos: the kept rows end there, everything before them is flushed.
    // Kept rows from pos on belong to drafts that were not confirmed: the call that follows replaces them.
    const int frow = pos - 1 - kept_pos0_;
    if (!ok_ || pos < 1 || anchor < 0 || anchor >= c.vocab_padded || frow < 0 || frow >= kept_ ||
        pos + B > max_ctx_)
        return 0;
    kept_ = frow + 1;
    done_ = std::min(done_, kept_);
    flush_(s);
    if (!ok_)
        return 0;
    // What the block reads from the device: position, local context length and first key, feature row,
    // the block's row in a local layer's K/V (its ring row when there is a ring).
    const int first = std::max(0, pos - c.window);
    par_[0] = pos, par_[1] = pos - first, par_[2] = ring_ > 0 ? first % ring_ : first, par_[3] = frow;
    par_[4] = par_[2] + par_[1];
    tok_ = anchor;
    if (cudaMemcpyAsync(d_par_, par_, sizeof(par_), cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(d_tok_, &tok_, sizeof(tok_), cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(emb_, d_emb, static_cast<size_t>(c.hidden) * sizeof(half), cudaMemcpyDeviceToDevice,
                        s) != cudaSuccess)
        return 0;
    // I3: a block graph of an older epoch is dropped and captured again (the last propose synced).
    if (exec_.get() && exec_epoch_ != graph_epoch()) {
        exec_.reset(nullptr);
        graph_.reset(nullptr);
    }
    if (graph_ok_ && !exec_.get() && s != nullptr)
        capture_(s);
    if (exec_.get() ? cudaGraphLaunch(exec_.get(), s) != cudaSuccess : !block_(s))
        return 0;
    drafted_pos_ = pos;
    float conf[InklingForward::kGraphRows] = {};
    if (cudaMemcpyAsync(out, d_tok_ + 1, static_cast<size_t>(n_out) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                        s) != cudaSuccess ||
        cudaMemcpyAsync(conf, d_conf_, static_cast<size_t>(n_out) * sizeof(float), cudaMemcpyDeviceToHost,
                        s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess)
        return 0;
    return inkling_draft_rows(out, conf, n_out, V);
}

}  // namespace imp
