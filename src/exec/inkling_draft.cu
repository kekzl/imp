// Inkling2 DSpark draft (inkling_draft.h): fp32 residual stream, fp16 at the linear layers.

#include "exec/inkling_draft.h"

#include "compute/gemv_i8b32.h"

#include "exec/inkling_draft_head.h"
#include "exec/inkling_draft_policy.h"

#include "compute/dspark_attention.h"
#include "compute/gemm.h"
#include "compute/inkling_ops.h"
#include "core/cuda_raii.h"
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

// dst[row][c] = src[row][col0 + c] for n rows of width columns; d_row (optional, device): dst row 0 is
// row *d_row of the array.
__global__ void draft_cols_k(const half* src, int stride, int col0, int width, half* dst, size_t n,
                             const int* d_row) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    dst[(d_row ? static_cast<size_t>(*d_row) * width : 0) + i] = src[(i / width) * stride + col0 + i % width];
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

// fold[v][col0 + row] = t[row][v] for rows rows; fold_b[v] = t[0][v] when fold is null.
__global__ void draft_fold_k(const half* t, half* fold, float* fold_b, int rows, int V, int r, int col0) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * V)
        return;
    const size_t row = i / V, v = i % V;
    if (fold)
        fold[v * r + col0 + row] = t[i];
    else
        fold_b[v] = __half2float(t[i]);
}

float json_float(const JValue& obj, const char* key, float dflt) {
    const JValue* v = jobj_find(obj, key);
    return (v && v->type == JType::NUMBER) ? static_cast<float>(v->num_val) : dflt;
}

bool draft_dims_ok(const InklingDraftConfig& c) {
    const bool tower = c.layers >= 1 && c.layers <= 8 && c.hidden >= 64 && c.hidden <= 8192 &&
                       c.head_dim == kHd;
    const bool heads = c.heads >= 1 && c.kv_heads >= 1 && c.heads % c.kv_heads == 0 &&
                       c.heads / c.kv_heads <= 8;
    const bool head = c.block >= 2 && c.block <= InklingForward::kGraphRows && c.rank == 256 && c.inter >= 1;
    return tower && heads && head && c.window >= 1 && c.vocab_padded >= 1;
}

// local_layer_ids and target_layer_ids; the taps must rise.
bool parse_draft_lists(const JValue& mc, InklingDraftConfig& c) {
    c.local.assign(static_cast<size_t>(c.layers), false);
    if (const JValue* v = jobj_find(mc, "local_layer_ids"); v && v->type == JType::ARRAY)
        for (const JValue& e : v->arr)
            if (e.type == JType::NUMBER && e.num_val >= 0 && e.num_val < c.layers)
                c.local[static_cast<size_t>(e.num_val)] = true;
    if (const JValue* v = jobj_find(mc, "target_layer_ids"); v && v->type == JType::ARRAY)
        for (const JValue& e : v->arr)
            c.taps.push_back(e.type == JType::NUMBER ? static_cast<int>(e.num_val) : -1);
    bool ok = c.taps.size() >= 2 && c.taps.size() <= 8;
    for (size_t i = 0; ok && i < c.taps.size(); ++i)
        ok = c.taps[i] >= 1 && (i == 0 || c.taps[i] > c.taps[i - 1]);
    return ok;
}

bool parse_draft_config(const std::string& dir, InklingDraftConfig& c) {
    JValue root;
    if (!parse_json_file(dir + "/config.json", root))
        return false;
    const JValue* mc = jobj_find(root, "mtp_config");
    if (!mc || mc->type != JType::OBJECT)
        return false;
    jobj_opt_int(*mc, "hidden_size", c.hidden);
    jobj_opt_int(*mc, "num_hidden_layers", c.layers);
    jobj_opt_int(*mc, "num_attention_heads", c.heads);
    jobj_opt_int(*mc, "num_key_value_heads", c.kv_heads);
    jobj_opt_int(*mc, "head_dim", c.head_dim);
    jobj_opt_int(*mc, "intermediate_size", c.inter);
    jobj_opt_int(*mc, "block_size", c.block);
    jobj_opt_int(*mc, "markov_rank", c.rank);
    jobj_opt_int(*mc, "sliding_window_size", c.window);
    c.theta = json_float(*mc, "rope_theta", c.theta);
    c.eps = json_float(*mc, "rms_norm_eps", c.eps);
    const JValue* tc = jobj_find(root, "text_config");
    if (tc && tc->type == JType::OBJECT)
        jobj_opt_int(*tc, "vocab_size", c.vocab_padded);
    return draft_dims_ok(c) && parse_draft_lists(*mc, c);
}

template <class LayerT>
void load_draft_layer(Bf16Loader& L, LayerT& ly, const std::string& p, const InklingDraftConfig& c) {
    const int64_t H = c.hidden, QD = static_cast<int64_t>(c.heads) * kHd,
                  KD = static_cast<int64_t>(c.kv_heads) * kHd;
    ly.in_norm = L.f32(p + "input_layernorm.weight", H);
    ly.post_norm = L.f32(p + "post_attention_layernorm.weight", H);
    ly.dflare = L.f32(p + "dflare_norm.weight", H);
    ly.q_norm = L.f32(p + "self_attn.q_norm.weight", kHd);
    ly.k_norm = L.f32(p + "self_attn.k_norm.weight", kHd);
    ly.k_norm_ctx = L.f32(p + "self_attn.k_norm_ctx.weight", kHd);
    // Stacked rows: [q; k; v], [k_ctx; v_ctx], [gate; up].
    ly.qkv = static_cast<half*>(L.dev(static_cast<size_t>((QD + 2 * KD) * H) * sizeof(half)));
    L.f16_into(p + "self_attn.q_proj.weight", QD * H, ly.qkv);
    L.f16_into(p + "self_attn.k_proj.weight", KD * H, ly.qkv ? ly.qkv + QD * H : nullptr);
    L.f16_into(p + "self_attn.v_proj.weight", KD * H, ly.qkv ? ly.qkv + (QD + KD) * H : nullptr);
    ly.kv_ctx = static_cast<half*>(L.dev(static_cast<size_t>(2 * KD * H) * sizeof(half)));
    L.f16_into(p + "self_attn.k_proj_ctx.weight", KD * H, ly.kv_ctx);
    L.f16_into(p + "self_attn.v_proj_ctx.weight", KD * H, ly.kv_ctx ? ly.kv_ctx + KD * H : nullptr);
    ly.wo = L.f16(p + "self_attn.o_proj.weight", H * QD);
    const int64_t I = c.inter;
    ly.gate_up = static_cast<half*>(L.dev(static_cast<size_t>(2 * I * H) * sizeof(half)));
    L.f16_into(p + "mlp.gate_proj.weight", I * H, ly.gate_up);
    L.f16_into(p + "mlp.up_proj.weight", I * H, ly.gate_up ? ly.gate_up + I * H : nullptr);
    ly.down = L.f16(p + "mlp.down_proj.weight", H * I);
}

// fuse_weight [layers + 1][taps] -> softmax per row.
std::vector<float> fuse_alpha(std::vector<float> fw, size_t taps) {
    for (size_t r = 0; r + taps <= fw.size(); r += taps) {
        const float m = *std::max_element(fw.begin() + r, fw.begin() + r + taps);
        float sum = 0.0f;
        for (size_t t = 0; t < taps; ++t)
            sum += fw[r + t] = std::exp(fw[r + t] - m);
        for (size_t t = 0; t < taps; ++t)
            fw[r + t] /= sum;
    }
    return fw;
}

}  // namespace

// The block projections as I8B32; an fp16 one is returned once its copy exists (no room: it stays).
bool InklingDraftWeights::quantize_blocks_(cudaStream_t s) {
    const InklingDraftConfig& c = cfg;
    const int64_t H = c.hidden, QD = static_cast<int64_t>(c.heads) * kHd,
                  KD = static_cast<int64_t>(c.kv_heads) * kHd;
    const auto to8 = [&](half*& w, int64_t rows, int64_t K, Q8& o) {
        const size_t n = static_cast<size_t>(rows * K);
        if (!w || K % 256 != 0)
            return;
        auto* q = static_cast<int8_t*>(va->allocate(n, "inkling_draft"));
        auto* sc = static_cast<half*>(va->allocate(n / 32 * sizeof(half), "inkling_draft"));
        bool done = q != nullptr && sc != nullptr;
        if (done) {
            quantize_i8b32(w, q, sc, static_cast<size_t>(rows), static_cast<int>(K), s);
            done = cudaStreamSynchronize(s) == cudaSuccess;
        }
        if (!done) {
            if (q)
                va->free(q);
            if (sc)
                va->free(sc);
            return;
        }
        owned.push_back(q);
        owned.push_back(sc);
        o.q = q, o.scales = sc;
        owned.erase(std::remove(owned.begin(), owned.end(), static_cast<void*>(w)), owned.end());
        va->free(w);
        w = nullptr;
    };
    for (Layer& ly : layers) {
        to8(ly.qkv, QD + (2 * KD), H, ly.qkv8);
        to8(ly.wo, H, QD, ly.wo8);
        to8(ly.gate_up, 2 * static_cast<int64_t>(c.inter), H, ly.gate_up8);
        to8(ly.down, H, c.inter, ly.down8);
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

InklingDraftWeights::~InklingDraftWeights() {
    for (void* p : owned)
        va->free(p);
}

bool InklingDraftWeights::load(const std::string& dir, VRAMAllocator& va_in, const float* embed_norm_in,
                               const InklingHeadView& head_in, cudaStream_t s) {
    va = &va_in;
    embed_norm = embed_norm_in;
    head = head_in;
    InklingDraftConfig& c = cfg;
    if (!embed_norm || !head.w8 || !head.row_scales || head.vocab < 1 || !parse_draft_config(dir, c)) {
        IMP_LOG_WARN("inkling draft: no usable mtp_config or no FP8 view of the LM head");
        return false;
    }
    Bf16Loader L{{}, va_in, owned, "inkling_draft"};
    void* base = nullptr;
    size_t size = 0;
    if (!load_raw_safetensors_file(dir + "/mtp.safetensors", L.t, base, size)) {
        IMP_LOG_WARN("inkling draft: %s/mtp.safetensors not readable", dir.c_str());
        return false;
    }
    const int64_t H = c.hidden, r = c.rank;
    const size_t taps = c.taps.size();
    layers.resize(static_cast<size_t>(c.layers));
    for (int i = 0; i < c.layers && L.ok; ++i)
        load_draft_layer(L, layers[static_cast<size_t>(i)], "layers." + std::to_string(i) + ".", c);
    norm = L.f32("norm.weight", H);
    mask = L.f16("mask_embed", H);
    alpha = L.upload(
        fuse_alpha(L.host("fuse_weight", static_cast<int64_t>(c.layers + 1) * static_cast<int64_t>(taps)),
                   taps));
    markov_dflare = L.f32("markov_head.dflare_norm.weight", H);
    w1 = L.f16("markov_head.markov_w1.weight", static_cast<int64_t>(c.vocab_padded) * r);
    rnn_b = L.f32("markov_head.rnn_proj.bias", 3 * r);
    {
        // rnn_proj [3r, 2r + 2H]: columns [state, previous token | hidden, f_rnn].
        const std::vector<float> rp = L.host("markov_head.rnn_proj.weight", 3 * r * (2 * r + 2 * H));
        std::vector<float> ws(static_cast<size_t>(3 * r * 2 * r)), wc(static_cast<size_t>(3 * r * 2 * H));
        for (int64_t row = 0; row < 3 * r; ++row) {
            const float* src = rp.data() + row * (2 * r + 2 * H);
            std::copy(src, src + 2 * r, ws.begin() + row * 2 * r);
            std::copy(src + 2 * r, src + 2 * r + 2 * H, wc.begin() + row * 2 * H);
        }
        rnn_state_w = L.upload_f16(ws);
        rnn_ctx_w = L.upload_f16(wc);
    }
    // hidden_out folded through the LM head: logits(hidden + W latent + b) = base + fold_w latent + fold_b.
    const std::vector<float> ho = L.host("markov_head.hidden_out.weight", H * r);
    const std::vector<float> hb = L.host("markov_head.hidden_out.bias", H);
    const size_t V = static_cast<size_t>(head.vocab);
    fold_w = static_cast<half*>(L.dev(V * static_cast<size_t>(r) * sizeof(half)));
    fold_b = static_cast<float*>(L.dev(V * sizeof(float)));
    bool ok = L.ok;
    if (ok) {
        const int chunk = 8;
        std::vector<half> xr(static_cast<size_t>(chunk) * static_cast<size_t>(H));
        // Scratch for the fold only, returned before load() ends.
        auto* d_x = static_cast<half*>(va_in.allocate(xr.size() * sizeof(half), "inkling_draft"));
        auto* d_t = static_cast<half*>(
            va_in.allocate(static_cast<size_t>(chunk) * V * sizeof(half), "inkling_draft"));
        ok = d_x != nullptr && d_t != nullptr;
        for (int64_t j0 = 0; ok && j0 <= r; j0 += chunk) {
            // Rows = columns j0.. of hidden_out; one more pass (j0 == r) carries the bias.
            const int rows = j0 == r ? 1 : chunk;
            for (int row = 0; row < rows; ++row)
                for (int64_t i = 0; i < H; ++i)
                    xr[static_cast<size_t>(row * H + i)] = __float2half(
                        j0 == r ? hb[static_cast<size_t>(i)] : ho[static_cast<size_t>(i * r + j0 + row)]);
            ok = cudaMemcpyAsync(d_x, xr.data(), xr.size() * sizeof(half), cudaMemcpyHostToDevice, s) ==
                     cudaSuccess &&
                 gemv_fp8_rowscale_f16(head.w8, head.row_scales, d_x, d_t, head.vocab, c.hidden, rows, s);
            if (ok) {
                draft_fold_k<<<blocks(static_cast<size_t>(rows) * V), kT, 0, s>>>(
                    d_t, j0 == r ? nullptr : fold_w, fold_b, rows, head.vocab, c.rank, static_cast<int>(j0));
                IMP_CUDA_CHECK_LAUNCH();
                ok = cudaStreamSynchronize(s) == cudaSuccess;
            }
        }
        if (d_x)
            va_in.free(d_x);
        if (d_t)
            va_in.free(d_t);
    }
    ok = ok && quantize_blocks_(s);
    munmap(base, size);
    if (ok)
        IMP_LOG_INFO("inkling draft: %d layers, block %d, %zu feature planes, Markov rank %d", c.layers,
                     c.block, taps, c.rank);
    return ok;
}

InklingDraft::~InklingDraft() {
    for (void* p : owned_)
        va_->free(p);
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
    void* p = va_->allocate(bytes, "inkling_draft_ctx");
    if (!p) {
        IMP_LOG_WARN("inkling draft: context allocation of %.1f MiB failed", bytes / 1048576.0);
        ok_ = false;
        return nullptr;
    }
    owned_.push_back(p);
    return p;
}

void InklingDraft::lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s) {
    const int64_t as[2] = {T, K}, bs[2] = {N, K}, cs[2] = {T, N};
    const Tensor A(const_cast<half*>(x), QType::F16, 2, as, true);
    const Tensor B(const_cast<half*>(w), QType::F16, 2, bs, true);
    Tensor C(y, QType::F16, 2, cs, true);
    gemm(A, B, C, 1.0f, 0.0f, s);
}

bool InklingDraft::init(const InklingDraftWeights& w, VRAMAllocator& va, int max_ctx, int cap, int back) {
    w_ = &w;
    va_ = &va;
    max_ctx_ = max_ctx;
    cap_ = cap;
    // A local layer reads window + block rows: its K/V live in a ring that also keeps the back positions a
    // restart may return to and the cap rows of one call. Not worth it below twice the ring.
    const int ring = w.cfg.window + w.cfg.block + cap + std::max(back, 0);
    ring_ = back >= 0 && 2 * static_cast<long long>(ring) < max_ctx + w.cfg.block ? ring : 0;
    const InklingDraftConfig& c = w.cfg;
    const size_t H = static_cast<size_t>(c.hidden), B = static_cast<size_t>(c.block), taps = c.taps.size();
    const size_t QD = static_cast<size_t>(c.heads) * kHd, KD = static_cast<size_t>(c.kv_heads) * kHd;
    const size_t V = static_cast<size_t>(w.head.vocab), I = static_cast<size_t>(c.inter),
                 C = static_cast<size_t>(cap);
    const auto F = [&](size_t n) { return static_cast<float*>(alloc_(n * sizeof(float))); };
    const auto Hf = [&](size_t n) { return static_cast<half*>(alloc_(n * sizeof(half))); };
    for (int i = 0; i < c.layers; ++i) {
        const size_t rows = ring_ > 0 && c.local[static_cast<size_t>(i)] ? 2 * static_cast<size_t>(ring_)
                                                                         : static_cast<size_t>(max_ctx) + B;
        k_.push_back(Hf(rows * KD));
        v_.push_back(Hf(rows * KD));
    }
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
    return h;
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
        const size_t nk = static_cast<size_t>(n) * KD, at = static_cast<size_t>(pos0) * KD;
        if (ring_ > 0 && c.local[static_cast<size_t>(i)]) {
            draft_ring_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, 0, KD, k_[static_cast<size_t>(i)], nk, pos0,
                                                   ring_);
            IMP_CUDA_CHECK_LAUNCH();
            draft_ring_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, KD, KD, v_[static_cast<size_t>(i)], nk, pos0,
                                                   ring_);
            IMP_CUDA_CHECK_LAUNCH();
            continue;
        }
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, 0, KD, k_[static_cast<size_t>(i)] + at, nk,
                                               nullptr);
        IMP_CUDA_CHECK_LAUNCH();
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(kvc_, 2 * KD, KD, KD, v_[static_cast<size_t>(i)] + at, nk,
                                               nullptr);
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
    if (!ok_ || n < 1 || n > cap_ || pos0 < 0 || pos0 + n > max_ctx_)
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
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(qkv_, QD + 2 * KD, QD, KD, kc, nk, d_row);
        IMP_CUDA_CHECK_LAUNCH();
        draft_cols_k<<<blocks(nk), kT, 0, s>>>(qkv_, QD + 2 * KD, QD + KD, KD, vc, nk, d_row);
        IMP_CUDA_CHECK_LAUNCH();
        if (dspark_attention_dev(q_, kc, vc, o_, work_, B, c.heads, c.kv_heads, local ? d_ctx_local : d_pos,
                                 (local ? std::min(c.window, max_ctx_) : max_ctx_) + B, scale, s,
                                 local ? d_first_local : nullptr) != cudaSuccess)
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
