// Inkling2 vision tower (inkling_vision.h): fp32 residual stream, fp16 at the linear layers,
// attention per head through two GEMMs (scores, values) around a row softmax.

#include "vision/inkling_vision.h"

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
#include <cstring>
#include <filesystem>
#include <unordered_map>

namespace imp {

namespace {

constexpr int kT = 256;

__device__ float block_sum(float v, float* sh) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (unsigned i = 0; i < blockDim.x / 32; ++i)
        t += sh[i];
    __syncthreads();
    return t;
}

// Row 0 = class token, row 1 + p = patch embedding p + its position embedding.
__global__ void vis_embed_k(const half* pe, const float* pos, const float* cls, float* x, size_t n, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const size_t t = i / H, c = i % H;
    x[i] = t == 0 ? cls[c] : __half2float(pe[(t - 1) * H + c]) + pos[(t - 1) * H + c];
}

// LayerNorm of one row per block; in place when out_f == x.
__global__ void vis_ln_k(const float* x, const float* w, const float* b, float* out_f, half* out_h, int H,
                         float eps) {
    __shared__ float sh[kT / 32];
    const float* r = x + (size_t)blockIdx.x * H;
    float s = 0.0f;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        s += r[c];
    const float mean = block_sum(s, sh) / H;
    float v = 0.0f;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        v += (r[c] - mean) * (r[c] - mean);
    const float inv = rsqrtf(block_sum(v, sh) / H + eps);
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        const float y = (r[c] - mean) * inv * w[c] + b[c];
        if (out_h)
            out_h[(size_t)blockIdx.x * H + c] = __float2half(y);
        if (out_f)
            out_f[(size_t)blockIdx.x * H + c] = y;
    }
}

// qkv [N, 3H] + bias -> q, k [heads][N][hd] with the axial rotation (q also scaled by hd^-0.5) and
// vt [heads][hd][N]. cs: [N][hd] as (cos, sin) per channel pair. One thread per (token, pair).
__global__ void vis_qkv_k(const half* qkv, const float* bias, const float* cs, half* q, half* k, half* vt,
                          int N, int H, int hd, float scale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = H / 2;
    if (i >= (size_t)N * pairs)
        return;
    const int t = (int)(i / pairs), p = (int)(i % pairs);
    const int head = p / (hd / 2), c = 2 * p;
    const size_t d = (size_t)(p % (hd / 2)) * 2;  // first channel of the pair inside its head
    const half* row = qkv + (size_t)t * 3 * H;
    const float co = cs[(size_t)t * hd + d], si = cs[(size_t)t * hd + d + 1];
    const float q0 = __half2float(row[c]) + bias[c], q1 = __half2float(row[c + 1]) + bias[c + 1];
    const float k0 = __half2float(row[H + c]) + bias[H + c],
                k1 = __half2float(row[H + c + 1]) + bias[H + c + 1];
    const size_t at = ((size_t)head * N + t) * hd + d;
    q[at] = __float2half((q0 * co - q1 * si) * scale);
    q[at + 1] = __float2half((q1 * co + q0 * si) * scale);
    k[at] = __float2half(k0 * co - k1 * si);
    k[at + 1] = __float2half(k1 * co + k0 * si);
    const size_t vat = ((size_t)head * hd + d) * N + t;
    vt[vat] = __float2half(__half2float(row[2 * H + c]) + bias[2 * H + c]);
    vt[vat + N] = __float2half(__half2float(row[2 * H + c + 1]) + bias[2 * H + c + 1]);
}

// Softmax of one score row per block, in place.
__global__ void vis_softmax_k(half* s, int N) {
    __shared__ float sh[kT / 32];
    half* r = s + (size_t)blockIdx.x * N;
    float m = -1e30f;
    for (int c = threadIdx.x; c < N; c += blockDim.x)
        m = fmaxf(m, __half2float(r[c]));
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = m;
    __syncthreads();
    for (unsigned i = 0; i < blockDim.x / 32; ++i)
        m = fmaxf(m, sh[i]);
    __syncthreads();
    float sum = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x)
        sum += expf(__half2float(r[c]) - m);
    const float inv = 1.0f / block_sum(sum, sh);
    for (int c = threadIdx.x; c < N; c += blockDim.x)
        r[c] = __float2half(expf(__half2float(r[c]) - m) * inv);
}

// o [heads][N][hd] -> y [N][H].
__global__ void vis_join_k(const half* o, half* y, int N, int H, int hd) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * H)
        return;
    const size_t t = i / H, c = i % H;
    y[i] = o[((c / hd) * N + t) * hd + c % hd];
}

// x += gamma * (y + bias).
__global__ void vis_residual_k(float* x, const half* y, const float* bias, const float* gamma, size_t n,
                               int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const size_t c = i % H;
    x[i] += gamma[c] * (__half2float(y[i]) + bias[c]);
}

// y = gelu(y + bias) (erf form); bias may be null.
__global__ void vis_gelu_k(half* y, const float* bias, size_t n, int C) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float v = __half2float(y[i]) + (bias ? bias[i % C] : 0.0f);
    y[i] = __float2half(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
}

unsigned blocks(size_t n) { return static_cast<unsigned>((n + kT - 1) / kT); }

// Source index and weight of bilinear resampling without aligned corners (F.interpolate).
void axis_tap(int i, int size, int side, int& i0, int& i1, float& w1) {
    const float src = std::max(
        (static_cast<float>(i) + 0.5f) * static_cast<float>(side) / static_cast<float>(size) - 0.5f, 0.0f);
    i0 = std::min(static_cast<int>(src), side - 1);
    i1 = std::min(i0 + 1, side - 1);
    w1 = src - static_cast<float>(i0);
}

}  // namespace

InklingVision::~InklingVision() {
    for (void* p : owned_)
        va_->free(p);
}

void InklingVision::grid(int h, int w, int& rows, int& cols) const {
    const int px = cfg_.token_px();
    rows = (h + px - 1) / px;
    cols = (w + px - 1) / px;
}

void InklingVision::lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s) {
    const int64_t as[2] = {T, K}, bs[2] = {N, K}, cs[2] = {T, N};
    const Tensor A(const_cast<half*>(x), QType::F16, 2, as, true);
    const Tensor B(const_cast<half*>(w), QType::F16, 2, bs, true);
    Tensor C(y, QType::F16, 2, cs, true);
    gemm(A, B, C, 1.0f, 0.0f, s);
}

namespace {

bool vision_config_ok(const InklingVisionConfig& c) {
    const int H = c.hidden, hd = c.heads > 0 ? H / c.heads : 0;
    const bool tower = c.layers >= 1 && c.layers <= 64 && H >= 64 && H <= 4096 && H % 2 == 0 &&
                       c.heads >= 1 && hd * c.heads == H && hd % 4 == 0;
    // The pixel staging in encode() fits 3 * patch^2 floats into 2 * hidden.
    const bool patches = c.merge == 2 && c.patch >= 1 && 3 * c.patch * c.patch <= 2 * H && c.grid >= 1 &&
                         c.grid <= 256;
    return tower && patches && c.proj_hidden == c.merge * c.merge * H && c.out >= 1;
}

bool parse_vision_config(const std::string& dir, InklingVisionConfig& c) {
    JValue root;
    if (!parse_json_file(dir + "/config.json", root))
        return false;
    const JValue* vc = jobj_find(root, "vision_config");
    if (!vc || vc->type != JType::OBJECT)
        return false;
    jobj_opt_int(*vc, "num_hidden_layers", c.layers);
    jobj_opt_int(*vc, "hidden_size", c.hidden);
    jobj_opt_int(*vc, "num_attention_heads", c.heads);
    jobj_opt_int(*vc, "native_patch_size", c.patch);
    jobj_opt_int(*vc, "adapter_merge_size", c.merge);
    jobj_opt_int(*vc, "posemb_grid_size", c.grid);
    jobj_opt_int(*vc, "projection_hidden_dim", c.proj_hidden);
    jobj_opt_int(*vc, "text_hidden_size", c.out);
    if (!vision_config_ok(c)) {
        IMP_LOG_ERROR("inkling vision: vision_config out of range");
        return false;
    }
    return true;
}

std::string model_file(const std::string& dir) {
    std::string file;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("model-", 0) == 0 && n.size() > 12 && n.substr(n.size() - 12) == ".safetensors")
            file = e.path().string();
    }
    return file;
}

// One encoder layer under prefix p; q, k, v weights and biases stacked in that order.
template <class LayerT>
void load_layer(Bf16Loader& L, LayerT& ly, const std::string& p, int H) {
    const int64_t HH = static_cast<int64_t>(H) * H, I = 4 * static_cast<int64_t>(H);
    ly.qkv = static_cast<half*>(L.dev(static_cast<size_t>(3 * HH) * sizeof(half)));
    std::vector<float> qb;
    int64_t part = 0;
    for (const char* n : {"q_proj", "k_proj", "v_proj"}) {
        L.f16_into(p + "self_attn." + n + ".weight", HH, ly.qkv ? ly.qkv + part++ * HH : nullptr);
        const std::vector<float> b = L.host(p + "self_attn." + n + ".bias", H);
        qb.insert(qb.end(), b.begin(), b.end());
    }
    ly.qkv_b = L.upload(qb);
    ly.wo = L.f16(p + "self_attn.out_proj.weight", HH);
    ly.wo_b = L.f32(p + "self_attn.out_proj.bias", H);
    ly.fc1 = L.f16(p + "mlp.c_fc.weight", I * H);
    ly.fc1_b = L.f32(p + "mlp.c_fc.bias", I);
    ly.fc2 = L.f16(p + "mlp.c_proj.weight", I * H);
    ly.fc2_b = L.f32(p + "mlp.c_proj.bias", H);
    ly.ln1_w = L.f32(p + "ln_1.weight", H);
    ly.ln1_b = L.f32(p + "ln_1.bias", H);
    ly.ln2_w = L.f32(p + "ln_2.weight", H);
    ly.ln2_b = L.f32(p + "ln_2.bias", H);
    ly.ls1 = L.f32(p + "ls_1.gamma", H);
    ly.ls2 = L.f32(p + "ls_2.gamma", H);
}

}  // namespace

bool InklingVision::alloc_workspace_() {
    // P patches, N = P + 1 rows.
    const InklingVisionConfig& c = cfg_;
    const size_t P = static_cast<size_t>(kMaxTokens) * 4, N = P + 1, Hs = static_cast<size_t>(c.hidden);
    bool ok = true;
    const auto take = [&](size_t bytes) {
        void* p = va_->allocate(bytes, "inkling_vision_ws");
        ok = ok && p != nullptr;
        if (p)
            owned_.push_back(p);
        return p;
    };
    const auto F = [&](size_t n) { return static_cast<float*>(take(n * sizeof(float))); };
    const auto Hf = [&](size_t n) { return static_cast<half*>(take(n * sizeof(half))); };
    x_ = F(N * Hs), pos_ = F(P * Hs), cs_ = F(N * (Hs / static_cast<size_t>(c.heads)));
    h16_ = Hf(N * Hs), qkv_ = Hf(N * 3 * Hs), q_ = Hf(N * Hs), k_ = Hf(N * Hs), vt_ = Hf(N * Hs);
    s_ = Hf(N * N), o_ = Hf(N * Hs), y_ = Hf(N * Hs), fc_ = Hf(N * 4 * Hs), pe_ = Hf(P * Hs);
    px_ = Hf(P * 3 * static_cast<size_t>(c.patch) * c.patch);
    m1_ = Hf(static_cast<size_t>(kMaxTokens) * c.proj_hidden);
    if (!ok)
        IMP_LOG_ERROR("inkling vision: workspace for %d tokens does not fit", kMaxTokens);
    return ok;
}

bool InklingVision::load(const std::string& dir, VRAMAllocator& va) {
    va_ = &va;
    InklingVisionConfig& c = cfg_;
    if (!parse_vision_config(dir, c))
        return false;
    const int H = c.hidden;
    Bf16Loader L{{}, va, owned_, "inkling_vision"};
    void* base = nullptr;
    size_t size = 0;
    const std::string file = model_file(dir);
    if (file.empty() || !load_raw_safetensors_file(file, L.t, base, size)) {
        IMP_LOG_ERROR("inkling vision: no readable model-*.safetensors in %s", dir.c_str());
        return false;
    }
    const std::string v = "model.visual.", bb = v + "backbone.";
    patch_w_ = L.f16(bb + "embeddings.patch_embedding.weight",
                     static_cast<int64_t>(H) * 3 * c.patch * c.patch);
    pos_table_ = L.host(bb + "embeddings.position_embedding", static_cast<int64_t>(c.grid) * c.grid * H);
    {
        std::vector<float> cls = L.host(bb + "embeddings.class_embedding", H);
        const std::vector<float> cp = L.host(bb + "embeddings.cls_position_embedding", H);
        for (int i = 0; i < H; ++i)
            cls[static_cast<size_t>(i)] += cp[static_cast<size_t>(i)];
        cls_ = L.upload(cls);
    }
    pre_w_ = L.f32(bb + "ln_pre.weight", H);
    pre_b_ = L.f32(bb + "ln_pre.bias", H);
    layers_.resize(static_cast<size_t>(c.layers));
    for (int i = 0; i < c.layers && L.ok; ++i)
        load_layer(L, layers_[static_cast<size_t>(i)], bb + "layers." + std::to_string(i) + ".", H);
    proj_in_ = L.f16(v + "proj_in.weight", static_cast<int64_t>(c.proj_hidden) * c.proj_hidden);
    proj_out_ = L.f16(v + "proj_out.weight", static_cast<int64_t>(c.out) * c.proj_hidden);
    const bool ok = L.ok && alloc_workspace_() && cudaDeviceSynchronize() == cudaSuccess;
    munmap(base, size);
    if (ok)
        IMP_LOG_INFO("inkling vision: %d layers, hidden %d, %d px per token, up to %d tokens per image",
                     c.layers, H, c.token_px(), kMaxTokens);
    return ok;
}

bool InklingVision::encode(const uint8_t* rgb, int h, int w, half* out, cudaStream_t s) {
    const InklingVisionConfig& c = cfg_;
    int rows = 0, cols = 0;
    grid(h, w, rows, cols);
    const int n_tok = rows * cols;
    if (!rgb || h < 1 || w < 1 || n_tok > kMaxTokens || layers_.empty()) {
        IMP_LOG_ERROR("inkling vision: image %d x %d (%d tokens) not servable, at most %d tokens", w, h,
                      n_tok, kMaxTokens);
        return false;
    }
    const int H = c.hidden, hd = H / c.heads, ps = c.patch, m = c.merge;
    const int P = n_tok * m * m, N = P + 1, in = 3 * ps * ps, Hp = rows * m, Wp = cols * m;
    // Host staging. Patch p of token (tr, tc): block-major (tr, tc, mr, mc); pixel order (y, x, channel);
    // a pixel outside the image is -1 on the 0..255 scale, then x / 127.5 - 1.
    h_px_.resize(static_cast<size_t>(P) * in);
    h_pos_.resize(static_cast<size_t>(P) * H);
    h_cs_.assign(static_cast<size_t>(N) * hd, 0.0f);
    for (int j = 0; j < hd / 2; ++j)
        h_cs_[static_cast<size_t>(j) * 2] = 1.0f;  // class token: position (0, 0)
    std::vector<float> inv(static_cast<size_t>(hd / 4));
    for (size_t j = 0; j < inv.size(); ++j)
        inv[j] = 1.0f /
                 std::pow(c.rope_theta, 2.0f * static_cast<float>(j) / (0.5f * static_cast<float>(hd)));
    for (int p = 0; p < P; ++p) {
        const int tok = p / (m * m), mr = (p / m) % m, mc = p % m;
        const int pr = (tok / cols) * m + mr, pc = (tok % cols) * m + mc;  // patch grid coordinates
        float* dst = &h_px_[static_cast<size_t>(p) * in];
        for (int y = 0; y < ps; ++y)
            for (int x = 0; x < ps; ++x) {
                const int iy = pr * ps + y, ix = pc * ps + x;
                for (int ch = 0; ch < 3; ++ch) {
                    const float v = (iy < h && ix < w)
                                        ? static_cast<float>(rgb[(static_cast<size_t>(iy) * w + ix) * 3 + ch])
                                        : -1.0f;
                    dst[(y * ps + x) * 3 + ch] = v / 127.5f - 1.0f;
                }
            }
        int r0, r1, c0, c1;
        float wr, wc;
        axis_tap(pr, Hp, c.grid, r0, r1, wr);
        axis_tap(pc, Wp, c.grid, c0, c1, wc);
        const float* t00 = &pos_table_[(static_cast<size_t>(r0) * c.grid + c0) * H];
        const float* t01 = &pos_table_[(static_cast<size_t>(r0) * c.grid + c1) * H];
        const float* t10 = &pos_table_[(static_cast<size_t>(r1) * c.grid + c0) * H];
        const float* t11 = &pos_table_[(static_cast<size_t>(r1) * c.grid + c1) * H];
        float* pd = &h_pos_[static_cast<size_t>(p) * H];
        for (int i = 0; i < H; ++i)
            pd[i] = (1 - wr) * ((1 - wc) * t00[i] + wc * t01[i]) + wr * ((1 - wc) * t10[i] + wc * t11[i]);
        // RoPE: channel pairs [0, hd/4) turn with the column, [hd/4, hd/2) with the row, both 1-based.
        float* cd = &h_cs_[static_cast<size_t>(p + 1) * hd];
        for (int j = 0; j < hd / 2; ++j) {
            const float a = static_cast<float>(j < hd / 4 ? pc + 1 : pr + 1) *
                            inv[static_cast<size_t>(j % (hd / 4))];
            cd[static_cast<size_t>(j) * 2] = std::cos(a);
            cd[static_cast<size_t>(j) * 2 + 1] = std::sin(a);
        }
    }
    const size_t nH = static_cast<size_t>(N) * H, pH = static_cast<size_t>(P) * H;
    bool ok = cudaMemcpyAsync(pos_, h_pos_.data(), pH * sizeof(float), cudaMemcpyHostToDevice, s) ==
                  cudaSuccess &&
              cudaMemcpyAsync(cs_, h_cs_.data(), h_cs_.size() * sizeof(float), cudaMemcpyHostToDevice, s) ==
                  cudaSuccess &&
              cudaMemcpyAsync(fc_, h_px_.data(), h_px_.size() * sizeof(float), cudaMemcpyHostToDevice, s) ==
                  cudaSuccess;
    if (!ok)
        return false;
    // fc_ ([N, 4H] fp16 = 2 N H floats) stages the fp32 pixels (P * 3 * patch^2 floats, patch <= 26).
    inkling::f32_to_f16(reinterpret_cast<const float*>(fc_), px_, h_px_.size(), s);
    lin_(px_, P, in, patch_w_, H, pe_, s);
    vis_embed_k<<<blocks(nH), kT, 0, s>>>(pe_, pos_, cls_, x_, nH, H);
    IMP_CUDA_CHECK_LAUNCH();
    vis_ln_k<<<N, kT, 0, s>>>(x_, pre_w_, pre_b_, x_, nullptr, H, c.eps);
    IMP_CUDA_CHECK_LAUNCH();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (const Layer& ly : layers_) {
        vis_ln_k<<<N, kT, 0, s>>>(x_, ly.ln1_w, ly.ln1_b, nullptr, h16_, H, c.eps);
        IMP_CUDA_CHECK_LAUNCH();
        lin_(h16_, N, H, ly.qkv, 3 * H, qkv_, s);
        vis_qkv_k<<<blocks(nH / 2), kT, 0, s>>>(qkv_, ly.qkv_b, cs_, q_, k_, vt_, N, H, hd, scale);
        IMP_CUDA_CHECK_LAUNCH();
        for (int head = 0; head < c.heads; ++head) {
            const size_t ho = static_cast<size_t>(head) * N * hd;
            lin_(q_ + ho, N, hd, k_ + ho, N, s_, s);
            vis_softmax_k<<<N, kT, 0, s>>>(s_, N);
            IMP_CUDA_CHECK_LAUNCH();
            lin_(s_, N, N, vt_ + ho, hd, o_ + ho, s);
        }
        vis_join_k<<<blocks(nH), kT, 0, s>>>(o_, y_, N, H, hd);
        IMP_CUDA_CHECK_LAUNCH();
        lin_(y_, N, H, ly.wo, H, h16_, s);
        vis_residual_k<<<blocks(nH), kT, 0, s>>>(x_, h16_, ly.wo_b, ly.ls1, nH, H);
        IMP_CUDA_CHECK_LAUNCH();
        vis_ln_k<<<N, kT, 0, s>>>(x_, ly.ln2_w, ly.ln2_b, nullptr, h16_, H, c.eps);
        IMP_CUDA_CHECK_LAUNCH();
        lin_(h16_, N, H, ly.fc1, 4 * H, fc_, s);
        vis_gelu_k<<<blocks(nH * 4), kT, 0, s>>>(fc_, ly.fc1_b, nH * 4, 4 * H);
        IMP_CUDA_CHECK_LAUNCH();
        lin_(fc_, N, 4 * H, ly.fc2, H, h16_, s);
        vis_residual_k<<<blocks(nH), kT, 0, s>>>(x_, h16_, ly.fc2_b, ly.ls2, nH, H);
        IMP_CUDA_CHECK_LAUNCH();
    }
    // The patch rows, four per token, are the merger's input rows.
    inkling::f32_to_f16(x_ + H, y_, pH, s);
    lin_(y_, n_tok, c.proj_hidden, proj_in_, c.proj_hidden, m1_, s);
    const size_t nm = static_cast<size_t>(n_tok) * c.proj_hidden;
    vis_gelu_k<<<blocks(nm), kT, 0, s>>>(m1_, nullptr, nm, c.proj_hidden);
    IMP_CUDA_CHECK_LAUNCH();
    lin_(m1_, n_tok, c.proj_hidden, proj_out_, c.out, out, s);
    return cudaStreamSynchronize(s) == cudaSuccess;
}

}  // namespace imp
