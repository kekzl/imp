// Inkling2 DSpark draft weights: mtp_config, mtp.safetensors, the I8B32 block projections.

#include "exec/inkling_draft.h"

#include "compute/gemv_i8b32.h"

#include "exec/inkling_draft_head.h"

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
#include <cstddef>

namespace imp {

using namespace inkling;

namespace {

constexpr int kT = 256;
constexpr int kHd = 128;

unsigned blocks(size_t n) { return static_cast<unsigned>((n + kT - 1) / kT); }

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

bool parse_inkling_draft_config(const std::string& dir, InklingDraftConfig& c) {
    return parse_draft_config(dir, c);
}

// The block projections as I8B32; an fp16 one is returned once its copy exists (no room: it stays).
bool InklingDraftWeights::quantize_blocks_(cudaStream_t s) {
    const InklingDraftConfig& c = cfg;
    const int64_t H = c.hidden, QD = static_cast<int64_t>(c.heads) * kHd,
                  KD = static_cast<int64_t>(c.kv_heads) * kHd;
    const auto to8 = [&](half*& w, int64_t rows, int64_t K, Q8& o) {
        const size_t n = static_cast<size_t>(rows * K);
        if (!w || K % 256 != 0)
            return;
        VramOwned<std::byte> q_own(*va, n, "inkling_draft");
        VramOwned<std::byte> sc_own(*va, n / 32 * sizeof(half), "inkling_draft");
        auto* q = reinterpret_cast<int8_t*>(q_own.get());
        auto* sc = reinterpret_cast<half*>(sc_own.get());
        bool done = q != nullptr && sc != nullptr;
        if (done) {
            quantize_i8b32(w, q, sc, static_cast<size_t>(rows), static_cast<int>(K), s);
            done = cudaStreamSynchronize(s) == cudaSuccess;
        }
        if (!done) {
            q_own.reset();
            sc_own.reset();
            return;
        }
        owned.push_back(std::move(q_own));
        owned.push_back(std::move(sc_own));
        o.q = q, o.scales = sc;
        // I5: the fp16 block leaves the list through its owner (every w here came from Bf16Loader::dev).
        const auto it = std::find_if(owned.begin(), owned.end(), [&](const VramOwned<std::byte>& b) {
            return static_cast<const void*>(b.get()) == static_cast<const void*>(w);
        });
        if (it != owned.end()) {
            it->reset();
            owned.erase(it);
        }
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
    for (auto& b : owned)
        b.reset();
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
        VramOwned<half> x_own(va_in, xr.size(), "inkling_draft");
        VramOwned<half> t_own(va_in, static_cast<size_t>(chunk) * V, "inkling_draft");
        half* const d_x = x_own.get();
        half* const d_t = t_own.get();
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
        x_own.reset();
        t_own.reset();
    }
    ok = ok && quantize_blocks_(s);
    munmap(base, size);
    if (ok)
        IMP_LOG_INFO("inkling draft: %d layers, block %d, %zu feature planes, Markov rank %d", c.layers,
                     c.block, taps, c.rank);
    return ok;
}

}  // namespace imp
