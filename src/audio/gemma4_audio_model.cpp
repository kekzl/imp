#include "audio/gemma4_audio_model.h"

#include "core/cuda_raii.h"
#include "core/fp_bits.h"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include "model/json_util.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <numbers>

namespace imp {

namespace {

constexpr const char* kTower = "model.audio_tower.";
constexpr const char* kEmbed = "model.embed_audio.";

float scalar_of(const Tensor& t) {
    switch (t.qtype) {
        case QType::BF16:
            return bf16_to_float(*static_cast<const uint16_t*>(t.data));
        case QType::F16:
            return __half2float(*static_cast<const half*>(t.data));
        case QType::F32:
            return *static_cast<const float*>(t.data);
        default:
            return std::nanf("");
    }
}

std::vector<float> to_f32(const Tensor& t) {
    const int64_t n = t.numel();
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        switch (t.qtype) {
            case QType::BF16:
                out[static_cast<size_t>(i)] = bf16_to_float(static_cast<const uint16_t*>(t.data)[i]);
                break;
            case QType::F16:
                out[static_cast<size_t>(i)] = __half2float(static_cast<const half*>(t.data)[i]);
                break;
            default:
                out[static_cast<size_t>(i)] = static_cast<const float*>(t.data)[i];
                break;
        }
    }
    return out;
}

class Fetcher {
public:
    explicit Fetcher(const std::unordered_map<std::string, Tensor>& t) : t_(t) {}

    // shape: expected dims; -1 = any.
    bool get(const std::string& name, Tensor& out, std::initializer_list<int64_t> shape) {
        const auto it = t_.find(name);
        if (it == t_.end()) {
            err_ = "audio tower tensor missing: " + name;
            return false;
        }
        const Tensor& t = it->second;
        const QType q = t.qtype;
        if (q != QType::BF16 && q != QType::F16 && q != QType::F32) {
            err_ = "audio tower tensor " + name + " has unsupported dtype " + qtype_name(q);
            return false;
        }
        bool ok = static_cast<size_t>(t.ndim) == shape.size();
        size_t d = 0;
        for (int64_t s : shape) {
            if (ok && s >= 0 && t.shape[d] != s)
                ok = false;
            ++d;
        }
        if (!ok) {
            err_ = "audio tower tensor " + name + " has an unexpected shape";
            return false;
        }
        out = t;
        return true;
    }

    bool linear(const std::string& base, Gemma4AudioLinear& out, int64_t n_out, int64_t n_in, bool clipped) {
        if (!get(base + ".linear.weight", out.w, {n_out, n_in}))
            return false;
        if (!clipped)
            return true;
        const char* names[] = {".input_min", ".input_max", ".output_min", ".output_max"};
        float* dst[] = {&out.in_min, &out.in_max, &out.out_min, &out.out_max};
        for (int i = 0; i < 4; ++i) {
            const auto it = t_.find(base + names[i]);
            // Scalars are shape [] in the checkpoint (ndim 0, which Tensor::numel() reports as 0).
            if (it == t_.end() || it->second.data == nullptr ||
                (it->second.ndim != 0 && it->second.numel() != 1)) {
                err_ = "audio tower clip bound missing: " + base + names[i];
                return false;
            }
            *dst[i] = scalar_of(it->second);
        }
        return true;
    }

    const std::string& error() const { return err_; }

private:
    const std::unordered_map<std::string, Tensor>& t_;
    std::string err_;
};

// HF Gemma4AudioRelPositionalEncoding: row r = distance (P-1-r), [sin(d * inv_ts) | cos(d * inv_ts)].
std::vector<float> rel_pos_table(const Gemma4AudioConfig& c) {
    const int P = c.rel_positions();
    const int H = c.hidden_size;
    const int half_h = H / 2;
    const double inc = std::log(10000.0) / std::max(half_h - 1, 1);
    std::vector<float> pe(static_cast<size_t>(P) * H);
    for (int r = 0; r < P; ++r) {
        const float pos = static_cast<float>(P - 1 - r);
        for (int i = 0; i < half_h; ++i) {
            // torch: float32 inv_timescales, float32 product
            const float inv = static_cast<float>(std::exp(static_cast<double>(i) * -inc));
            const float t = pos * inv;
            pe[(static_cast<size_t>(r) * H) + i] = std::sin(t);
            pe[(static_cast<size_t>(r) * H) + half_h + i] = std::cos(t);
        }
    }
    return pe;
}

Tensor host_f32(Gemma4AudioModel& m, std::vector<float> v, std::initializer_list<int64_t> shape) {
    m.host_owned.push_back(std::move(v));
    int64_t dims[4] = {};
    int nd = 0;
    for (int64_t s : shape)
        dims[nd++] = s;
    return Tensor(m.host_owned.back().data(), QType::F32, nd, dims, false);
}

void read_audio_keys(const JValue& ac, const JValue& root, Gemma4AudioConfig& c) {
    jobj_opt_int(ac, "hidden_size", c.hidden_size);
    jobj_opt_int(ac, "num_attention_heads", c.num_heads);
    jobj_opt_int(ac, "num_hidden_layers", c.num_layers);
    jobj_opt_int(ac, "attention_chunk_size", c.chunk_size);
    jobj_opt_int(ac, "attention_context_left", c.context_left);
    jobj_opt_int(ac, "attention_context_right", c.context_right);
    jobj_opt_float(ac, "attention_logit_cap", c.logit_cap);
    jobj_opt_int(ac, "conv_kernel_size", c.conv_kernel);
    jobj_opt_int(ac, "output_proj_dims", c.output_proj_dims);
    jobj_opt_float(ac, "residual_weight", c.residual_weight);
    jobj_opt_float(ac, "rms_norm_eps", c.rms_norm_eps);
    jobj_opt_float(ac, "gradient_clipping", c.gradient_clipping);
    int clipped = 1;
    jobj_opt_int(ac, "use_clipped_linears", clipped);
    c.clipped_linears = clipped != 0;
    if (const JValue* ch = jobj_find(ac, "subsampling_conv_channels");
        ch && ch->type == JType::ARRAY && ch->arr.size() == 2) {
        c.sub_channels0 = static_cast<int>(ch->arr[0].num_val);
        c.sub_channels1 = static_cast<int>(ch->arr[1].num_val);
    }
    if (const JValue* tc = jobj_find(root, "text_config"); tc && tc->type == JType::OBJECT)
        jobj_opt_int(*tc, "hidden_size", c.lm_d_model);
    jobj_opt_int(root, "audio_token_id", c.audio_token_id);
    jobj_opt_int(root, "boa_token_id", c.boa_token_id);
    jobj_opt_int(root, "eoa_token_id", c.eoa_token_id);
}

// Bounds the kernels assume (attention warp: head_dim % 32, <= 256; window <= 64) and the
// checkpoint's own counts before they size anything.
bool geometry_ok(const Gemma4AudioConfig& c) {
    const bool attn = c.hidden_size > 0 && c.hidden_size <= 8192 && c.num_heads > 0 &&
                      c.hidden_size % c.num_heads == 0 && c.head_dim() % 32 == 0 && c.head_dim() <= 256;
    const bool window = c.chunk_size > 0 && c.context_left >= 2 && c.context_left <= 64 &&
                        c.context_right == 0;
    const bool rest = c.num_layers > 0 && c.num_layers <= 64 && c.conv_kernel > 0 && c.conv_kernel <= 16;
    const bool sub = c.sub_channels0 > 0 && c.sub_channels0 % 4 == 0 && c.sub_channels1 > 0;
    return attn && window && rest && sub && c.output_proj_dims > 0 && c.lm_d_model > 0;
}

bool load_ffw(Fetcher& f, const std::string& p, Gemma4AudioFfw& w, int64_t H, bool clip) {
    return f.get(p + ".pre_layer_norm.weight", w.pre_norm, {H}) &&
           f.get(p + ".post_layer_norm.weight", w.post_norm, {H}) &&
           f.linear(p + ".ffw_layer_1", w.up, 4 * H, H, clip) &&
           f.linear(p + ".ffw_layer_2", w.down, H, 4 * H, clip);
}

bool load_layer(Fetcher& f, const Gemma4AudioConfig& cfg, const std::string& p, Gemma4AudioLayer& L,
                Tensor& rel_w, Tensor& pds) {
    const int64_t H = cfg.hidden_size;
    const bool clip = cfg.clipped_linears;
    return load_ffw(f, p + "feed_forward1", L.ffw1, H, clip) &&
           load_ffw(f, p + "feed_forward2", L.ffw2, H, clip) &&
           f.get(p + "norm_pre_attn.weight", L.norm_pre_attn, {H}) &&
           f.get(p + "norm_post_attn.weight", L.norm_post_attn, {H}) &&
           f.get(p + "norm_out.weight", L.norm_out, {H}) &&
           f.linear(p + "self_attn.q_proj", L.q, H, H, clip) &&
           f.linear(p + "self_attn.k_proj", L.k, H, H, clip) &&
           f.linear(p + "self_attn.v_proj", L.v, H, H, clip) &&
           f.linear(p + "self_attn.post", L.post, H, H, clip) &&
           f.get(p + "self_attn.relative_k_proj.weight", rel_w, {H, H}) &&
           f.get(p + "self_attn.per_dim_scale", pds, {cfg.head_dim()}) &&
           f.get(p + "lconv1d.pre_layer_norm.weight", L.lconv_pre_norm, {H}) &&
           f.get(p + "lconv1d.conv_norm.weight", L.lconv_conv_norm, {H}) &&
           f.get(p + "lconv1d.depthwise_conv1d.weight", L.dw_conv, {H, 1, cfg.conv_kernel}) &&
           f.linear(p + "lconv1d.linear_start", L.lin_start, 2 * H, H, clip) &&
           f.linear(p + "lconv1d.linear_end", L.lin_end, H, H, clip);
}

// rel_k = pe @ relative_k_proj^T and q_scale = hd^-0.5 / ln2 * softplus(per_dim_scale): weight-only,
// computed once here in FP32.
void derive_layer_tables(Gemma4AudioModel& m, Gemma4AudioLayer& L, const Tensor& rel_w, const Tensor& pds,
                         const std::vector<float>& pe) {
    const int P = m.config.rel_positions();
    const int64_t H = m.config.hidden_size;
    const int hd = m.config.head_dim();
    const std::vector<float> w = to_f32(rel_w);
    std::vector<float> rk(static_cast<size_t>(P) * H);
    for (int r = 0; r < P; ++r)
        for (int64_t o = 0; o < H; ++o) {
            const float* wr = w.data() + (o * H);
            const float* x = pe.data() + (static_cast<size_t>(r) * H);
            float acc = 0.0f;
            for (int64_t k = 0; k < H; ++k)
                acc += x[k] * wr[k];
            rk[(static_cast<size_t>(r) * H) + o] = acc;
        }
    L.rel_k = host_f32(m, std::move(rk), {P, H});
    const std::vector<float> s = to_f32(pds);
    std::vector<float> qs(static_cast<size_t>(hd));
    const float base = static_cast<float>((1.0 / std::sqrt(static_cast<double>(hd))) / std::numbers::ln2);
    for (int d = 0; d < hd; ++d) {
        const float x = s[static_cast<size_t>(d)];
        qs[static_cast<size_t>(d)] = base * (x > 20.0f ? x : std::log1p(std::exp(x)));  // torch softplus
    }
    L.q_scale = host_f32(m, std::move(qs), {hd});
}

}  // namespace

bool is_gemma4_audio_tensor_name(const std::string& name) {
    return name.starts_with(kTower) || name.starts_with(kEmbed);
}

std::expected<Gemma4AudioConfig, std::string> parse_gemma4_audio_config(const JValue& root) {
    const JValue* ac = jobj_find(root, "audio_config");
    if (!ac || ac->type != JType::OBJECT)
        return std::unexpected(std::string());
    std::string type;
    jobj_opt_string(*ac, "model_type", type);
    if (type != "gemma4_audio")
        return std::unexpected("audio_config model_type '" + type + "' has no encoder in imp");
    Gemma4AudioConfig c;
    read_audio_keys(*ac, root, c);
    std::string act = "silu";
    jobj_opt_string(*ac, "hidden_act", act);
    if (act != "silu")
        return std::unexpected("gemma4_audio hidden_act '" + act + "' is not implemented (silu only)");
    if (!geometry_ok(c))
        return std::unexpected(std::string("gemma4_audio config outside the implemented geometry"));
    if (c.audio_token_id < 0 || c.boa_token_id < 0 || c.eoa_token_id < 0)
        return std::unexpected(std::string("config.json lacks audio_token_id / boa_token_id / eoa_token_id"));
    return c;
}

std::expected<std::unique_ptr<Gemma4AudioModel>, std::string> load_gemma4_audio_tower(
    const Gemma4AudioConfig& cfg, const std::unordered_map<std::string, Tensor>& tensors) {
    auto m = std::make_unique<Gemma4AudioModel>();
    m->config = cfg;
    const int64_t H = cfg.hidden_size;
    const int64_t c0 = cfg.sub_channels0;
    const int64_t c1 = cfg.sub_channels1;
    Fetcher f(tensors);
    const std::string t = kTower;
    const std::string sub = t + "subsample_conv_projection.";
    bool ok = f.get(sub + "layer0.conv.weight", m->sub0_conv, {c0, 1, 3, 3}) &&
              f.get(sub + "layer0.norm.weight", m->sub0_norm, {c0}) &&
              f.get(sub + "layer1.conv.weight", m->sub1_conv, {c1, c0, 3, 3}) &&
              f.get(sub + "layer1.norm.weight", m->sub1_norm, {c1}) &&
              f.get(sub + "input_proj_linear.weight", m->input_proj, {H, (cfg.n_mels / 4) * c1}) &&
              f.get(t + "output_proj.weight", m->out_proj_w, {cfg.output_proj_dims, H}) &&
              f.get(t + "output_proj.bias", m->out_proj_b, {cfg.output_proj_dims}) &&
              f.get(std::string(kEmbed) + "embedding_projection.weight", m->embed_proj,
                    {cfg.lm_d_model, cfg.output_proj_dims});
    m->layers.resize(static_cast<size_t>(cfg.num_layers));
    const std::vector<float> pe = rel_pos_table(cfg);
    for (int i = 0; ok && i < cfg.num_layers; ++i) {
        Tensor rel_w, pds;
        auto& L = m->layers[static_cast<size_t>(i)];
        ok = load_layer(f, cfg, t + "layers." + std::to_string(i) + ".", L, rel_w, pds);
        if (ok)
            derive_layer_tables(*m, L, rel_w, pds, pe);
    }
    if (!ok)
        return std::unexpected(f.error());
    return m;
}

std::unique_ptr<Gemma4AudioModel> load_gemma4_audio_from_checkpoint(
    const std::string& model_dir, const std::unordered_map<std::string, Tensor>& tensors) {
    JValue root;
    if (!parse_json_file(model_dir + "/config.json", root))
        return nullptr;
    auto cfg = parse_gemma4_audio_config(root);
    if (!cfg) {
        if (!cfg.error().empty())
            IMP_LOG_WARN("Audio: %s; audio input unavailable", cfg.error().c_str());
        return nullptr;
    }
    if (!tensors.contains(std::string(kTower) + "output_proj.weight")) {
        IMP_LOG_WARN(
            "Audio: audio_config declares a gemma4_audio encoder but the checkpoint carries no "
            "model.audio_tower.* tensors; audio input unavailable");
        return nullptr;
    }
    auto tower = load_gemma4_audio_tower(*cfg, tensors);
    if (!tower) {
        IMP_LOG_WARN("Audio: tower not loaded (%s); audio input unavailable", tower.error().c_str());
        return nullptr;
    }
    IMP_LOG_INFO("Audio: gemma4_audio tower (%d layers, hidden %d, %d heads) -> LM width %d", cfg->num_layers,
                 cfg->hidden_size, cfg->num_heads, cfg->lm_d_model);
    return std::move(*tower);
}

void gemma4_audio_visit_tensors(Gemma4AudioModel& m,
                                const std::function<void(Tensor&, const std::string&, bool fp16)>& fn) {
    fn(m.sub0_conv, "sub0_conv", false);
    fn(m.sub0_norm, "sub0_norm", false);
    fn(m.sub1_conv, "sub1_conv", false);
    fn(m.sub1_norm, "sub1_norm", false);
    fn(m.input_proj, "input_proj", true);
    for (size_t i = 0; i < m.layers.size(); ++i) {
        auto& L = m.layers[i];
        const std::string p = "layers." + std::to_string(i) + ".";
        for (auto* w : {&L.ffw1, &L.ffw2}) {
            const std::string n = p + (w == &L.ffw1 ? "ffw1." : "ffw2.");
            fn(w->pre_norm, n + "pre_norm", false);
            fn(w->post_norm, n + "post_norm", false);
            fn(w->up.w, n + "up", true);
            fn(w->down.w, n + "down", true);
        }
        fn(L.norm_pre_attn, p + "norm_pre_attn", false);
        fn(L.norm_post_attn, p + "norm_post_attn", false);
        fn(L.norm_out, p + "norm_out", false);
        fn(L.q.w, p + "q", true);
        fn(L.k.w, p + "k", true);
        fn(L.v.w, p + "v", true);
        fn(L.post.w, p + "post", true);
        fn(L.rel_k, p + "rel_k", false);
        fn(L.q_scale, p + "q_scale", false);
        fn(L.lconv_pre_norm, p + "lconv_pre_norm", false);
        fn(L.lconv_conv_norm, p + "lconv_conv_norm", false);
        fn(L.dw_conv, p + "dw_conv", false);
        fn(L.lin_start.w, p + "lin_start", true);
        fn(L.lin_end.w, p + "lin_end", true);
    }
    fn(m.out_proj_w, "out_proj_w", true);
    fn(m.out_proj_b, "out_proj_b", false);
    fn(m.embed_proj, "embed_proj", true);
}

size_t gemma4_audio_tower_device_bytes(Gemma4AudioModel& m) {
    size_t total = 0;
    gemma4_audio_visit_tensors(m, [&](Tensor& t, const std::string&, bool fp16) {
        total += ((static_cast<size_t>(t.numel()) * (fp16 ? 2 : 4)) + 255) & ~size_t{255};
    });
    return total;
}

std::expected<size_t, std::string> gemma4_audio_upload(Gemma4AudioModel& m) {
    struct Pending {
        Tensor* slot;
        void* device;
        bool fp16;
    };
    std::vector<Pending> pending;
    std::vector<uint8_t> staging;
    size_t total = 0;
    std::string err;
    // Own non-blocking stream, drained per tensor (staging is reused); no legacy-stream write (#2275).
    CudaStream up;
    if (!up.create())
        return std::unexpected(std::string("audio tower upload: no CUDA stream"));
    gemma4_audio_visit_tensors(m, [&](Tensor& t, const std::string& what, bool fp16) {
        if (!err.empty())
            return;
        if (t.data == nullptr || t.on_device) {
            err = "audio tower slot '" + what + "' is not a host tensor";
            return;
        }
        const std::vector<float> f = to_f32(t);
        const size_t bytes = f.size() * (fp16 ? 2 : 4);
        staging.resize(bytes);
        if (fp16) {
            half* h = reinterpret_cast<half*>(staging.data());
            for (size_t i = 0; i < f.size(); ++i)
                h[i] = __float2half(f[i]);
        } else {
            std::memcpy(staging.data(), f.data(), bytes);
        }
        auto slab = engine_arena().take_bytes(bytes);
        if (slab.empty()) {
            err = "engine arena exhausted uploading audio tower tensor '" + what + "'";
            return;
        }
        if (cudaMemcpyAsync(slab.data(), staging.data(), bytes, cudaMemcpyHostToDevice, up) != cudaSuccess ||
            cudaStreamSynchronize(up) != cudaSuccess) {
            err = "upload failed for audio tower tensor '" + what + "'";
            return;
        }
        pending.push_back({&t, slab.data(), fp16});
        total += (bytes + 255) & ~size_t{255};  // as gemma4_audio_tower_device_bytes counts it
    });
    if (!err.empty())
        return std::unexpected(err);
    for (const auto& p : pending) {
        p.slot->data = p.device;
        p.slot->qtype = p.fp16 ? QType::F16 : QType::F32;
        p.slot->on_device = true;
        p.slot->compute_strides();
    }
    IMP_LOG_INFO("Audio tower: %zu tensors uploaded, %.1f MiB", pending.size(),
                 static_cast<double>(total) / (1024.0 * 1024.0));
    return total;
}

}  // namespace imp
