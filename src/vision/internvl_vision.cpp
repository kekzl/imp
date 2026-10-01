#include "vision/internvl_vision.h"

#include "core/logging.h"

#include <cmath>
#include <cstring>
#include <optional>

namespace imp {

namespace {

std::optional<int> get_int(const JValue& v, const char* key) {
    const JValue* f = jobj_find(v, key);
    if (!f || f->type != JType::NUMBER)
        return std::nullopt;
    return static_cast<int>(f->as_int());
}

// image_size / patch_size ship as [h, w] or as a scalar; only square is implemented.
std::optional<int> get_square(const JValue& v, const char* key) {
    const JValue* f = jobj_find(v, key);
    if (!f)
        return std::nullopt;
    if (f->type == JType::NUMBER)
        return static_cast<int>(f->as_int());
    if (f->type == JType::ARRAY && f->arr.size() == 2 && f->arr[0].type == JType::NUMBER &&
        f->arr[0].as_int() == f->arr[1].as_int())
        return static_cast<int>(f->arr[0].as_int());
    return std::nullopt;
}

// JSON true/false parse as NUMBER 1/0 (json_util.cpp parse_bool).
bool flag(const JValue& v, const char* key, bool fallback) {
    const JValue* f = jobj_find(v, key);
    return f && f->type == JType::NUMBER ? f->num_val != 0.0 : fallback;
}

std::string str(const JValue& v, const char* key) {
    std::string s;
    jobj_opt_string(v, key, s);
    return s;
}

// What the encoder does not implement, or "" when the config is inside its envelope.
std::string unsupported_features(const JValue& root, const JValue& vc) {
    if (flag(vc, "use_qk_norm", false) || str(vc, "norm_type") != "layer_norm" ||
        !flag(vc, "use_absolute_position_embeddings", true) || str(vc, "hidden_act") != "gelu")
        return "InternVL vision_config: only layer_norm, gelu, no qk norm, absolute position embeddings are "
               "implemented";
    const JValue* ds = jobj_find(root, "downsample_ratio");
    if (!ds || ds->type != JType::NUMBER || std::fabs(ds->num_val - 0.5) > 1e-6)
        return "InternVL config: downsample_ratio must be 0.5";
    return {};
}

}  // namespace

std::expected<VisionConfig, std::string> parse_internvl_vision_config(const JValue& root) {
    const JValue* vc = jobj_find(root, "vision_config");
    const JValue* tc = jobj_find(root, "text_config");
    if (!vc || vc->type != JType::OBJECT || !tc || tc->type != JType::OBJECT)
        return std::unexpected("InternVL config: vision_config and text_config objects required");
    VisionConfig c;
    const auto hidden = get_int(*vc, "hidden_size"), heads = get_int(*vc, "num_attention_heads"),
               inter = get_int(*vc, "intermediate_size"), depth = get_int(*vc, "num_hidden_layers"),
               out_hidden = get_int(*tc, "hidden_size");
    const auto image = get_square(*vc, "image_size"), patch = get_square(*vc, "patch_size");
    if (!hidden || !heads || !inter || !depth || !out_hidden || !image || !patch)
        return std::unexpected(
            "InternVL vision_config: missing hidden_size, num_attention_heads, "
            "intermediate_size, num_hidden_layers, square image_size/patch_size or "
            "text_config.hidden_size");
    if (*hidden <= 0 || *heads <= 0 || *hidden % *heads != 0 || *patch <= 0 || *image % *patch != 0)
        return std::unexpected("InternVL vision_config: inconsistent hidden/heads or image/patch sizes");
    if (const std::string why = unsupported_features(root, *vc); !why.empty())
        return std::unexpected(why);
    const int side = *image / *patch;
    if (side % 2 != 0)
        return std::unexpected("InternVL config: patch grid side must be even for the 2x2 pixel shuffle");

    c.is_internvl = true;
    c.image_size = *image;
    c.patch_size = *patch;
    c.hidden_size = *hidden;
    c.num_heads = *heads;
    c.head_dim = *hidden / *heads;
    c.intermediate_size = *inter;
    c.num_layers = *depth;
    c.num_patches = side * side;
    c.merge_size = 2;
    c.num_image_tokens = c.num_patches / 4;
    c.out_hidden_size = *out_hidden;
    c.pos_embed_grid = side;
    c.n_merge = 1;
    const JValue* eps = jobj_find(*vc, "layer_norm_eps");
    if (eps && eps->type == JType::NUMBER)
        c.layer_norm_eps = static_cast<float>(eps->num_val);
    // ImageNet statistics (GotOcr2 preprocessor_config.json).
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    std::memcpy(c.image_mean, mean, sizeof(mean));
    std::memcpy(c.image_std, stdv, sizeof(stdv));
    return c;
}

void internvl_visit_vision_tensors(VisionModel& model,
                                   const std::function<void(Tensor&, const std::string&)>& fn) {
    fn(model.patch_embd_w, "embeddings.patch_embeddings.projection.weight");
    fn(model.patch_embd_b, "embeddings.patch_embeddings.projection.bias");
    fn(model.position_embd, "embeddings.position_embeddings");
    fn(model.cls_token, "embeddings.cls_token");
    for (size_t i = 0; i < model.layers.size(); ++i) {
        VisionLayerWeights& L = model.layers[i];
        const std::string p = "encoder.layer." + std::to_string(i) + ".";
        fn(L.ln1_w, p + "layernorm_before.weight");
        fn(L.ln1_b, p + "layernorm_before.bias");
        fn(L.wq, p + "attention.qkv.weight (fused)");
        fn(L.bq, p + "attention.qkv.bias (fused)");
        fn(L.wo, p + "attention.projection_layer.weight");
        fn(L.bo, p + "attention.projection_layer.bias");
        fn(L.ls1, p + "lambda_1");
        fn(L.ln2_w, p + "layernorm_after.weight");
        fn(L.ln2_b, p + "layernorm_after.bias");
        fn(L.ffn_up_w, p + "mlp.fc1.weight");
        fn(L.ffn_up_b, p + "mlp.fc1.bias");
        fn(L.ffn_down_w, p + "mlp.fc2.weight");
        fn(L.ffn_down_b, p + "mlp.fc2.bias");
        fn(L.ls2, p + "lambda_2");
    }
    fn(model.merger.norm_w, "multi_modal_projector.layer_norm.weight");
    fn(model.merger.norm_b, "multi_modal_projector.layer_norm.bias");
    fn(model.merger.fc1_w, "multi_modal_projector.linear_1.weight");
    fn(model.merger.fc1_b, "multi_modal_projector.linear_1.bias");
    fn(model.merger.fc2_w, "multi_modal_projector.linear_2.weight");
    fn(model.merger.fc2_b, "multi_modal_projector.linear_2.bias");
}

namespace {

// Checkpoint shape -> the 2-D (or 1-D) view the encoder reads; empty optional = shape mismatch.
std::optional<Tensor> as_shape(const Tensor& t, int64_t d0, int64_t d1) {
    if (t.numel() != d0 * (d1 > 0 ? d1 : 1))
        return std::nullopt;
    Tensor v = t;
    v.ndim = d1 > 0 ? 2 : 1;
    v.shape[0] = d0;
    v.shape[1] = d1 > 0 ? d1 : 0;
    for (int i = 2; i < kMaxDims; ++i)
        v.shape[i] = 0;
    v.compute_strides();
    return v;
}

// q|k|v rows concatenated into one host buffer the model owns: one GEMM per block.
std::optional<Tensor> fuse3(const Tensor& a, const Tensor& b, const Tensor& c, int64_t rows, int64_t cols,
                            VisionModel& out) {
    if (a.qtype != b.qtype || a.qtype != c.qtype)
        return std::nullopt;
    const size_t each = a.nbytes();
    if (b.nbytes() != each || c.nbytes() != each || a.numel() != rows * (cols > 0 ? cols : 1))
        return std::nullopt;
    auto& buf = out.host_owned.emplace_back(3 * each);
    std::memcpy(buf.data(), a.data, each);
    std::memcpy(buf.data() + each, b.data, each);
    std::memcpy(buf.data() + 2 * each, c.data, each);
    Tensor f = a;
    f.data = buf.data();
    f.ndim = 1;
    f.shape[0] = 3 * a.numel();  // numel of the fused buffer; as_shape sets the 2-D view
    return as_shape(f, 3 * rows, cols);
}

}  // namespace

std::expected<int, std::string> load_internvl_vision_tensors(
    const std::unordered_map<std::string, Tensor>& tensors, VisionModel& out) {
    const VisionConfig& c = out.config;
    if (!c.is_internvl || c.num_layers <= 0)
        return std::unexpected("InternVL vision config was not parsed before loading tensors");
    const int64_t H = c.hidden_size, I = c.intermediate_size, P = c.patch_size, W = 4 * H,
                  D = c.out_hidden_size;
    out.layers.assign(static_cast<size_t>(c.num_layers), VisionLayerWeights{});

    int used = 0;
    std::string err;
    auto get = [&](const std::string& name) -> const Tensor* {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            if (err.empty())
                err = "InternVL vision tower is missing '" + name + "'";
            return nullptr;
        }
        ++used;
        return &it->second;
    };
    auto put = [&](Tensor& slot, const std::string& name, int64_t d0, int64_t d1) {
        const Tensor* t = get(name);
        if (!t)
            return;
        auto v = as_shape(*t, d0, d1);
        if (!v) {
            if (err.empty())
                err = "InternVL vision tensor '" + name + "' has " + std::to_string(t->numel()) +
                      " elements, config implies " + std::to_string(d0 * (d1 > 0 ? d1 : 1));
            return;
        }
        slot = *v;
    };

    const std::string e = "vision_tower.embeddings.";
    put(out.patch_embd_w, e + "patch_embeddings.projection.weight", H, 3 * P * P);
    put(out.patch_embd_b, e + "patch_embeddings.projection.bias", H, 0);
    put(out.position_embd, e + "position_embeddings", 1 + c.num_patches, H);
    put(out.cls_token, e + "cls_token", H, 0);
    for (int l = 0; l < c.num_layers; ++l) {
        VisionLayerWeights& L = out.layers[static_cast<size_t>(l)];
        const std::string p = "vision_tower.encoder.layer." + std::to_string(l) + ".";
        const Tensor *wq = get(p + "attention.q_proj.weight"), *wk = get(p + "attention.k_proj.weight"),
                     *wv = get(p + "attention.v_proj.weight"), *bq = get(p + "attention.q_proj.bias"),
                     *bk = get(p + "attention.k_proj.bias"), *bv = get(p + "attention.v_proj.bias");
        if (wq && wk && wv && bq && bk && bv) {
            auto fw = fuse3(*wq, *wk, *wv, H, H, out);
            auto fb = fuse3(*bq, *bk, *bv, H, 0, out);
            if (!fw || !fb) {
                if (err.empty())
                    err = "InternVL vision layer " + std::to_string(l) + ": q/k/v disagree in dtype or shape";
            } else {
                L.wq = *fw;
                L.bq = *fb;
            }
        }
        put(L.ln1_w, p + "layernorm_before.weight", H, 0);
        put(L.ln1_b, p + "layernorm_before.bias", H, 0);
        put(L.wo, p + "attention.projection_layer.weight", H, H);
        put(L.bo, p + "attention.projection_layer.bias", H, 0);
        put(L.ls1, p + "lambda_1", H, 0);
        put(L.ln2_w, p + "layernorm_after.weight", H, 0);
        put(L.ln2_b, p + "layernorm_after.bias", H, 0);
        put(L.ffn_up_w, p + "mlp.fc1.weight", I, H);
        put(L.ffn_up_b, p + "mlp.fc1.bias", I, 0);
        put(L.ffn_down_w, p + "mlp.fc2.weight", H, I);
        put(L.ffn_down_b, p + "mlp.fc2.bias", H, 0);
        put(L.ls2, p + "lambda_2", H, 0);
    }
    const std::string m = "multi_modal_projector.";
    put(out.merger.norm_w, m + "layer_norm.weight", W, 0);
    put(out.merger.norm_b, m + "layer_norm.bias", W, 0);
    put(out.merger.fc1_w, m + "linear_1.weight", D, W);
    put(out.merger.fc1_b, m + "linear_1.bias", D, 0);
    put(out.merger.fc2_w, m + "linear_2.weight", D, D);
    put(out.merger.fc2_b, m + "linear_2.bias", D, 0);
    if (!err.empty())
        return std::unexpected(err);

    // Every vision_tower./multi_modal_projector. tensor must have been read: an extra one means
    // the checkpoint is shaped differently from what this loader assumes.
    int vision_names = 0;
    for (const auto& [name, t] : tensors)
        vision_names += name.starts_with("vision_tower.") || name.starts_with("multi_modal_projector.");
    if (vision_names != used)
        return std::unexpected("InternVL vision: checkpoint holds " + std::to_string(vision_names) +
                               " vision tensors, loader read " + std::to_string(used));
    return used;
}

}  // namespace imp
