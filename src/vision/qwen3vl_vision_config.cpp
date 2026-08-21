#include "vision/qwen3vl_vision_config.h"

#include <cmath>
#include <optional>

namespace imp {

namespace {

const JValue* find(const JValue& v, const char* key) {
    if (v.type != JType::OBJECT)
        return nullptr;
    for (const auto& [k, val] : v.obj)
        if (k == key)
            return &val;
    return nullptr;
}

// Missing and present-but-not-a-number are the same failure here: both mean the
// geometry is not what this parser assumes.
std::optional<int> get_int(const JValue& v, const char* key) {
    const JValue* f = find(v, key);
    if (!f || f->type != JType::NUMBER)
        return std::nullopt;
    return static_cast<int>(f->as_int());
}

}  // namespace

bool vision_tower_supported(const std::string& vision_model_type) {
    // Qwen3.6 (`qwen3_5_moe`) ships the same tower under a different name: 333
    // `model.visual.*` tensors whose names are a strict subset of Qwen3-VL's
    // patterns, the same nine geometry fields, and an empty
    // `deepstack_visual_indexes` (which the loader already handles).
    //
    // Qwen3.8 (`qwen3_5`, the dense sibling) is the same tower again, checked
    // field by field against Qwen3.6-35B: depth 27, hidden 1152, heads 16,
    // intermediate 4304, patch 16, merge 2, temporal 2, pos-grid 2304, empty
    // deepstack, and the same `image_token_id` 248056. Only `out_hidden_size`
    // differs (5120 vs 2048), which is the LM's hidden size, not a tower
    // property. Its checkpoint carries the same 333 `model.visual.*` tensors.
    //
    // An allowlist rather than a shape fingerprint on purpose — anything
    // unrecognised must keep hitting the loud text-only path rather than being
    // parsed on a resemblance.
    return vision_model_type == "qwen3_vl" || vision_model_type == "qwen3_5_moe" ||
           vision_model_type == "qwen3_5";
}

std::expected<VisionConfig, std::string> parse_qwen3vl_vision_config(const JValue& vision_cfg) {
    VisionConfig c;
    int depth = 0, hidden = 0, heads = 0, inter = 0, patch = 0, merge = 0, temporal = 0;
    int out_hidden = 0, num_pos = 0;

    struct Field {
        const char* key;
        int* dst;
    };
    const Field required[] = {
        {"depth", &depth},
        {"hidden_size", &hidden},
        {"num_heads", &heads},
        {"intermediate_size", &inter},
        {"patch_size", &patch},
        {"spatial_merge_size", &merge},
        {"temporal_patch_size", &temporal},
        {"out_hidden_size", &out_hidden},
        {"num_position_embeddings", &num_pos},
    };
    for (const auto& f : required) {
        const std::optional<int> v = get_int(vision_cfg, f.key);
        if (!v)
            return std::unexpected(std::string("vision_config: missing or non-numeric '") + f.key + "'");
        if (*v <= 0)
            return std::unexpected(std::string("vision_config: '") + f.key + "' must be positive");
        *f.dst = *v;
    }

    if (hidden % heads != 0)
        return std::unexpected("vision_config: hidden_size is not divisible by num_heads");

    // The learned position embedding is a SQUARE grid that gets resampled per
    // image. A non-square count means this parser has the wrong model.
    const int grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(num_pos))));
    if (grid * grid != num_pos)
        return std::unexpected("vision_config: num_position_embeddings is not a perfect square");

    c.is_qwen3vl = true;
    c.num_layers = depth;
    c.hidden_size = hidden;
    c.num_heads = heads;
    c.head_dim = hidden / heads;
    c.intermediate_size = inter;
    c.patch_size = patch;
    c.merge_size = merge;
    c.temporal_patch_size = temporal;
    c.out_hidden_size = out_hidden;
    c.pos_embed_grid = grid;
    // Dynamic resolution: there is no fixed image_size or patch count. Leaving
    // the inherited defaults in place would be a lie the encoder could read.
    c.image_size = 0;
    c.num_patches = 0;
    c.num_image_tokens = 0;

    const JValue* ds = find(vision_cfg, "deepstack_visual_indexes");
    if (ds && ds->type == JType::ARRAY) {
        for (const auto& e : ds->arr) {
            if (e.type != JType::NUMBER)
                return std::unexpected("vision_config: deepstack_visual_indexes holds a non-number");
            const int idx = static_cast<int>(e.as_int());
            if (idx < 0 || idx >= depth)
                return std::unexpected("vision_config: deepstack index out of range for depth");
            c.deepstack_indexes.push_back(idx);
        }
    }

    return c;
}

}  // namespace imp
