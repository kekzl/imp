#include "vision/qwen3vl_vision_load.h"

#include "core/logging.h"
#include "vision/qwen3vl_vision_map.h"

namespace imp {

namespace {

using Slot = Qwen3VLVisionSlot;

constexpr const char* kVisualPrefix = "model.visual.";

VisionMergerWeights* merger_for(VisionModel& m, int index) {
    if (index < 0)
        return &m.merger;
    if (index < static_cast<int>(m.deepstack_mergers.size()))
        return &m.deepstack_mergers[static_cast<size_t>(index)];
    return nullptr;
}

// Assign into the slot, refusing to overwrite. A second tensor landing in the
// same slot means the mapper and this router disagree, which would otherwise
// show up as "the encoder uses whichever came last in hash order".
bool put(Tensor& dst, const Tensor& src, const char* what, std::string& err) {
    if (dst.data != nullptr) {
        err = std::string("two tensors claim the same slot: ") + what;
        return false;
    }
    dst = src;
    return true;
}

// The shape every slot must have, derived from the config alone. `ndim == 0`
// means "no expectation".
struct Expect {
    int ndim = 0;
    int64_t d0 = 0;
    int64_t d1 = 0;
};

Expect expected_for(Slot s, int index, const VisionConfig& c) {
    const int64_t h = c.hidden_size;
    const int64_t f = c.intermediate_size;
    // A merged token is the 2x2 spatial block concatenated, so every merger
    // tensor is sized in this unit rather than in hidden_size.
    const int64_t m = static_cast<int64_t>(c.merge_size) * c.merge_size * h;
    const int64_t grid = static_cast<int64_t>(c.pos_embed_grid) * c.pos_embed_grid;
    // in_channels is 3 for every published Qwen3-VL. It is not parsed from the
    // config, so a hypothetical single-channel variant is refused here with an
    // explicit shape mismatch rather than silently mis-strided.
    const int64_t patch_in = int64_t(3) * c.temporal_patch_size * c.patch_size * c.patch_size;

    switch (s) {
        case Slot::PatchEmbedWeight:
            return {2, h, patch_in};  // conv3d weight, flattened by the loader
        case Slot::PosEmbed:
            return {2, grid, h};
        case Slot::QkvWeight:
            return {2, 3 * h, h};
        case Slot::QkvBias:
            return {1, 3 * h, 0};
        case Slot::ProjWeight:
            return {2, h, h};
        case Slot::Fc1Weight:
            return {2, f, h};
        case Slot::Fc1Bias:
            return {1, f, 0};
        case Slot::Fc2Weight:
            return {2, h, f};
        case Slot::PatchEmbedBias:
        case Slot::Norm1Weight:
        case Slot::Norm1Bias:
        case Slot::Norm2Weight:
        case Slot::Norm2Bias:
        case Slot::ProjBias:
        case Slot::Fc2Bias:
            return {1, h, 0};
        // Merger norm placement is only visible in the shape: the main merger
        // normalises BEFORE the 2x2 concat (width hidden_size), the DeepStack
        // mergers AFTER it (width merge^2 * hidden_size). Getting this backwards
        // normalises the wrong axis and still runs, so it is checked here.
        case Slot::MergerNormWeight:
        case Slot::MergerNormBias:
            return {1, index < 0 ? h : m, 0};
        case Slot::MergerFc1Weight:
            return {2, m, m};
        case Slot::MergerFc1Bias:
            return {1, m, 0};
        case Slot::MergerFc2Weight:
            return {2, c.out_hidden_size, m};
        case Slot::MergerFc2Bias:
            return {1, c.out_hidden_size, 0};
        default:
            return {};
    }
}

bool shape_ok(const Tensor& t, const Expect& e, const char* what, std::string& err) {
    if (e.ndim == 0)
        return true;
    const bool ok = t.ndim == e.ndim && t.shape[0] == e.d0 && (e.ndim < 2 || t.shape[1] == e.d1);
    if (ok)
        return true;
    err = std::string("vision tensor '") + what + "' has shape [" + std::to_string(t.shape[0]) +
          (t.ndim > 1 ? ", " + std::to_string(t.shape[1]) : "") + "] (ndim " + std::to_string(t.ndim) +
          "), config implies [" + std::to_string(e.d0) + (e.ndim > 1 ? ", " + std::to_string(e.d1) : "") +
          "]";
    return false;
}

}  // namespace

bool load_qwen3vl_vision_tensors(const std::unordered_map<std::string, Tensor>& tensors, VisionModel& out,
                                 Qwen3VLVisionLoadStats& stats, std::string& err) {
    const VisionConfig& cfg = out.config;
    if (!cfg.is_qwen3vl || cfg.num_layers <= 0) {
        err = "vision config was not parsed before loading tensors";
        return false;
    }
    out.layers.resize(static_cast<size_t>(cfg.num_layers));
    out.deepstack_mergers.resize(cfg.deepstack_indexes.size());

    const size_t plen = std::char_traits<char>::length(kVisualPrefix);
    for (const auto& [name, t] : tensors) {
        if (name.compare(0, plen, kVisualPrefix) != 0)
            continue;
        const std::string local = name.substr(plen);
        const Qwen3VLVisionRef ref = qwen3vl_map_vision_tensor(local);
        if (ref.slot == Slot::Unknown) {
            // Reported, never ignored: an unrecognised vision tensor means this
            // checkpoint is not shaped the way the mapper assumes.
            IMP_LOG_WARN("Qwen3-VL vision: unrecognised tensor '%s'", name.c_str());
            stats.unknown++;
            continue;
        }
        // Checked before routing: a shape that contradicts the config means the
        // config and the checkpoint describe different models, and every later
        // consumer would read it as a stride.
        if (!shape_ok(t, expected_for(ref.slot, ref.index, cfg), local.c_str(), err))
            return false;

        bool ok = true;
        switch (ref.slot) {
            case Slot::PatchEmbedWeight:
                ok = put(out.patch_embd_w, t, local.c_str(), err);
                break;
            case Slot::PatchEmbedBias:
                ok = put(out.patch_embd_b, t, local.c_str(), err);
                break;
            case Slot::PosEmbed:
                ok = put(out.position_embd, t, local.c_str(), err);
                break;
            case Slot::MergerNormWeight:
            case Slot::MergerNormBias:
            case Slot::MergerFc1Weight:
            case Slot::MergerFc1Bias:
            case Slot::MergerFc2Weight:
            case Slot::MergerFc2Bias: {
                VisionMergerWeights* m = merger_for(out, ref.index);
                if (!m) {
                    err = "DeepStack merger index " + std::to_string(ref.index) +
                          " has no slot (config lists " + std::to_string(out.deepstack_mergers.size()) + ")";
                    return false;
                }
                Tensor* dst = nullptr;
                switch (ref.slot) {
                    case Slot::MergerNormWeight:
                        dst = &m->norm_w;
                        break;
                    case Slot::MergerNormBias:
                        dst = &m->norm_b;
                        break;
                    case Slot::MergerFc1Weight:
                        dst = &m->fc1_w;
                        break;
                    case Slot::MergerFc1Bias:
                        dst = &m->fc1_b;
                        break;
                    case Slot::MergerFc2Weight:
                        dst = &m->fc2_w;
                        break;
                    default:
                        dst = &m->fc2_b;
                        break;
                }
                ok = put(*dst, t, local.c_str(), err);
                break;
            }
            default: {
                if (ref.index < 0 || ref.index >= cfg.num_layers) {
                    err = "vision block index " + std::to_string(ref.index) + " exceeds depth " +
                          std::to_string(cfg.num_layers);
                    return false;
                }
                VisionLayerWeights& L = out.layers[static_cast<size_t>(ref.index)];
                Tensor* dst = nullptr;
                switch (ref.slot) {
                    case Slot::Norm1Weight:
                        dst = &L.ln1_w;
                        break;
                    case Slot::Norm1Bias:
                        dst = &L.ln1_b;
                        break;
                    // The fused QKV stays whole here; splitting it into wq/wk/wv
                    // is the uploader's job, since the split is a view of device
                    // memory and this runs before upload.
                    case Slot::QkvWeight:
                        dst = &L.wq;
                        break;
                    case Slot::QkvBias:
                        dst = &L.bq;
                        break;
                    case Slot::ProjWeight:
                        dst = &L.wo;
                        break;
                    case Slot::ProjBias:
                        dst = &L.bo;
                        break;
                    case Slot::Norm2Weight:
                        dst = &L.ln2_w;
                        break;
                    case Slot::Norm2Bias:
                        dst = &L.ln2_b;
                        break;
                    case Slot::Fc1Weight:
                        dst = &L.ffn_up_w;
                        break;
                    case Slot::Fc1Bias:
                        dst = &L.ffn_up_b;
                        break;
                    case Slot::Fc2Weight:
                        dst = &L.ffn_down_w;
                        break;
                    default:
                        dst = &L.ffn_down_b;
                        break;
                }
                ok = put(*dst, t, local.c_str(), err);
                break;
            }
        }
        if (!ok)
            return false;
        stats.assigned++;
    }

    // A null slot is the failure this function exists to catch: the encoder
    // would read it as a garbage embedding many layers later.
    auto require = [&](const Tensor& t, const char* what) {
        if (t.data == nullptr) {
            stats.missing++;
            if (err.empty())
                err = std::string("vision tower is missing ") + what;
        }
    };
    require(out.patch_embd_w, "patch_embed.proj.weight");
    require(out.patch_embd_b, "patch_embed.proj.bias");
    require(out.position_embd, "pos_embed.weight");
    for (size_t i = 0; i < out.layers.size(); ++i) {
        const VisionLayerWeights& L = out.layers[i];
        const std::string p = "blocks." + std::to_string(i) + ".";
        require(L.ln1_w, (p + "norm1.weight").c_str());
        require(L.ln1_b, (p + "norm1.bias").c_str());
        require(L.wq, (p + "attn.qkv.weight").c_str());
        require(L.bq, (p + "attn.qkv.bias").c_str());
        require(L.wo, (p + "attn.proj.weight").c_str());
        require(L.bo, (p + "attn.proj.bias").c_str());
        require(L.ln2_w, (p + "norm2.weight").c_str());
        require(L.ln2_b, (p + "norm2.bias").c_str());
        require(L.ffn_up_w, (p + "mlp.linear_fc1.weight").c_str());
        require(L.ffn_up_b, (p + "mlp.linear_fc1.bias").c_str());
        require(L.ffn_down_w, (p + "mlp.linear_fc2.weight").c_str());
        require(L.ffn_down_b, (p + "mlp.linear_fc2.bias").c_str());
    }
    auto require_merger = [&](const VisionMergerWeights& m, const char* what) {
        require(m.norm_w, what);
        require(m.norm_b, what);
        require(m.fc1_w, what);
        require(m.fc1_b, what);
        require(m.fc2_w, what);
        require(m.fc2_b, what);
    };
    require_merger(out.merger, "merger");
    for (size_t i = 0; i < out.deepstack_mergers.size(); ++i)
        require_merger(out.deepstack_mergers[i], "a deepstack merger");

    return stats.missing == 0;
}

}  // namespace imp
