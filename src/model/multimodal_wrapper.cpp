#include "model/multimodal_wrapper.h"

#include "core/logging.h"
#include "model/model.h"
#include "vision/internvl_vision.h"
#include "vision/qwen3vl_vision_config.h"
#include "vision/qwen3vl_vision_load.h"
#include "vision/vision_family.h"

#include <string_view>
#include <utility>

namespace imp {

std::string wrapped_lm_architecture(const JValue& root, const std::string& top_arch) {
    if (top_arch != "InternVLForConditionalGeneration")
        return top_arch;
    const JValue* tc = jobj_find(root, "text_config");
    const JValue* ta = tc ? jobj_find(*tc, "architectures") : nullptr;
    if (!ta || ta->type != JType::ARRAY || ta->arr.empty())
        return top_arch;
    IMP_LOG_INFO("%s: language model from text_config: %s", top_arch.c_str(), ta->arr[0].str_val.c_str());
    return ta->arr[0].str_val;
}

bool is_wrapper_vision_name(const std::string& n) {
    static constexpr std::string_view kPrefixes[] = {"model.vision_tower.", "model.visual.",
                                                     "model.embed_vision.", "vision_tower.",
                                                     "multi_modal_projector."};
    for (const std::string_view p : kPrefixes)
        if (n.starts_with(p))
            return true;
    return false;
}

std::string strip_wrapper_lm_prefix(const std::string& n) {
    static constexpr std::pair<std::string_view, std::string_view> kRenames[] = {
        {"model.language_model.", "model."},
        {"language_model.model.", "model."},
        {"language_model.lm_head.", "lm_head."},
        {"language_model.embed_tokens.", "model.embed_tokens."},
    };
    for (const auto& [from, to] : kRenames)
        if (n.starts_with(from))
            return std::string(to) + n.substr(from.size());
    return n;
}

// Multimodal vision-tower detection from vision_config. Family from the registry
// (vision_family.h); the SafeTensors keep-the-vision-tensors gate asks the same registry one step
// earlier (vision_tower_supported()). Anything not parsed says so: no silent text-only degrade.
void load_vision_tower_config(const JValue& root, std::unique_ptr<VisionModel>* out_vision_tower) {
    const JValue* vc = jobj_find(root, "vision_config");
    if (!vc || vc->type != JType::OBJECT)
        return;
    std::string vision_type;
    jobj_opt_string(*vc, "model_type", vision_type);
    const VisionFamily family = vision_family_of(vision_type);
    if (family != VisionFamily::None && !vision_family_loadable(family))
        IMP_LOG_WARN(
            "%s vision tower recognised (model_type='%s'); this build has no encoder for it, text-only",
            vision_family_name(family), vision_type.c_str());
    bool parsed = false;
    if (vision_family_loadable(family) && out_vision_tower) {
        const auto parsed_cfg = family == VisionFamily::InternVL ? parse_internvl_vision_config(root)
                                                                 : parse_qwen3vl_vision_config(*vc);
        if (parsed_cfg) {
            *out_vision_tower = std::make_unique<VisionModel>();
            (*out_vision_tower)->config = *parsed_cfg;
            parsed = true;
            const VisionConfig& c = (*out_vision_tower)->config;
            IMP_LOG_INFO(
                "%s vision tower: depth=%d hidden=%d heads=%d patch=%d merge=%d pos_grid=%dx%d out=%d "
                "deepstack=%zu",
                vision_family_name(family), c.num_layers, c.hidden_size, c.num_heads, c.patch_size,
                c.merge_size, c.pos_embed_grid, c.pos_embed_grid, c.out_hidden_size,
                c.deepstack_indexes.size());
        } else {
            // Refusing beats loading a tower whose geometry we misread.
            IMP_LOG_WARN("%s vision_config rejected (%s), text-only", vision_family_name(family),
                         parsed_cfg.error().c_str());
        }
    }
    if (!parsed)
        IMP_LOG_WARN(
            "Multimodal model detected (vision_config present, model_type='%s'). "
            "imp's SafeTensors loader handles only the language head; the "
            "vision tower will be skipped. Multimodal SafeTensors support is "
            "a separate audit item (#18).",
            vision_type.empty() ? "?" : vision_type.c_str());
}

// Routes the vision tower's tensors through its family's loader. An incomplete tower is dropped
// (text-only) rather than handed to the encoder with null slots.
void load_vision_tower_weights(const std::unordered_map<std::string, Tensor>& tensors, Model& model) {
    VisionModel& tower = *model.vision_tower;
    if (tower.config.is_internvl) {
        const auto used = load_internvl_vision_tensors(tensors, tower);
        if (used) {
            IMP_LOG_INFO("Vision tower (InternVL): %d tensors assigned (%zu blocks)", *used,
                         tower.layers.size());
            return;
        }
        IMP_LOG_WARN("Vision tower incomplete (%s), continuing text-only", used.error().c_str());
        model.vision_tower.reset();
        return;
    }
    const auto loaded = load_qwen3vl_vision_tensors(tensors, tower);
    if (loaded) {
        IMP_LOG_INFO("Vision tower: %d tensors assigned (%zu blocks, %zu deepstack mergers)",
                     loaded->assigned, tower.layers.size(), tower.deepstack_mergers.size());
        return;
    }
    const auto& e = loaded.error();
    IMP_LOG_WARN("Vision tower incomplete (%s; %d assigned, %d unknown, %d missing), continuing text-only",
                 e.what.c_str(), e.stats.assigned, e.stats.unknown, e.stats.missing);
    model.vision_tower.reset();
}

}  // namespace imp
