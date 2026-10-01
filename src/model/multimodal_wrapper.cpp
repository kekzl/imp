#include "model/multimodal_wrapper.h"

#include "core/logging.h"

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

}  // namespace imp
