#include "vision/vision_family.h"

#include "model/image_placeholders.h"
#include "model/tokenizer.h"

namespace imp {

namespace {

struct Entry {
    std::string_view model_type;
    VisionFamily family;
};

// Qwen3.6 (qwen3_5_moe) and Qwen3.8 (qwen3_5) ship the Qwen3-VL tower under their text
// model_type: 333 model.visual.* tensors, a subset of Qwen3-VL's names, same geometry fields.
constexpr Entry kRegistry[] = {
    {"qwen3_vl", VisionFamily::Qwen3VL},
    {"qwen3_5_moe", VisionFamily::Qwen3VL},
    {"qwen3_5", VisionFamily::Qwen3VL},
    {"internvl_vision", VisionFamily::InternVL},
};

}  // namespace

VisionFamily vision_family_of(std::string_view vision_model_type) {
    for (const Entry& e : kRegistry)
        if (e.model_type == vision_model_type)
            return e.family;
    return VisionFamily::None;
}

const char* vision_family_name(VisionFamily f) {
    switch (f) {
        case VisionFamily::Qwen3VL:
            return "Qwen3-VL";
        case VisionFamily::InternVL:
            return "InternVL";
        case VisionFamily::None:
            break;
    }
    return "none";
}

bool vision_family_loadable(VisionFamily f) {
    return f == VisionFamily::Qwen3VL || f == VisionFamily::InternVL;
}

namespace {

// Qwen3-VL chat_template.jinja image / video blocks; HF Qwen3VLProcessor expansion.
constexpr VisionPromptLayout kQwen3VLLayout = {
    .image_block = "<|vision_start|><|image_pad|><|vision_end|>",
    .video_block = "<|vision_start|><|video_pad|><|vision_end|>",
    .image_pad = "<|image_pad|>",
    .video_pad = "<|video_pad|>",
    .start = "<|vision_start|>",
    .end = "<|vision_end|>",
    .wrap_each_image = false,
};

// InternVL chat_template.jinja: one context token + newline; HF InternVLProcessor, one tile per image.
constexpr VisionPromptLayout kInternVLLayout = {
    .image_block = "<IMG_CONTEXT>\n",
    .video_block = "",
    .image_pad = "<IMG_CONTEXT>",
    .video_pad = "",
    .start = "<img>",
    .end = "</img>",
    .wrap_each_image = true,
};

}  // namespace

const VisionPromptLayout& vision_prompt_layout(VisionFamily f) {
    return f == VisionFamily::InternVL ? kInternVLLayout : kQwen3VLLayout;
}

std::string vision_prompt_blocks(VisionFamily f, std::string_view order) {
    const VisionPromptLayout& l = vision_prompt_layout(f);
    std::string blocks;
    for (const char kind : order)
        blocks += kind == 'v' ? l.video_block : l.image_block;
    return blocks;
}

std::expected<void, std::string> expand_vision_prompt(VisionFamily f, std::vector<int32_t>& tokens,
                                                      const VisionTokenLookup& find,
                                                      const std::vector<int>& image_tokens,
                                                      const std::vector<VideoPlaceholderLayout>& videos) {
    const VisionPromptLayout& l = vision_prompt_layout(f);
    const int32_t pad_id = find(l.image_pad);
    if (l.wrap_each_image)
        return expand_internvl_image_placeholders(tokens, pad_id, find(l.start), find(l.end),
                                                  static_cast<int>(image_tokens.size()),
                                                  image_tokens.empty() ? 0 : image_tokens[0]);
    if (pad_id < 0)
        return std::unexpected("tokenizer has no " + std::string(l.image_pad));
    auto expanded = expand_image_placeholders(tokens, pad_id, image_tokens);
    if (expanded && !videos.empty())
        expanded = expand_video_placeholders(tokens, find(l.video_pad), find(l.start), find(l.end), videos);
    return expanded;
}

std::expected<void, std::string> expand_vision_prompt(VisionFamily f, std::vector<int32_t>& tokens,
                                                      const Tokenizer& tok,
                                                      const std::vector<int>& image_tokens,
                                                      const std::vector<VideoPlaceholderLayout>& videos) {
    return expand_vision_prompt(
        f, tokens, [&tok](std::string_view s) { return tok.find_token(std::string(s)); }, image_tokens,
        videos);
}

}  // namespace imp
