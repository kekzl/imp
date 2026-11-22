#include "vision/vision_family.h"

#include "model/image_placeholders.h"
#include "model/tokenizer.h"

#include <algorithm>

namespace imp {

namespace {

struct Entry {
    std::string_view model_type;
    VisionFamily family;
};

// Qwen3.6 (qwen3_5_moe) and Qwen3.8 (qwen3_5) ship the Qwen3-VL tower under their text
// model_type: 333 model.visual.* tensors, a subset of Qwen3-VL's names, same geometry fields.
// Qwen3.8-Flash-Next (qwen4_exp_vision): the same 333 names and geometry as Qwen3.8, out 2560.
// qwen3_5_vision: nvidia/Qwen3.8-27B-NVFP4-QAD spells the Qwen3.8 tower's type that way.
// clang-format off: one model_type per line
constexpr Entry kRegistry[] = {
    {"qwen3_vl", VisionFamily::Qwen3VL},
    {"qwen3_5_moe", VisionFamily::Qwen3VL},
    {"qwen3_5", VisionFamily::Qwen3VL},
    {"qwen3_5_vision", VisionFamily::Qwen3VL},
    {"qwen4_exp_vision", VisionFamily::Qwen3VL},
    {"internvl_vision", VisionFamily::InternVL},
    {"inkling2_vision", VisionFamily::Inkling2},
};
// clang-format on

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
        case VisionFamily::Inkling2:
            return "Inkling2";
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
// A video ('<video>\n' in the template) is "\n".join("Frame{i+1}: <img>...</img>"): frame blocks.
constexpr VisionPromptLayout kInternVLLayout = {
    .image_block = "<IMG_CONTEXT>\n",
    .video_block = "",
    .image_pad = "<IMG_CONTEXT>",
    .video_pad = "",
    .start = "<img>",
    .end = "</img>",
    .wrap_each_image = true,
    .frame_label = "Frame",
};

// Inkling2 processor_config.json: one <|unused_200054|> per 40 x 40 pixel cell, no markers of its own;
// the part markers around it come from inkling_split_image_parts.
constexpr VisionPromptLayout kInkling2Layout = {
    .image_block = "<|unused_200054|>",
    .video_block = "",
    .image_pad = "<|unused_200054|>",
    .video_pad = "",
    .start = "",
    .end = "",
    .wrap_each_image = false,
};

}  // namespace

const VisionPromptLayout& vision_prompt_layout(VisionFamily f) {
    if (f == VisionFamily::Inkling2)
        return kInkling2Layout;
    return f == VisionFamily::InternVL ? kInternVLLayout : kQwen3VLLayout;
}

void inkling_split_image_parts(std::vector<int32_t>& tokens, const InklingPartIds& ids) {
    if (ids.pad < 0 || ids.text < 0 || ids.image < 0 || ids.end < 0)
        return;
    const auto is_role = [&](int32_t t) {
        return std::find(ids.roles.begin(), ids.roles.end(), t) != ids.roles.end();
    };
    std::vector<int32_t> out, hdr;
    out.reserve(tokens.size() + 16);
    bool collecting = false, in_text = false;
    size_t part_start = 0, text_len = 0;  // out index of the open text part's header, its tokens so far
    for (size_t i = 0; i < tokens.size();) {
        const int32_t t = tokens[i];
        if (in_text && t == ids.pad) {
            size_t n = 0;
            while (i + n < tokens.size() && tokens[i + n] == ids.pad)
                ++n;
            if (text_len == 0)
                out.resize(part_start);  // nothing before the picture: no empty text part
            else
                out.push_back(ids.end);
            out.insert(out.end(), hdr.begin(), hdr.end());
            out.push_back(ids.image);
            out.insert(out.end(), n, ids.pad);
            out.push_back(ids.end);
            i += n;
            if (i < tokens.size() && tokens[i] == ids.end) {
                ++i;  // nothing after it either
                in_text = false;
            } else {
                part_start = out.size();
                out.insert(out.end(), hdr.begin(), hdr.end());
                out.push_back(ids.text);
                text_len = 0;
            }
            continue;
        }
        if (is_role(t)) {
            hdr.assign(1, t);
            collecting = true;
            in_text = false;
        } else if (collecting && t == ids.text) {
            collecting = false;
            in_text = true;
            text_len = 0;
            part_start = out.size() - hdr.size();
        } else if (t == ids.end) {
            collecting = in_text = false;
        } else if (collecting) {
            hdr.push_back(t);  // a name behind the role token
        } else if (in_text) {
            ++text_len;
        }
        out.push_back(t);
        ++i;
    }
    tokens = std::move(out);
}

std::string vision_prompt_blocks(VisionFamily f, std::string_view order, std::span<const int> video_frames) {
    const VisionPromptLayout& l = vision_prompt_layout(f);
    std::string blocks;
    size_t nv = 0;
    for (const char kind : order) {
        if (kind != 'v') {
            blocks += l.image_block;
        } else if (l.frame_label.empty()) {
            blocks += l.video_block;
        } else {
            const int frames = nv < video_frames.size() ? video_frames[nv] : 0;
            for (int i = 0; i < frames; ++i)
                blocks += std::string(l.frame_label) + std::to_string(i + 1) + ": " +
                          std::string(l.image_block);
        }
        nv += kind == 'v' ? 1 : 0;
    }
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
    if (expanded && f == VisionFamily::Inkling2) {
        InklingPartIds ids{
            pad_id, find("<|content_text|>"), find("<|content_image|>"), find("<|end_message|>"), {}};
        for (const char* r :
             {"<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>"})
            ids.roles.push_back(find(r));
        inkling_split_image_parts(tokens, ids);
    }
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
