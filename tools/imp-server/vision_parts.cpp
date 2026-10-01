#include "vision_parts.h"

#include "model/image_placeholders.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"

#include <memory>
#include <span>
#include <vector>

size_t vision_parts_hash(ChatRequestParams& params) {
    if (params.vision_order.size() != params.images.size() + params.videos.size())
        params.vision_order = std::string(params.images.size(), 'i') + std::string(params.videos.size(), 'v');
    size_t hash = 0;
    size_t ni = 0, nv = 0;
    for (const char kind : params.vision_order) {
        if (kind == 'i') {
            hash = imp::combine_image_hash(hash, imp::image_content_hash(params.images[ni++]));
            continue;
        }
        for (const auto& frame : params.videos[nv++].frames)
            hash = imp::combine_image_hash(hash, imp::image_content_hash(frame));
    }
    return hash;
}

std::string qwen_preprocess_vision_parts(imp::Engine& engine, const ChatRequestParams& params,
                                         ChatStateSnapshot& snap) {
    snap.vision_internvl = engine.vision_is_internvl();
    if (snap.vision_internvl && !params.videos.empty())
        return "video parts need a Qwen3-VL model; this InternVL model takes images only";
    size_t ni = 0, nv = 0;
    for (const char kind : params.vision_order) {
        if (kind == 'v') {
            const ChatVideoInput& v = params.videos[nv++];
            std::vector<std::span<const uint8_t>> spans(v.frames.begin(), v.frames.end());
            std::vector<imp::QwenPatches> groups;
            if (!engine.preprocess_video_qwen(spans, groups) || groups.empty())
                return "Failed to process video frames (same size, decodable images, 2.." +
                       std::to_string(imp::Engine::kQwenVideoMaxFrames) + " frames)";
            const int per_group = engine.image_tokens_of(groups[0]);
            if (per_group <= 0)
                return "Failed to process video frames";
            const imp::Tokenizer* tok = snap.tok;
            snap.qwen_video_layouts.push_back(
                imp::qwen_video_layout(per_group, v.seconds, 2,
                                       [&](const std::string& s) { return tok->encode(s); }));
            for (auto& g : groups)
                snap.qwen_patches.push_back(std::make_shared<imp::QwenPatches>(std::move(g)));
            continue;
        }
        auto patches = std::make_shared<imp::QwenPatches>();
        if (!engine.preprocess_image_qwen(params.images[ni++], *patches))
            return "Failed to process image";
        const int tokens = engine.image_tokens_of(*patches);
        if (tokens <= 0)
            return "Failed to process image";
        snap.qwen_image_tokens.push_back(tokens);
        snap.qwen_patches.push_back(std::move(patches));
    }
    return {};
}

std::string qwen_vision_blocks(const std::string& order, bool internvl) {
    std::string blocks;
    for (const char kind : order) {
        if (internvl)
            blocks += "<IMG_CONTEXT>\n";  // InternVL chat_template.jinja: one context token + newline
        else
            blocks += kind == 'v' ? "<|vision_start|><|video_pad|><|vision_end|>"
                                  : "<|vision_start|><|image_pad|><|vision_end|>";
    }
    return blocks;
}

std::expected<void, std::string> qwen_expand_vision_placeholders(ChatStateSnapshot& snap) {
    if (snap.vision_internvl) {
        const int per_image = snap.qwen_image_tokens.empty() ? 0 : snap.qwen_image_tokens[0];
        return imp::expand_internvl_image_placeholders(snap.tokens, snap.tok->find_token("<IMG_CONTEXT>"),
                                                       snap.tok->find_token("<img>"),
                                                       snap.tok->find_token("</img>"),
                                                       static_cast<int>(snap.qwen_image_tokens.size()),
                                                       per_image);
    }
    const int32_t pad_id = snap.tok->find_token("<|image_pad|>");
    if (pad_id < 0)
        return std::unexpected(std::string("tokenizer has no <|image_pad|>"));
    auto expanded = imp::expand_image_placeholders(snap.tokens, pad_id, snap.qwen_image_tokens);
    if (expanded && !snap.qwen_video_layouts.empty())
        expanded = imp::expand_video_placeholders(snap.tokens, snap.tok->find_token("<|video_pad|>"),
                                                  snap.tok->find_token("<|vision_start|>"),
                                                  snap.tok->find_token("<|vision_end|>"),
                                                  snap.qwen_video_layouts);
    return expanded;
}
