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
    snap.vision_family = engine.vision_family();
    if (imp::vision_prompt_layout(snap.vision_family).video_block.empty() && !params.videos.empty())
        return "video parts need a Qwen3-VL model; this InternVL model takes images only";
    size_t ni = 0, nv = 0;
    for (const char kind : params.vision_order) {
        if (kind == 'v') {
            const ChatVideoInput& v = params.videos[nv++];
            std::vector<std::span<const uint8_t>> spans(v.frames.begin(), v.frames.end());
            std::vector<imp::QwenPatches> groups;
            if (!engine.preprocess_video_qwen(spans, groups) || groups.empty())
                return "Failed to process video frames (same size, JPEG or PNG, 2.." +
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
            return "Failed to process image (accepted formats: JPEG, PNG)";
        const int tokens = engine.image_tokens_of(*patches);
        if (tokens <= 0)
            return "Failed to process image (accepted formats: JPEG, PNG)";
        snap.qwen_image_tokens.push_back(tokens);
        snap.qwen_patches.push_back(std::move(patches));
    }
    return {};
}
