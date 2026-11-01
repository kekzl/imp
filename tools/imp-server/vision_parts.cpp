#include "vision_parts.h"

#include "model/image_placeholders.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"

#include <bit>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace {

// runtime.video_pruning_rate as the decimal it was written as; 0 without an engine.
double default_pruning_rate(const imp::Engine* engine) {
    return engine ? imp::evs_rate_from_config(engine->runtime_config().runtime.video_pruning_rate) : 0.0;
}

}  // namespace

size_t vision_parts_hash(ChatRequestParams& params, const imp::Engine* engine) {
    const double default_rate = default_pruning_rate(engine);
    if (params.vision_order.size() != params.images.size() + params.videos.size())
        params.vision_order = std::string(params.images.size(), 'i') + std::string(params.videos.size(), 'v');
    size_t hash = 0;
    size_t ni = 0, nv = 0;
    for (const char kind : params.vision_order) {
        if (kind == 'i') {
            hash = imp::combine_image_hash(hash, imp::image_content_hash(params.images[ni++]));
            continue;
        }
        const ChatVideoInput& v = params.videos[nv++];
        for (const auto& frame : v.frames)
            hash = imp::combine_image_hash(hash, imp::image_content_hash(frame));
        // Same frames at another rate splice other rows; rate 0 leaves the salt as it was.
        if (const double rate = video_pruning_rate(v, default_rate); rate > 0.0)
            hash = imp::combine_image_hash(hash, static_cast<size_t>(std::bit_cast<uint64_t>(rate)));
    }
    return hash;
}

void bind_vision_parts(imp::Request& req, const ChatStateSnapshot& snap) {
    req.images = snap.vision_images;
    req.qwen_patches = snap.qwen_patches;  // dynamic-resolution route (empty otherwise)
    req.qwen_video_evs = snap.qwen_video_evs;
    req.vision_content_hash = snap.vision_content_hash;
}

std::string qwen_preprocess_vision_parts(imp::Engine& engine, const ChatRequestParams& params,
                                         ChatStateSnapshot& snap) {
    snap.vision_family = engine.vision_family();
    const imp::VisionPromptLayout& layout = imp::vision_prompt_layout(snap.vision_family);
    if (layout.video_block.empty() && layout.frame_label.empty() && !params.videos.empty())
        return "video parts need a vision model that takes video";
    const double default_rate = default_pruning_rate(&engine);
    size_t ni = 0, nv = 0;
    for (const char kind : params.vision_order) {
        if (kind == 'v' && !layout.frame_label.empty()) {
            const ChatVideoInput& v = params.videos[nv++];
            if (v.frames.size() < 2 || v.frames.size() > imp::Engine::kQwenVideoMaxFrames)
                return "Failed to process video frames (JPEG or PNG, 2.." +
                       std::to_string(imp::Engine::kQwenVideoMaxFrames) + " frames)";
            for (const auto& frame : v.frames) {
                auto patches = std::make_shared<imp::QwenPatches>();
                if (!engine.preprocess_image_qwen(frame, *patches) || engine.image_tokens_of(*patches) <= 0)
                    return "Failed to process video frames (JPEG or PNG)";
                snap.qwen_image_tokens.push_back(engine.image_tokens_of(*patches));
                snap.qwen_patches.push_back(std::move(patches));
            }
            snap.video_frame_counts.push_back(static_cast<int>(v.frames.size()));
            continue;
        }
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
            // EVS: the kept count fixes the prompt now; which tokens survive is decided after encode.
            const double rate = video_pruning_rate(v, default_rate);
            const int n_groups = static_cast<int>(groups.size());
            const int keep = imp::evs_retained_count(n_groups, per_group, rate);
            if (keep < n_groups * per_group) {
                imp::evs_prune_layout(snap.qwen_video_layouts.back(), keep);
                snap.qwen_video_evs.push_back(
                    {static_cast<int>(snap.qwen_patches.size()), n_groups, per_group, keep, rate});
            }
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
