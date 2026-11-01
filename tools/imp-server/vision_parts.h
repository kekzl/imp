#pragma once

// Chat-request vision parts (images and client-frame videos) in content order, Qwen3-VL side:
// prefix-cache salt, CPU patchify. Template blocks and expansion: imp::vision_prompt_blocks,
// imp::expand_vision_prompt.

#include "handlers_internal.h"

#include <string>

namespace imp {
class Engine;
}  // namespace imp

// Normalises params.vision_order (a dialect that fills only `images` gets all-images) and folds
// every image and video frame, in that order, into one prefix-cache salt. A video's EVS rate
// (its "pruning_rate", else the engine's runtime.video_pruning_rate) joins the salt when > 0.
size_t vision_parts_hash(ChatRequestParams& params, const imp::Engine* engine = nullptr);

// Copies the snapshot's vision state onto a request: images, Qwen patches, EVS plans, content salt.
void bind_vision_parts(imp::Request& req, const ChatStateSnapshot& snap);

// The EVS rate a video part runs at: its own "pruning_rate", else the server default.
inline double video_pruning_rate(const ChatVideoInput& v, double default_rate) {
    return v.pruning_rate >= 0.0 ? v.pruning_rate : default_rate;
}

// Patchifies every part into snap.qwen_patches (a video adds one entry per frame pair; on InternVL
// one image per frame), with snap.qwen_image_tokens, snap.qwen_video_layouts and
// snap.video_frame_counts. Returns "" or the 400 message.
std::string qwen_preprocess_vision_parts(imp::Engine& engine, const ChatRequestParams& params,
                                         ChatStateSnapshot& snap);
