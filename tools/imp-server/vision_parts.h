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
// every image and video frame, in that order, into one prefix-cache salt.
size_t vision_parts_hash(ChatRequestParams& params);

// Patchifies every part into snap.qwen_patches (a video adds one entry per frame pair), with
// snap.qwen_image_tokens and snap.qwen_video_layouts. Returns "" or the 400 message.
std::string qwen_preprocess_vision_parts(imp::Engine& engine, const ChatRequestParams& params,
                                         ChatStateSnapshot& snap);
