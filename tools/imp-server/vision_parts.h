#pragma once

// Chat-request vision parts (images and client-frame videos) in content order, Qwen3-VL side:
// prefix-cache salt, CPU patchify, template blocks, placeholder expansion.

#include "handlers_internal.h"

#include <expected>
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

// One template block per part, content order: the template's own image and video blocks.
std::string qwen_vision_blocks(const std::string& order, bool internvl);

// Expands image then video placeholders in snap.tokens.
std::expected<void, std::string> qwen_expand_vision_placeholders(ChatStateSnapshot& snap);
