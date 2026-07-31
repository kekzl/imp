#pragma once

// Expand the single image placeholder a chat template emits into the number of
// tokens the vision encoder actually produced.
//
// Qwen-VL templates render `<|vision_start|><|image_pad|><|vision_end|>` with
// exactly ONE `<|image_pad|>`, because at template time nobody knows how big
// the image will be after `smart_resize`. The processor expands it afterwards.
// Doing the same on the token sequence — rather than teaching each chat-template
// family about images — keeps multi-turn, multiple images and system prompts
// working without duplicating template logic.

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Replaces the k-th occurrence of `pad_id` with `counts[k]` copies of it.
// Refuses when the number of placeholders and the number of images disagree:
// that means the prompt and the encoder are describing different inputs, and
// every position after the mismatch would be shifted.
bool expand_image_placeholders(std::vector<int32_t>& tokens, int32_t pad_id, const std::vector<int>& counts,
                               std::string& err);

}  // namespace imp
