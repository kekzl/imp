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
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace imp {

// Replaces the k-th occurrence of `pad_id` with `counts[k]` copies of it.
// Refuses when the number of placeholders and the number of images disagree:
// that means the prompt and the encoder are describing different inputs, and
// every position after the mismatch would be shifted. `tokens` is untouched on
// a refusal.
[[nodiscard]] std::expected<void, std::string> expand_image_placeholders(std::vector<int32_t>& tokens,
                                                                         int32_t pad_id,
                                                                         const std::vector<int>& counts);

// FNV-1a over an image's bytes. Used as the prefix cache's content salt, so a
// hit needs the same tokens AND the same picture. Never returns 0 — that value
// means "no image" to the cache, and an all-zero image must not claim it.
size_t image_content_hash(std::span<const uint8_t> data);

// Fold one more image's hash into a running salt for a request carrying
// several. Order-sensitive on purpose: the same two pictures the other way
// round are a different prompt and must not share a prefix-cache key. Seed with
// 0 and call once per image, in prompt order.
size_t combine_image_hash(size_t running, size_t next);

// How many image tokens sit before `upto` — the embedding index a chunk
// starting there must resume from. The vision kernels scan the chunk they are
// handed, so without this a run of image tokens crossing a chunk boundary
// silently restarts at the image's first embedding. `upto` is clamped, so a
// prefix-cache offset past the prompt is not a special case for the caller.
int image_tokens_before(const std::vector<int32_t>& tokens, int32_t pad_id, int upto);

}  // namespace imp
