#pragma once

// Expands one image placeholder into the number of tokens the vision encoder actually
// produced. Qwen-VL templates render exactly one <|image_pad|> since image size (post
// smart_resize) is unknown at template time; expansion happens on tokens, not per-template.

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace imp {

// Replaces the k-th occurrence of pad_id with counts[k] copies. Refuses if placeholder
// count disagrees with image count (prompt/encoder mismatch would shift every later
// position); tokens is untouched on refusal.
[[nodiscard]] std::expected<void, std::string> expand_image_placeholders(std::vector<int32_t>& tokens,
                                                                         int32_t pad_id,
                                                                         const std::vector<int>& counts);

// FNV-1a over an image's bytes. Used as the prefix cache's content salt, so a
// hit needs the same tokens AND the same picture. Never returns 0 — that value
// means "no image" to the cache, and an all-zero image must not claim it.
size_t image_content_hash(std::span<const uint8_t> data);

// Folds one image's hash into a running salt for a multi-image request. Order-sensitive:
// swapping two images is a different prompt and must not share a prefix-cache key. Seed 0,
// call once per image in prompt order.
size_t combine_image_hash(size_t running, size_t next);

// Count of image tokens before `upto`: the embedding index a chunk starting there must
// resume from. Without this, image tokens crossing a chunk boundary restart at the image's
// first embedding. `upto` is clamped, so a prefix-cache offset past the prompt is not special.
int image_tokens_before(const std::vector<int32_t>& tokens, int32_t pad_id, int upto);

}  // namespace imp
