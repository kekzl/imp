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

// One video's prompt layout. HF Qwen3VLProcessor.replace_video_token: per frame pair g,
// stamp_ids[g] (tokenized "<x.x seconds>") + vision_start + video_pad*tokens_per_group + vision_end.
struct VideoPlaceholderLayout {
    int tokens_per_group = 0;                     // merged rows * cols of one frame pair
    std::vector<std::vector<int32_t>> stamp_ids;  // one per frame pair; size() = group count
};

// Replaces the k-th video_pad with videos[k]'s layout. The template's own vision_start/end around
// the pad stay, so each group sits inside a second start/end pair, as upstream emits it. Refuses on
// a count mismatch or an empty layout; tokens untouched on refusal.
[[nodiscard]] std::expected<void, std::string> expand_video_placeholders(
    std::vector<int32_t>& tokens, int32_t video_pad_id, int32_t vision_start_id, int32_t vision_end_id,
    const std::vector<VideoPlaceholderLayout>& videos);

// Per frame-pair timestamp: mean of the first and last frame time of the pair, odd tail padded with
// the last frame (Qwen3VLProcessor._calculate_timestamps). frame_seconds = frame index / fps.
std::vector<double> qwen_video_group_seconds(std::span<const double> frame_seconds, int temporal_patch_size);

// "<%.1f seconds>", the text tokenized into VideoPlaceholderLayout::stamp_ids.
std::string qwen_video_timestamp_text(double seconds);

// HF default when a video carries no fps (Qwen3VLProcessor.replace_video_token: fps 24).
inline constexpr double kQwenVideoDefaultFps = 24.0;

// A client video's layout: `groups` frame pairs of `tokens_per_group` tokens; frame k sits at
// frame_seconds[k]; stamps tokenized with `encode` (no special tokens).
template <typename Encode>
VideoPlaceholderLayout qwen_video_layout(int tokens_per_group, std::span<const double> frame_seconds,
                                         int temporal_patch_size, const Encode& encode) {
    VideoPlaceholderLayout v;
    v.tokens_per_group = tokens_per_group;
    for (double s : qwen_video_group_seconds(frame_seconds, temporal_patch_size))
        v.stamp_ids.push_back(encode(qwen_video_timestamp_text(s)));
    return v;
}

// Rewrites every video_pad to image_pad and returns the HF mm_token_type_ids (0 text, 1 image,
// 2 video) of the original sequence; empty when there was no video_pad (tokens untouched). Both pads
// are replaced by encoder rows, so the GPU path counts one id; the types keep M-RoPE exact.
std::vector<uint8_t> fold_video_pads(std::vector<int32_t>& tokens, int32_t image_pad_id,
                                     int32_t video_pad_id);

// InternVL (HF InternVLProcessor.replace_image_token, one tile per image): each context_id the
// template rendered becomes start_id + tokens_per_image x context_id + end_id. Refuses when the
// prompt's context_id count differs from n_images; tokens untouched on refusal.
[[nodiscard]] std::expected<void, std::string> expand_internvl_image_placeholders(
    std::vector<int32_t>& tokens, int32_t context_id, int32_t start_id, int32_t end_id, int n_images,
    int tokens_per_image);

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
