#pragma once

// Vision tower family registry: vision_config.model_type -> the family whose config parser,
// weight names and encoder this build uses. Exact names only: a tower recognised on resemblance
// loads with misread geometry, so an unlisted name stays on the loud text-only path.

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace imp {

class Tokenizer;
struct VideoPlaceholderLayout;

enum class VisionFamily {
    None,      // not listed: text-only, WARN at load
    Qwen3VL,   // qwen3_vl, qwen3_5_moe (Qwen3.6), qwen3_5 (Qwen3.8): one tower layout
    InternVL,  // internvl_vision (InternVL3.5)
};

VisionFamily vision_family_of(std::string_view vision_model_type);
const char* vision_family_name(VisionFamily f);

// True when this build loads and encodes the family (config parser, weight map, encoder).
// The SafeTensors keep-the-vision-tensors gate and the config loader both ask this, so they
// cannot disagree.
[[nodiscard]] bool vision_family_loadable(VisionFamily f);

// A family's prompt layout: the only place vision token strings live. CLI and server render
// vision_prompt_blocks() ahead of the user text, apply the template, then expand_vision_prompt().
struct VisionPromptLayout {
    std::string_view image_block;  // rendered once per image
    std::string_view video_block;  // rendered once per video; "" = family takes no video
    std::string_view image_pad;    // token the image embeddings splice at
    std::string_view video_pad;    // token the video embeddings splice at; "" = no video
    std::string_view start, end;   // vision start / end markers
    // true: each image_pad -> start + n x image_pad + end, n = first image's count (InternVL).
    // false: the k-th image_pad -> counts[k] x image_pad inside the block's own markers.
    bool wrap_each_image = false;
};

// None maps to the Qwen3-VL layout: a build without a tower renders no blocks.
const VisionPromptLayout& vision_prompt_layout(VisionFamily f);

// One block per part in `order` ('i' image, 'v' video), content order.
std::string vision_prompt_blocks(VisionFamily f, std::string_view order);

// Expands the rendered blocks in `tokens` to image_tokens (one count per image, prompt order) and
// `videos`. Tokens untouched on refusal.
// `find` maps a token string to its id, -1 when absent.
using VisionTokenLookup = std::function<int32_t(std::string_view)>;
[[nodiscard]] std::expected<void, std::string> expand_vision_prompt(
    VisionFamily f, std::vector<int32_t>& tokens, const VisionTokenLookup& find,
    const std::vector<int>& image_tokens, const std::vector<VideoPlaceholderLayout>& videos);
[[nodiscard]] std::expected<void, std::string> expand_vision_prompt(
    VisionFamily f, std::vector<int32_t>& tokens, const Tokenizer& tok, const std::vector<int>& image_tokens,
    const std::vector<VideoPlaceholderLayout>& videos);

}  // namespace imp
