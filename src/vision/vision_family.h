#pragma once

// Vision tower family registry: vision_config.model_type -> the family whose config parser,
// weight names and encoder this build uses. Exact names only: a tower recognised on resemblance
// loads with misread geometry, so an unlisted name stays on the loud text-only path.

#include <string>
#include <string_view>

namespace imp {

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

}  // namespace imp
