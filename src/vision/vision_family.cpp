#include "vision/vision_family.h"

namespace imp {

namespace {

struct Entry {
    std::string_view model_type;
    VisionFamily family;
};

// Qwen3.6 (qwen3_5_moe) and Qwen3.8 (qwen3_5) ship the Qwen3-VL tower under their text
// model_type: 333 model.visual.* tensors, a subset of Qwen3-VL's names, same geometry fields.
constexpr Entry kRegistry[] = {
    {"qwen3_vl", VisionFamily::Qwen3VL},
    {"qwen3_5_moe", VisionFamily::Qwen3VL},
    {"qwen3_5", VisionFamily::Qwen3VL},
    {"internvl_vision", VisionFamily::InternVL},
};

}  // namespace

VisionFamily vision_family_of(std::string_view vision_model_type) {
    for (const Entry& e : kRegistry)
        if (e.model_type == vision_model_type)
            return e.family;
    return VisionFamily::None;
}

const char* vision_family_name(VisionFamily f) {
    switch (f) {
        case VisionFamily::Qwen3VL:
            return "Qwen3-VL";
        case VisionFamily::InternVL:
            return "InternVL";
        case VisionFamily::None:
            break;
    }
    return "none";
}

bool vision_family_loadable(VisionFamily f) {
    // InternVL: recognised, encoder not built yet (roadmap row 10, units T3-T5).
    return f == VisionFamily::Qwen3VL;
}

}  // namespace imp
