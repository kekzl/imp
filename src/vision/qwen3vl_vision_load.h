#pragma once

// Fill a Qwen3-VL VisionModel's tensors from a loaded shard map.
//
// The config half must already be in place (parse_qwen3vl_vision_config); this
// only routes weights. Kept apart from the mapping and the config parse because
// each has its own failure mode, and this one's is "a slot stayed null" —
// which the encoder would hit much later as a garbage embedding.

#include "core/tensor.h"
#include "vision/vision_model.h"

#include <string>
#include <unordered_map>

namespace imp {

struct Qwen3VLVisionLoadStats {
    int assigned = 0;
    int unknown = 0;  // model.visual.* names the mapper did not recognise
    int missing = 0;  // slots the checkpoint never filled
};

// `tensors` is the full shard map (names as they appear in the checkpoint).
// Only `model.visual.*` entries are touched. Returns false with `err` set when a
// required slot is missing or a shape contradicts the config — a vision tower
// with a null slot must not reach the encoder.
bool load_qwen3vl_vision_tensors(const std::unordered_map<std::string, Tensor>& tensors, VisionModel& out,
                                 Qwen3VLVisionLoadStats& stats, std::string& err);

}  // namespace imp
