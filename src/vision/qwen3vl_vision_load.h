#pragma once

// Fills a Qwen3-VL VisionModel's tensors from a loaded shard map; config half must already
// be in place (parse_qwen3vl_vision_config). Kept apart from mapping/config-parse since each
// has its own failure mode; this one's is "a slot stayed null", which the encoder would hit
// much later as a garbage embedding.

#include "core/tensor.h"
#include "vision/vision_model.h"

#include <expected>
#include <functional>
#include <string>
#include <unordered_map>

namespace imp {

struct Qwen3VLVisionLoadStats {
    int assigned = 0;
    int unknown = 0;  // model.visual.* names the mapper did not recognise
    int missing = 0;  // slots the checkpoint never filled
};

// A refusal carries the counts too: the caller logs how far the load got, and
// dropping that on the failure path is what an out-parameter made easy.
struct Qwen3VLVisionLoadError {
    std::string what;
    Qwen3VLVisionLoadStats stats;
};

// `tensors` is the full shard map (checkpoint names). Only model.visual.* entries are
// touched. Returns counts, or a refusal when a required slot is missing or a shape
// contradicts the config: a vision tower with a null slot must not reach the encoder.
[[nodiscard]] std::expected<Qwen3VLVisionLoadStats, Qwen3VLVisionLoadError> load_qwen3vl_vision_tensors(
    const std::unordered_map<std::string, Tensor>& tensors, VisionModel& out);

// Visits every tensor slot of the tower exactly once with the checkpoint name it came from.
// Single source of truth: the load-completeness check and device upload both walk this
// list, so a slot added to one cannot be forgotten by the other (which would leave the
// encoder reading a host pointer from the device). model.layers/model.deepstack_mergers
// must already be sized.
void qwen3vl_visit_vision_tensors(VisionModel& model,
                                  const std::function<void(Tensor&, const std::string&)>& fn);

// Device bytes the tower will occupy once uploaded, walking the same list as
// qwen3vl_visit_vision_tensors (for the same drift-proofing) and reading shapes only,
// never data - answerable BEFORE upload, letting the engine arena be sized at open time,
// long before the vision warmup runs.
size_t qwen3vl_vision_tower_device_bytes(VisionModel& model);

// Everything the T2 arena owes the Qwen3-VL half: tower plus pipeline/encoder scratch at
// the budget the engine will use. Engine::init asks this one question instead of composing
// three, keeping arena sizing from knowing the vision layer's internal split.
size_t qwen3vl_vision_arena_bytes(VisionModel& tower, int configured_max_patches);

}  // namespace imp
