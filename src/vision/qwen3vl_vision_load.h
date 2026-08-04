#pragma once

// Fill a Qwen3-VL VisionModel's tensors from a loaded shard map.
//
// The config half must already be in place (parse_qwen3vl_vision_config); this
// only routes weights. Kept apart from the mapping and the config parse because
// each has its own failure mode, and this one's is "a slot stayed null" —
// which the encoder would hit much later as a garbage embedding.

#include "core/tensor.h"
#include "vision/vision_model.h"

#include <functional>
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

// Visit every tensor slot of the tower exactly once, with the checkpoint name it
// came from. Single source of truth on purpose: the load-completeness check and
// the device upload both walk this list, so a slot added to one cannot be
// forgotten by the other — which would leave the encoder reading a host pointer
// from the device.
//
// `model.layers` and `model.deepstack_mergers` must already be sized.
void qwen3vl_visit_vision_tensors(VisionModel& model,
                                  const std::function<void(Tensor&, const std::string&)>& fn);

// Device bytes the tower will occupy once uploaded. Walks the same list as
// qwen3vl_visit_vision_tensors for the reason given above — a slot added to the
// upload cannot be missed here — and reads shapes only, never data, so it is
// answerable BEFORE the upload. That is what lets the engine arena be sized for
// the tower at open time, which happens long before the vision warmup runs.
size_t qwen3vl_vision_tower_device_bytes(VisionModel& model);

}  // namespace imp
