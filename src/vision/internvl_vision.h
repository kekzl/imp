#pragma once

// InternVL (HF layout, InternVL3.5) vision tower: config parse, tensor load, slot visitor.
// Checkpoint names: vision_tower.embeddings.*, vision_tower.encoder.layer.N.*,
// multi_modal_projector.{layer_norm,linear_1,linear_2}.*.

#include "core/tensor.h"
#include "model/json_util.h"
#include "vision/vision_model.h"

#include <expected>
#include <functional>
#include <string>
#include <unordered_map>

namespace imp {

// `root` is the whole config.json: vision_config for the tower, downsample_ratio and
// text_config.hidden_size for the projector. Refuses what the encoder does not implement
// (qk norm, RMSNorm, non-0.5 downsample, no absolute position table, non-square grid).
[[nodiscard]] std::expected<VisionConfig, std::string> parse_internvl_vision_config(const JValue& root);

// Fills `out` (config already set) from the checkpoint map; q/k/v fused into wq/bq ([3H, H],
// [3H]) on the host. Returns the number of assigned checkpoint tensors, or why not: an unknown
// vision_tower./multi_modal_projector. name, a missing slot or a wrong shape refuses.
[[nodiscard]] std::expected<int, std::string> load_internvl_vision_tensors(
    const std::unordered_map<std::string, Tensor>& tensors, VisionModel& out);

// Every device slot of an InternVL tower exactly once (upload, device bytes, release).
void internvl_visit_vision_tensors(VisionModel& model,
                                   const std::function<void(Tensor&, const std::string&)>& fn);

}  // namespace imp
