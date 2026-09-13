#pragma once

// Own TU for two reasons: hf_config_loader.cpp is already at its size limit, and this is a
// different failure mode from the tensor-name mapping next door - a wrong name silently
// misroutes one weight, a wrong geometry silently mis-shapes every buffer.

#include "model/json_util.h"
#include "vision/vision_model.h"

#include <expected>
#include <string>

namespace imp {

// Builds VisionConfig from a vision_config object; returns the error text instead of a
// config on a missing/inconsistent field so a half-filled geometry cannot exist to be
// returned. Checked: every dimension positive, hidden_size divisible by num_heads,
// num_position_embeddings a perfect square (2304=48^2).
[[nodiscard]] std::expected<VisionConfig, std::string> parse_qwen3vl_vision_config(const JValue& vision_cfg);

// True if vision_config.model_type names a tower this parser covers. One definition on
// purpose: the config parser and the SafeTensors loader's keep-the-vision-tensors gate must
// agree exactly, or tensors ride along dead or the tower loads with null slots.
bool vision_tower_supported(const std::string& vision_model_type);

}  // namespace imp
