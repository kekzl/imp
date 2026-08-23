#pragma once

// Parse a Qwen3-VL `vision_config` block into VisionConfig.
//
// Its own TU for two reasons: hf_config_loader.cpp is already at its size limit,
// and this is a different failure mode from the tensor-name mapping next door —
// a wrong name silently misroutes one weight, a wrong geometry silently
// mis-shapes every buffer.

#include "model/json_util.h"
#include "vision/vision_model.h"

#include <expected>
#include <string>

namespace imp {

// Builds the Qwen3-VL VisionConfig from a `vision_config` object.
//
// Returns the error text instead of a config when a required field is missing
// or inconsistent. A half-filled geometry can never reach the encoder because
// there is no half-filled value to hand back: the old signature filled an
// out-parameter and relied on the caller checking the bool first. Checked:
// every dimension positive, hidden_size divisible by num_heads, and
// num_position_embeddings a perfect square (it is a square grid: 2304 = 48^2).
[[nodiscard]] std::expected<VisionConfig, std::string> parse_qwen3vl_vision_config(const JValue& vision_cfg);

// True if `vision_config.model_type` names a tower this parser covers.
//
// One definition on purpose: the config parser and the SafeTensors loader's
// keep-the-vision-tensors gate must agree exactly. If the loader keeps a family
// the parser then rejects, the tensors ride along as dead weight; if the parser
// accepts one the loader dropped, the tower loads with null slots.
bool vision_tower_supported(const std::string& vision_model_type);

}  // namespace imp
