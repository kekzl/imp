#pragma once

// Parse a Qwen3-VL `vision_config` block into VisionConfig.
//
// Its own TU for two reasons: hf_config_loader.cpp is already at its size limit,
// and this is a different failure mode from the tensor-name mapping next door —
// a wrong name silently misroutes one weight, a wrong geometry silently
// mis-shapes every buffer.

#include "model/json_util.h"
#include "vision/vision_model.h"

#include <string>

namespace imp {

// Fills the Qwen3-VL fields of `out` from a `vision_config` object.
//
// Returns false — leaving `out` UNTOUCHED — when a required field is missing or
// inconsistent, so a half-filled geometry can never reach the encoder. `err`
// then names what was wrong. Checked: every dimension positive, hidden_size
// divisible by num_heads, and num_position_embeddings a perfect square (it is a
// square grid: 2304 = 48^2).
bool parse_qwen3vl_vision_config(const JValue& vision_cfg, VisionConfig& out, std::string& err);

}  // namespace imp
