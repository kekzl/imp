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

// True if `vision_config.model_type` names a tower this parser covers.
//
// One definition on purpose: the config parser and the SafeTensors loader's
// keep-the-vision-tensors gate must agree exactly. If the loader keeps a family
// the parser then rejects, the tensors ride along as dead weight; if the parser
// accepts one the loader dropped, the tower loads with null slots.
bool vision_tower_supported(const std::string& vision_model_type);

}  // namespace imp
