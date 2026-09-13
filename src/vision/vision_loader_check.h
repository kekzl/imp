#pragma once

// The two refusals the mmproj GGUF path was missing. This loader speaks exactly one
// dialect: the LLaVA/SigLIP layout Gemma-3/Gemma-4v export (separate attn_q/k/v, one
// projection tensor, no side towers). A checkpoint in another dialect parses *partly*:
// shared names land in their slots, the rest are dropped at DEBUG, and the model looks
// loaded while the encoder gets null slots. Qwen3-VL's mmproj is the live example (247/316
// tensors land; imp reads that model from SafeTensors instead, nothing here can read it).
// Two gates: name the dialect we know we cannot read, and behind it verify every slot the
// encoder dereferences unconditionally is actually filled - catching dialects nobody has
// named yet.

#include "vision/vision_model.h"

#include <string>

namespace imp {

// Empty when this loader can read `projector_type`; otherwise the reason,
// phrased for whoever has to pick a different flag.
std::string vision_projector_reject_reason(const std::string& projector);

// Empty when every slot the encoder dereferences unconditionally is filled; else the GGUF
// name of the first one that isn't. Predicate is ndim==0, not data==nullptr: the probe
// walks the loader with uploads counted not performed, so every slot has a null pointer and
// a real shape there. Optional slots (patch_embd_b, attn/FFN biases, post_ln, mm_post_norm,
// guarded by `if(.data)`; gemma4v has none of these) are deliberately absent from the walk.
std::string vision_model_missing_slot(const VisionModel& model);

}  // namespace imp
