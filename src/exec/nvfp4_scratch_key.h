#pragma once

// NVFP4 scratch key ("L5.wq", "L5.expert_w_gate.7", "out_proj") -> the weight it describes.
// Defined in pre_dequant_phase0_nvfp4_loader.cpp.

#include "model/model.h"

#include <string>

namespace imp {

// nullptr for an unknown slot, an index >= n_layers / expert count, or a malformed index
// ("x", "3x", overflow): a malformed index is never read as layer or expert 0.
[[nodiscard]] Tensor* resolve_nvfp4_scratch_key(Model& model, int n_layers, const std::string& key);

}  // namespace imp
