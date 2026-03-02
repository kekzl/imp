#pragma once

#include "vision/vision_model.h"
#include <string>
#include <memory>

namespace imp {

// Load a SigLIP vision model from an mmproj GGUF file.
// Weights are uploaded to GPU as FP16.
std::unique_ptr<VisionModel> load_vision_gguf(const std::string& path);

} // namespace imp
