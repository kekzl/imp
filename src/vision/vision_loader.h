#pragma once

#include "vision/vision_model.h"
#include <string>
#include <memory>

namespace imp {

// Loads a SigLIP vision model from an mmproj GGUF file; weights uploaded as FP16.
// out_device_bytes reports the raw device bytes taken, the same accumulator
// vision_gguf_probe() returns, so a caller can assert the two agree.
std::unique_ptr<VisionModel> load_vision_gguf(const std::string& path, size_t* out_device_bytes = nullptr);

// Device bytes the tower will take from the T2 arena, plus the parsed config, answered
// without allocating: runs the SAME load path in counting mode so the reservation cannot
// drift from what the upload takes. Needed since the arena opens long before the vision
// warmup loads the mmproj. Returns 0 if the file cannot be parsed.
size_t vision_gguf_probe(const std::string& path, VisionConfig* out_cfg = nullptr,
                         int* out_lm_d_model = nullptr);

}  // namespace imp
