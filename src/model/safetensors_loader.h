#pragma once

#include "model/model.h"
#include <string>
#include <memory>

namespace imp {

// Load model from SafeTensors format.
// path can be:
//   - A single .safetensors file
//   - A directory containing .safetensors files (+ config.json, etc.)
std::unique_ptr<Model> load_safetensors(const std::string& path);

}  // namespace imp
