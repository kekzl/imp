#pragma once

#include "model/model.h"
#include <string>
#include <memory>

namespace imp {

std::unique_ptr<Model> load_safetensors(const std::string& path);

} // namespace imp
