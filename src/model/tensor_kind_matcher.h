#pragma once

#include "core/tensor_kind.h"
#include <string_view>

namespace imp {

// Map a GGUF or SafeTensors tensor name to its semantic TensorKind.
// Returns TensorKind::UNKNOWN if no rule matches.
TensorKind match_tensor_kind(std::string_view name);

}  // namespace imp
