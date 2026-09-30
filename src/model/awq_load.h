#pragma once

#include "model/model.h"

#include <string>
#include <unordered_map>

namespace imp {

// AWQ SafeTensors (#2205, #2196): true = refuse the load (error logged). A supported variant
// (bits=4, zero_point, version gemm) marks every projection for dequant_awq4 and returns false.
[[nodiscard]] bool awq_refuses(Model& model, ModelConfig& cfg, const std::unordered_map<std::string, Tensor>& tensor_map,
                 const std::string& model_dir);

}  // namespace imp
