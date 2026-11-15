#pragma once

// A checkpoint with ModelOpt NVFP4 tensors (weight + weight_scale + weight_scale_2) whose declaration this
// build does not read would run the packed codes as weights (Flint, quant_algo W4A16_NVFP4: PPL 1.3e22).

#include "core/logging.h"

#include <string>
#include <string_view>

namespace imp {

// Module name of the first such Linear in a tensor-name map, "" when there is none.
template <class Map>
[[nodiscard]] std::string first_undeclared_nvfp4_module(const Map& tensor_map) {
    constexpr std::string_view kScale2 = ".weight_scale_2";
    for (const auto& kv : tensor_map) {
        const std::string& name = kv.first;
        if (!name.ends_with(kScale2))
            continue;
        const std::string module = name.substr(0, name.size() - kScale2.size());
        if (tensor_map.contains(module + ".weight") && tensor_map.contains(module + ".weight_scale"))
            return module;
    }
    return "";
}

// True (and logs why) when the loader must refuse: NVFP4 tensors, no declaration read.
template <class Map>
[[nodiscard]] bool nvfp4_undeclared(const Map& tensor_map, bool declared, const std::string& model_dir) {
    if (declared)
        return false;
    const std::string module = first_undeclared_nvfp4_module(tensor_map);
    if (module.empty())
        return false;
    IMP_LOG_ERROR(
        "%s: NVFP4 tensors (%s.weight_scale_2) but no quantization declaration this build reads "
        "(hf_quant_config.json quant_algo NVFP4 / W4A16_NVFP4 / MIXED_PRECISION, or recipe.yaml); "
        "refusing to load the packed codes as weights",
        model_dir.c_str(), module.c_str());
    return true;
}

}  // namespace imp
