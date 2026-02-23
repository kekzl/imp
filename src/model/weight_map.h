#pragma once

#include "model/model.h"
#include <string>
#include <unordered_map>

namespace imp {

// Maps weight names from file format to internal layer structure
class WeightMap {
public:
    explicit WeightMap(ModelArch arch);

    std::string map_name(const std::string& name) const;
    bool apply_weights(Model& model,
                       const std::unordered_map<std::string, Tensor>& tensors);

private:
    ModelArch arch_;
    std::unordered_map<std::string, std::string> name_map_;
};

} // namespace imp
