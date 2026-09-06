#pragma once

#include "model/model.h"
#include <string>
#include <unordered_map>

namespace imp {

// Maps weight names from file format to internal layer structure
class WeightMap {
public:
    // Why a tensor was not assigned. `total` is the number the load summary has
    // always printed; the rest is the breakdown it used to hide. `audio` is the
    // one class nothing downstream owns: no encoder, no input type, no
    // tokenizer route, so those tensors are a lost modality rather than a lost
    // weight, and the loader says so.
    struct SkipStats {
        int total = 0;
        int vision = 0;        // model.{vision_tower,visual,embed_vision}.*
        int audio = 0;         // model.embed_audio.*
        int mtp = 0;           // mtp.* / model.mtp.* stripped by the multimodal path
        int unrecognised = 0;  // no matcher claimed the name
    };

    explicit WeightMap(ModelArch arch);

    std::string map_name(const std::string& name) const;
    bool apply_weights(Model& model, const std::unordered_map<std::string, Tensor>& tensors);

    // Valid after apply_weights(); all zero before it.
    const SkipStats& skip_stats() const { return skip_stats_; }

private:
    ModelArch arch_;
    std::unordered_map<std::string, std::string> name_map_;
    SkipStats skip_stats_{};
};

}  // namespace imp
