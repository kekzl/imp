#pragma once
// load_safetensors' Inkling2-Loop branch points. The arch reads its layers under the vendor names at
// engine init (model/inkling_weights.h); the standard loader only keeps embedding, final norm and lm_head.

#include "core/tensor.h"
#include "model/hf_config_loader.h"
#include "model/model_config.h"
#include "model/weight_map.h"

#include <string>
#include <unordered_map>

namespace imp {

// Inkling2-Loop: moves the three standard-path tensors out and empties the map; returns model_dir.
// Every other arch: map untouched, returns "". A missing tensor stays null (the executor refuses it).
inline std::string inkling_take_vocab(const ModelConfig& cfg, const std::string& model_dir,
                                      std::unordered_map<std::string, Tensor>& tensor_map, Tensor& tok_emb,
                                      Tensor& out_norm, Tensor& out_proj) {
    if (cfg.arch != ModelArch::INKLING2)
        return {};
    auto take = [&](const char* suffix) {
        for (const auto& [name, t] : tensor_map)
            if (name.ends_with(suffix))
                return t;
        return Tensor{};
    };
    tok_emb = take("llm.embed.weight");
    out_norm = take("llm.norm.weight");
    out_proj = take("llm.unembed.weight");
    // The lm_head is the unpadded rows of the unembedding.
    if (out_proj.shape[0] >= cfg.vocab_size)
        out_proj.shape[0] = cfg.vocab_size;
    else
        out_proj = Tensor{};
    tensor_map.clear();
    return model_dir;
}

[[nodiscard]] inline bool apply_weights_unless(bool skip, WeightMap& wmap, Model& model,
                                               std::unordered_map<std::string, Tensor>& tensor_map) {
    return skip || wmap.apply_weights(model, tensor_map);
}

[[nodiscard]] inline bool load_nvfp4_config_unless(bool skip, const std::string& model_dir,
                                                   HFConfigLoader::NvFP4Config& cfg) {
    return !skip && HFConfigLoader::load_nvfp4_config(model_dir, cfg);
}

}  // namespace imp
