#pragma once

#include "model/model_config.h"
#include "model/model_arch.h"

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

struct HFConfigLoader {
    // Load config.json from model directory, populate ModelConfig.
    // Returns true if config.json was found and parsed successfully.
    // Only overwrites cfg fields that are present in the JSON.
    static bool load_config(const std::string& model_dir, ModelConfig& cfg);

    // Load generation_config.json — populates eos_token_ids.
    // Returns true if file found and parsed.
    static bool load_generation_config(const std::string& model_dir,
                                       std::vector<int32_t>& eos_token_ids);

    // Load chat_template string from tokenizer_config.json.
    // Returns empty string if not found.
    static std::string load_chat_template(const std::string& model_dir);

    // Load added_tokens from tokenizer_config.json.
    // Returns pairs of (token_string, special_flag).
    struct AddedToken {
        std::string content;
        bool special = false;
        int id = -1;
    };
    static std::vector<AddedToken> load_added_tokens(const std::string& model_dir);

    // GPTQ quantization config from quantize_config.json
    struct GPTQConfig {
        int bits = 0;        // 4 or 8
        int group_size = 128;
        bool desc_act = false;
    };
    static bool load_gptq_config(const std::string& model_dir, GPTQConfig& cfg);

    // Map HF architecture class name to imp ModelArch.
    static ModelArch map_architecture(const std::string& hf_arch);
};

} // namespace imp
