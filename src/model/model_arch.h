#pragma once

#include <string>

namespace imp {

enum class ModelArch {
    LLAMA,
    MISTRAL,
    MIXTRAL,
    DEEPSEEK,
    NEMOTRON_H_MOE,
    QWEN3,
    QWEN3_MOE,
    QWEN35,
    QWEN35_MOE,
    QWEN36_MOE,
    GPT_OSS,
    GEMMA3,
    GEMMA4,
    LLAMA4,
    GENERIC,
};

const char* model_arch_name(ModelArch arch);

// True iff this arch family has been empirically verified safe to honor a model
// author's kv_cache_quant_algo=FP8 hint BY DEFAULT (long-context FP8 KV). This is
// the long-context quality gate for kv_cache.dtype=auto; see the definition in
// model.cpp for the measured per-family evidence. Keep the list conservative.
bool kv_fp8_hint_default_safe(ModelArch arch);

// C API enum value for this architecture.
int model_arch_c_api_id(ModelArch arch);

// Sampling defaults from registry.
struct SamplingDefaults;
void model_arch_sampling_defaults(ModelArch arch, float& temperature, float& top_p, int& top_k);

// Parse architecture string (e.g. from GGUF "general.architecture")
ModelArch parse_model_arch(const std::string& s);

// Apply arch-specific config defaults (call after loading metadata)
struct ModelConfig;
void apply_arch_defaults(ModelConfig& cfg);

}  // namespace imp
