#pragma once

// HF config.json parsing split (#2537): load_config runs the arch-independent parsers below,
// then the one hook arch_registry.cpp registers for cfg.arch. A new arch's config parsing is
// one file in src/model/hf_config/, one declaration here and one kHfConfigHooks row.

#include "model/model_arch.h"

namespace imp {

struct JValue;
struct ModelConfig;

// root = config.json, eff = text_config when present, else root. false = refuse the config.
using HfConfigHook = bool (*)(const JValue& root, const JValue& eff, ModelConfig& cfg);

// Hook for `arch`; nullptr when the arch has no arch-specific keys.
[[nodiscard]] HfConfigHook find_hf_config_hook(ModelArch arch);

// Arch-independent parsers (hf_config_generic.cpp), in load_config order.
[[nodiscard]] bool parse_hf_core_dims(const JValue& eff, ModelConfig& cfg);
[[nodiscard]] bool parse_hf_rope(const JValue& eff, ModelConfig& cfg);
void parse_hf_ffn(const JValue& root, const JValue& eff, ModelConfig& cfg);
void parse_hf_moe(const JValue& eff, ModelConfig& cfg);
void parse_hf_tristate_flags(const JValue& root, const JValue& eff, ModelConfig& cfg);

// Per-arch hooks, one file each in src/model/hf_config/.
[[nodiscard]] bool parse_qwen_gdn_config(const JValue& root, const JValue& eff, ModelConfig& cfg);
[[nodiscard]] bool parse_nemotron_h_config(const JValue& root, const JValue& eff, ModelConfig& cfg);
[[nodiscard]] bool parse_gpt_oss_config(const JValue& root, const JValue& eff, ModelConfig& cfg);
[[nodiscard]] bool parse_gemma4_config(const JValue& root, const JValue& eff, ModelConfig& cfg);
[[nodiscard]] bool parse_gemma3_config(const JValue& root, const JValue& eff, ModelConfig& cfg);
[[nodiscard]] bool parse_deepseek_config(const JValue& root, const JValue& eff, ModelConfig& cfg);
[[nodiscard]] bool parse_olmo3_config(const JValue& root, const JValue& eff, ModelConfig& cfg);

}  // namespace imp
