#include "model/arch_registry.h"
#include "model/hf_config_hooks.h"

namespace imp {

namespace {

using enum ArchSource;

// Grouped by ModelArch. GGUF rows: llama.cpp general.architecture ids. HF_CLASS rows serve
// both loaders. HF_MODEL_TYPE rows: config.json model_type when architectures is absent.
constexpr ArchSpelling kSpellings[] = {
    // LLAMA (Llama-shaped families without their own arch)
    {GGUF, "llama", ModelArch::LLAMA},
    {GGUF, "qwen2", ModelArch::LLAMA},
    {GGUF, "phi3", ModelArch::LLAMA},
    {HF_CLASS, "LlamaForCausalLM", ModelArch::LLAMA},
    {HF_CLASS, "PhiForCausalLM", ModelArch::LLAMA},
    {HF_CLASS, "Phi3ForCausalLM", ModelArch::LLAMA},
    {HF_CLASS, "Phi3SmallForCausalLM", ModelArch::LLAMA},
    {HF_CLASS, "InternLM2ForCausalLM", ModelArch::LLAMA},
    {HF_CLASS, "Starcoder2ForCausalLM", ModelArch::LLAMA},
    {HF_CLASS, "CohereForCausalLM", ModelArch::LLAMA},
    {HF_MODEL_TYPE, "llama", ModelArch::LLAMA},
    {HF_MODEL_TYPE, "phi", ModelArch::LLAMA},
    {HF_MODEL_TYPE, "phi3", ModelArch::LLAMA},
    {HF_MODEL_TYPE, "cohere", ModelArch::LLAMA},
    {HF_MODEL_TYPE, "starcoder2", ModelArch::LLAMA},

    // MISTRAL
    {GGUF, "mistral", ModelArch::MISTRAL},
    {GGUF, "mistral3", ModelArch::MISTRAL},  // Ministral 3 / Devstral-Small-2 (#2411)
    {HF_CLASS, "MistralForCausalLM", ModelArch::MISTRAL},
    {HF_CLASS, "Mistral3ForConditionalGeneration", ModelArch::MISTRAL},
    {HF_MODEL_TYPE, "mistral", ModelArch::MISTRAL},

    // MIXTRAL
    {GGUF, "mixtral", ModelArch::MIXTRAL},
    {HF_CLASS, "MixtralForCausalLM", ModelArch::MIXTRAL},
    {HF_MODEL_TYPE, "mixtral", ModelArch::MIXTRAL},

    // DEEPSEEK
    {GGUF, "deepseek", ModelArch::DEEPSEEK},
    {GGUF, "deepseek2", ModelArch::DEEPSEEK},
    {HF_CLASS, "DeepseekV2ForCausalLM", ModelArch::DEEPSEEK},
    {HF_CLASS, "DeepseekV3ForCausalLM", ModelArch::DEEPSEEK},
    {HF_MODEL_TYPE, "deepseek_v2", ModelArch::DEEPSEEK},
    {HF_MODEL_TYPE, "deepseek_v3", ModelArch::DEEPSEEK},

    // NEMOTRON_H_MOE
    {GGUF, "nemotron_h_moe", ModelArch::NEMOTRON_H_MOE},
    {HF_CLASS, "NemotronHForCausalLM", ModelArch::NEMOTRON_H_MOE},
    {HF_MODEL_TYPE, "nemotron_h", ModelArch::NEMOTRON_H_MOE},

    // QWEN3 (Qwen2 dense loads through it; Qwen3-VL text tower is a plain Qwen3)
    {GGUF, "qwen3", ModelArch::QWEN3},
    {HF_CLASS, "Qwen2ForCausalLM", ModelArch::QWEN3},
    {HF_CLASS, "Qwen3ForCausalLM", ModelArch::QWEN3},
    {HF_CLASS, "Qwen3VLForConditionalGeneration", ModelArch::QWEN3},
    {HF_MODEL_TYPE, "qwen2", ModelArch::QWEN3},
    {HF_MODEL_TYPE, "qwen3", ModelArch::QWEN3},

    // QWEN3_MOE
    {GGUF, "qwen3moe", ModelArch::QWEN3_MOE},
    {HF_CLASS, "Qwen2MoeForCausalLM", ModelArch::QWEN3_MOE},
    {HF_CLASS, "Qwen3MoeForCausalLM", ModelArch::QWEN3_MOE},
    {HF_CLASS, "Qwen3VLMoeForConditionalGeneration", ModelArch::QWEN3_MOE},
    {HF_MODEL_TYPE, "qwen2_moe", ModelArch::QWEN3_MOE},
    {HF_MODEL_TYPE, "qwen3_moe", ModelArch::QWEN3_MOE},

    // QWEN35
    {GGUF, "qwen35", ModelArch::QWEN35},
    {HF_CLASS, "Qwen3_5ForCausalLM", ModelArch::QWEN35},
    {HF_CLASS, "Qwen3_5ForConditionalGeneration", ModelArch::QWEN35},
    {HF_MODEL_TYPE, "qwen3_5", ModelArch::QWEN35},
    {HF_MODEL_TYPE, "qwen3_5_text", ModelArch::QWEN35},

    // QWEN35_MOE
    {GGUF, "qwen35moe", ModelArch::QWEN35_MOE},

    // QWEN36_MOE (HF ships Qwen3.6 MoE under the Qwen3_5Moe class names)
    {GGUF, "qwen36moe", ModelArch::QWEN36_MOE},
    {GGUF, "qwen3.6_moe", ModelArch::QWEN36_MOE},
    {GGUF, "qwen3.6moe", ModelArch::QWEN36_MOE},
    {HF_CLASS, "Qwen3_5MoeForCausalLM", ModelArch::QWEN36_MOE},
    {HF_CLASS, "Qwen3_5MoeForConditionalGeneration", ModelArch::QWEN36_MOE},
    {HF_MODEL_TYPE, "qwen3_5_moe", ModelArch::QWEN36_MOE},
    {HF_MODEL_TYPE, "qwen3_5_moe_text", ModelArch::QWEN36_MOE},

    // QWEN4_EXP
    {GGUF, "qwen4exp", ModelArch::QWEN4_EXP},
    {GGUF, "qwen4_exp", ModelArch::QWEN4_EXP},
    {HF_CLASS, "Qwen4ExpForCausalLM", ModelArch::QWEN4_EXP},
    {HF_CLASS, "Qwen4ExpForConditionalGeneration", ModelArch::QWEN4_EXP},
    {HF_MODEL_TYPE, "qwen4_exp", ModelArch::QWEN4_EXP},
    {HF_MODEL_TYPE, "qwen4_exp_text", ModelArch::QWEN4_EXP},

    // GPT_OSS
    {GGUF, "gpt_oss", ModelArch::GPT_OSS},
    {GGUF, "gpt-oss", ModelArch::GPT_OSS},
    {HF_CLASS, "GptOssForCausalLM", ModelArch::GPT_OSS},

    // GEMMA3 (Gemma 1/2 load through it)
    {GGUF, "gemma3", ModelArch::GEMMA3},
    {GGUF, "gemma", ModelArch::GEMMA3},
    {GGUF, "gemma2", ModelArch::GEMMA3},
    {HF_CLASS, "GemmaForCausalLM", ModelArch::GEMMA3},
    {HF_CLASS, "Gemma2ForCausalLM", ModelArch::GEMMA3},
    {HF_CLASS, "Gemma3ForCausalLM", ModelArch::GEMMA3},
    {HF_CLASS, "Gemma3ForConditionalGeneration", ModelArch::GEMMA3},
    {HF_MODEL_TYPE, "gemma", ModelArch::GEMMA3},
    {HF_MODEL_TYPE, "gemma2", ModelArch::GEMMA3},
    {HF_MODEL_TYPE, "gemma3", ModelArch::GEMMA3},

    // GEMMA4
    {GGUF, "gemma4", ModelArch::GEMMA4},
    {HF_CLASS, "Gemma4ForCausalLM", ModelArch::GEMMA4},
    {HF_CLASS, "Gemma4ForConditionalGeneration", ModelArch::GEMMA4},
    {HF_CLASS, "Gemma4UnifiedForConditionalGeneration", ModelArch::GEMMA4},
    {HF_MODEL_TYPE, "gemma4", ModelArch::GEMMA4},
    {HF_MODEL_TYPE, "gemma4_unified", ModelArch::GEMMA4},

    // LLAMA4
    {GGUF, "llama4", ModelArch::LLAMA4},
    {HF_CLASS, "Llama4ForCausalLM", ModelArch::LLAMA4},
    {HF_CLASS, "Llama4ForConditionalGeneration", ModelArch::LLAMA4},
    {HF_MODEL_TYPE, "llama4", ModelArch::LLAMA4},

    // NOMIC_BERT (encoder-only embedder, #836)
    {GGUF, "nomic-bert", ModelArch::NOMIC_BERT},

    // GRANITE (#2412): GGUF multipliers in gguf_loader.cpp, HF in hf_config_loader.cpp
    {GGUF, "granite", ModelArch::GRANITE},
    {HF_CLASS, "GraniteForCausalLM", ModelArch::GRANITE},
    {HF_MODEL_TYPE, "granite", ModelArch::GRANITE},

    // QWEN3_NEXT (#2410): fused in_proj_qkvz / in_proj_ba split in weight_map.cpp
    {GGUF, "qwen3next", ModelArch::QWEN3_NEXT},  // refused by load_gguf
    {HF_CLASS, "Qwen3NextForCausalLM", ModelArch::QWEN3_NEXT},
    {HF_MODEL_TYPE, "qwen3_next", ModelArch::QWEN3_NEXT},
    {GGUF, "olmo3", ModelArch::OLMO3},  // refused at load (gguf_loader.cpp)
    {HF_CLASS, "Olmo3ForCausalLM", ModelArch::OLMO3},
    {HF_MODEL_TYPE, "olmo3", ModelArch::OLMO3},
    {GGUF, "lfm2", ModelArch::LFM2},  // refused at load (gguf_loader.cpp)
    {HF_CLASS, "Lfm2ForCausalLM", ModelArch::LFM2},
    {HF_CLASS, "Lfm2MoeForCausalLM", ModelArch::LFM2},
    {HF_MODEL_TYPE, "lfm2", ModelArch::LFM2},
    {HF_MODEL_TYPE, "lfm2_moe", ModelArch::LFM2},
};

struct HfConfigHookRow {
    ModelArch arch;
    HfConfigHook hook;
};

// config.json keys only one arch reads (#2537): one row per arch, hook in src/model/hf_config/.
// Archs without a row use the generic parsers in hf_config_generic.cpp only.
// clang-format off
constexpr HfConfigHookRow kHfConfigHooks[] = {
    {ModelArch::DEEPSEEK, parse_deepseek_config},
    {ModelArch::NEMOTRON_H_MOE, parse_nemotron_h_config},
    {ModelArch::QWEN35, parse_qwen_gdn_config},
    {ModelArch::QWEN35_MOE, parse_qwen_gdn_config},
    {ModelArch::QWEN36_MOE, parse_qwen_gdn_config},
    {ModelArch::QWEN4_EXP, parse_qwen_gdn_config},
    {ModelArch::QWEN3_NEXT, parse_qwen_gdn_config},
    {ModelArch::GPT_OSS, parse_gpt_oss_config},
    {ModelArch::GEMMA3, parse_gemma3_config},
    {ModelArch::GEMMA4, parse_gemma4_config},
    {ModelArch::OLMO3, parse_olmo3_config},
    {ModelArch::LFM2, parse_lfm2_config},
};
// clang-format on

}  // namespace

HfConfigHook find_hf_config_hook(ModelArch arch) {
    for (const auto& e : kHfConfigHooks)
        if (e.arch == arch)
            return e.hook;
    return nullptr;
}

std::span<const ArchSpelling> arch_spellings() { return kSpellings; }

std::optional<ModelArch> find_arch(ArchSource source, std::string_view s) {
    for (const auto& e : kSpellings)
        if (e.source == source && e.spelling == s)
            return e.arch;
    return std::nullopt;
}

}  // namespace imp
