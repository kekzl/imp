#pragma once

#include "model/model_config.h"
#include "model/model_arch.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imp {

struct VisionModel;  // vision/vision_model.h — only a pointer is needed here

struct HFConfigLoader {
    // Loads config.json, populates ModelConfig; only overwrites fields present in the JSON.
    // out_vision_tower, when non-null and the checkpoint carries a supported vision tower,
    // receives a VisionModel with only its config filled; weights are weight_map's half.
    static bool load_config(const std::string& model_dir, ModelConfig& cfg,
                            std::unique_ptr<VisionModel>* out_vision_tower = nullptr);

    // Sampling/stop-condition defaults from generation_config.json. Sentinels mean "not
    // present": <0 for floats, -1 for ints, empty vector for eos. Caller falls back to arch
    // defaults or the hardcoded ImpGenerateParams baseline.
    struct GenerationConfig {
        float temperature = -1.0f;
        float top_p = -1.0f;
        int top_k = -1;
        float repetition_penalty = -1.0f;
        std::vector<int32_t> eos_token_ids;
    };

    // Load generation_config.json. Returns true if file found and parsed.
    // Only fields present in the JSON are populated; the rest stay at sentinel.
    static bool load_generation_config(const std::string& model_dir, GenerationConfig& cfg);

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

    // Authoritative special-token declarations from special_tokens_map.json. tokenizer.json's
    // special:true flag is normally a faithful copy but conversions can drop it; cross-checking
    // patches the gap before the engine builds its banned-token list.
    struct SpecialTokensMap {
        std::vector<std::string> additional_special_tokens;
        std::string bos_token;  // empty if not specified
        std::string eos_token;
        std::string pad_token;
        std::string unk_token;
    };
    static bool load_special_tokens_map(const std::string& model_dir, SpecialTokensMap& out);

    // Tokenizer-side flags from tokenizer_config.json; GGUF equivalent is
    // tokenizer.ggml.add_bos_token / add_space_prefix. Sentinel -1=absent, 0=false, 1=true.
    // Caller default matches GGUF: gpt2-style tokenizers default add_bos=false, else true.
    struct TokenizerFlags {
        int add_bos_token = -1;
        int add_eos_token = -1;
        int add_prefix_space = -1;
        // use_default_system_prompt:false means the chat-template apply path must NOT auto-fill
        // the template's hardcoded default system message (Mistral-Small-3.2's 600-token default).
        int use_default_system_prompt = -1;
        // BOS/EOS token strings from tokenizer_config.json (plain string or AddedToken object).
        // Empty if absent; resolves the BOS/EOS id for models whose BOS string isn't in the
        // hardcoded detection list.
        std::string bos_token;
        std::string eos_token;
    };
    static bool load_tokenizer_flags(const std::string& model_dir, TokenizerFlags& out);

    // GPTQ quantization config from quantize_config.json
    struct GPTQConfig {
        int bits = 0;  // 4 or 8
        int group_size = 128;
        bool desc_act = false;
    };
    static bool load_gptq_config(const std::string& model_dir, GPTQConfig& cfg);

    // Source format of NVFP4 quantization metadata.
    enum class NvFP4Format {
        MODELOPT,        // hf_quant_config.json from NVIDIA Model Optimizer
        LLM_COMPRESSOR,  // recipe.yaml from vllm-project/llm-compressor
    };

    // NVFP4 quantization config. Sourced from hf_quant_config.json (modelopt)
    // or recipe.yaml (llm-compressor) — see `format` field for which.
    struct NvFP4Config {
        int group_size = 16;                       // micro-scale group (default: 16 for NVFP4)
        std::string kv_cache_quant_algo;           // "FP8" or empty (modelopt only)
        std::vector<std::string> exclude_modules;  // e.g. ["lm_head"]
        NvFP4Format format = NvFP4Format::MODELOPT;
        // Modelopt quant_algo:MIXED_PRECISION: a per-tensor quantized_layers table instead of one
        // global algorithm (Nemotron-3.5: Mamba in/out on FP8, MoE experts on NVFP4). Only counts
        // are kept; storage tier is decided from each tensor's actual dtype at load.
        bool mixed_precision = false;
        int n_nvfp4_tensors = 0;
        int n_fp8_tensors = 0;
        int n_other_tensors = 0;  // an algorithm this build does not recognise
    };
    static bool load_nvfp4_config(const std::string& model_dir, NvFP4Config& cfg);

    // True if config.json declares a vision_config this build can build a tower from. Must
    // run before load_config(): shards are mapped first and a dropped shard cannot be
    // recovered. Only answers "worth keeping tensors for"; geometry is validated by load_config().
    static bool probe_vision_tower(const std::string& model_dir);

    // MXFP4 quantization config (e.g. GPT-OSS): config.json quantization_config,
    // quant_method=="mxfp4". Only metadata is parsed; the SafeTensors decode path is not
    // implemented (use GGUF for MXFP4 inference).
    struct MxFP4Config {
        int block_size = 32;  // E8M0 scale per 32 elements is the standard
    };
    static bool load_mxfp4_config(const std::string& model_dir, MxFP4Config& cfg);

    // AWQ (Activation-aware Weight Quantization) config. Sourced from
    // `quantization_config` in `config.json` or a separate `quant_config.json`.
    // Detection-only today; imp does not yet have an AWQ dequant kernel.
    struct AWQConfig {
        int bits = 4;            // typically 4
        int group_size = 128;    // typical AWQ group_size
        bool zero_point = true;  // AWQ uses zero-points by default
        std::string version;     // "gemm", "gemv", "marlin", or empty
    };
    static bool load_awq_config(const std::string& model_dir, AWQConfig& cfg);

    // Map HF architecture class name to imp ModelArch.
    static ModelArch map_architecture(const std::string& hf_arch);
};

}  // namespace imp
