#pragma once

#include <string>

#include "core/qtype.h"

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
    NOMIC_BERT,  // encoder-only embedder (#836): bidirectional, post-LN, mean-pool
    GENERIC,
};

const char* model_arch_name(ModelArch arch);

// True iff this arch family is empirically verified safe to honor a kv_cache_quant_algo=FP8
// hint by default. Long-context quality gate for kv_cache.dtype=auto; see model.cpp for
// per-family evidence. Keep conservative.
bool kv_fp8_hint_default_safe(ModelArch arch);

// True iff safe for default FP8 KV with no checkpoint hint (GGUF never carries one).
// Stricter bar than the hint list: the family must gate ~neutral, not merely <=1.5%. See
// model.cpp for per-family evidence and exclusions.
bool kv_fp8_no_hint_default_safe(ModelArch arch);

// True iff measured safe for default NVFP4 KV. A capacity gate, not a speed one: on a GDN
// hybrid only attention layers hold a KV cache, so the dtype decides how much context fits.
// See model.cpp for per-family measurements.
bool kv_nvfp4_default_safe(ModelArch arch);

// How many times the auto default's KV bytes/token an explicit dtype pin costs; 0 or 1 =
// free. Only families whose auto default is NVFP4 can lose here. A pin can invert without
// the config changing: e.g. IMP_KV_FP8=1 doubled bytes/token once the default flipped to
// NVFP4 (docs/plans/2026-08-29-qwen38-long-context-posture.md, trap 1).
int kv_pin_context_cost_factor(ModelArch arch, QType pinned);

// True when the KV dtype was CHOSEN rather than auto-resolved: a CLI flag sets the engine
// enum directly, imp.conf sets a string the resolver reads. Answers only "did the operator
// pick this"; whether the pick costs anything is kv_pin_context_cost_factor's question.
bool kv_dtype_is_explicit_pin(QType cli_dtype, const std::string& conf_dtype);

// max_seq_len from its two operator surfaces: preset (--max-seq-len/C-API) and file_key
// (runtime.max_seq_len). Rule: preset>0 wins, else file_key, else 0 (auto) - CLI beats
// file (AUDIT_arch_2026 G-5, the resolver used to let the file key overwrite the flag).
int max_seq_len_operator_value(int preset, int file_key);

// C API enum value for this architecture.
int model_arch_c_api_id(ModelArch arch);

// Sampling defaults from registry.
struct SamplingDefaults;
void model_arch_sampling_defaults(ModelArch arch, float& temperature, float& top_p, int& top_k);

// Parse architecture string (e.g. from GGUF "general.architecture")
ModelArch parse_model_arch(const std::string& s);

// True for encoder-only (BERT-style) architecture strings (bge/e5/nomic/jina GGUF ids). No
// causal LM head, need pooling; the generic-decoder fallback "succeeds" then hits a CUDA
// IMA on first request (#818). Loaders must reject with a clear error instead.
bool is_encoder_only_arch(const std::string& s);

// Apply arch-specific config defaults (call after loading metadata)
struct ModelConfig;
void apply_arch_defaults(ModelConfig& cfg);

}  // namespace imp
