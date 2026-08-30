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

// True iff this arch family has been empirically verified safe to honor a model
// author's kv_cache_quant_algo=FP8 hint BY DEFAULT (long-context FP8 KV). This is
// the long-context quality gate for kv_cache.dtype=auto; see the definition in
// model.cpp for the measured per-family evidence. Keep the list conservative.
bool kv_fp8_hint_default_safe(ModelArch arch);

// True iff this arch family has been empirically verified safe for DEFAULT FP8 KV
// even when the checkpoint declares NO kv_cache_quant_algo hint (GGUF exports never
// carry one, so the hint gate alone left GGUF long-context decode on FP16 KV —
// −39% at 16k on Qwen3-8B Q8_0). Stricter evidence bar than the hint list: the
// author didn't opt in, so the family must gate ~neutral, not merely ≤1.5%. See
// model.cpp for the per-family evidence and the measured exclusions.
bool kv_fp8_no_hint_default_safe(ModelArch arch);

// True iff this arch family has been measured safe for DEFAULT NVFP4 KV. This
// is a capacity gate, not a speed one: the KV cache is what bounds context
// length, and on a GDN hybrid only the attention layers hold one at all, so the
// dtype decides how much context fits rather than how fast decode runs. See
// model.cpp for the per-family measurements.
bool kv_nvfp4_default_safe(ModelArch arch);

// How many times the auto default's KV bytes per token an EXPLICIT dtype pin
// costs on this family; 0 or 1 when the pin costs nothing. Only families whose
// auto default is NVFP4 can lose here - everywhere else `auto` already resolves
// to FP8 or FP16, so pinning those is not a downgrade.
//
// This exists because a pin can invert without changing: `IMP_KV_FP8=1` in a
// compose file was correct when `auto` meant FP16, and since the NVFP4 default
// it DOUBLES the bytes per token on this family (max_model_len 90 528 instead
// of 126 432), while `--kv-fp8` sets the enum directly and skips the resolver
// that would have logged the trade. See
// docs/plans/2026-08-29-qwen38-long-context-posture.md, trap 1.
int kv_pin_context_cost_factor(ModelArch arch, QType pinned);

// True when the KV dtype was CHOSEN rather than resolved by `auto`. Two
// independent surfaces reach this: a CLI flag (--kv-fp8 and friends) sets the
// engine enum before the resolver runs, so anything but F16 there is the flag
// itself; imp.conf sets a string the resolver reads. Answers only "did the
// operator pick this" - whether the pick costs anything is
// kv_pin_context_cost_factor's question.
bool kv_dtype_is_explicit_pin(QType cli_dtype, const std::string& conf_dtype);

// C API enum value for this architecture.
int model_arch_c_api_id(ModelArch arch);

// Sampling defaults from registry.
struct SamplingDefaults;
void model_arch_sampling_defaults(ModelArch arch, float& temperature, float& top_p, int& top_k);

// Parse architecture string (e.g. from GGUF "general.architecture")
ModelArch parse_model_arch(const std::string& s);

// True for encoder-only (BERT-style embedding) architecture strings — the
// llama.cpp GGUF ids used by bge/e5/nomic/jina checkpoints. These have no
// causal LM head and need pooling; loading them via the generic-decoder
// fallback "succeeds" and then hits a CUDA illegal memory access on the first
// request (#818). Loaders must reject them with a clear error instead.
bool is_encoder_only_arch(const std::string& s);

// Apply arch-specific config defaults (call after loading metadata)
struct ModelConfig;
void apply_arch_defaults(ModelConfig& cfg);

}  // namespace imp
