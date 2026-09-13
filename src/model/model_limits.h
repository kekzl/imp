#pragma once

// Structural limits on what a checkpoint may declare about itself: every count that sizes
// a container comes from the file (SafeTensors header, config.json, GGUF metadata, tensor
// name) and needs a ceiling before a resize. A declared count (config.json/GGUF metadata)
// is refused outright; an index parsed from a tensor NAME is dropped with a counted warning.

#include "model/model_config.h"

#include <charconv>
#include <string>

namespace imp {

inline constexpr int kMaxModelLayers = 1024;
inline constexpr int kMaxModelExperts = 4096;
// Per-dimension ceilings (AUDIT_arch_2026 F1-10): head_dim sizes RoPE factor tables,
// others size scratch/vocab. Largest today: head_dim 512 (Gemma-4), hidden 16384
// (Llama-3.1-405B), intermediate 53248, vocab 262144 (Gemma), 128 heads.
inline constexpr int kMaxHeadDim = 4096;
inline constexpr int kMaxModelDim = 1 << 17;
inline constexpr int kMaxFfnDim = 1 << 20;
inline constexpr int kMaxVocabSize = 1 << 22;
inline constexpr int kMaxHeads = 4096;

// Parses a non-negative decimal index from a tensor-name component; -1 for empty,
// non-digit, or overflow. std::atoi is unfit here: UB on overflow, stops at the first
// non-digit instead of rejecting it.
inline int parse_index(const std::string& s) {
    if (s.empty())
        return -1;
    int value = 0;
    const char* begin = s.data();
    const char* end = begin + s.size();
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end || value < 0)
        return -1;
    return value;
}

// Check the counts a checkpoint declares about itself, before anything is
// sized from them. Returns false and fills `err` when a count is impossible;
// the caller refuses the load.
inline bool validate_declared_dimensions(const ModelConfig& cfg, std::string* err) {
    auto fail = [err](const char* what, int got, int limit) {
        if (err)
            *err = std::string(what) + " is " + std::to_string(got) + ", which exceeds the limit of " +
                   std::to_string(limit) + " (broken or hostile checkpoint)";
        return false;
    };
    if (cfg.n_layers < 0 || cfg.n_layers > kMaxModelLayers)
        return fail("declared layer count", cfg.n_layers, kMaxModelLayers);
    if (cfg.n_experts < 0 || cfg.n_experts > kMaxModelExperts)
        return fail("declared expert count", cfg.n_experts, kMaxModelExperts);
    // 0 is "not declared, infer later" for every field below.
    if (cfg.n_heads < 0 || cfg.n_heads > kMaxHeads)
        return fail("declared head count", cfg.n_heads, kMaxHeads);
    if (cfg.n_kv_heads < 0 || cfg.n_kv_heads > kMaxHeads)
        return fail("declared KV head count", cfg.n_kv_heads, kMaxHeads);
    if (cfg.head_dim < 0 || cfg.head_dim > kMaxHeadDim)
        return fail("declared head_dim", cfg.head_dim, kMaxHeadDim);
    if (cfg.rope_dim < 0 || cfg.rope_dim > kMaxHeadDim)
        return fail("declared rope_dim", cfg.rope_dim, kMaxHeadDim);
    if (cfg.d_model < 0 || cfg.d_model > kMaxModelDim)
        return fail("declared hidden size", cfg.d_model, kMaxModelDim);
    if (cfg.d_ff < 0 || cfg.d_ff > kMaxFfnDim)
        return fail("declared intermediate size", cfg.d_ff, kMaxFfnDim);
    if (cfg.expert_d_ff < 0 || cfg.expert_d_ff > kMaxFfnDim)
        return fail("declared expert intermediate size", cfg.expert_d_ff, kMaxFfnDim);
    if (cfg.vocab_size < 0 || cfg.vocab_size > kMaxVocabSize)
        return fail("declared vocab size", cfg.vocab_size, kMaxVocabSize);
    return true;
}

}  // namespace imp
