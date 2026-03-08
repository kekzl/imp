#pragma once

#include "imp/config.h"

#include <string>
#include <vector>

namespace imp {

// A loaded preset configuration (from TOML or built-in defaults)
struct PresetConfig {
    std::string name;           // section ID, e.g. "qwen3-32b"
    std::string description;    // human-readable name
    std::vector<std::string> detect_patterns;  // glob patterns for auto-detection

    // Config overrides (only set values are applied)
    int max_seq_len = 0;        // 0 = not set
    int max_batch_size = 0;
    float temperature = -1.0f;  // <0 = not set
    float top_p = -1.0f;
    int top_k = -1;
    int gpu_layers = 0;         // 0 = not set (-1 means all on GPU)
    int kv_cache_dtype = -1;    // ImpDType or -1 = not set
    int ssm_state_dtype = -1;
    int use_nvfp4_decode = -2;  // -2 = not set
    int use_fp8_prefill = -1;   // -1 = not set
    int enable_cuda_graphs = -1;
    int enable_pdl = -1;
    int enable_speculative = -1;
    int spec_k = 0;
    int prefill_chunk_size = -1;
};

// Load presets from file. Search order:
//   1) explicit_path (if non-empty)
//   2) next to executable: <exe_dir>/presets.toml
//   3) ~/.config/imp/presets.toml
//   4) built-in defaults (always available)
void load_presets(const std::string& explicit_path = "");

// Auto-detect preset from model filename. Returns nullptr if no match.
const PresetConfig* detect_preset(const std::string& model_path);

// Find preset by explicit name. Returns nullptr if not found.
const PresetConfig* find_preset(const std::string& name);

// Apply preset to ImpConfig (merges defaults + preset overrides).
void apply_preset(const PresetConfig* preset, ImpConfig& config);

// Print all available presets to stderr.
void print_presets();

// Check if presets are loaded.
bool presets_loaded();

} // namespace imp
