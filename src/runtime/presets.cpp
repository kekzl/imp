#include "runtime/presets.h"
#include "core/toml_parser.h"
#include "core/logging.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace imp {

// --- Global state ---

static std::vector<PresetConfig> g_presets;
static PresetConfig g_defaults;  // [defaults] section values
static bool g_loaded = false;

// --- Helpers ---

static std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

static std::string file_basename(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    return (slash != std::string::npos) ? path.substr(slash + 1) : path;
}

static std::string exe_dir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "";
    buf[len] = '\0';
    std::string p(buf);
    size_t slash = p.find_last_of('/');
    return (slash != std::string::npos) ? p.substr(0, slash) : ".";
}

// Simple glob: * matches zero or more chars, rest is literal (case-insensitive)
static bool glob_match(const std::string& pattern, const std::string& text) {
    size_t pi = 0, ti = 0;
    size_t star_p = std::string::npos, star_t = 0;
    while (ti < text.size()) {
        if (pi < pattern.size() && pattern[pi] == '*') {
            star_p = pi++;
            star_t = ti;
        } else if (pi < pattern.size() && pattern[pi] == text[ti]) {
            ++pi; ++ti;
        } else if (star_p != std::string::npos) {
            pi = star_p + 1;
            ti = ++star_t;
        } else {
            return false;
        }
    }
    while (pi < pattern.size() && pattern[pi] == '*') ++pi;
    return pi == pattern.size();
}

// Parse dtype string to ImpDType value
static int parse_dtype_string(const std::string& s) {
    std::string l = to_lower(s);
    if (l == "fp32")      return IMP_DTYPE_FP32;
    if (l == "fp16")      return IMP_DTYPE_FP16;
    if (l == "bf16")      return IMP_DTYPE_BF16;
    if (l == "fp8_e4m3" || l == "fp8") return IMP_DTYPE_FP8_E4M3;
    if (l == "fp8_e5m2")  return IMP_DTYPE_FP8_E5M2;
    if (l == "int8")      return IMP_DTYPE_INT8;
    if (l == "int4")      return IMP_DTYPE_INT4;
    if (l == "fp4_e2m1" || l == "fp4" || l == "nvfp4") return IMP_DTYPE_FP4_E2M1;
    return -1;
}

// Load a PresetConfig from a TOML table
static PresetConfig load_preset_from_table(const std::string& name, const TomlTable& table) {
    PresetConfig p;
    p.name = name;
    p.description = toml_get_string(table, "description", name);
    p.detect_patterns = toml_get_string_array(table, "detect");

    if (toml_has(table, "max_seq_len"))
        p.max_seq_len = static_cast<int>(toml_get_int(table, "max_seq_len"));
    if (toml_has(table, "max_batch_size"))
        p.max_batch_size = static_cast<int>(toml_get_int(table, "max_batch_size"));
    if (toml_has(table, "temperature"))
        p.temperature = static_cast<float>(toml_get_float(table, "temperature"));
    if (toml_has(table, "top_p"))
        p.top_p = static_cast<float>(toml_get_float(table, "top_p"));
    if (toml_has(table, "top_k"))
        p.top_k = static_cast<int>(toml_get_int(table, "top_k"));
    if (toml_has(table, "gpu_layers"))
        p.gpu_layers = static_cast<int>(toml_get_int(table, "gpu_layers"));
    if (toml_has(table, "use_nvfp4_decode"))
        p.use_nvfp4_decode = static_cast<int>(toml_get_int(table, "use_nvfp4_decode"));
    if (toml_has(table, "use_fp8_prefill"))
        p.use_fp8_prefill = static_cast<int>(toml_get_int(table, "use_fp8_prefill"));
    if (toml_has(table, "enable_cuda_graphs"))
        p.enable_cuda_graphs = static_cast<int>(toml_get_int(table, "enable_cuda_graphs"));
    if (toml_has(table, "enable_pdl"))
        p.enable_pdl = static_cast<int>(toml_get_int(table, "enable_pdl"));
    if (toml_has(table, "enable_speculative"))
        p.enable_speculative = static_cast<int>(toml_get_int(table, "enable_speculative"));
    if (toml_has(table, "spec_k"))
        p.spec_k = static_cast<int>(toml_get_int(table, "spec_k"));
    if (toml_has(table, "prefill_chunk_size"))
        p.prefill_chunk_size = static_cast<int>(toml_get_int(table, "prefill_chunk_size"));

    std::string kv_dtype = toml_get_string(table, "kv_cache_dtype");
    if (!kv_dtype.empty()) p.kv_cache_dtype = parse_dtype_string(kv_dtype);

    std::string ssm_dtype = toml_get_string(table, "ssm_state_dtype");
    if (!ssm_dtype.empty()) p.ssm_state_dtype = parse_dtype_string(ssm_dtype);

    return p;
}

// Merge defaults into a preset (preset values take precedence)
static void merge_defaults(PresetConfig& preset, const PresetConfig& defaults) {
    if (preset.gpu_layers == 0 && defaults.gpu_layers != 0)
        preset.gpu_layers = defaults.gpu_layers;
    if (preset.kv_cache_dtype == -1 && defaults.kv_cache_dtype != -1)
        preset.kv_cache_dtype = defaults.kv_cache_dtype;
    if (preset.use_nvfp4_decode == -2 && defaults.use_nvfp4_decode != -2)
        preset.use_nvfp4_decode = defaults.use_nvfp4_decode;
    if (preset.use_fp8_prefill == -1 && defaults.use_fp8_prefill != -1)
        preset.use_fp8_prefill = defaults.use_fp8_prefill;
    if (preset.enable_cuda_graphs == -1 && defaults.enable_cuda_graphs != -1)
        preset.enable_cuda_graphs = defaults.enable_cuda_graphs;
    if (preset.enable_pdl == -1 && defaults.enable_pdl != -1)
        preset.enable_pdl = defaults.enable_pdl;
    if (preset.temperature < 0.0f && defaults.temperature >= 0.0f)
        preset.temperature = defaults.temperature;
    if (preset.top_p < 0.0f && defaults.top_p >= 0.0f)
        preset.top_p = defaults.top_p;
    if (preset.top_k == -1 && defaults.top_k != -1)
        preset.top_k = defaults.top_k;
}

// --- Built-in presets (fallback when no TOML file found) ---

static void load_builtin() {
    // Common defaults for all RTX 5090 presets
    g_defaults.gpu_layers = -1;
    g_defaults.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    g_defaults.use_nvfp4_decode = 2;
    g_defaults.use_fp8_prefill = 1;
    g_defaults.enable_cuda_graphs = 1;
    g_defaults.enable_pdl = 1;
    g_defaults.temperature = 0.6f;
    g_defaults.top_p = 0.95f;
    g_defaults.top_k = 0;

    auto add = [](const char* name, const char* desc,
                  std::vector<std::string> patterns,
                  int seq_len, int batch, int top_k = 0) {
        PresetConfig p;
        p.name = name;
        p.description = desc;
        p.detect_patterns = std::move(patterns);
        p.max_seq_len = seq_len;
        p.max_batch_size = batch;
        p.top_k = top_k;
        merge_defaults(p, g_defaults);
        g_presets.push_back(std::move(p));
    };

    // Top-5
    add("qwen3-coder-next", "Qwen3-Coder-Next (80B MoE, 3B active)",
        {"*qwen3*coder*next*", "*qwen3*coder*moe*"}, 65536, 1, 20);
    add("qwen3-coder-32b", "Qwen3-Coder-32B (dense, code)",
        {"*qwen3*coder*32b*"}, 131072, 4, 20);
    add("qwen3-coder-14b", "Qwen3-Coder-14B (dense, code)",
        {"*qwen3*coder*14b*"}, 131072, 8, 20);
    add("qwen3-coder-7b", "Qwen3-Coder-7B (dense, code)",
        {"*qwen3*coder*7b*"}, 131072, 8, 20);
    add("qwen3-32b", "Qwen3-32B (dense)",
        {"*qwen3*32b*"}, 131072, 4, 20);
    {
        PresetConfig p;
        p.name = "glm-4.7-flash";
        p.description = "GLM-4.7-Flash (30B MoE, 3B active)";
        p.detect_patterns = {"*glm*4.7*", "*glm*47*"};
        p.max_seq_len = 131072;
        p.max_batch_size = 1;
        p.enable_speculative = 1;
        p.spec_k = 3;
        merge_defaults(p, g_defaults);
        g_presets.push_back(std::move(p));
    }
    add("deepseek-r1-32b", "DeepSeek-R1-Distill-Qwen-32B (dense)",
        {"*deepseek*r1*"}, 65536, 1);
    add("mimo-v2-flash", "MiMo-V2-Flash (~7B dense)",
        {"*mimo*v2*", "*mimo*flash*"}, 262144, 16);

    // Qwen3 family
    add("qwen3-8b", "Qwen3-8B (dense)",
        {"*qwen3*8b*"}, 131072, 8, 20);
    add("qwen3-4b", "Qwen3-4B (dense)",
        {"*qwen3*4b*"}, 131072, 16, 20);
    add("qwen3-moe", "Qwen3-MoE",
        {"*qwen3*moe*"}, 65536, 1, 20);

    // Gemma-3
    add("gemma3-27b", "Gemma-3-27B (dense, text+vision)",
        {"*gemma*3*27b*"}, 131072, 4);
    add("gemma3-12b", "Gemma-3-12B (dense, text+vision)",
        {"*gemma*3*12b*"}, 131072, 8);
    add("gemma3-4b", "Gemma-3-4B (dense, text+vision)",
        {"*gemma*3*4b*"}, 131072, 16);

    // LLaMA 3
    add("llama3-70b", "LLaMA-3-70B (dense)",
        {"*llama*3*70b*"}, 32768, 1);
    add("llama3-8b", "LLaMA-3-8B (dense)",
        {"*llama*3*8b*"}, 131072, 8);

    // Mistral / Mixtral
    add("mistral-24b", "Mistral Small 3.2 (24B dense)",
        {"*mistral*24b*", "*mistral*small*3*"}, 131072, 4);
    add("mistral-7b", "Mistral-7B (dense)",
        {"*mistral*7b*"}, 131072, 8);
    add("mixtral-8x7b", "Mixtral-8x7B (MoE)",
        {"*mixtral*7b*", "*mixtral*8x7*"}, 32768, 4);
    {
        PresetConfig p;
        p.name = "mixtral-8x22b";
        p.description = "Mixtral-8x22B (MoE)";
        p.detect_patterns = {"*mixtral*22b*", "*mixtral*8x22*"};
        p.max_seq_len = 32768;
        p.max_batch_size = 1;
        p.use_nvfp4_decode = -1;  // auto
        merge_defaults(p, g_defaults);
        p.use_nvfp4_decode = -1;  // override after merge
        g_presets.push_back(std::move(p));
    }
    {
        PresetConfig p;
        p.name = "deepseek-v3";
        p.description = "DeepSeek-V3 (MoE, 37B active)";
        p.detect_patterns = {"*deepseek*v3*"};
        p.max_seq_len = 32768;
        p.max_batch_size = 1;
        p.use_nvfp4_decode = -1;
        merge_defaults(p, g_defaults);
        p.use_nvfp4_decode = -1;
        g_presets.push_back(std::move(p));
    }
    {
        PresetConfig p;
        p.name = "nemotron-h";
        p.description = "Nemotron-H (Mamba2+Attn+MoE hybrid)";
        p.detect_patterns = {"*nemotron*"};
        p.max_seq_len = 32768;
        p.max_batch_size = 1;
        p.ssm_state_dtype = IMP_DTYPE_FP16;
        merge_defaults(p, g_defaults);
        g_presets.push_back(std::move(p));
    }

    // Coding models
    add("devstral", "Devstral (24B dense, coding agent)",
        {"*devstral*"}, 131072, 4);
    add("codestral-25b", "Codestral 25.01 (22B dense, code)",
        {"*codestral*"}, 131072, 4);
    add("qwen2.5-coder-32b", "Qwen2.5-Coder-32B (dense, code)",
        {"*qwen2.5*coder*32b*", "*qwen25*coder*32b*"}, 131072, 4, 20);
    add("qwen2.5-coder-7b", "Qwen2.5-Coder-7B (dense, code)",
        {"*qwen2.5*coder*", "*qwen25*coder*"}, 131072, 8, 20);
    add("deepseek-coder", "DeepSeek-Coder-V2 (MoE, code)",
        {"*deepseek*coder*"}, 65536, 1);
}

// --- Load from TOML file ---

static void load_from_toml(const std::string& path) {
    TomlDocument doc = toml_parse_file(path);
    if (doc.empty()) return;

    // Load [defaults] section first
    const TomlTable* defaults_table = toml_find_section(doc, "defaults");
    if (defaults_table)
        g_defaults = load_preset_from_table("defaults", *defaults_table);

    // Load all other sections as presets
    for (auto& [section, table] : doc) {
        if (section == "defaults" || section.empty()) continue;
        PresetConfig p = load_preset_from_table(section, table);
        merge_defaults(p, g_defaults);
        g_presets.push_back(std::move(p));
    }

    IMP_LOG_INFO("Loaded %zu presets from %s", g_presets.size(), path.c_str());
}

// --- Public API ---

void load_presets(const std::string& explicit_path) {
    g_presets.clear();
    g_defaults = {};
    g_loaded = true;

    // 1) Explicit path
    if (!explicit_path.empty()) {
        load_from_toml(explicit_path);
        if (!g_presets.empty()) return;
        IMP_LOG_WARN("Presets file not found: %s — using built-in defaults", explicit_path.c_str());
    }

    // 2) Next to executable
    std::string dir = exe_dir();
    if (!dir.empty()) {
        std::string p = dir + "/presets.toml";
        load_from_toml(p);
        if (!g_presets.empty()) return;
    }

    // 3) ~/.config/imp/presets.toml
    if (const char* home = std::getenv("HOME")) {
        std::string p = std::string(home) + "/.config/imp/presets.toml";
        load_from_toml(p);
        if (!g_presets.empty()) return;
    }

    // 4) Built-in fallback
    load_builtin();
    IMP_LOG_INFO("Using %zu built-in presets", g_presets.size());
}

const PresetConfig* detect_preset(const std::string& model_path) {
    if (!g_loaded) load_presets();

    std::string name = to_lower(file_basename(model_path));

    // First match wins — order in file/builtin determines priority
    for (const auto& preset : g_presets) {
        for (const auto& pattern : preset.detect_patterns) {
            if (glob_match(to_lower(pattern), name))
                return &preset;
        }
    }
    return nullptr;
}

const PresetConfig* find_preset(const std::string& name) {
    if (!g_loaded) load_presets();
    if (name.empty()) return nullptr;

    std::string lower = to_lower(name);
    for (const auto& preset : g_presets) {
        if (to_lower(preset.name) == lower)
            return &preset;
    }
    return nullptr;
}

void apply_preset(const PresetConfig* preset, ImpConfig& config) {
    if (!preset) return;

    // Apply all set values
    if (preset->gpu_layers != 0)
        config.gpu_layers = preset->gpu_layers;
    if (preset->kv_cache_dtype != -1)
        config.kv_cache_dtype = static_cast<ImpDType>(preset->kv_cache_dtype);
    if (preset->ssm_state_dtype != -1)
        config.ssm_state_dtype = static_cast<ImpDType>(preset->ssm_state_dtype);
    if (preset->use_nvfp4_decode != -2)
        config.use_nvfp4_decode = preset->use_nvfp4_decode;
    if (preset->use_fp8_prefill != -1)
        config.use_fp8_prefill = preset->use_fp8_prefill;
    if (preset->enable_cuda_graphs != -1)
        config.enable_cuda_graphs = preset->enable_cuda_graphs;
    if (preset->enable_pdl != -1)
        config.enable_pdl = preset->enable_pdl;
    if (preset->max_seq_len > 0)
        config.max_seq_len = preset->max_seq_len;
    if (preset->max_batch_size > 0)
        config.max_batch_size = preset->max_batch_size;
    if (preset->temperature >= 0.0f)
        config.temperature = preset->temperature;
    if (preset->top_p >= 0.0f)
        config.top_p = preset->top_p;
    if (preset->top_k >= 0)
        config.top_k = preset->top_k;
    if (preset->enable_speculative != -1)
        config.enable_speculative = preset->enable_speculative;
    if (preset->spec_k > 0)
        config.spec_k = preset->spec_k;
    if (preset->prefill_chunk_size >= 0)
        config.prefill_chunk_size = preset->prefill_chunk_size;
}

void print_presets() {
    if (!g_loaded) load_presets();

    fprintf(stderr, "Available presets (auto-detected from model filename):\n\n");
    for (const auto& p : g_presets) {
        fprintf(stderr, "  %-24s %s\n", p.name.c_str(), p.description.c_str());
    }
    fprintf(stderr,
        "\nAuto-detected from filename. Override: --preset <name> / --preset none\n"
        "Custom presets: edit ~/.config/imp/presets.toml\n");
}

bool presets_loaded() {
    return g_loaded;
}

} // namespace imp
