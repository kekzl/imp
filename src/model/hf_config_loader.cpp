#include "model/hf_config_loader.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// ---- Minimal JSON parser (local copy from safetensors_loader.cpp) ----
// TODO: extract to a shared json.h header

namespace {

enum class JType { NUL, STRING, NUMBER, ARRAY, OBJECT };

struct JValue {
    JType type = JType::NUL;
    std::string str_val;
    double num_val = 0.0;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj;

    int64_t as_int() const { return static_cast<int64_t>(num_val); }
};

class JsonParser {
public:
    explicit JsonParser(const char* data, size_t len)
        : data_(data), len_(len), pos_(0) {}

    JValue parse() {
        skip_ws();
        return parse_value();
    }

    bool ok() const { return !error_; }

private:
    const char* data_;
    size_t len_;
    size_t pos_;
    bool error_ = false;

    char peek() const {
        if (pos_ >= len_) return '\0';
        return data_[pos_];
    }

    char advance() {
        if (pos_ >= len_) { error_ = true; return '\0'; }
        return data_[pos_++];
    }

    void skip_ws() {
        while (pos_ < len_ && (data_[pos_] == ' ' || data_[pos_] == '\t' ||
                                data_[pos_] == '\n' || data_[pos_] == '\r')) {
            pos_++;
        }
    }

    bool expect(char c) {
        skip_ws();
        if (peek() == c) { advance(); return true; }
        error_ = true;
        return false;
    }

    JValue parse_value() {
        skip_ws();
        if (error_) return {};
        char c = peek();
        if (c == '"') return parse_string_value();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        error_ = true;
        return {};
    }

    JValue parse_string_value() {
        JValue v;
        v.type = JType::STRING;
        v.str_val = parse_string_raw();
        return v;
    }

    std::string parse_string_raw() {
        if (!expect('"')) return "";
        std::string s;
        while (pos_ < len_) {
            char c = advance();
            if (c == '"') return s;
            if (c == '\\') {
                if (pos_ >= len_) { error_ = true; return s; }
                char esc = advance();
                switch (esc) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/':  s += '/'; break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    case 'u': {
                        for (int i = 0; i < 4 && pos_ < len_; i++) advance();
                        s += '?';
                        break;
                    }
                    default: s += esc; break;
                }
            } else {
                s += c;
            }
        }
        error_ = true;
        return s;
    }

    JValue parse_number() {
        JValue v;
        v.type = JType::NUMBER;
        size_t start = pos_;
        if (peek() == '-') advance();
        while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9') advance();
        if (pos_ < len_ && data_[pos_] == '.') {
            advance();
            while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9') advance();
        }
        if (pos_ < len_ && (data_[pos_] == 'e' || data_[pos_] == 'E')) {
            advance();
            if (pos_ < len_ && (data_[pos_] == '+' || data_[pos_] == '-')) advance();
            while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9') advance();
        }
        std::string num_str(data_ + start, pos_ - start);
        v.num_val = std::stod(num_str);
        return v;
    }

    JValue parse_object() {
        JValue v;
        v.type = JType::OBJECT;
        if (!expect('{')) return v;
        skip_ws();
        if (peek() == '}') { advance(); return v; }
        while (!error_) {
            skip_ws();
            std::string key = parse_string_raw();
            if (!expect(':')) break;
            JValue val = parse_value();
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        expect('}');
        return v;
    }

    JValue parse_array() {
        JValue v;
        v.type = JType::ARRAY;
        if (!expect('[')) return v;
        skip_ws();
        if (peek() == ']') { advance(); return v; }
        while (!error_) {
            v.arr.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        expect(']');
        return v;
    }

    JValue parse_bool() {
        JValue v;
        v.type = JType::NUMBER;
        if (peek() == 't') {
            for (int i = 0; i < 4 && pos_ < len_; i++) advance();
            v.num_val = 1.0;
        } else {
            for (int i = 0; i < 5 && pos_ < len_; i++) advance();
            v.num_val = 0.0;
        }
        return v;
    }

    JValue parse_null() {
        JValue v;
        v.type = JType::NUL;
        for (int i = 0; i < 4 && pos_ < len_; i++) advance();
        return v;
    }
};

const JValue* jobj_find(const JValue& obj, const std::string& key) {
    for (const auto& kv : obj.obj) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

// ---- File I/O helper ----

std::string read_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return "";
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return ""; }
    std::string data(st.st_size, '\0');
    [[maybe_unused]] auto n = ::read(fd, data.data(), st.st_size);
    close(fd);
    return data;
}

// ---- Helper: parse a JSON file into a JValue ----

bool parse_json_file(const std::string& path, JValue& out) {
    std::string data = read_file(path);
    if (data.empty()) return false;

    JsonParser parser(data.c_str(), data.size());
    out = parser.parse();
    if (!parser.ok() || out.type != JType::OBJECT) {
        IMP_LOG_WARN("failed to parse JSON: %s", path.c_str());
        return false;
    }
    return true;
}

// ---- Helper: extract numeric field from JSON object ----

template<typename T>
bool jobj_get_int(const JValue& obj, const std::string& key, T& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::NUMBER) return false;
    out = static_cast<T>(v->num_val);
    return true;
}

bool jobj_get_float(const JValue& obj, const std::string& key, float& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::NUMBER) return false;
    out = static_cast<float>(v->num_val);
    return true;
}

bool jobj_get_string(const JValue& obj, const std::string& key, std::string& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::STRING) return false;
    out = v->str_val;
    return true;
}

} // anonymous namespace

// ---- Architecture mapping ----

ModelArch HFConfigLoader::map_architecture(const std::string& hf_arch) {
    static const std::unordered_map<std::string, ModelArch> arch_map = {
        {"LlamaForCausalLM",        ModelArch::LLAMA},
        {"MistralForCausalLM",      ModelArch::MISTRAL},
        {"MixtralForCausalLM",      ModelArch::MIXTRAL},
        {"Qwen2ForCausalLM",        ModelArch::QWEN3},
        {"Qwen2MoeForCausalLM",     ModelArch::QWEN3_MOE},
        {"Qwen3ForCausalLM",        ModelArch::QWEN3},
        {"Qwen3MoeForCausalLM",     ModelArch::QWEN3_MOE},
        {"Gemma2ForCausalLM",       ModelArch::GEMMA3},
        {"GemmaForCausalLM",        ModelArch::GEMMA3},
        {"Gemma3ForCausalLM",       ModelArch::GEMMA3},
        {"DeepseekV2ForCausalLM",   ModelArch::DEEPSEEK},
        {"DeepseekV3ForCausalLM",   ModelArch::DEEPSEEK},
        {"PhiForCausalLM",          ModelArch::LLAMA},
        {"Phi3ForCausalLM",         ModelArch::LLAMA},
        {"Phi3SmallForCausalLM",    ModelArch::LLAMA},
        {"InternLM2ForCausalLM",    ModelArch::LLAMA},
        {"Starcoder2ForCausalLM",   ModelArch::LLAMA},
        {"CohereForCausalLM",       ModelArch::LLAMA},
    };

    auto it = arch_map.find(hf_arch);
    if (it != arch_map.end()) return it->second;

    IMP_LOG_WARN("unknown HF architecture: %s, falling back to GENERIC", hf_arch.c_str());
    return ModelArch::GENERIC;
}

// ---- load_config ----

bool HFConfigLoader::load_config(const std::string& model_dir, ModelConfig& cfg) {
    std::string path = model_dir + "/config.json";
    JValue root;
    if (!parse_json_file(path, root)) return false;

    IMP_LOG_INFO("loading HF config from %s", path.c_str());

    // Architecture detection: prefer "architectures" array, fall back to "model_type"
    const JValue* archs = jobj_find(root, "architectures");
    if (archs && archs->type == JType::ARRAY && !archs->arr.empty()) {
        cfg.arch = map_architecture(archs->arr[0].str_val);
    } else {
        // Fallback: map model_type string to arch
        std::string model_type;
        if (jobj_get_string(root, "model_type", model_type)) {
            // Common model_type values → architecture class names
            static const std::unordered_map<std::string, std::string> type_to_class = {
                {"llama",     "LlamaForCausalLM"},
                {"mistral",   "MistralForCausalLM"},
                {"mixtral",   "MixtralForCausalLM"},
                {"qwen2",     "Qwen2ForCausalLM"},
                {"qwen2_moe", "Qwen2MoeForCausalLM"},
                {"qwen3",     "Qwen3ForCausalLM"},
                {"qwen3_moe", "Qwen3MoeForCausalLM"},
                {"gemma",     "GemmaForCausalLM"},
                {"gemma2",    "Gemma2ForCausalLM"},
                {"gemma3",    "Gemma3ForCausalLM"},
                {"deepseek_v2", "DeepseekV2ForCausalLM"},
                {"deepseek_v3", "DeepseekV3ForCausalLM"},
                {"phi",       "PhiForCausalLM"},
                {"phi3",      "Phi3ForCausalLM"},
                {"cohere",    "CohereForCausalLM"},
                {"starcoder2", "Starcoder2ForCausalLM"},
            };
            auto it = type_to_class.find(model_type);
            if (it != type_to_class.end()) {
                cfg.arch = map_architecture(it->second);
            } else {
                IMP_LOG_WARN("unknown model_type '%s', using GENERIC", model_type.c_str());
                cfg.arch = ModelArch::GENERIC;
            }
        }
    }

    // Core dimensions
    jobj_get_int(root, "hidden_size", cfg.d_model);
    jobj_get_int(root, "num_attention_heads", cfg.n_heads);
    jobj_get_int(root, "intermediate_size", cfg.d_ff);
    jobj_get_int(root, "num_hidden_layers", cfg.n_layers);
    jobj_get_int(root, "vocab_size", cfg.vocab_size);
    jobj_get_int(root, "max_position_embeddings", cfg.max_seq_len);
    jobj_get_int(root, "head_dim", cfg.head_dim);

    // KV heads: default to n_heads (MHA) if not specified
    if (!jobj_get_int(root, "num_key_value_heads", cfg.n_kv_heads)) {
        cfg.n_kv_heads = cfg.n_heads;
    }

    // Norm epsilon: try rms_norm_eps first, then layer_norm_eps
    if (!jobj_get_float(root, "rms_norm_eps", cfg.rms_norm_eps)) {
        jobj_get_float(root, "layer_norm_eps", cfg.rms_norm_eps);
    }

    // RoPE
    jobj_get_float(root, "rope_theta", cfg.rope_theta);

    // RoPE scaling (object with type, factor, and optional YaRN/LongRoPE params)
    const JValue* rope_scaling = jobj_find(root, "rope_scaling");
    if (rope_scaling && rope_scaling->type == JType::OBJECT) {
        std::string rope_type;
        jobj_get_string(*rope_scaling, "type", rope_type);
        // Also check "rope_type" (some HF configs use this instead)
        if (rope_type.empty()) {
            jobj_get_string(*rope_scaling, "rope_type", rope_type);
        }

        float factor = 1.0f;
        jobj_get_float(*rope_scaling, "factor", factor);

        if (rope_type == "linear") {
            cfg.rope_freq_scale = 1.0f / factor;
        } else if (rope_type == "yarn") {
            cfg.rope_freq_scale = 1.0f / factor;
            jobj_get_float(*rope_scaling, "attn_factor", cfg.yarn_attn_factor);
            jobj_get_float(*rope_scaling, "beta_fast", cfg.yarn_beta_fast);
            jobj_get_float(*rope_scaling, "beta_slow", cfg.yarn_beta_slow);
            jobj_get_int(*rope_scaling, "original_max_position_embeddings",
                         cfg.rope_n_ctx_orig);
            // YaRN uses ext_factor=1.0 by default
            cfg.yarn_ext_factor = 1.0f;
        } else if (rope_type == "longrope" || rope_type == "long_rope") {
            // LongRoPE: per-dimension frequency scaling factors
            jobj_get_int(*rope_scaling, "original_max_position_embeddings",
                         cfg.rope_scaling_orig_max_pos);

            const JValue* short_f = jobj_find(*rope_scaling, "short_factor");
            if (short_f && short_f->type == JType::ARRAY) {
                cfg.rope_short_factor.reserve(short_f->arr.size());
                for (const auto& v : short_f->arr) {
                    cfg.rope_short_factor.push_back(static_cast<float>(v.num_val));
                }
            }
            const JValue* long_f = jobj_find(*rope_scaling, "long_factor");
            if (long_f && long_f->type == JType::ARRAY) {
                cfg.rope_long_factor.reserve(long_f->arr.size());
                for (const auto& v : long_f->arr) {
                    cfg.rope_long_factor.push_back(static_cast<float>(v.num_val));
                }
            }
        }
        // "dynamic" uses the same factor as linear at runtime
        else if (rope_type == "dynamic") {
            cfg.rope_freq_scale = 1.0f / factor;
        }
    }

    // Sliding window attention
    jobj_get_int(root, "sliding_window", cfg.sliding_window);

    // Softcapping (Gemma-2/3)
    jobj_get_float(root, "attn_logit_softcapping", cfg.attn_logit_softcap);
    jobj_get_float(root, "final_logit_softcapping", cfg.final_logit_softcap);

    // FFN activation
    std::string hidden_act;
    if (jobj_get_string(root, "hidden_act", hidden_act) ||
        jobj_get_string(root, "hidden_activation", hidden_act)) {
        if (hidden_act == "silu" || hidden_act == "swiglu") {
            cfg.ffn_activation = FFNActivation::SWIGLU;
        } else if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh" ||
                   hidden_act == "geglu") {
            cfg.ffn_activation = FFNActivation::GEGLU;
        }
    }

    // MoE config
    if (!jobj_get_int(root, "num_local_experts", cfg.n_experts)) {
        jobj_get_int(root, "num_experts", cfg.n_experts);
    }
    jobj_get_int(root, "num_experts_per_tok", cfg.n_experts_active);
    if (!jobj_get_int(root, "moe_intermediate_size", cfg.expert_d_ff)) {
        jobj_get_int(root, "expert_intermediate_size", cfg.expert_d_ff);
    }

    // tie_word_embeddings is informational (logged but not stored in ModelConfig)
    const JValue* tie = jobj_find(root, "tie_word_embeddings");
    if (tie && tie->type == JType::NUMBER && tie->num_val != 0.0) {
        IMP_LOG_INFO("  tie_word_embeddings = true");
    }

    IMP_LOG_INFO("  arch=%s layers=%d d_model=%d heads=%d kv_heads=%d d_ff=%d vocab=%d",
                 model_arch_name(cfg.arch), cfg.n_layers, cfg.d_model,
                 cfg.n_heads, cfg.n_kv_heads, cfg.d_ff, cfg.vocab_size);

    return true;
}

// ---- load_generation_config ----

bool HFConfigLoader::load_generation_config(const std::string& model_dir,
                                             std::vector<int32_t>& eos_token_ids) {
    std::string path = model_dir + "/generation_config.json";
    JValue root;
    if (!parse_json_file(path, root)) return false;

    const JValue* eos = jobj_find(root, "eos_token_id");
    if (!eos) return false;

    if (eos->type == JType::NUMBER) {
        // Single EOS token ID
        eos_token_ids.push_back(static_cast<int32_t>(eos->num_val));
    } else if (eos->type == JType::ARRAY) {
        // Array of EOS token IDs
        for (const auto& v : eos->arr) {
            if (v.type == JType::NUMBER) {
                eos_token_ids.push_back(static_cast<int32_t>(v.num_val));
            }
        }
    } else {
        return false;
    }

    IMP_LOG_INFO("loaded %zu EOS token IDs from generation_config.json",
                 eos_token_ids.size());
    return true;
}

// ---- load_chat_template ----

std::string HFConfigLoader::load_chat_template(const std::string& model_dir) {
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root)) return "";

    // Case 1: chat_template is a plain string
    std::string chat_template;
    if (jobj_get_string(root, "chat_template", chat_template)) {
        IMP_LOG_INFO("loaded chat_template from tokenizer_config.json (%zu chars)",
                     chat_template.size());
        return chat_template;
    }

    // Case 2: chat_template is an array of {name, template} objects
    // HuggingFace format: [{"name": "default", "template": "..."}, ...]
    const JValue* ct = jobj_find(root, "chat_template");
    if (ct && ct->type == JType::ARRAY) {
        // Prefer "default" entry
        for (const auto& entry : ct->arr) {
            if (entry.type != JType::OBJECT) continue;
            std::string name;
            if (!jobj_get_string(entry, "name", name)) continue;
            if (name == "default") {
                if (jobj_get_string(entry, "template", chat_template)) {
                    IMP_LOG_INFO("loaded chat_template (default) from tokenizer_config.json (%zu chars)",
                                 chat_template.size());
                    return chat_template;
                }
            }
        }
        // Fallback: first entry with a valid template string
        for (const auto& entry : ct->arr) {
            if (entry.type != JType::OBJECT) continue;
            if (jobj_get_string(entry, "template", chat_template)) {
                std::string name;
                jobj_get_string(entry, "name", name);
                IMP_LOG_INFO("loaded chat_template (%s) from tokenizer_config.json (%zu chars)",
                             name.empty() ? "unnamed" : name.c_str(), chat_template.size());
                return chat_template;
            }
        }
        IMP_LOG_WARN("chat_template array found but no usable entry in %s", path.c_str());
    }

    return "";
}

// ---- load_added_tokens ----

std::vector<HFConfigLoader::AddedToken> HFConfigLoader::load_added_tokens(
        const std::string& model_dir) {
    std::vector<AddedToken> result;
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root)) return result;

    const JValue* added = jobj_find(root, "added_tokens_decoder");
    if (!added || added->type != JType::OBJECT) return result;

    for (const auto& [id_str, val] : added->obj) {
        if (val.type != JType::OBJECT) continue;
        AddedToken tok;
        tok.id = std::atoi(id_str.c_str());
        jobj_get_string(val, "content", tok.content);
        // "special" field — treat as bool via number (true=1.0, false=0.0)
        const JValue* sp = jobj_find(val, "special");
        tok.special = sp && sp->type == JType::NUMBER && sp->num_val != 0.0;
        if (!tok.content.empty()) {
            result.push_back(std::move(tok));
        }
    }

    if (!result.empty()) {
        IMP_LOG_INFO("loaded %zu added tokens from tokenizer_config.json", result.size());
    }
    return result;
}

// ---- load_gptq_config ----

bool HFConfigLoader::load_gptq_config(const std::string& model_dir, GPTQConfig& cfg) {
    std::string path = model_dir + "/quantize_config.json";
    JValue root;
    if (!parse_json_file(path, root)) return false;

    IMP_LOG_INFO("loading GPTQ config from %s", path.c_str());

    jobj_get_int(root, "bits", cfg.bits);
    jobj_get_int(root, "group_size", cfg.group_size);

    const JValue* da = jobj_find(root, "desc_act");
    cfg.desc_act = da && da->type == JType::NUMBER && da->num_val != 0.0;

    IMP_LOG_INFO("  GPTQ: bits=%d group_size=%d desc_act=%s",
                 cfg.bits, cfg.group_size, cfg.desc_act ? "true" : "false");
    return true;
}

// ---- load_nvfp4_config ----

bool HFConfigLoader::load_nvfp4_config(const std::string& model_dir, NvFP4Config& cfg) {
    std::string path = model_dir + "/hf_quant_config.json";
    JValue root;
    if (!parse_json_file(path, root)) return false;

    const JValue* quant = jobj_find(root, "quantization");
    if (!quant || quant->type != JType::OBJECT) return false;

    // Check algorithm is NVFP4
    const JValue* algo = jobj_find(*quant, "quant_algo");
    if (!algo || algo->type != JType::STRING) return false;
    if (algo->str_val != "NVFP4" && algo->str_val != "nvfp4") return false;

    jobj_get_int(*quant, "group_size", cfg.group_size);

    const JValue* kv_algo = jobj_find(*quant, "kv_cache_quant_algo");
    if (kv_algo && kv_algo->type == JType::STRING)
        cfg.kv_cache_quant_algo = kv_algo->str_val;

    const JValue* exclude = jobj_find(*quant, "exclude_modules");
    if (exclude && exclude->type == JType::ARRAY) {
        for (const auto& v : exclude->arr) {
            if (v.type == JType::STRING)
                cfg.exclude_modules.push_back(v.str_val);
        }
    }

    IMP_LOG_INFO("NVFP4 model (Model Optimizer): group_size=%d, kv_cache=%s, exclude=%zu modules",
                 cfg.group_size, cfg.kv_cache_quant_algo.c_str(), cfg.exclude_modules.size());
    return true;
}

} // namespace imp
