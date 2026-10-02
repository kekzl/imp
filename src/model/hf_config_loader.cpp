#include "model/hf_config_loader.h"
#include "model/arch_registry.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "vision/qwen3vl_vision_config.h"
#include "vision/vision_model.h"
#include "model/multimodal_wrapper.h"
#include "model/llm_compressor_loader.h"
#include "core/logging.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace imp {

// ---- Architecture mapping ----

ModelArch HFConfigLoader::map_architecture(const std::string& hf_arch) {
    if (auto a = find_arch(ArchSource::HF_CLASS, hf_arch))
        return *a;

    IMP_LOG_WARN(
        "unknown HF architecture '%s' — falling back to GENERIC + tensor-name "
        "heuristics. If inference is incoherent, the architecture is likely "
        "unsupported. Add a class mapping in arch_registry.cpp.",
        hf_arch.c_str());
    return ModelArch::GENERIC;
}

namespace {

// Encoder-only (BERT/RoBERTa/embedding) models have no causal LM head; the SafeTensors/HF
// path must reject them like the GGUF loader does, or NomicBertModel falls through to the
// generic decoder and hits a CUDA IMA on first request (#818).
void reject_encoder_only(const std::string& name, const char* what) {
    if (is_encoder_only_arch(name)) {
        throw std::runtime_error(std::string("encoder-only ") + what + " '" + name +
                                 "' is not supported (imp runs causal decoder LMs; "
                                 "embedding encoders need pooling support)");
    }
}

// Architecture detection: prefer "architectures" array, fall back to "model_type".
void detect_arch(const JValue& root, ModelConfig& cfg) {
    const JValue* archs = jobj_find(root, "architectures");
    if (archs && archs->type == JType::ARRAY && !archs->arr.empty()) {
        reject_encoder_only(archs->arr[0].str_val, "architecture");
        cfg.arch = HFConfigLoader::map_architecture(wrapped_lm_architecture(root, archs->arr[0].str_val));
        if (archs->arr.size() > 1) {
            std::string dropped;
            for (size_t i = 1; i < archs->arr.size(); i++) {
                if (i > 1)
                    dropped += ", ";
                dropped += archs->arr[i].str_val;
            }
            IMP_LOG_WARN("config.json `architectures` has %zu entries; using '%s' and ignoring [%s]",
                         archs->arr.size(), archs->arr[0].str_val.c_str(), dropped.c_str());
        }
        return;
    }
    std::string model_type;
    if (!jobj_get_string(root, "model_type", model_type))
        return;
    reject_encoder_only(model_type, "model_type");
    if (auto a = find_arch(ArchSource::HF_MODEL_TYPE, model_type)) {
        cfg.arch = *a;
    } else {
        IMP_LOG_WARN(
            "unknown model_type '%s': falling back to GENERIC + tensor-name heuristics. If "
            "inference is incoherent, the architecture is likely unsupported.",
            model_type.c_str());
        cfg.arch = ModelArch::GENERIC;
    }
}

// Audio modality: unlike vision, an omni checkpoint with a lost audio path said nothing
// (roadmap Open 8). Test OBJECT not presence: Gemma-4-26B ships audio_config:null with no
// audio tensor; audio_token_id is set regardless of an encoder, so it is not a signal either.
void note_audio_config(const JValue& root, bool has_vision, ModelConfig& cfg) {
    const JValue* ac = jobj_find(root, "audio_config");
    if (!ac || ac->type != JType::OBJECT)
        return;
    cfg.has_audio_config = true;
    std::string audio_type;
    jobj_opt_string(*ac, "model_type", audio_type);
    IMP_LOG_WARN(
        "Audio modality present (audio_config, model_type='%s') and unsupported. imp has no "
        "audio encoder: `model.embed_audio.*` tensors are dropped at load and audio input "
        "cannot be sent. The model loads as text%s only (roadmap Open 8).",
        audio_type.empty() ? "?" : audio_type.c_str(), has_vision ? "+vision" : "");
}

}  // namespace

// ---- load_config ----

// Generic parsers (hf_config_generic.cpp), then the one hook the arch registry holds for
// cfg.arch (src/model/hf_config/, #2537).
bool HFConfigLoader::load_config(const std::string& model_dir, ModelConfig& cfg,
                                 std::unique_ptr<VisionModel>* out_vision_tower) {
    std::string path = model_dir + "/config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;
    IMP_LOG_INFO("loading HF config from %s", path.c_str());
    detect_arch(root, cfg);

    // Multimodal wrappers nest the text-model hyperparameters under `text_config`: that is the
    // effective root for every read below, and marks the checkpoint as a wrapper (weight_map
    // strips model.language_model. and drops the vision tower).
    const JValue* text_cfg = jobj_find(root, "text_config");
    const JValue& eff = (text_cfg && text_cfg->type == JType::OBJECT) ? *text_cfg : root;
    if (&eff != &root)
        cfg.multimodal_wrapper = true;

    if (!parse_hf_core_dims(eff, cfg) || !parse_hf_rope(eff, cfg))
        return false;
    parse_hf_ffn(root, eff, cfg);
    parse_hf_moe(eff, cfg);
    parse_hf_tristate_flags(root, eff, cfg);
    if (HfConfigHook hook = find_hf_config_hook(cfg.arch); hook && !hook(root, eff, cfg))
        return false;

    load_vision_tower_config(root, out_vision_tower);
    note_audio_config(root, out_vision_tower && *out_vision_tower, cfg);
    if (cfg.arch == ModelArch::GENERIC)
        cfg.arch_inferred_fallback = true;
    IMP_LOG_INFO("  arch=%s layers=%d d_model=%d heads=%d kv_heads=%d d_ff=%d vocab=%d",
                 model_arch_name(cfg.arch), cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff,
                 cfg.vocab_size);
    return true;
}

// ---- load_generation_config ----

bool HFConfigLoader::load_generation_config(const std::string& model_dir, GenerationConfig& cfg) {
    std::string path = model_dir + "/generation_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    // EOS IDs (single number or array)
    if (const JValue* eos = jobj_find(root, "eos_token_id")) {
        if (eos->type == JType::NUMBER) {
            cfg.eos_token_ids.push_back(static_cast<int32_t>(eos->num_val));
        } else if (eos->type == JType::ARRAY) {
            for (const auto& v : eos->arr) {
                if (v.type == JType::NUMBER) {
                    cfg.eos_token_ids.push_back(static_cast<int32_t>(v.num_val));
                }
            }
        }
    }

    // Sampling defaults. Each field is only overwritten if the JSON has it;
    // otherwise the struct's sentinel survives.
    jobj_opt_float(root, "temperature", cfg.temperature);
    jobj_opt_float(root, "top_p", cfg.top_p);
    jobj_opt_int(root, "top_k", cfg.top_k);
    jobj_opt_float(root, "repetition_penalty", cfg.repetition_penalty);

    // do_sample=false → force greedy. Authors set this when they want
    // deterministic output regardless of temperature.
    if (const JValue* ds = jobj_find(root, "do_sample")) {
        if (ds->type == JType::NUMBER && ds->num_val == 0.0) {
            cfg.temperature = 0.0f;
        }
    }

    IMP_LOG_INFO(
        "generation_config.json: eos=%zu temp=%.3g top_p=%.3g top_k=%d "
        "rep_penalty=%.3g",
        cfg.eos_token_ids.size(), cfg.temperature, cfg.top_p, cfg.top_k, cfg.repetition_penalty);
    return true;
}

// ---- load_chat_template ----

std::string HFConfigLoader::load_chat_template(const std::string& model_dir) {
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return "";

    // Case 1: chat_template is a plain string in tokenizer_config.json
    std::string chat_template;
    if (jobj_get_string(root, "chat_template", chat_template) && !chat_template.empty()) {
        IMP_LOG_INFO("loaded chat_template from tokenizer_config.json (%zu chars)", chat_template.size());
        return chat_template;
    }

    // Case 2: chat_template is an array of {name, template} objects in
    // tokenizer_config.json (HuggingFace format: [{"name":"default","template":"..."}, …]).
    const JValue* ct = jobj_find(root, "chat_template");
    if (ct && ct->type == JType::ARRAY) {
        // Prefer "default" entry
        for (const auto& entry : ct->arr) {
            if (entry.type != JType::OBJECT)
                continue;
            std::string name;
            if (!jobj_get_string(entry, "name", name))
                continue;
            if (name == "default") {
                if (jobj_get_string(entry, "template", chat_template) && !chat_template.empty()) {
                    IMP_LOG_INFO("loaded chat_template (default) from tokenizer_config.json (%zu chars)",
                                 chat_template.size());
                    return chat_template;
                }
            }
        }
        // Fallback: first entry with a valid template string
        for (const auto& entry : ct->arr) {
            if (entry.type != JType::OBJECT)
                continue;
            if (jobj_get_string(entry, "template", chat_template) && !chat_template.empty()) {
                std::string name;
                jobj_opt_string(entry, "name", name);
                IMP_LOG_INFO("loaded chat_template (%s) from tokenizer_config.json (%zu chars)",
                             name.empty() ? "unnamed" : name.c_str(), chat_template.size());
                return chat_template;
            }
        }
        IMP_LOG_WARN("chat_template array found but no usable entry in %s", path.c_str());
    }

    // Case 3: standalone chat_template.jinja beside tokenizer_config.json (e.g. Gemma-4
    // llm-compressor NVFP4 exports with no chat_template field in tokenizer_config.json).
    // Read raw; the Jinja2 engine consumes the string, no parsing needed here.
    std::string jinja_path = model_dir + "/chat_template.jinja";
    if (std::ifstream in(jinja_path); in.good()) {
        std::stringstream buf;
        buf << in.rdbuf();
        chat_template = buf.str();
        if (!chat_template.empty()) {
            IMP_LOG_INFO("loaded chat_template from chat_template.jinja (%zu chars)", chat_template.size());
            return chat_template;
        }
    }

    return "";
}

// ---- load_added_tokens ----

std::vector<HFConfigLoader::AddedToken> HFConfigLoader::load_added_tokens(const std::string& model_dir) {
    std::vector<AddedToken> result;
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return result;

    const JValue* added = jobj_find(root, "added_tokens_decoder");
    if (!added || added->type != JType::OBJECT)
        return result;

    for (const auto& [id_str, val] : added->obj) {
        if (val.type != JType::OBJECT)
            continue;
        AddedToken tok;
        tok.id = std::atoi(id_str.c_str());
        jobj_opt_string(val, "content", tok.content);
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

// ---- load_special_tokens_map ----

namespace {

// special_tokens_map.json token entries come as a plain string or a {"content":...} object.
// Extracts the content string from either; returns empty if neither shape matches.
std::string extract_token_content(const JValue& v) {
    if (v.type == JType::STRING)
        return v.str_val;
    if (v.type == JType::OBJECT) {
        std::string s;
        jobj_opt_string(v, "content", s);
        return s;
    }
    return {};
}

}  // namespace

bool HFConfigLoader::load_special_tokens_map(const std::string& model_dir, SpecialTokensMap& out) {
    std::string path = model_dir + "/special_tokens_map.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    if (const JValue* arr = jobj_find(root, "additional_special_tokens"); arr && arr->type == JType::ARRAY) {
        out.additional_special_tokens.reserve(arr->arr.size());
        for (const auto& v : arr->arr) {
            std::string s = extract_token_content(v);
            if (!s.empty())
                out.additional_special_tokens.push_back(std::move(s));
        }
    }

    if (const JValue* bos = jobj_find(root, "bos_token"))
        out.bos_token = extract_token_content(*bos);
    if (const JValue* eos = jobj_find(root, "eos_token"))
        out.eos_token = extract_token_content(*eos);
    if (const JValue* pad = jobj_find(root, "pad_token"))
        out.pad_token = extract_token_content(*pad);
    if (const JValue* unk = jobj_find(root, "unk_token"))
        out.unk_token = extract_token_content(*unk);

    IMP_LOG_INFO(
        "special_tokens_map.json: %zu additional_special_tokens, "
        "bos='%s' eos='%s' pad='%s' unk='%s'",
        out.additional_special_tokens.size(), out.bos_token.c_str(), out.eos_token.c_str(),
        out.pad_token.c_str(), out.unk_token.c_str());
    return true;
}

// ---- load_tokenizer_flags ----

bool HFConfigLoader::load_tokenizer_flags(const std::string& model_dir, TokenizerFlags& out) {
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    auto read_bool = [&](const char* key, int& field) {
        const JValue* v = jobj_find(root, key);
        if (v && v->type == JType::NUMBER) {
            field = (v->num_val != 0.0) ? 1 : 0;
        }
    };

    read_bool("add_bos_token", out.add_bos_token);
    read_bool("add_eos_token", out.add_eos_token);
    read_bool("add_prefix_space", out.add_prefix_space);
    read_bool("use_default_system_prompt", out.use_default_system_prompt);

    // Read BOS/EOS token content strings — may be plain strings or AddedToken
    // objects ({"__type": "AddedToken", "content": "...", ...}).
    // DeepSeek models use AddedToken objects for their custom BOS/EOS strings.
    if (const JValue* bos = jobj_find(root, "bos_token"))
        out.bos_token = extract_token_content(*bos);
    if (const JValue* eos = jobj_find(root, "eos_token"))
        out.eos_token = extract_token_content(*eos);

    IMP_LOG_INFO(
        "tokenizer_config.json: add_bos=%s add_eos=%s add_prefix_space=%s "
        "use_default_system_prompt=%s bos_token='%s' eos_token='%s'",
        out.add_bos_token < 0 ? "unset" : (out.add_bos_token ? "true" : "false"),
        out.add_eos_token < 0 ? "unset" : (out.add_eos_token ? "true" : "false"),
        out.add_prefix_space < 0 ? "unset" : (out.add_prefix_space ? "true" : "false"),
        out.use_default_system_prompt < 0 ? "unset" : (out.use_default_system_prompt ? "true" : "false"),
        out.bos_token.c_str(), out.eos_token.c_str());
    return true;
}

// ---- load_nvfp4_config ----

namespace {

bool file_exists_at(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

}  // namespace

bool HFConfigLoader::probe_vision_tower(const std::string& model_dir) {
    JValue root;
    if (!parse_json_file(model_dir + "/config.json", root))
        return false;
    const JValue* vc = jobj_find(root, "vision_config");
    if (!vc || vc->type != JType::OBJECT)
        return false;
    std::string vision_type;
    jobj_opt_string(*vc, "model_type", vision_type);
    return vision_tower_supported(vision_type);
}

bool HFConfigLoader::load_nvfp4_config(const std::string& model_dir, NvFP4Config& cfg) {
    bool has_modelopt = file_exists_at(model_dir + "/hf_quant_config.json");
    bool has_compressor = file_exists_at(model_dir + "/recipe.yaml");

    if (has_modelopt && has_compressor) {
        IMP_LOG_WARN("Both quant config files present in %s — preferring modelopt", model_dir.c_str());
    }

    if (has_modelopt) {
        // Existing modelopt parsing (unchanged from before).
        std::string path = model_dir + "/hf_quant_config.json";
        JValue root;
        if (!parse_json_file(path, root))
            return false;

        const JValue* quant = jobj_find(root, "quantization");
        if (!quant || quant->type != JType::OBJECT)
            return false;

        const JValue* algo = jobj_find(*quant, "quant_algo");
        if (!algo || algo->type != JType::STRING)
            return false;
        const bool plain_nvfp4 = (algo->str_val == "NVFP4" || algo->str_val == "nvfp4");
        const bool mixed = (algo->str_val == "MIXED_PRECISION");
        if (!plain_nvfp4 && !mixed)
            return false;

        // Read first so the per-tensor table below can override it: on a
        // MIXED_PRECISION export the group size belongs to the NVFP4 entries,
        // and a top-level value (if any) is the less specific statement.
        jobj_opt_int(*quant, "group_size", cfg.group_size);

        // MIXED_PRECISION carries no top-level algorithm; quantized_layers maps each tensor to its
        // own. is_nvfp4_prequant is set only if NVFP4 is actually present (drives the MoE expert
        // cache + VRAM budget). Per-tensor algorithms are counted only, to log any unexpected one.
        if (mixed) {
            const JValue* layers = jobj_find(*quant, "quantized_layers");
            if (!layers || layers->type != JType::OBJECT) {
                IMP_LOG_WARN(
                    "MIXED_PRECISION quant config without a quantized_layers table — "
                    "treating %s as unquantized",
                    path.c_str());
                return false;
            }
            for (const auto& [tensor_name, spec] : layers->obj) {
                (void)tensor_name;
                std::string ta;
                if (spec.type == JType::OBJECT)
                    jobj_opt_string(spec, "quant_algo", ta);
                if (ta.find("NVFP4") != std::string::npos) {
                    cfg.n_nvfp4_tensors++;
                    // group_size rides on the NVFP4 entries, not the top level.
                    if (cfg.n_nvfp4_tensors == 1)
                        jobj_opt_int(spec, "group_size", cfg.group_size);
                } else if (ta.find("FP8") != std::string::npos) {
                    cfg.n_fp8_tensors++;
                } else {
                    cfg.n_other_tensors++;
                }
            }
            if (cfg.n_nvfp4_tensors == 0) {
                IMP_LOG_WARN(
                    "MIXED_PRECISION quant config names no NVFP4 tensor (fp8=%d other=%d) — "
                    "not an NVFP4 checkpoint, treating %s as unquantized",
                    cfg.n_fp8_tensors, cfg.n_other_tensors, path.c_str());
                return false;
            }
            cfg.mixed_precision = true;
        }

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

        cfg.format = NvFP4Format::MODELOPT;
        if (cfg.mixed_precision) {
            IMP_LOG_INFO(
                "NVFP4 model (Model Optimizer, MIXED_PRECISION): group_size=%d, kv_cache=%s, "
                "per-tensor algos: %d NVFP4, %d FP8, %d unrecognised",
                cfg.group_size, cfg.kv_cache_quant_algo.c_str(), cfg.n_nvfp4_tensors, cfg.n_fp8_tensors,
                cfg.n_other_tensors);
            if (cfg.n_other_tensors > 0) {
                IMP_LOG_WARN(
                    "  %d tensor(s) name a quantization algorithm this build does not know. "
                    "They load as whatever dtype they carry — check the output.",
                    cfg.n_other_tensors);
            }
        } else {
            IMP_LOG_INFO("NVFP4 model (Model Optimizer): group_size=%d, kv_cache=%s, exclude=%zu modules",
                         cfg.group_size, cfg.kv_cache_quant_algo.c_str(), cfg.exclude_modules.size());
        }
        return true;
    }

    if (has_compressor && imp::llm_compressor::parse_recipe_yaml(model_dir, cfg)) {
        // The scheme comes from the recipe; the ignore list must come from config.json when
        // present. recipe.yaml records RUN patterns (re:.*router), config.json the expanded module
        // names (...router.proj); the two do not match the same modules (#1962).
        std::vector<std::string> expanded;
        if (imp::llm_compressor::read_config_ignore_list(model_dir, expanded)) {
            IMP_LOG_INFO("NVFP4 ignore list: %zu expanded names from config.json replace %zu recipe pattern(s)",
                         expanded.size(), cfg.exclude_modules.size());
            cfg.exclude_modules = std::move(expanded);
        }
        return true;
    }

    // No usable recipe.yaml: fall back to config.json's quantization_config (what HF/vLLM
    // read). A compressed-tensors export without a recipe used to read as "not quantized"; its
    // scales are the RECIPROCAL of Modelopt's, so weights load and run wrong by amax^2/36.
    return imp::llm_compressor::parse_compressed_tensors_config(model_dir, cfg);
}

// ---- load_mxfp4_config ----

bool HFConfigLoader::load_mxfp4_config(const std::string& model_dir, MxFP4Config& cfg) {
    // MXFP4 is declared at config.json:quantization_config (top-level, unlike NVFP4's separate
    // file). GPT-OSS/TorchAO exports use quant_method=="mxfp4" (case-insensitive) plus
    // block_size (typically 32, E8M0 scale per 32 elements).
    std::string path = model_dir + "/config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    const JValue* qc = jobj_find(root, "quantization_config");
    if (!qc || qc->type != JType::OBJECT)
        return false;

    std::string method;
    if (!jobj_get_string(*qc, "quant_method", method))
        return false;

    // Accept the common spellings.
    bool is_mxfp4 = (method == "mxfp4" || method == "MXFP4" || method == "mx_fp4");
    if (!is_mxfp4)
        return false;

    int bs = 0;
    if (!jobj_get_int(*qc, "block_size", bs)) {
        // Some exports nest the block size under a "weight" sub-object.
        const JValue* w = jobj_find(*qc, "weight");
        if (w && w->type == JType::OBJECT)
            jobj_opt_int(*w, "block_size", bs);
    }
    if (bs > 0)
        cfg.block_size = bs;

    IMP_LOG_INFO("MXFP4 config: quant_method=%s block_size=%d", method.c_str(), cfg.block_size);
    return true;
}

// ---- load_awq_config ----

bool HFConfigLoader::load_awq_config(const std::string& model_dir, AWQConfig& cfg) {
    // AWQ config lives in one of two places: config.json:quantization_config (HF-standard) or
    // a separate quant_config.json (older AutoAWQ). Fields: quant_method=="awq", bits,
    // group_size, zero_point, version (gemm/gemv/marlin).
    auto try_parse = [&](const std::string& path, bool nested_under_qc) -> bool {
        JValue root;
        if (!parse_json_file(path, root))
            return false;
        const JValue* base = &root;
        if (nested_under_qc) {
            const JValue* qc = jobj_find(root, "quantization_config");
            if (!qc || qc->type != JType::OBJECT)
                return false;
            base = qc;
        }
        std::string method;
        if (!jobj_get_string(*base, "quant_method", method))
            return false;
        if (method != "awq" && method != "AWQ")
            return false;

        jobj_opt_int(*base, "bits", cfg.bits);
        jobj_opt_int(*base, "w_bit", cfg.bits);  // older AutoAWQ field name
        jobj_opt_int(*base, "group_size", cfg.group_size);
        jobj_opt_int(*base, "q_group_size", cfg.group_size);  // older field name

        const JValue* zp = jobj_find(*base, "zero_point");
        if (zp && zp->type == JType::NUMBER)
            cfg.zero_point = (zp->num_val != 0.0);

        jobj_opt_string(*base, "version", cfg.version);
        return true;
    };

    if (try_parse(model_dir + "/config.json", /*nested_under_qc=*/true))
        return true;
    if (try_parse(model_dir + "/quant_config.json", /*nested_under_qc=*/false))
        return true;
    return false;
}

}  // namespace imp
