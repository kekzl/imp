#include "checkpoint_out.h"

#include "model/json_util.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

namespace imp::quantize {
namespace {

// FP4 E2M1's largest magnitude, which is what a tensor scale is placed against.
// A local copy so this stays free of the CUDA header that defines it for the
// kernel.
constexpr float kFp4E2M1Max = 6.0f;

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// The module suffixes an engine merges, and the name of the merged layer. Order
// matters only for readability; the sets are disjoint.
struct FusedSet {
    const char* group;
    std::array<const char*, 3> members;  // nullptr-terminated when shorter
};
constexpr std::array<FusedSet, 4> kFusedSets = {{
    {"qkv_proj", {"q_proj", "k_proj", "v_proj"}},
    {"gate_up_proj", {"gate_proj", "up_proj", nullptr}},
    {"in_proj_qkvz", {"in_proj_qkv", "in_proj_z", nullptr}},
    {"in_proj_ba", {"in_proj_b", "in_proj_a", nullptr}},
}};

// Minimal JSON string escape. Module names are plain identifiers today, but an
// unescaped quote would produce a config.json no reader accepts, so this is not
// left to the shape of the input.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ---- a scanner over one JSON object's top level -------------------------
//
// Enough of a parser to find where a key's VALUE starts and ends in the source
// text, and no more. Everything outside that span is copied byte for byte.

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        i++;
}

// Advance past a string literal whose opening quote is at s[i].
bool skip_string(const std::string& s, size_t& i) {
    if (i >= s.size() || s[i] != '"')
        return false;
    i++;
    while (i < s.size()) {
        if (s[i] == '\\') {
            i += 2;
            continue;
        }
        if (s[i] == '"') {
            i++;
            return true;
        }
        i++;
    }
    return false;
}

// Read a string literal at s[i] into `out`, resolving only the escapes that can
// appear in a key we compare against ASCII.
bool read_string(const std::string& s, size_t& i, std::string& out) {
    const size_t start = i;
    if (!skip_string(s, i))
        return false;
    out.assign(s, start + 1, i - start - 2);
    return true;
}

// Advance past one JSON value starting at s[i].
bool skip_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size())
        return false;
    const char c = s[i];
    if (c == '"')
        return skip_string(s, i);
    if (c == '{' || c == '[') {
        int depth = 0;
        while (i < s.size()) {
            if (s[i] == '"') {
                if (!skip_string(s, i))
                    return false;
                continue;
            }
            if (s[i] == '{' || s[i] == '[')
                depth++;
            else if (s[i] == '}' || s[i] == ']') {
                depth--;
                i++;
                if (depth == 0)
                    return true;
                continue;
            }
            i++;
        }
        return false;
    }
    // A scalar runs until the structural character that ends it.
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']')
        i++;
    return true;
}

}  // namespace

bool parse_output_format(const std::string& s, OutputFormat& out) {
    if (s == "modelopt") {
        out = OutputFormat::Modelopt;
        return true;
    }
    // `vllm` is accepted because that is what the format is FOR, and a user who
    // wants a checkpoint vLLM reads should not have to know the exporter's name.
    if (s == "compressed-tensors" || s == "compressed_tensors" || s == "vllm") {
        out = OutputFormat::CompressedTensors;
        return true;
    }
    return false;
}

const char* format_name(OutputFormat fmt) {
    return fmt == OutputFormat::Modelopt ? "modelopt" : "compressed-tensors";
}

QuantTensorNames quant_tensor_names(const std::string& base, OutputFormat fmt) {
    if (fmt == OutputFormat::CompressedTensors)
        return {base + ".weight_packed", base + ".weight_scale", base + ".weight_global_scale"};
    return {base + ".weight", base + ".weight_scale", base + ".weight_scale_2"};
}

float global_scale_value(float tensor_scale, OutputFormat fmt) {
    if (fmt == OutputFormat::Modelopt)
        return tensor_scale;
    return tensor_scale == 0.0f ? 0.0f : 1.0f / tensor_scale;
}

namespace {

// FP16 bits -> float. Written out rather than taken from cuda_fp16.h so this
// translation unit stays host-only and the CPU test lane can reach it.
float fp16_bits_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1Fu;
    const uint32_t mant = bits & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;  // +/- 0
        } else {
            // Subnormal: renormalize into a float32 exponent.
            uint32_t e = 0, m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e++;
            }
            m &= 0x3FFu;
            out = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);  // Inf / NaN
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

}  // namespace

float fp16_absmax(const uint16_t* data, size_t n) {
    uint16_t best = 0;
    for (size_t i = 0; i < n; i++) {
        const uint16_t mag = data[i] & 0x7FFFu;
        if (mag >= 0x7C00u)  // Inf or NaN — not a magnitude this scale should follow
            continue;
        if (mag > best)
            best = mag;
    }
    return fp16_bits_to_float(best);
}

float export_tensor_scale(float absmax) {
    // An all-zero tensor has no scale to derive; 1.0 keeps the division defined
    // and every value still quantizes to zero.
    if (!(absmax > 0.0f))
        return 1.0f;
    return absmax / kFp4E2M1Max;
}

const char* portability_warning(OutputFormat fmt, bool quantize_lm_head) {
    if (fmt == OutputFormat::CompressedTensors && quantize_lm_head)
        return "--lm-head with --format vllm: vLLM cannot load the result. Its ParallelLMHead\n"
               "takes no scales, so loading stops at 'no module or parameter named\n"
               "lm_head.weight_global_scale'. imp reads it fine and costs nothing EXTRA there,\n"
               "because its default already quantizes a native head at load — but that also\n"
               "makes the trade irreversible (gemm.nvfp4_lm_head=off can no longer buy the\n"
               "0.99%% perplexity back). Use it with --format modelopt when the model would\n"
               "otherwise not fit, or drop it for a checkpoint both engines read.";
    return nullptr;
}

std::string fusion_group_key(const std::string& weight_name) {
    static const std::string kW = ".weight";
    if (!ends_with(weight_name, kW))
        return "";
    const std::string base = weight_name.substr(0, weight_name.size() - kW.size());
    const size_t dot = base.rfind('.');
    if (dot == std::string::npos)
        return "";
    const std::string module = base.substr(dot + 1);
    const std::string prefix = base.substr(0, dot);
    for (const auto& set : kFusedSets) {
        for (const char* m : set.members) {
            if (m && module == m)
                return prefix + "|" + set.group;
        }
    }
    return "";
}

std::string compressed_tensors_quant_config(const std::vector<std::string>& ignore, bool calibrated) {
    // The weights entry mirrors what llm-compressor writes for
    // `nvfp4-pack-quantized`, because that is what vLLM's _is_nvfp4_format()
    // tests: tensor_group strategy, group_size 16, symmetric, float, 4 bits.
    // Any one of those wrong and vLLM silently picks a different scheme.
    std::string s;
    s += "{\n";
    s += "    \"config_groups\": {\n";
    s += "      \"group_0\": {\n";
    s += "        \"input_activations\": null,\n";
    s += "        \"output_activations\": null,\n";
    s += "        \"targets\": [\"Linear\"],\n";
    s += "        \"weights\": {\n";
    s += "          \"actorder\": null,\n";
    s += "          \"block_structure\": null,\n";
    s += "          \"dynamic\": false,\n";
    s += "          \"group_size\": 16,\n";
    s += "          \"num_bits\": 4,\n";
    s += std::string("          \"observer\": \"") + (calibrated ? "mse" : "memoryless_minmax") + "\",\n";
    s += "          \"observer_kwargs\": {},\n";
    s += "          \"scale_dtype\": \"torch.float8_e4m3fn\",\n";
    s += "          \"strategy\": \"tensor_group\",\n";
    s += "          \"symmetric\": true,\n";
    s += "          \"type\": \"float\",\n";
    s += "          \"zp_dtype\": null\n";
    s += "        }\n";
    s += "      }\n";
    s += "    },\n";
    s += "    \"format\": \"nvfp4-pack-quantized\",\n";
    s += "    \"global_compression_ratio\": null,\n";
    s += "    \"ignore\": [";
    for (size_t i = 0; i < ignore.size(); i++) {
        s += i ? ",\n      " : "\n      ";
        s += '"' + json_escape(ignore[i]) + '"';
    }
    s += ignore.empty() ? "]" : "\n    ]";
    s += ",\n";
    s += "    \"kv_cache_scheme\": null,\n";
    s += "    \"producer\": {\"name\": \"imp-quantize\", \"calibration\": \"";
    s += calibrated ? "awq" : "none";
    s += "\"},\n";
    s += "    \"quant_method\": \"compressed-tensors\",\n";
    s += "    \"quantization_status\": \"compressed\",\n";
    s += "    \"sparsity_config\": {},\n";
    s += "    \"transform_config\": {}\n";
    s += "  }";
    return s;
}

bool patch_config_json(const std::string& src, const std::string& quant_config_obj, std::string& out,
                       std::string& err) {
    static const std::string kKey = "quantization_config";
    size_t i = 0;
    skip_ws(src, i);
    if (i >= src.size() || src[i] != '{') {
        err = "config.json does not start with a JSON object";
        return false;
    }
    const size_t obj_open = i;
    i++;

    // Walk the top level looking for an existing key. Nested objects are
    // skipped whole, so a `quantization_config` inside some other section — or
    // the same word appearing in a string value — is not mistaken for this one.
    while (true) {
        skip_ws(src, i);
        if (i >= src.size()) {
            err = "config.json ends inside its top-level object";
            return false;
        }
        if (src[i] == '}') {
            // No such key: insert one right after the opening brace. The comma
            // is omitted for an otherwise empty object, which would otherwise
            // produce `{"k": {...},}`.
            const bool empty = (i == obj_open + 1) || [&] {
                size_t j = obj_open + 1;
                skip_ws(src, j);
                return j == i;
            }();
            out = src.substr(0, obj_open + 1) + "\n  \"" + kKey + "\": " + quant_config_obj +
                  (empty ? "\n" : ",\n") + src.substr(obj_open + 1);
            return true;
        }
        std::string key;
        if (!read_string(src, i, key)) {
            err = "malformed key in config.json";
            return false;
        }
        skip_ws(src, i);
        if (i >= src.size() || src[i] != ':') {
            err = "expected ':' after key '" + key + "' in config.json";
            return false;
        }
        i++;
        skip_ws(src, i);
        const size_t value_start = i;
        if (!skip_value(src, i)) {
            err = "malformed value for key '" + key + "' in config.json";
            return false;
        }
        if (key == kKey) {
            out = src.substr(0, value_start) + quant_config_obj + src.substr(i);
            return true;
        }
        skip_ws(src, i);
        if (i < src.size() && src[i] == ',') {
            i++;
            continue;
        }
        if (i < src.size() && src[i] == '}')
            continue;  // handled at the top of the loop
        err = "expected ',' or '}' in config.json";
        return false;
    }
}

// ---- the files -----------------------------------------------------------

namespace fs = std::filesystem;

bool can_declare_quantization(const std::string& in_dir, OutputFormat fmt, std::string& err) {
    if (fmt != OutputFormat::CompressedTensors || fs::exists(fs::path(in_dir) / "config.json"))
        return true;
    err = "compressed-tensors output needs a config.json to declare the quantization in, and " + in_dir +
          " has none";
    return false;
}

bool copy_aux_files(const std::string& in_dir, const std::string& out_dir, OutputFormat fmt,
                    const std::vector<std::string>& excluded_modules, bool calibrated, std::string& err) {
    static const char* kNames[] = {"config.json",
                                   "generation_config.json",
                                   "tokenizer.json",
                                   "tokenizer_config.json",
                                   "special_tokens_map.json",
                                   "added_tokens.json",
                                   "vocab.json",
                                   "merges.txt",
                                   "tokenizer.model",
                                   "chat_template.jinja",
                                   "chat_template.json"};
    std::set<std::string> copied;
    std::error_code ec;
    auto copy_one = [&](const fs::path& src, const std::string& name) {
        fs::copy_file(src, fs::path(out_dir) / name, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            err = "failed to copy " + name + ": " + ec.message();
            return false;
        }
        copied.insert(name);
        return true;
    };
    if (!can_declare_quantization(in_dir, fmt, err))
        return false;

    for (const char* n : kNames) {
        const fs::path src = fs::path(in_dir) / n;
        if (!fs::exists(src))
            continue;
        if (fmt == OutputFormat::CompressedTensors && std::string(n) == "config.json") {
            const std::string text = read_file(src.string());
            if (text.empty()) {
                err = "cannot read " + src.string();
                return false;
            }
            std::string patched;
            if (!patch_config_json(text, compressed_tensors_quant_config(excluded_modules, calibrated),
                                   patched, err))
                return false;
            std::ofstream f(fs::path(out_dir) / "config.json");
            f << patched;
            if (!f) {
                err = "cannot write config.json";
                return false;
            }
            copied.insert("config.json");
            continue;
        }
        if (!copy_one(src, n))
            return false;
    }

    // Then every remaining `*_config.json`, by pattern rather than by name, so a
    // preprocessor, video preprocessor or processor config travels without this
    // list having to learn each one. Two are excluded on purpose: the shard
    // index and hf_quant_config.json describe THIS write and are produced
    // separately, so copying the source's would contradict what was written.
    for (const auto& e : fs::directory_iterator(in_dir, ec)) {
        if (!e.is_regular_file())
            continue;
        const std::string name = e.path().filename().string();
        if (copied.count(name) || !ends_with(name, "_config.json"))
            continue;
        if (name == "hf_quant_config.json")
            continue;
        if (!copy_one(e.path(), name))
            return false;
    }
    return true;
}

bool write_modelopt_quant_config(const std::string& out_dir, const std::vector<std::string>& excluded,
                                 bool calibrated, std::string& err) {
    std::ofstream f(fs::path(out_dir) / "hf_quant_config.json");
    if (!f) {
        err = "cannot write hf_quant_config.json";
        return false;
    }
    f << "{\n"
      << "  \"producer\": { \"name\": \"imp-quantize\", \"version\": \"2\", \"calibration\": \""
      << (calibrated ? "awq" : "none") << "\" },\n"
      << "  \"quantization\": {\n"
      << "    \"quant_algo\": \"NVFP4\",\n"
      << "    \"kv_cache_quant_algo\": null,\n"
      << "    \"group_size\": 16,\n"
      << "    \"exclude_modules\": [";
    for (size_t i = 0; i < excluded.size(); i++)
        f << (i ? ", " : "") << '"' << json_escape(excluded[i]) << '"';
    f << "]\n  }\n}\n";
    return static_cast<bool>(f);
}

bool write_recipe_yaml(const std::string& out_dir, const std::vector<std::string>& excluded,
                       std::string& err) {
    std::ofstream f(fs::path(out_dir) / "recipe.yaml");
    if (!f) {
        err = "cannot write recipe.yaml";
        return false;
    }
    f << "default_stage:\n"
      << "  default_modifiers:\n"
      << "    QuantizationModifier:\n"
      << "      targets: [Linear]\n"
      << "      ignore: [";
    // Quoted individually: a module name is a plain identifier, but YAML flow
    // sequences treat a bare `re:...` entry as a mapping key.
    for (size_t i = 0; i < excluded.size(); i++)
        f << (i ? ", " : "") << '\'' << excluded[i] << '\'';
    f << "]\n"
      << "      scheme: NVFP4\n";
    return static_cast<bool>(f);
}

bool write_shard_index(const std::string& out_dir,
                       const std::vector<std::pair<std::string, std::string>>& tensor_to_shard,
                       size_t total_bytes, std::string& err) {
    std::ofstream f(fs::path(out_dir) / "model.safetensors.index.json");
    if (!f) {
        err = "cannot write model.safetensors.index.json";
        return false;
    }
    f << "{\n  \"metadata\": { \"total_size\": " << total_bytes << " },\n  \"weight_map\": {\n";
    for (size_t i = 0; i < tensor_to_shard.size(); i++)
        f << "    \"" << json_escape(tensor_to_shard[i].first) << "\": \"" << tensor_to_shard[i].second << '"'
          << (i + 1 < tensor_to_shard.size() ? ",\n" : "\n");
    f << "  }\n}\n";
    return static_cast<bool>(f);
}

}  // namespace imp::quantize
