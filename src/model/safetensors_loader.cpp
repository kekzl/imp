#include "model/safetensors_loader.h"
#include "model/model_arch.h"
#include "model/weight_map.h"
#include "model/hf_config_loader.h"
#include "model/llm_compressor_loader.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

namespace imp {

// ---- Minimal JSON parser for SafeTensors headers ----

// JSON value types we care about
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
    explicit JsonParser(const char* data, size_t len) : data_(data), len_(len), pos_(0) {}

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
        if (pos_ >= len_)
            return '\0';
        return data_[pos_];
    }

    char advance() {
        if (pos_ >= len_) {
            error_ = true;
            return '\0';
        }
        return data_[pos_++];
    }

    void skip_ws() {
        while (pos_ < len_ &&
               (data_[pos_] == ' ' || data_[pos_] == '\t' || data_[pos_] == '\n' || data_[pos_] == '\r')) {
            pos_++;
        }
    }

    bool expect(char c) {
        skip_ws();
        if (peek() == c) {
            advance();
            return true;
        }
        error_ = true;
        return false;
    }

    JValue parse_value() {
        skip_ws();
        if (error_)
            return {};
        char c = peek();
        if (c == '"')
            return parse_string_value();
        if (c == '{')
            return parse_object();
        if (c == '[')
            return parse_array();
        if (c == 't' || c == 'f')
            return parse_bool();
        if (c == 'n')
            return parse_null();
        if (c == '-' || (c >= '0' && c <= '9'))
            return parse_number();
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
        if (!expect('"'))
            return "";
        std::string s;
        while (pos_ < len_) {
            char c = advance();
            if (c == '"')
                return s;
            if (c == '\\') {
                if (pos_ >= len_) {
                    error_ = true;
                    return s;
                }
                char esc = advance();
                switch (esc) {
                    case '"':
                        s += '"';
                        break;
                    case '\\':
                        s += '\\';
                        break;
                    case '/':
                        s += '/';
                        break;
                    case 'b':
                        s += '\b';
                        break;
                    case 'f':
                        s += '\f';
                        break;
                    case 'n':
                        s += '\n';
                        break;
                    case 'r':
                        s += '\r';
                        break;
                    case 't':
                        s += '\t';
                        break;
                    case 'u': {
                        // Skip unicode escapes (consume 4 hex digits, emit '?')
                        for (int i = 0; i < 4 && pos_ < len_; i++)
                            advance();
                        s += '?';
                        break;
                    }
                    default:
                        s += esc;
                        break;
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
        if (peek() == '-')
            advance();
        while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9')
            advance();
        if (pos_ < len_ && data_[pos_] == '.') {
            advance();
            while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9')
                advance();
        }
        if (pos_ < len_ && (data_[pos_] == 'e' || data_[pos_] == 'E')) {
            advance();
            if (pos_ < len_ && (data_[pos_] == '+' || data_[pos_] == '-'))
                advance();
            while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9')
                advance();
        }
        std::string num_str(data_ + start, pos_ - start);
        v.num_val = std::stod(num_str);
        return v;
    }

    JValue parse_object() {
        JValue v;
        v.type = JType::OBJECT;
        if (!expect('{'))
            return v;
        skip_ws();
        if (peek() == '}') {
            advance();
            return v;
        }
        while (!error_) {
            skip_ws();
            std::string key = parse_string_raw();
            if (!expect(':'))
                break;
            JValue val = parse_value();
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (peek() == ',') {
                advance();
                continue;
            }
            break;
        }
        expect('}');
        return v;
    }

    JValue parse_array() {
        JValue v;
        v.type = JType::ARRAY;
        if (!expect('['))
            return v;
        skip_ws();
        if (peek() == ']') {
            advance();
            return v;
        }
        while (!error_) {
            v.arr.push_back(parse_value());
            skip_ws();
            if (peek() == ',') {
                advance();
                continue;
            }
            break;
        }
        expect(']');
        return v;
    }

    JValue parse_bool() {
        JValue v;
        v.type = JType::NUMBER;
        if (peek() == 't') {
            // true
            for (int i = 0; i < 4 && pos_ < len_; i++)
                advance();
            v.num_val = 1.0;
        } else {
            // false
            for (int i = 0; i < 5 && pos_ < len_; i++)
                advance();
            v.num_val = 0.0;
        }
        return v;
    }

    JValue parse_null() {
        JValue v;
        v.type = JType::NUL;
        for (int i = 0; i < 4 && pos_ < len_; i++)
            advance();
        return v;
    }
};

// ---- Helper: find a key in a JValue object ----

static const JValue* jobj_find(const JValue& obj, const std::string& key) {
    for (const auto& kv : obj.obj) {
        if (kv.first == key)
            return &kv.second;
    }
    return nullptr;
}

// ---- SafeTensors dtype string to QType ----

static QType safetensors_dtype(const std::string& s) {
    if (s == "F32")
        return QType::F32;
    if (s == "F16")
        return QType::F16;
    if (s == "BF16")
        return QType::BF16;
    if (s == "F64")
        return QType::F32;  // closest proxy
    if (s == "I8")
        return QType::INT8;
    if (s == "U8")
        return QType::INT8;  // treat unsigned byte as INT8
    if (s == "I16")
        return QType::INT32;  // closest proxy
    if (s == "I32")
        return QType::INT32;
    if (s == "I64")
        return QType::INT32;  // closest proxy
    if (s == "BOOL")
        return QType::INT8;
    if (s == "F8_E4M3")
        return QType::FP8_E4M3;
    if (s == "F8_E5M2") {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_WARN(
                "SafeTensors F8_E5M2 tensors found; mapping to FP8_E4M3 as a "
                "lossy proxy (no native E5M2 path). Activation-style tensors "
                "may lose precision. (Logged once.)");
        }
        return QType::FP8_E4M3;
    }
    IMP_LOG_WARN("Unknown SafeTensors dtype '%s', defaulting to FP32", s.c_str());
    return QType::F32;
}

// ---- Architecture detection from weight names ----

static ModelArch detect_arch_from_weights(const std::unordered_map<std::string, Tensor>& tensors) {
    bool has_block_sparse_moe = false;
    bool has_mlp_experts = false;
    bool has_ssm = false;
    bool has_layers = false;
    bool has_gptq = false;

    for (const auto& kv : tensors) {
        const auto& name = kv.first;
        if (name.find("model.layers") != std::string::npos)
            has_layers = true;
        if (name.find("block_sparse_moe") != std::string::npos)
            has_block_sparse_moe = true;
        if (name.find("mlp.experts") != std::string::npos)
            has_mlp_experts = true;
        if (name.find("mamba") != std::string::npos || name.find("ssm") != std::string::npos)
            has_ssm = true;
        if (name.find(".qweight") != std::string::npos)
            has_gptq = true;
    }

    if (has_gptq) {
        IMP_LOG_INFO("Detected GPTQ quantized weights");
    }

    if (has_ssm)
        return ModelArch::NEMOTRON_H_MOE;
    if (has_mlp_experts)
        return ModelArch::DEEPSEEK;
    if (has_block_sparse_moe)
        return ModelArch::MIXTRAL;
    if (has_layers)
        return ModelArch::LLAMA;
    return ModelArch::GENERIC;
}

// ---- Extract layer index from a HuggingFace weight name ----
// e.g. "model.layers.5.self_attn.q_proj.weight" -> 5
// Returns -1 if not a layer weight.

static int extract_layer_index(const std::string& name) {
    const char* prefix = "model.layers.";
    size_t plen = std::strlen(prefix);
    if (name.compare(0, plen, prefix) != 0)
        return -1;

    int idx = 0;
    size_t i = plen;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        idx = idx * 10 + (name[i] - '0');
        i++;
    }
    if (i == plen)
        return -1;  // no digits found
    return idx;
}

// ---- Infer max layer index to determine n_layers ----

static int infer_n_layers(const std::unordered_map<std::string, Tensor>& tensors) {
    int max_idx = -1;
    for (const auto& kv : tensors) {
        int idx = extract_layer_index(kv.first);
        if (idx > max_idx)
            max_idx = idx;
    }
    return max_idx + 1;  // 0-indexed, so count = max + 1
}

// ---- Infer model config from weight shapes ----

static void infer_config(ModelConfig& cfg, const std::unordered_map<std::string, Tensor>& tensors) {
    // Only infer fields that are still at their default (zero) values.
    // config.json (via HFConfigLoader) is authoritative when present.

    if (cfg.n_layers == 0)
        cfg.n_layers = infer_n_layers(tensors);

    // token embedding: shape = [vocab_size, d_model]
    auto it = tensors.find("model.embed_tokens.weight");
    if (it != tensors.end() && it->second.ndim == 2) {
        if (cfg.vocab_size == 0)
            cfg.vocab_size = static_cast<int>(it->second.shape[0]);
        if (cfg.d_model == 0)
            cfg.d_model = static_cast<int>(it->second.shape[1]);
    }

    // Only infer heads from weights if config.json didn't set them
    if (cfg.n_heads == 0) {
        auto it_q = tensors.find("model.layers.0.self_attn.q_proj.weight");
        auto it_k = tensors.find("model.layers.0.self_attn.k_proj.weight");
        if (it_q != tensors.end() && it_q->second.ndim == 2 && cfg.d_model > 0) {
            int q_out = static_cast<int>(it_q->second.shape[0]);
            int head_dim = cfg.d_model;
            for (int hd : {128, 64, 96, 80, 256}) {
                if (q_out % hd == 0) {
                    cfg.n_heads = q_out / hd;
                    head_dim = hd;
                    break;
                }
            }
            if (cfg.n_kv_heads == 0 && it_k != tensors.end() && it_k->second.ndim == 2 && head_dim > 0) {
                cfg.n_kv_heads = static_cast<int>(it_k->second.shape[0]) / head_dim;
            }
        }
    }

    if (cfg.d_ff == 0) {
        auto it_gate = tensors.find("model.layers.0.mlp.gate_proj.weight");
        if (it_gate != tensors.end() && it_gate->second.ndim == 2) {
            cfg.d_ff = static_cast<int>(it_gate->second.shape[0]);
        }
    }

    // MoE inference (only if not set by config.json)
    if (cfg.n_experts == 0) {
        auto it_moe = tensors.find("model.layers.0.block_sparse_moe.gate.weight");
        if (it_moe != tensors.end() && it_moe->second.ndim == 2) {
            cfg.n_experts = static_cast<int>(it_moe->second.shape[0]);
            cfg.n_experts_active = std::min(2, cfg.n_experts);
        }
    }

    if (cfg.expert_d_ff == 0 && cfg.n_experts > 0) {
        // Try Mixtral-style (w1) then DeepSeek/Qwen-style (gate_proj)
        for (const char* name : {"model.layers.0.block_sparse_moe.experts.0.w1.weight",
                                 "model.layers.0.mlp.experts.0.gate_proj.weight"}) {
            auto it_expert = tensors.find(name);
            if (it_expert != tensors.end() && it_expert->second.ndim == 2) {
                cfg.expert_d_ff = static_cast<int>(it_expert->second.shape[0]);
                break;
            }
        }
    }

    // Defaults for fields we couldn't infer
    if (cfg.max_seq_len == 0)
        cfg.max_seq_len = 4096;
    if (cfg.n_kv_heads == 0)
        cfg.n_kv_heads = cfg.n_heads;
}

// ---- Per-shard loading helper ----

struct ShardInfo {
    void* mmap_base = nullptr;
    size_t mmap_size = 0;
};

static bool load_shard(const std::string& path, std::unordered_map<std::string, Tensor>& tensor_map,
                       ShardInfo& shard, bool llm_compressor_format,
                       imp::llm_compressor::TranslationCounters& counters) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("Failed to open: %s", path.c_str());
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return false;
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < 8) {
        close(fd);
        return false;
    }

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (mmap_base == MAP_FAILED) {
        // MAP_POPULATE may fail on some filesystems; retry without it.
        int fd2 = open(path.c_str(), O_RDONLY);
        if (fd2 < 0)
            return false;
        mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd2, 0);
        close(fd2);
        if (mmap_base == MAP_FAILED)
            return false;
    }
    madvise(mmap_base, file_size, MADV_WILLNEED);
    madvise(mmap_base, file_size, MADV_SEQUENTIAL);

    shard.mmap_base = mmap_base;
    shard.mmap_size = file_size;

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);
    uint64_t header_size = 0;
    std::memcpy(&header_size, data, sizeof(uint64_t));
    if (8 + header_size > file_size) {
        munmap(mmap_base, file_size);
        return false;
    }

    const char* json_data = reinterpret_cast<const char*>(data + 8);
    JsonParser parser(json_data, static_cast<size_t>(header_size));
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        munmap(mmap_base, file_size);
        return false;
    }

    size_t tensor_data_offset = 8 + static_cast<size_t>(header_size);
    uint8_t* tensor_data_base = const_cast<uint8_t*>(data + tensor_data_offset);

    for (const auto& kv : root.obj) {
        std::string tensor_name = kv.first;  // copy — may be mutated by translation
        const JValue& tensor_meta = kv.second;

        if (tensor_name == "__metadata__")
            continue;
        if (tensor_meta.type != JType::OBJECT)
            continue;

        // Translate llm-compressor names → modelopt names if applicable.
        if (llm_compressor_format) {
            auto translated = imp::llm_compressor::translate_name(tensor_name, counters);
            if (translated.action == imp::llm_compressor::NameTranslation::SKIP)
                continue;
            tensor_name = std::move(translated.out_name);
        }

        const JValue* dtype_val = jobj_find(tensor_meta, "dtype");
        if (!dtype_val || dtype_val->type != JType::STRING)
            continue;
        QType dtype = safetensors_dtype(dtype_val->str_val);

        const JValue* shape_val = jobj_find(tensor_meta, "shape");
        if (!shape_val || shape_val->type != JType::ARRAY)
            continue;

        int ndim = static_cast<int>(shape_val->arr.size());
        if (ndim > kMaxDims)
            continue;

        int64_t shape[kMaxDims] = {};
        for (int d = 0; d < ndim; d++) {
            shape[d] = shape_val->arr[d].as_int();
        }

        const JValue* offsets_val = jobj_find(tensor_meta, "data_offsets");
        if (!offsets_val || offsets_val->type != JType::ARRAY || offsets_val->arr.size() != 2)
            continue;

        uint64_t offset_start = static_cast<uint64_t>(offsets_val->arr[0].as_int());
        uint64_t offset_end = static_cast<uint64_t>(offsets_val->arr[1].as_int());

        if (tensor_data_offset + offset_end > file_size)
            continue;

        void* tensor_ptr = tensor_data_base + offset_start;
        Tensor t(tensor_ptr, dtype, ndim, shape, /*on_device=*/false);
        tensor_map.emplace(tensor_name, t);

        IMP_LOG_DEBUG("Tensor: %s dtype=%s shape=[%ld%s%s%s%s] offsets=[%lu,%lu]", tensor_name.c_str(),
                      dtype_val->str_val.c_str(), (long)shape[0], ndim > 1 ? "," : "",
                      ndim > 1 ? std::to_string(shape[1]).c_str() : "", ndim > 2 ? "," : "",
                      ndim > 2 ? std::to_string(shape[2]).c_str() : "", (unsigned long)offset_start,
                      (unsigned long)offset_end);
    }

    return true;
}

// ---- Sharded SafeTensors loading ----

static bool load_sharded(const std::string& model_dir, std::unordered_map<std::string, Tensor>& tensor_map,
                         std::vector<ShardInfo>& shards) {
    std::string index_path = model_dir + "/model.safetensors.index.json";

    // Read the index file
    std::ifstream ifs(index_path);
    if (!ifs.is_open()) {
        IMP_LOG_ERROR("Failed to open index: %s", index_path.c_str());
        return false;
    }
    std::string index_json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    JsonParser parser(index_json.data(), index_json.size());
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        IMP_LOG_ERROR("Failed to parse index JSON: %s", index_path.c_str());
        return false;
    }

    const JValue* weight_map = jobj_find(root, "weight_map");
    if (!weight_map || weight_map->type != JType::OBJECT) {
        IMP_LOG_ERROR("No weight_map in index: %s", index_path.c_str());
        return false;
    }

    // Collect tensors per shard (need this to decide whether a shard is
    // entirely skippable — e.g. an MTP-only shard when spec decode is off,
    // or a vision-only shard when no mmproj is configured).
    std::map<std::string, std::vector<std::string>> shard_tensors;
    for (const auto& kv : weight_map->obj) {
        if (kv.second.type == JType::STRING) {
            shard_tensors[kv.second.str_val].push_back(kv.first);
        }
    }

    // Drop shards where every tensor would be skipped by translate_name.
    // Saves the mmap + header parse + page cache pressure for an unused file.
    std::set<std::string> shard_files;
    for (auto& [fname, tensors] : shard_tensors) {
        bool all_skip = !tensors.empty() &&
                        std::all_of(tensors.begin(), tensors.end(), imp::llm_compressor::name_is_skipped);
        if (all_skip) {
            IMP_LOG_INFO("Skipping shard %s (%zu tensors are MTP/vision-only and unused)", fname.c_str(),
                         tensors.size());
            continue;
        }
        shard_files.insert(fname);
    }

    IMP_LOG_INFO("Sharded SafeTensors: %zu shards", shard_files.size());

    // Detect format ONCE so all shards translate consistently.
    imp::HFConfigLoader::NvFP4Config probe_cfg;
    bool probe_ok = imp::HFConfigLoader::load_nvfp4_config(model_dir, probe_cfg);
    bool llm_compressor_format = probe_ok &&
                                 probe_cfg.format == imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
    imp::llm_compressor::TranslationCounters tcounters{};

    // Parse shards in parallel (mmap + header decode are independent per file).
    std::vector<std::string> shard_list(shard_files.begin(), shard_files.end());
    std::vector<std::unordered_map<std::string, Tensor>> per_shard_maps(shard_list.size());
    std::vector<ShardInfo> per_shard_info(shard_list.size());
    std::vector<imp::llm_compressor::TranslationCounters> per_shard_counters(shard_list.size());
    std::atomic<bool> any_failure{false};

    std::atomic<size_t> shards_done{0};
    const size_t total_shards = shard_list.size();
    auto worker = [&](size_t i) {
        std::string shard_path = model_dir + "/" + shard_list[i];
        if (!load_shard(shard_path, per_shard_maps[i], per_shard_info[i], llm_compressor_format,
                        per_shard_counters[i])) {
            IMP_LOG_ERROR("Failed to load shard: %s", shard_path.c_str());
            any_failure.store(true);
        }
        const size_t done = shards_done.fetch_add(1) + 1;
        IMP_LOG_INFO("  [%zu/%zu] mmap'd shard: %s (%zu tensors)", done, total_shards,
                     shard_list[i].c_str(), per_shard_maps[i].size());
    };

    {
        std::vector<std::thread> ts;
        ts.reserve(shard_list.size());
        for (size_t i = 0; i < shard_list.size(); ++i)
            ts.emplace_back(worker, i);
        for (auto& t : ts)
            t.join();
    }

    if (any_failure.load())
        return false;

    // Merge per-shard results into the caller-owned aggregates.
    for (size_t i = 0; i < shard_list.size(); ++i) {
        for (auto& kv : per_shard_maps[i])
            tensor_map.emplace(kv.first, kv.second);
        shards.push_back(per_shard_info[i]);
        const auto& c = per_shard_counters[i];
        tcounters.suffix_renames += c.suffix_renames;
        tcounters.prefix_strips += c.prefix_strips;
        tcounters.vision_skipped += c.vision_skipped;
        tcounters.gemma4_extras += c.gemma4_extras;
        tcounters.passed_through += c.passed_through;
        IMP_LOG_INFO("Loaded shard: %s (%zu tensors total)", shard_list[i].c_str(), tensor_map.size());
    }

    if (llm_compressor_format) {
        imp::llm_compressor::log_summary(tcounters);
    }

    return true;
}

// ---- Main SafeTensors loader ----

std::unique_ptr<Model> load_safetensors(const std::string& path) {
    namespace fs = std::filesystem;

    std::string model_dir;
    std::string single_file;

    if (fs::is_directory(path)) {
        model_dir = path;
    } else if (fs::is_regular_file(path)) {
        single_file = path;
        model_dir = fs::path(path).parent_path().string();
    } else {
        IMP_LOG_ERROR("Path does not exist: %s", path.c_str());
        return nullptr;
    }

    std::unordered_map<std::string, Tensor> tensor_map;
    std::vector<ShardInfo> shards;

    // Detect format ONCE so all shards translate consistently.
    imp::HFConfigLoader::NvFP4Config probe_cfg;
    bool probe_ok = imp::HFConfigLoader::load_nvfp4_config(model_dir, probe_cfg);
    bool llm_compressor_format = probe_ok &&
                                 probe_cfg.format == imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
    imp::llm_compressor::TranslationCounters tcounters{};

    // Try loading tensors
    bool loaded = false;

    if (!single_file.empty()) {
        // Single file mode
        ShardInfo shard;
        loaded = load_shard(single_file, tensor_map, shard, llm_compressor_format, tcounters);
        if (loaded)
            shards.push_back(shard);
    } else {
        // Directory mode: try sharded first, then single
        std::string index_path = model_dir + "/model.safetensors.index.json";
        if (fs::exists(index_path)) {
            loaded = load_sharded(model_dir, tensor_map, shards);
        }
        if (!loaded) {
            std::string st_path = model_dir + "/model.safetensors";
            if (fs::exists(st_path)) {
                ShardInfo shard;
                loaded = load_shard(st_path, tensor_map, shard, llm_compressor_format, tcounters);
                if (loaded)
                    shards.push_back(shard);
            }
        }
    }

    // For the single-file paths (single_file mode or directory fallback to model.safetensors),
    // emit the summary here. The sharded path (load_sharded) emits its own summary internally
    // with its own counters — tcounters is only populated by the two load_shard calls above.
    if (llm_compressor_format &&
        (tcounters.suffix_renames + tcounters.prefix_strips + tcounters.vision_skipped +
         tcounters.gemma4_extras + tcounters.passed_through) > 0) {
        imp::llm_compressor::log_summary(tcounters);
    }

    if (!loaded || tensor_map.empty()) {
        IMP_LOG_ERROR("Failed to load SafeTensors from %s", path.c_str());
        return nullptr;
    }

    IMP_LOG_INFO("Parsed %zu tensors from SafeTensors", tensor_map.size());

    // Create model
    auto model = std::make_unique<Model>();

    // Store mmap info for cleanup
    model->mmap_base_ = shards[0].mmap_base;
    model->mmap_size_ = shards[0].mmap_size;
    for (size_t i = 1; i < shards.size(); i++) {
        model->split_mmaps_.emplace_back(shards[i].mmap_base, shards[i].mmap_size);
    }

    ModelConfig& cfg = model->config_;

    // 1. Try config.json (authoritative for all hyperparams)
    bool has_config = HFConfigLoader::load_config(model_dir, cfg);

    // 2. Detect architecture from weights if config.json didn't provide it
    if (!has_config || cfg.arch == ModelArch::GENERIC) {
        ModelArch detected = detect_arch_from_weights(tensor_map);
        if (cfg.arch == ModelArch::GENERIC && detected != ModelArch::GENERIC) {
            cfg.arch = detected;
        }
    }

    // 3. Infer remaining config from weight shapes (fills fields still at defaults)
    infer_config(cfg, tensor_map);

    // 4. Apply arch-specific defaults
    apply_arch_defaults(cfg);

    IMP_LOG_INFO("Architecture: %s", model_arch_name(cfg.arch));
    IMP_LOG_INFO("Config: layers=%d d_model=%d d_ff=%d heads=%d kv_heads=%d vocab=%d ctx=%d", cfg.n_layers,
                 cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.max_seq_len);
    IMP_LOG_INFO(
        "RoPE: theta=%.0f freq_scale=%.4f head_dim=%d sliding_window=%d "
        "rope_theta_swa=%.0f rope_local_theta=%.0f rope_n_ctx_orig=%d",
        cfg.rope_theta, cfg.rope_freq_scale, cfg.head_dim, cfg.sliding_window, cfg.rope_theta_swa,
        cfg.rope_local_theta, cfg.rope_n_ctx_orig);
    if (cfg.n_experts > 0) {
        IMP_LOG_INFO("MoE: %d experts, %d active, expert_d_ff=%d", cfg.n_experts, cfg.n_experts_active,
                     cfg.expert_d_ff);
    }

    // 5. Allocate layers and expert vectors
    model->layers_.resize(cfg.n_layers);
    if (cfg.n_experts > 0) {
        for (auto& layer : model->layers_) {
            layer.expert_w_gate.resize(cfg.n_experts);
            layer.expert_w_up.resize(cfg.n_experts);
            layer.expert_w_down.resize(cfg.n_experts);
        }
    }

    // 6. Assign tensors via WeightMap
    WeightMap wmap(cfg.arch);
    wmap.apply_weights(*model, tensor_map);

    // 6b. GPTQ config: set bit width and group size on all GPTQ weight structs
    HFConfigLoader::GPTQConfig gptq_cfg;
    bool is_gptq = HFConfigLoader::load_gptq_config(model_dir, gptq_cfg);
    if (is_gptq) {
        for (auto& layer : model->layers_) {
            for (auto* gw : {&layer.gptq_q, &layer.gptq_k, &layer.gptq_v, &layer.gptq_o, &layer.gptq_gate,
                             &layer.gptq_up, &layer.gptq_down}) {
                gw->bits = gptq_cfg.bits;
                gw->group_size = gptq_cfg.group_size;
            }
        }
        IMP_LOG_INFO("GPTQ model: %d-bit, group_size=%d, desc_act=%s", gptq_cfg.bits, gptq_cfg.group_size,
                     gptq_cfg.desc_act ? "true" : "false");
    }

    // 6c. NVFP4 config detection. Scale tensors were already routed into
    // model->nvfp4_scratch_ by weight_map.cpp during the layer-pattern pass;
    // executor_pre_dequant.cu's Phase 0 promote() resolves each scratch key
    // back to the main weight tensor and writes the device pointer onto its
    // .scales / .tensor_scale sidecars. No load-side linking needed here.
    HFConfigLoader::NvFP4Config nvfp4_cfg;
    bool is_nvfp4 = HFConfigLoader::load_nvfp4_config(model_dir, nvfp4_cfg);
    if (is_nvfp4) {
        cfg.is_nvfp4_prequant = true;
        cfg.nvfp4_group_size = nvfp4_cfg.group_size;
        cfg.is_llm_compressor_nvfp4 = (nvfp4_cfg.format == HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);
        cfg.kv_cache_quant_hint = nvfp4_cfg.kv_cache_quant_algo;
        IMP_LOG_INFO("NVFP4 pre-quantized: %zu scratch entries (group_size=%d)", model->nvfp4_scratch_.size(),
                     nvfp4_cfg.group_size);
        if (!cfg.kv_cache_quant_hint.empty()) {
            IMP_LOG_INFO(
                "Model author declared kv_cache_quant_algo=%s; imp keeps the "
                "engine's KV-cache dtype default (FP16). Pass --kv-fp8 to honor "
                "the author's hint (correctness varies by family — see docs).",
                cfg.kv_cache_quant_hint.c_str());
        }
    }

    // 7. Tie output projection if not found.
    // Cross-check the author's `tie_word_embeddings` flag (parsed by
    // HFConfigLoader::load_config) against the actual lm_head.weight
    // presence. Mismatch is a real surprise — most models tie, so silent
    // tying when the author said `tie=false` would mask a missing lm_head.
    const bool out_proj_missing =
        (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr);
    if (cfg.tie_word_embeddings == 0 && out_proj_missing) {
        IMP_LOG_WARN(
            "config.json declares tie_word_embeddings=false but lm_head.weight "
            "is absent in the SafeTensors files; tying anyway as a fallback.");
    }
    if (cfg.tie_word_embeddings == 1 && model->out_proj_.data != nullptr &&
        model->tok_emb_.data != nullptr && model->out_proj_.data != model->tok_emb_.data) {
        IMP_LOG_INFO(
            "config.json declares tie_word_embeddings=true but lm_head.weight "
            "was loaded as a separate tensor; honoring the file (no tying).");
    }
    if (out_proj_missing) {
        model->out_proj_ = model->tok_emb_;
        IMP_LOG_INFO("Tied output projection to token embedding");
    }

    // 8. Load chat template from tokenizer_config.json
    if (!model_dir.empty()) {
        std::string chat_tpl = HFConfigLoader::load_chat_template(model_dir);
        if (!chat_tpl.empty()) {
            if (!model->tokenizer_) {
                auto tok = std::make_unique<Tokenizer>();
                tok->set_chat_template_str(chat_tpl);
                model->set_tokenizer(std::move(tok));
            } else {
                model->tokenizer_->set_chat_template_str(chat_tpl);
            }
        }
    }

    // 9. Load tokenizer from tokenizer.json (if available)
    if (!model_dir.empty()) {
        std::string tok_json_path = model_dir + "/tokenizer.json";
        if (std::filesystem::exists(tok_json_path)) {
            auto tok = std::make_unique<Tokenizer>();
            if (tok->load(tok_json_path)) {
                // Preserve chat template if already set
                if (model->tokenizer_ && !model->tokenizer_->chat_template_str().empty()) {
                    tok->set_chat_template_str(model->tokenizer_->chat_template_str());
                }
                model->set_tokenizer(std::move(tok));
                IMP_LOG_INFO("Loaded tokenizer from %s", tok_json_path.c_str());
            }
        }
    }

    // 9b. tokenizer_config.json — tokenizer-side flags (add_bos_token,
    // add_prefix_space). Mirrors gguf_loader.cpp's read of
    // tokenizer.ggml.add_bos_token / add_space_prefix. Without this the
    // SafeTensors path used Tokenizer's hardcoded default `add_bos_=true`
    // — wrong for any model that ships add_bos_token=false in its config
    // (e.g. Qwen3-Coder-30B-A3B-FP4 which auto-prepends <|endoftext|>
    // unwantedly otherwise).
    if (!model_dir.empty() && model->tokenizer_) {
        HFConfigLoader::TokenizerFlags tflags;
        if (HFConfigLoader::load_tokenizer_flags(model_dir, tflags)) {
            if (tflags.add_bos_token >= 0) {
                model->tokenizer_->set_add_bos(tflags.add_bos_token != 0);
            } else if (model->tokenizer_->type() == "gpt2") {
                // Match GGUF default: BPE tokenizers without an explicit
                // flag don't add BOS.
                model->tokenizer_->set_add_bos(false);
            }
            if (tflags.add_prefix_space >= 0) {
                model->tokenizer_->set_add_space_prefix(tflags.add_prefix_space != 0);
            }
            if (tflags.use_default_system_prompt >= 0) {
                model->tokenizer_->set_use_default_system_prompt(tflags.use_default_system_prompt != 0);
            }
        }
    }

    // Re-infer vocab_size from token embedding if needed
    if (cfg.vocab_size == 0 && model->tok_emb_.data != nullptr) {
        cfg.vocab_size = static_cast<int>(model->tok_emb_.shape[0]);
    }

    // 10. generation_config.json — sampling/EOS defaults shipped by the model
    // author. Loaded into model->generation_config_ for engine + CLI consumers.
    // EOS IDs additionally pushed onto the tokenizer's eos list so the engine's
    // existing stop-condition path picks them up without further plumbing.
    if (!model_dir.empty()) {
        HFConfigLoader::load_generation_config(model_dir, model->generation_config_);
        if (model->tokenizer_) {
            for (int32_t eid : model->generation_config_.eos_token_ids) {
                model->tokenizer_->add_eos_id(eid);
            }
        }
    }

    // 11. Cross-check special_tokens_map.json against the loaded tokenizer's
    // special-flag column. The model author's list is authoritative; if a
    // string from `additional_special_tokens` exists in vocab but isn't
    // marked CONTROL (token_type=3), patch it. Caught by the engine's
    // banned-token scan in engine.cpp.
    if (!model_dir.empty() && model->tokenizer_) {
        HFConfigLoader::SpecialTokensMap stm;
        if (HFConfigLoader::load_special_tokens_map(model_dir, stm)) {
            int patched = 0, missing = 0;
            for (const auto& s : stm.additional_special_tokens) {
                int32_t id = model->tokenizer_->find_token(s);
                if (id < 0) {
                    missing++;
                    continue;
                }
                if (!model->tokenizer_->is_control_token(id)) {
                    model->tokenizer_->mark_as_control(id);
                    patched++;
                }
            }
            if (patched > 0 || missing > 0) {
                IMP_LOG_INFO(
                    "special_tokens_map: cross-check patched %d, "
                    "missing-from-vocab %d",
                    patched, missing);
            }
        }
    }

    // 12. Validate config-promised biases against actual tensor presence.
    // Some HF configs declare attention_bias/mlp_bias but the SafeTensors
    // export omits the bias tensors. Without this check the loader silently
    // leaves bias slots null and inference proceeds with undefined behaviour
    // depending on which kernels short-circuit on null biases.
    if (cfg.attention_bias == 1) {
        int missing_q = 0, missing_k = 0, missing_v = 0;
        for (const auto& layer : model->layers_) {
            if (layer.q_bias.data == nullptr) missing_q++;
            if (layer.k_bias.data == nullptr) missing_k++;
            if (layer.v_bias.data == nullptr) missing_v++;
        }
        if (missing_q || missing_k || missing_v) {
            IMP_LOG_WARN(
                "config.json says attention_bias=true but %d/%d Q-biases, "
                "%d/%d K-biases, %d/%d V-biases are missing from the SafeTensors "
                "export. Inference will proceed without those biases.",
                missing_q, static_cast<int>(model->layers_.size()),
                missing_k, static_cast<int>(model->layers_.size()),
                missing_v, static_cast<int>(model->layers_.size()));
        }
    }

    if (cfg.arch_inferred_fallback) {
        IMP_LOG_WARN(
            "Architecture detection fell back to GENERIC + tensor-name "
            "heuristics (config.json had no recognized architectures/model_type). "
            "Inference may be incoherent.");
    }

    IMP_LOG_INFO("SafeTensors model loaded successfully from %s", path.c_str());
    return model;
}

}  // namespace imp
