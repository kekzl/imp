#include "model/safetensors_loader.h"
#include "model/model_arch.h"
#include "model/weight_map.h"
#include "model/hf_config_loader.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

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
                        // Skip unicode escapes (consume 4 hex digits, emit '?')
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
            // true
            for (int i = 0; i < 4 && pos_ < len_; i++) advance();
            v.num_val = 1.0;
        } else {
            // false
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

// ---- Helper: find a key in a JValue object ----

static const JValue* jobj_find(const JValue& obj, const std::string& key) {
    for (const auto& kv : obj.obj) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

// ---- SafeTensors dtype string to DType ----

static DType safetensors_dtype(const std::string& s) {
    if (s == "F32")  return DType::FP32;
    if (s == "F16")  return DType::FP16;
    if (s == "BF16") return DType::BF16;
    if (s == "F64")  return DType::FP32;  // closest proxy
    if (s == "I8")   return DType::INT8;
    if (s == "U8")   return DType::INT8;   // treat unsigned byte as INT8
    if (s == "I16")  return DType::INT32;  // closest proxy
    if (s == "I32")  return DType::INT32;
    if (s == "I64")  return DType::INT32;  // closest proxy
    if (s == "BOOL") return DType::INT8;
    if (s == "F8_E4M3") return DType::FP8_E4M3;
    if (s == "F8_E5M2") return DType::FP8_E4M3;  // closest proxy
    IMP_LOG_WARN("Unknown SafeTensors dtype '%s', defaulting to FP32", s.c_str());
    return DType::FP32;
}

// ---- Architecture detection from weight names ----

static ModelArch detect_arch_from_weights(
        const std::unordered_map<std::string, Tensor>& tensors) {
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
        if (name.find("mamba") != std::string::npos ||
            name.find("ssm") != std::string::npos)
            has_ssm = true;
        if (name.find(".qweight") != std::string::npos)
            has_gptq = true;
    }

    if (has_gptq) {
        IMP_LOG_INFO("Detected GPTQ quantized weights");
    }

    if (has_ssm)              return ModelArch::NEMOTRON_H_MOE;
    if (has_mlp_experts)      return ModelArch::DEEPSEEK;
    if (has_block_sparse_moe) return ModelArch::MIXTRAL;
    if (has_layers)           return ModelArch::LLAMA;
    return ModelArch::GENERIC;
}

// ---- Extract layer index from a HuggingFace weight name ----
// e.g. "model.layers.5.self_attn.q_proj.weight" -> 5
// Returns -1 if not a layer weight.

static int extract_layer_index(const std::string& name) {
    const char* prefix = "model.layers.";
    size_t plen = std::strlen(prefix);
    if (name.compare(0, plen, prefix) != 0) return -1;

    int idx = 0;
    size_t i = plen;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        idx = idx * 10 + (name[i] - '0');
        i++;
    }
    if (i == plen) return -1;  // no digits found
    return idx;
}

// ---- Infer max layer index to determine n_layers ----

static int infer_n_layers(const std::unordered_map<std::string, Tensor>& tensors) {
    int max_idx = -1;
    for (const auto& kv : tensors) {
        int idx = extract_layer_index(kv.first);
        if (idx > max_idx) max_idx = idx;
    }
    return max_idx + 1;  // 0-indexed, so count = max + 1
}

// ---- Infer model config from weight shapes ----

static void infer_config(ModelConfig& cfg,
                         const std::unordered_map<std::string, Tensor>& tensors) {
    // Only infer fields that are still at their default (zero) values.
    // config.json (via HFConfigLoader) is authoritative when present.

    if (cfg.n_layers == 0)
        cfg.n_layers = infer_n_layers(tensors);

    // token embedding: shape = [vocab_size, d_model]
    auto it = tensors.find("model.embed_tokens.weight");
    if (it != tensors.end() && it->second.ndim == 2) {
        if (cfg.vocab_size == 0) cfg.vocab_size = static_cast<int>(it->second.shape[0]);
        if (cfg.d_model == 0) cfg.d_model = static_cast<int>(it->second.shape[1]);
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
        auto it_expert = tensors.find("model.layers.0.block_sparse_moe.experts.0.w1.weight");
        if (it_expert != tensors.end() && it_expert->second.ndim == 2) {
            cfg.expert_d_ff = static_cast<int>(it_expert->second.shape[0]);
        }
    }

    // Defaults for fields we couldn't infer
    if (cfg.max_seq_len == 0) cfg.max_seq_len = 4096;
    if (cfg.n_kv_heads == 0) cfg.n_kv_heads = cfg.n_heads;
}

// ---- Per-shard loading helper ----

struct ShardInfo {
    void* mmap_base = nullptr;
    size_t mmap_size = 0;
};

static bool load_shard(const std::string& path,
                       std::unordered_map<std::string, Tensor>& tensor_map,
                       ShardInfo& shard) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { IMP_LOG_ERROR("Failed to open: %s", path.c_str()); return false; }

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < 8) { close(fd); return false; }

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_base == MAP_FAILED) return false;
    madvise(mmap_base, file_size, MADV_SEQUENTIAL);

    shard.mmap_base = mmap_base;
    shard.mmap_size = file_size;

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);
    uint64_t header_size = 0;
    std::memcpy(&header_size, data, sizeof(uint64_t));
    if (8 + header_size > file_size) { munmap(mmap_base, file_size); return false; }

    const char* json_data = reinterpret_cast<const char*>(data + 8);
    JsonParser parser(json_data, static_cast<size_t>(header_size));
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) { munmap(mmap_base, file_size); return false; }

    size_t tensor_data_offset = 8 + static_cast<size_t>(header_size);
    uint8_t* tensor_data_base = const_cast<uint8_t*>(data + tensor_data_offset);

    for (const auto& kv : root.obj) {
        const std::string& tensor_name = kv.first;
        const JValue& tensor_meta = kv.second;

        if (tensor_name == "__metadata__") continue;
        if (tensor_meta.type != JType::OBJECT) continue;

        const JValue* dtype_val = jobj_find(tensor_meta, "dtype");
        if (!dtype_val || dtype_val->type != JType::STRING) continue;
        DType dtype = safetensors_dtype(dtype_val->str_val);

        const JValue* shape_val = jobj_find(tensor_meta, "shape");
        if (!shape_val || shape_val->type != JType::ARRAY) continue;

        int ndim = static_cast<int>(shape_val->arr.size());
        if (ndim > kMaxDims) continue;

        int64_t shape[kMaxDims] = {};
        for (int d = 0; d < ndim; d++) {
            shape[d] = shape_val->arr[d].as_int();
        }

        const JValue* offsets_val = jobj_find(tensor_meta, "data_offsets");
        if (!offsets_val || offsets_val->type != JType::ARRAY || offsets_val->arr.size() != 2) continue;

        uint64_t offset_start = static_cast<uint64_t>(offsets_val->arr[0].as_int());
        uint64_t offset_end = static_cast<uint64_t>(offsets_val->arr[1].as_int());

        if (tensor_data_offset + offset_end > file_size) continue;

        void* tensor_ptr = tensor_data_base + offset_start;
        Tensor t(tensor_ptr, dtype, ndim, shape, /*on_device=*/false);
        tensor_map.emplace(tensor_name, t);

        IMP_LOG_DEBUG("Tensor: %s dtype=%s shape=[%ld%s%s%s%s] offsets=[%lu,%lu]",
                      tensor_name.c_str(), dtype_val->str_val.c_str(),
                      (long)shape[0],
                      ndim > 1 ? "," : "", ndim > 1 ? std::to_string(shape[1]).c_str() : "",
                      ndim > 2 ? "," : "", ndim > 2 ? std::to_string(shape[2]).c_str() : "",
                      (unsigned long)offset_start, (unsigned long)offset_end);
    }

    return true;
}

// ---- Sharded SafeTensors loading ----

static bool load_sharded(const std::string& model_dir,
                         std::unordered_map<std::string, Tensor>& tensor_map,
                         std::vector<ShardInfo>& shards) {
    std::string index_path = model_dir + "/model.safetensors.index.json";

    // Read the index file
    std::ifstream ifs(index_path);
    if (!ifs.is_open()) {
        IMP_LOG_ERROR("Failed to open index: %s", index_path.c_str());
        return false;
    }
    std::string index_json((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
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

    // Collect unique shard filenames
    std::set<std::string> shard_files;
    for (const auto& kv : weight_map->obj) {
        if (kv.second.type == JType::STRING) {
            shard_files.insert(kv.second.str_val);
        }
    }

    IMP_LOG_INFO("Sharded SafeTensors: %zu shards", shard_files.size());

    // Load each shard
    for (const auto& filename : shard_files) {
        std::string shard_path = model_dir + "/" + filename;
        ShardInfo shard;
        if (!load_shard(shard_path, tensor_map, shard)) {
            IMP_LOG_ERROR("Failed to load shard: %s", shard_path.c_str());
            return false;
        }
        shards.push_back(shard);
        IMP_LOG_INFO("Loaded shard: %s (%zu tensors total)", filename.c_str(), tensor_map.size());
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

    // Try loading tensors
    bool loaded = false;

    if (!single_file.empty()) {
        // Single file mode
        ShardInfo shard;
        loaded = load_shard(single_file, tensor_map, shard);
        if (loaded) shards.push_back(shard);
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
                loaded = load_shard(st_path, tensor_map, shard);
                if (loaded) shards.push_back(shard);
            }
        }
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
    IMP_LOG_INFO("Config: layers=%d d_model=%d d_ff=%d heads=%d kv_heads=%d vocab=%d ctx=%d",
                 cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv_heads,
                 cfg.vocab_size, cfg.max_seq_len);
    if (cfg.n_experts > 0) {
        IMP_LOG_INFO("MoE: %d experts, %d active, expert_d_ff=%d",
                     cfg.n_experts, cfg.n_experts_active, cfg.expert_d_ff);
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
            for (auto* gw : {&layer.gptq_q, &layer.gptq_k, &layer.gptq_v, &layer.gptq_o,
                             &layer.gptq_gate, &layer.gptq_up, &layer.gptq_down}) {
                gw->bits = gptq_cfg.bits;
                gw->group_size = gptq_cfg.group_size;
            }
        }
        IMP_LOG_INFO("GPTQ model: %d-bit, group_size=%d, desc_act=%s",
                     gptq_cfg.bits, gptq_cfg.group_size,
                     gptq_cfg.desc_act ? "true" : "false");
    }

    // 6c. NVFP4 config: link weight tensors to NvFP4PreQuantWeight structs
    HFConfigLoader::NvFP4Config nvfp4_cfg;
    bool is_nvfp4 = HFConfigLoader::load_nvfp4_config(model_dir, nvfp4_cfg);
    if (is_nvfp4) {
        cfg.is_nvfp4_prequant = true;
        cfg.nvfp4_group_size = nvfp4_cfg.group_size;
        // Link the main weight tensors to nvfp4 structs (they share the same data pointer)
        for (auto& layer : model->layers_) {
            auto link = [](TransformerLayer::NvFP4PreQuantWeight& nw, const Tensor& w) {
                if (nw.weight_scale.data != nullptr) nw.weight = w;
            };
            // Dense weights
            link(layer.nvfp4_q, layer.wq);
            link(layer.nvfp4_k, layer.wk);
            link(layer.nvfp4_v, layer.wv);
            link(layer.nvfp4_o, layer.wo);
            link(layer.nvfp4_gate, layer.w_gate);
            link(layer.nvfp4_up, layer.w_up);
            link(layer.nvfp4_down, layer.w_down);
            // Expert weights
            for (size_t e = 0; e < layer.expert_nvfp4_gate.size(); e++) {
                if (e < layer.expert_w_gate.size()) link(layer.expert_nvfp4_gate[e], layer.expert_w_gate[e]);
                if (e < layer.expert_w_up.size())   link(layer.expert_nvfp4_up[e],   layer.expert_w_up[e]);
                if (e < layer.expert_w_down.size()) link(layer.expert_nvfp4_down[e], layer.expert_w_down[e]);
            }
        }
        int nvfp4_count = 0;
        int nvfp4_expert_count = 0;
        for (const auto& layer : model->layers_) {
            for (const auto* nw : {&layer.nvfp4_q, &layer.nvfp4_k, &layer.nvfp4_v, &layer.nvfp4_o,
                                   &layer.nvfp4_gate, &layer.nvfp4_up, &layer.nvfp4_down}) {
                if (nw->valid()) nvfp4_count++;
            }
            for (const auto* vec : {&layer.expert_nvfp4_gate, &layer.expert_nvfp4_up, &layer.expert_nvfp4_down}) {
                for (const auto& nw : *vec) {
                    if (nw.valid()) nvfp4_expert_count++;
                }
            }
        }
        IMP_LOG_INFO("NVFP4 pre-quantized: %d dense + %d expert weight tensors (group_size=%d)",
                     nvfp4_count, nvfp4_expert_count, nvfp4_cfg.group_size);
    }

    // 7. Tie output projection if not found
    if (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr) {
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

    // Re-infer vocab_size from token embedding if needed
    if (cfg.vocab_size == 0 && model->tok_emb_.data != nullptr) {
        cfg.vocab_size = static_cast<int>(model->tok_emb_.shape[0]);
    }

    IMP_LOG_INFO("SafeTensors model loaded successfully from %s", path.c_str());
    return model;
}

} // namespace imp
