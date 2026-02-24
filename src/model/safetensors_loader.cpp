#include "model/safetensors_loader.h"
#include "model/model_arch.h"
#include "model/weight_map.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
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
    cfg.n_layers = infer_n_layers(tensors);

    // token embedding: shape = [vocab_size, d_model]
    auto it = tensors.find("model.embed_tokens.weight");
    if (it != tensors.end() && it->second.ndim == 2) {
        cfg.vocab_size = static_cast<int>(it->second.shape[0]);
        cfg.d_model = static_cast<int>(it->second.shape[1]);
    }

    // q_proj.weight on layer 0: shape = [n_heads * head_dim, d_model]
    // k_proj.weight on layer 0: shape = [n_kv_heads * head_dim, d_model]
    auto it_q = tensors.find("model.layers.0.self_attn.q_proj.weight");
    auto it_k = tensors.find("model.layers.0.self_attn.k_proj.weight");
    if (it_q != tensors.end() && it_q->second.ndim == 2 && cfg.d_model > 0) {
        int q_out = static_cast<int>(it_q->second.shape[0]);
        int head_dim = cfg.d_model > 0 ? cfg.d_model : q_out;
        // Standard: q_out == d_model, head_dim = d_model / n_heads
        // Try common head dimensions: 128, 64, 96
        for (int hd : {128, 64, 96, 80, 256}) {
            if (q_out % hd == 0) {
                cfg.n_heads = q_out / hd;
                head_dim = hd;
                break;
            }
        }

        if (it_k != tensors.end() && it_k->second.ndim == 2 && head_dim > 0) {
            int k_out = static_cast<int>(it_k->second.shape[0]);
            cfg.n_kv_heads = k_out / head_dim;
        } else {
            cfg.n_kv_heads = cfg.n_heads;
        }
    }

    // gate_proj.weight on layer 0: shape = [d_ff, d_model]
    auto it_gate = tensors.find("model.layers.0.mlp.gate_proj.weight");
    if (it_gate != tensors.end() && it_gate->second.ndim == 2) {
        cfg.d_ff = static_cast<int>(it_gate->second.shape[0]);
    }

    // MoE: check for expert weights
    // Pattern: model.layers.0.block_sparse_moe.experts.0.w1.weight
    auto it_moe_gate = tensors.find("model.layers.0.block_sparse_moe.gate.weight");
    if (it_moe_gate != tensors.end() && it_moe_gate->second.ndim == 2) {
        cfg.n_experts = static_cast<int>(it_moe_gate->second.shape[0]);
        cfg.n_experts_active = std::min(2, cfg.n_experts);  // common default
    }

    // MoE expert d_ff: model.layers.0.block_sparse_moe.experts.0.w1.weight -> [expert_d_ff, d_model]
    auto it_expert = tensors.find("model.layers.0.block_sparse_moe.experts.0.w1.weight");
    if (it_expert != tensors.end() && it_expert->second.ndim == 2) {
        cfg.expert_d_ff = static_cast<int>(it_expert->second.shape[0]);
    }

    // Defaults for fields we couldn't infer
    if (cfg.max_seq_len == 0) cfg.max_seq_len = 4096;
    if (cfg.n_kv_heads == 0) cfg.n_kv_heads = cfg.n_heads;
}

// ---- Assign tensor to model fields by HuggingFace name ----

static bool assign_tensor_hf(Model& model, const std::string& name,
                              const Tensor& tensor) {
    // Global tensors
    if (name == "model.embed_tokens.weight" || name == "lm_head.weight" ||
        name == "model.norm.weight") {
        if (name == "model.embed_tokens.weight") { model.tok_emb_ = tensor; return true; }
        if (name == "model.norm.weight")         { model.out_norm_ = tensor; return true; }
        if (name == "lm_head.weight")            { model.out_proj_ = tensor; return true; }
    }

    int layer_idx = extract_layer_index(name);
    if (layer_idx < 0 || layer_idx >= model.n_layers()) return false;

    auto& layer = model.layers_[layer_idx];

    // Strip "model.layers.{N}." prefix to get suffix
    // Find third dot to get suffix
    size_t dot_count = 0;
    size_t suffix_start = 0;
    for (size_t i = 0; i < name.size(); i++) {
        if (name[i] == '.') {
            dot_count++;
            if (dot_count == 3) { suffix_start = i + 1; break; }
        }
    }
    std::string suffix = name.substr(suffix_start);

    // Standard (non-MoE) attention
    if      (suffix == "self_attn.q_proj.weight") { layer.wq = tensor; return true; }
    else if (suffix == "self_attn.k_proj.weight") { layer.wk = tensor; return true; }
    else if (suffix == "self_attn.v_proj.weight") { layer.wv = tensor; return true; }
    else if (suffix == "self_attn.o_proj.weight") { layer.wo = tensor; return true; }
    else if (suffix == "input_layernorm.weight")  { layer.attn_norm = tensor; return true; }
    else if (suffix == "post_attention_layernorm.weight") { layer.ffn_norm = tensor; return true; }

    // Standard (non-MoE) FFN
    else if (suffix == "mlp.gate_proj.weight") { layer.w_gate = tensor; return true; }
    else if (suffix == "mlp.up_proj.weight")   { layer.w_up = tensor; return true; }
    else if (suffix == "mlp.down_proj.weight") { layer.w_down = tensor; return true; }

    // MoE router gate
    else if (suffix == "block_sparse_moe.gate.weight") { layer.moe_gate = tensor; return true; }

    // MoE expert weights: block_sparse_moe.experts.{E}.w1/w2/w3.weight
    if (suffix.find("block_sparse_moe.experts.") == 0) {
        // Parse expert index
        const char* exp_prefix = "block_sparse_moe.experts.";
        size_t ep_len = std::strlen(exp_prefix);
        size_t dot_pos = suffix.find('.', ep_len);
        if (dot_pos == std::string::npos) return false;

        int expert_idx = 0;
        for (size_t i = ep_len; i < dot_pos; i++) {
            expert_idx = expert_idx * 10 + (suffix[i] - '0');
        }

        std::string field = suffix.substr(dot_pos + 1);

        if (expert_idx < 0 || expert_idx >= static_cast<int>(layer.expert_w_gate.size()))
            return false;

        // w1 = gate, w2 = down, w3 = up (Mixtral convention)
        if      (field == "w1.weight") { layer.expert_w_gate[expert_idx] = tensor; return true; }
        else if (field == "w2.weight") { layer.expert_w_down[expert_idx] = tensor; return true; }
        else if (field == "w3.weight") { layer.expert_w_up[expert_idx] = tensor; return true; }
    }

    return false;
}

// ---- Main SafeTensors loader ----

std::unique_ptr<Model> load_safetensors(const std::string& path) {
    // 1. Open and mmap the file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("Failed to open SafeTensors file: %s", path.c_str());
        return nullptr;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        IMP_LOG_ERROR("Failed to stat SafeTensors file: %s", path.c_str());
        close(fd);
        return nullptr;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    if (file_size < 8) {
        IMP_LOG_ERROR("SafeTensors file too small: %s (%zu bytes)", path.c_str(), file_size);
        close(fd);
        return nullptr;
    }

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_base == MAP_FAILED) {
        IMP_LOG_ERROR("Failed to mmap SafeTensors file: %s (size=%zu)", path.c_str(), file_size);
        return nullptr;
    }

    madvise(mmap_base, file_size, MADV_SEQUENTIAL);

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);

    // 2. Read 8-byte header size (little-endian uint64_t)
    uint64_t header_size = 0;
    std::memcpy(&header_size, data, sizeof(uint64_t));

    if (8 + header_size > file_size) {
        IMP_LOG_ERROR("SafeTensors header size (%lu) exceeds file size (%zu)",
                      (unsigned long)header_size, file_size);
        munmap(mmap_base, file_size);
        return nullptr;
    }

    IMP_LOG_INFO("SafeTensors header: %lu bytes", (unsigned long)header_size);

    // 3. Parse JSON header
    const char* json_data = reinterpret_cast<const char*>(data + 8);
    JsonParser parser(json_data, static_cast<size_t>(header_size));
    JValue root = parser.parse();

    if (!parser.ok() || root.type != JType::OBJECT) {
        IMP_LOG_ERROR("Failed to parse SafeTensors JSON header");
        munmap(mmap_base, file_size);
        return nullptr;
    }

    // Tensor data begins right after the header
    size_t tensor_data_offset = 8 + static_cast<size_t>(header_size);
    uint8_t* tensor_data_base = const_cast<uint8_t*>(data + tensor_data_offset);

    // 4. Create Tensor descriptors from JSON metadata
    std::unordered_map<std::string, Tensor> tensor_map;

    for (const auto& kv : root.obj) {
        const std::string& tensor_name = kv.first;
        const JValue& tensor_meta = kv.second;

        // Skip __metadata__ entry
        if (tensor_name == "__metadata__") continue;

        if (tensor_meta.type != JType::OBJECT) {
            IMP_LOG_WARN("Skipping non-object entry: %s", tensor_name.c_str());
            continue;
        }

        // Extract dtype
        const JValue* dtype_val = jobj_find(tensor_meta, "dtype");
        if (!dtype_val || dtype_val->type != JType::STRING) {
            IMP_LOG_WARN("Tensor '%s' missing dtype, skipping", tensor_name.c_str());
            continue;
        }
        DType dtype = safetensors_dtype(dtype_val->str_val);

        // Extract shape
        const JValue* shape_val = jobj_find(tensor_meta, "shape");
        if (!shape_val || shape_val->type != JType::ARRAY) {
            IMP_LOG_WARN("Tensor '%s' missing shape, skipping", tensor_name.c_str());
            continue;
        }

        int ndim = static_cast<int>(shape_val->arr.size());
        if (ndim > kMaxDims) {
            IMP_LOG_WARN("Tensor '%s' has %d dims (max %d), skipping",
                         tensor_name.c_str(), ndim, kMaxDims);
            continue;
        }

        int64_t shape[kMaxDims] = {};
        for (int d = 0; d < ndim; d++) {
            shape[d] = shape_val->arr[d].as_int();
        }

        // Extract data_offsets [start, end]
        const JValue* offsets_val = jobj_find(tensor_meta, "data_offsets");
        if (!offsets_val || offsets_val->type != JType::ARRAY || offsets_val->arr.size() != 2) {
            IMP_LOG_WARN("Tensor '%s' missing data_offsets, skipping", tensor_name.c_str());
            continue;
        }

        uint64_t offset_start = static_cast<uint64_t>(offsets_val->arr[0].as_int());
        uint64_t offset_end = static_cast<uint64_t>(offsets_val->arr[1].as_int());

        // Validate offsets fit within file
        if (tensor_data_offset + offset_end > file_size) {
            IMP_LOG_WARN("Tensor '%s' data offset out of bounds [%lu, %lu], skipping",
                         tensor_name.c_str(),
                         (unsigned long)offset_start, (unsigned long)offset_end);
            continue;
        }

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

    IMP_LOG_INFO("Parsed %zu tensors from SafeTensors header", tensor_map.size());

    // 5. Detect architecture from weight names
    ModelArch arch = detect_arch_from_weights(tensor_map);

    // 6. Create model and infer config from weight shapes
    auto model = std::make_unique<Model>();
    model->mmap_base_ = mmap_base;
    model->mmap_size_ = file_size;

    ModelConfig& cfg = model->config_;
    cfg.arch = arch;
    infer_config(cfg, tensor_map);
    apply_arch_defaults(cfg);

    IMP_LOG_INFO("Architecture: %s", model_arch_name(cfg.arch));
    IMP_LOG_INFO("Config: layers=%d d_model=%d d_ff=%d heads=%d kv_heads=%d vocab=%d ctx=%d",
                 cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv_heads,
                 cfg.vocab_size, cfg.max_seq_len);

    if (cfg.n_experts > 0) {
        IMP_LOG_INFO("MoE: %d experts, %d active, expert_d_ff=%d",
                     cfg.n_experts, cfg.n_experts_active, cfg.expert_d_ff);
    }

    // 7. Allocate layers and expert vectors
    model->layers_.resize(cfg.n_layers);

    if (cfg.n_experts > 0) {
        for (auto& layer : model->layers_) {
            layer.expert_w_gate.resize(cfg.n_experts);
            layer.expert_w_up.resize(cfg.n_experts);
            layer.expert_w_down.resize(cfg.n_experts);
        }
    }

    // 8. Assign tensors to model fields
    // Try the WeightMap first (for architecture-aware mapping)
    WeightMap wmap(arch);
    bool wmap_ok = wmap.apply_weights(*model, tensor_map);

    if (!wmap_ok) {
        // Fall back to direct HuggingFace name assignment
        IMP_LOG_DEBUG("WeightMap did not assign weights, using direct HF name mapping");

        int assigned = 0, skipped = 0;

        for (const auto& kv : tensor_map) {
            if (assign_tensor_hf(*model, kv.first, kv.second)) {
                assigned++;
            } else {
                IMP_LOG_DEBUG("Unassigned tensor: %s", kv.first.c_str());
                skipped++;
            }
        }

        IMP_LOG_INFO("Weights: %d assigned, %d skipped (direct mapping)", assigned, skipped);
    }

    // If output projection wasn't found, tie it to token embedding
    if (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr) {
        model->out_proj_ = model->tok_emb_;
        IMP_LOG_INFO("Tied output projection to token embedding");
    }

    // Re-infer vocab_size from token embedding if needed
    if (cfg.vocab_size == 0 && model->tok_emb_.data != nullptr) {
        cfg.vocab_size = static_cast<int>(model->tok_emb_.shape[0]);
    }

    IMP_LOG_INFO("SafeTensors model loaded successfully from %s", path.c_str());
    return model;
}

} // namespace imp
