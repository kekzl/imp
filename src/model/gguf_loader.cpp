#include "model/gguf_loader.h"
#include "model/model_arch.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// ---- GGML type tables ----

int ggml_blck_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:     return 1;
        case GGMLType::F16:     return 1;
        case GGMLType::BF16:    return 1;
        case GGMLType::F64:     return 1;
        case GGMLType::I8:      return 1;
        case GGMLType::I16:     return 1;
        case GGMLType::I32:     return 1;
        case GGMLType::I64:     return 1;
        case GGMLType::Q4_0:    return 32;
        case GGMLType::Q4_1:    return 32;
        case GGMLType::Q5_0:    return 32;
        case GGMLType::Q5_1:    return 32;
        case GGMLType::Q8_0:    return 32;
        case GGMLType::Q8_1:    return 32;
        case GGMLType::IQ4_NL:  return 32;
        case GGMLType::Q2_K:    return 256;
        case GGMLType::Q3_K:    return 256;
        case GGMLType::Q4_K:    return 256;
        case GGMLType::Q5_K:    return 256;
        case GGMLType::Q6_K:    return 256;
        case GGMLType::Q8_K:    return 256;
        case GGMLType::IQ2_XXS: return 256;
        case GGMLType::IQ2_XS:  return 256;
        case GGMLType::IQ2_S:   return 256;
        case GGMLType::IQ3_XXS: return 256;
        case GGMLType::IQ3_S:   return 256;
        case GGMLType::IQ1_S:   return 256;
        case GGMLType::IQ1_M:   return 256;
        case GGMLType::IQ4_XS:  return 256;
        default: return 0;
    }
}

size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:     return 4;
        case GGMLType::F16:     return 2;
        case GGMLType::BF16:    return 2;
        case GGMLType::F64:     return 8;
        case GGMLType::I8:      return 1;
        case GGMLType::I16:     return 2;
        case GGMLType::I32:     return 4;
        case GGMLType::I64:     return 8;
        case GGMLType::Q4_0:    return 18;   // 32*4/8 + 2 (fp16 scale)
        case GGMLType::Q4_1:    return 20;   // 32*4/8 + 2 + 2 (scale + min)
        case GGMLType::Q5_0:    return 22;   // 32*5/8 + 4 (high bits) + 2
        case GGMLType::Q5_1:    return 24;   // 32*5/8 + 4 + 2 + 2
        case GGMLType::Q8_0:    return 34;   // 32*1 + 2
        case GGMLType::Q8_1:    return 36;   // 32*1 + 2 + 2
        case GGMLType::Q2_K:    return 84;
        case GGMLType::Q3_K:    return 110;
        case GGMLType::Q4_K:    return 144;
        case GGMLType::Q5_K:    return 176;
        case GGMLType::Q6_K:    return 210;
        case GGMLType::Q8_K:    return 292;
        case GGMLType::IQ2_XXS: return 66;
        case GGMLType::IQ2_XS:  return 74;
        case GGMLType::IQ2_S:   return 82;
        case GGMLType::IQ3_XXS: return 98;
        case GGMLType::IQ3_S:   return 110;
        case GGMLType::IQ1_S:   return 50;
        case GGMLType::IQ1_M:   return 56;
        case GGMLType::IQ4_NL:  return 18;
        case GGMLType::IQ4_XS:  return 136;
        default: return 0;
    }
}

size_t ggml_row_size(GGMLType type, int64_t n_elements) {
    int bs = ggml_blck_size(type);
    if (bs == 0) return 0;
    return static_cast<size_t>((n_elements + bs - 1) / bs) * ggml_type_size(type);
}

DType ggml_type_to_dtype(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return DType::FP32;
        case GGMLType::F16:  return DType::FP16;
        case GGMLType::BF16: return DType::BF16;
        case GGMLType::I8:   return DType::INT8;
        case GGMLType::I32:  return DType::INT32;
        // Quantized types: use INT4 for 4-bit, INT8 as proxy for others
        case GGMLType::Q4_0: case GGMLType::Q4_1:
        case GGMLType::Q4_K: case GGMLType::IQ4_NL: case GGMLType::IQ4_XS:
            return DType::INT4;
        default:
            return DType::INT8;  // other quantized types stored as blocks
    }
}

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:     return "F32";
        case GGMLType::F16:     return "F16";
        case GGMLType::BF16:    return "BF16";
        case GGMLType::F64:     return "F64";
        case GGMLType::I8:      return "I8";
        case GGMLType::I16:     return "I16";
        case GGMLType::I32:     return "I32";
        case GGMLType::I64:     return "I64";
        case GGMLType::Q4_0:    return "Q4_0";
        case GGMLType::Q4_1:    return "Q4_1";
        case GGMLType::Q5_0:    return "Q5_0";
        case GGMLType::Q5_1:    return "Q5_1";
        case GGMLType::Q8_0:    return "Q8_0";
        case GGMLType::Q8_1:    return "Q8_1";
        case GGMLType::Q2_K:    return "Q2_K";
        case GGMLType::Q3_K:    return "Q3_K";
        case GGMLType::Q4_K:    return "Q4_K";
        case GGMLType::Q5_K:    return "Q5_K";
        case GGMLType::Q6_K:    return "Q6_K";
        case GGMLType::Q8_K:    return "Q8_K";
        case GGMLType::IQ2_XXS: return "IQ2_XXS";
        case GGMLType::IQ2_XS:  return "IQ2_XS";
        case GGMLType::IQ2_S:   return "IQ2_S";
        case GGMLType::IQ3_XXS: return "IQ3_XXS";
        case GGMLType::IQ3_S:   return "IQ3_S";
        case GGMLType::IQ1_S:   return "IQ1_S";
        case GGMLType::IQ1_M:   return "IQ1_M";
        case GGMLType::IQ4_NL:  return "IQ4_NL";
        case GGMLType::IQ4_XS:  return "IQ4_XS";
        default:                return "UNKNOWN";
    }
}

// ---- Binary reader over mmap'd memory ----

class BinaryReader {
public:
    BinaryReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    size_t pos() const { return pos_; }
    size_t remaining() const { return size_ - pos_; }
    const uint8_t* ptr() const { return data_ + pos_; }

    bool check(size_t n) const { return pos_ + n <= size_; }

    void skip(size_t n) {
        pos_ += n;
    }

    void align(size_t alignment) {
        size_t rem = pos_ % alignment;
        if (rem != 0) pos_ += alignment - rem;
    }

    template<typename T>
    T read() {
        T val;
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    uint8_t  read_u8()  { return read<uint8_t>(); }
    int8_t   read_i8()  { return read<int8_t>(); }
    uint16_t read_u16() { return read<uint16_t>(); }
    int16_t  read_i16() { return read<int16_t>(); }
    uint32_t read_u32() { return read<uint32_t>(); }
    int32_t  read_i32() { return read<int32_t>(); }
    uint64_t read_u64() { return read<uint64_t>(); }
    int64_t  read_i64() { return read<int64_t>(); }
    float    read_f32() { return read<float>(); }
    double   read_f64() { return read<double>(); }

    std::string read_string() {
        uint64_t len = read_u64();
        if (!check(len)) return "";
        std::string s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += static_cast<size_t>(len);
        return s;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

// ---- GGUF metadata value (variant-like) ----

struct GGUFValue {
    GGUFValueType type = GGUFValueType::UINT32;
    uint64_t uint_val = 0;
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string str_val;
    std::vector<std::string> str_array;
    std::vector<float> float_array;
    std::vector<int32_t> int_array;
};

static GGUFValue read_gguf_value(BinaryReader& r, GGUFValueType type) {
    GGUFValue v;
    v.type = type;
    switch (type) {
        case GGUFValueType::UINT8:   v.uint_val = r.read_u8(); break;
        case GGUFValueType::INT8:    v.int_val  = r.read_i8(); break;
        case GGUFValueType::UINT16:  v.uint_val = r.read_u16(); break;
        case GGUFValueType::INT16:   v.int_val  = r.read_i16(); break;
        case GGUFValueType::UINT32:  v.uint_val = r.read_u32(); break;
        case GGUFValueType::INT32:   v.int_val  = r.read_i32(); break;
        case GGUFValueType::FLOAT32: v.float_val = r.read_f32(); break;
        case GGUFValueType::BOOL:    v.uint_val = r.read_u8(); break;
        case GGUFValueType::STRING:  v.str_val = r.read_string(); break;
        case GGUFValueType::UINT64:  v.uint_val = r.read_u64(); break;
        case GGUFValueType::INT64:   v.int_val  = r.read_i64(); break;
        case GGUFValueType::FLOAT64: v.float_val = r.read_f64(); break;
        case GGUFValueType::ARRAY: {
            auto arr_type = static_cast<GGUFValueType>(r.read_u32());
            uint64_t count = r.read_u64();
            if (arr_type == GGUFValueType::STRING) {
                v.str_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++)
                    v.str_array.push_back(r.read_string());
            } else if (arr_type == GGUFValueType::FLOAT32) {
                v.float_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++)
                    v.float_array.push_back(r.read_f32());
            } else if (arr_type == GGUFValueType::INT32) {
                v.int_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++)
                    v.int_array.push_back(r.read_i32());
            } else if (arr_type == GGUFValueType::UINT32) {
                v.int_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++)
                    v.int_array.push_back(static_cast<int32_t>(r.read_u32()));
            } else {
                // Skip unknown array element types
                for (uint64_t i = 0; i < count; i++)
                    read_gguf_value(r, arr_type);
            }
            break;
        }
    }
    return v;
}

static uint64_t val_uint(const GGUFValue& v) {
    switch (v.type) {
        case GGUFValueType::UINT8: case GGUFValueType::UINT16:
        case GGUFValueType::UINT32: case GGUFValueType::UINT64:
        case GGUFValueType::BOOL:
            return v.uint_val;
        case GGUFValueType::INT8: case GGUFValueType::INT16:
        case GGUFValueType::INT32: case GGUFValueType::INT64:
            return static_cast<uint64_t>(v.int_val);
        case GGUFValueType::FLOAT32: case GGUFValueType::FLOAT64:
            return static_cast<uint64_t>(v.float_val);
        default: return 0;
    }
}

static double val_float(const GGUFValue& v) {
    switch (v.type) {
        case GGUFValueType::FLOAT32: case GGUFValueType::FLOAT64:
            return v.float_val;
        case GGUFValueType::UINT8: case GGUFValueType::UINT16:
        case GGUFValueType::UINT32: case GGUFValueType::UINT64:
            return static_cast<double>(v.uint_val);
        case GGUFValueType::INT8: case GGUFValueType::INT16:
        case GGUFValueType::INT32: case GGUFValueType::INT64:
            return static_cast<double>(v.int_val);
        default: return 0.0;
    }
}

// ---- Split string by delimiter ----

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == delim) {
            parts.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return parts;
}

// ---- Assign a single tensor to the model by GGUF name ----

static bool assign_tensor(Model& model, const std::string& name,
                           const Tensor& tensor, GGMLType gtype) {
    auto qtype = static_cast<GGMLQuantType>(static_cast<uint32_t>(gtype));
    if (name == "token_embd.weight") {
        model.tok_emb_ = tensor;
        model.tok_emb_qtype_ = qtype;
        return true;
    }
    if (name == "output_norm.weight") {
        model.out_norm_ = tensor;
        model.out_norm_qtype_ = qtype;
        return true;
    }
    if (name == "output.weight") {
        model.out_proj_ = tensor;
        model.out_proj_qtype_ = qtype;
        return true;
    }

    // Layer weights: "blk.{i}.{field}" or "blk.{i}.{field}.{expert}.weight"
    if (name.substr(0, 4) != "blk.") return false;

    auto parts = split(name, '.');
    // Minimum: ["blk", "0", "ssm_a"] = 3 parts (some SSM tensors have no suffix)
    if (parts.size() < 3) return false;

    int layer_idx = 0;
    try { layer_idx = std::stoi(parts[1]); }
    catch (...) { return false; }

    if (layer_idx < 0 || layer_idx >= model.n_layers()) return false;
    auto& layer = model.layers_[layer_idx];

    // 3-part: "blk.{i}.{name}" — SSM scalar/vector tensors without .weight/.bias suffix
    if (parts.size() == 3) {
        const auto& field = parts[2];
        if      (field == "ssm_a") layer.ssm_a = tensor;
        else if (field == "ssm_d") layer.ssm_d = tensor;
        else return false;
        return true;
    }

    // 4-part: "blk.{i}.{name}.weight" or "blk.{i}.{name}.bias"
    if (parts.size() == 4) {
        const auto& field = parts[2];
        if      (field == "attn_q")      { layer.wq = tensor; layer.wq_qtype = qtype; }
        else if (field == "attn_k")      { layer.wk = tensor; layer.wk_qtype = qtype; }
        else if (field == "attn_v")      { layer.wv = tensor; layer.wv_qtype = qtype; }
        else if (field == "attn_output") { layer.wo = tensor; layer.wo_qtype = qtype; }
        else if (field == "attn_norm")     layer.attn_norm = tensor;
        else if (field == "attn_q_norm") layer.attn_q_norm = tensor;
        else if (field == "attn_k_norm") layer.attn_k_norm = tensor;
        else if (field == "ffn_gate")    { layer.w_gate = tensor; layer.w_gate_qtype = qtype; }
        else if (field == "ffn_up")      { layer.w_up = tensor; layer.w_up_qtype = qtype; }
        else if (field == "ffn_down")    { layer.w_down = tensor; layer.w_down_qtype = qtype; }
        else if (field == "ffn_norm")      layer.ffn_norm = tensor;
        else if (field == "ffn_gate_inp")  layer.moe_gate = tensor;
        // Packed expert tensors: 3D [n_experts, rows, cols]
        else if (field == "ffn_gate_exps") { layer.expert_gate_packed = tensor; layer.expert_gate_qtype = qtype; }
        else if (field == "ffn_up_exps")   { layer.expert_up_packed = tensor; layer.expert_up_qtype = qtype; }
        else if (field == "ffn_down_exps") { layer.expert_down_packed = tensor; layer.expert_down_qtype = qtype; }
        // Shared expert (always-active, e.g. Nemotron/DeepSeek)
        else if (field == "ffn_gate_shexp") { layer.w_gate_shared = tensor; layer.w_gate_shared_qtype = qtype; }
        else if (field == "ffn_up_shexp")   { layer.w_up_shared = tensor; layer.w_up_shared_qtype = qtype; }
        else if (field == "ffn_down_shexp") { layer.w_down_shared = tensor; layer.w_down_shared_qtype = qtype; }
        // SSM weights (Mamba2)
        else if (field == "ssm_in")   { layer.ssm_in = tensor; layer.ssm_in_qtype = qtype; }
        else if (field == "ssm_out")  { layer.ssm_out = tensor; layer.ssm_out_qtype = qtype; }
        else if (field == "ssm_dt")     layer.ssm_dt_b = tensor;  // dt bias
        else if (field == "ssm_norm")   layer.ssm_norm_w = tensor;
        // SSM conv1d: "blk.{i}.ssm_conv1d.weight" / "blk.{i}.ssm_conv1d.bias"
        else if (field == "ssm_conv1d") {
            if (parts[3] == "weight")     layer.ssm_conv1d_w = tensor;
            else if (parts[3] == "bias")  layer.ssm_conv1d_b = tensor;
            else return false;
        }
        // Router bias (Nemotron MoE)
        else if (field == "exp_probs_b") layer.moe_router_bias = tensor;
        else return false;
        return true;
    }

    // 5-part: "blk.{i}.ffn_*.{expert_idx}.weight" — MoE per-expert weights
    if (parts.size() == 5) {
        const auto& field = parts[2];

        // MoE expert weights: "blk.{i}.ffn_*.{expert_idx}.weight"
        int expert_idx = 0;
        try { expert_idx = std::stoi(parts[3]); }
        catch (...) { return false; }

        int n_experts = model.config().n_experts;
        if (expert_idx < 0 || expert_idx >= n_experts) return false;

        if      (field == "ffn_gate") layer.expert_w_gate[expert_idx] = tensor;
        else if (field == "ffn_up")   layer.expert_w_up[expert_idx] = tensor;
        else if (field == "ffn_down") layer.expert_w_down[expert_idx] = tensor;
        else return false;
        return true;
    }

    return false;
}

// ---- Main GGUF loader ----

std::unique_ptr<Model> load_gguf(const std::string& path) {
    // 1. Open and mmap the file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("Failed to open GGUF file: %s", path.c_str());
        return nullptr;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        IMP_LOG_ERROR("Failed to stat GGUF file: %s", path.c_str());
        close(fd);
        return nullptr;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_base == MAP_FAILED) {
        IMP_LOG_ERROR("Failed to mmap GGUF file: %s (size=%zu)", path.c_str(), file_size);
        return nullptr;
    }

    // Advise the kernel we'll read sequentially
    madvise(mmap_base, file_size, MADV_SEQUENTIAL);

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);
    BinaryReader reader(data, file_size);

    // 2. Parse header
    uint32_t magic = reader.read_u32();
    if (magic != GGUF_MAGIC) {
        IMP_LOG_ERROR("Invalid GGUF magic: 0x%08x", magic);
        munmap(mmap_base, file_size);
        return nullptr;
    }

    uint32_t version = reader.read_u32();
    if (version < 2 || version > 3) {
        IMP_LOG_ERROR("Unsupported GGUF version: %u (expected 2 or 3)", version);
        munmap(mmap_base, file_size);
        return nullptr;
    }

    uint64_t tensor_count = reader.read_u64();
    uint64_t kv_count = reader.read_u64();

    IMP_LOG_INFO("GGUF v%u: %lu tensors, %lu metadata KVs",
                 version, (unsigned long)tensor_count, (unsigned long)kv_count);

    // 3. Parse metadata key-value pairs
    std::unordered_map<std::string, GGUFValue> metadata;
    metadata.reserve(static_cast<size_t>(kv_count));

    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key = reader.read_string();
        auto vtype = static_cast<GGUFValueType>(reader.read_u32());
        GGUFValue value = read_gguf_value(reader, vtype);
        metadata.emplace(std::move(key), std::move(value));
    }

    // 4. Parse tensor info entries
    std::vector<GGUFTensorInfo> tensor_infos;
    tensor_infos.reserve(static_cast<size_t>(tensor_count));

    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFTensorInfo info;
        info.name = reader.read_string();
        info.n_dims = reader.read_u32();
        for (uint32_t d = 0; d < info.n_dims && d < 4; d++) {
            info.dims[d] = static_cast<int64_t>(reader.read_u64());
        }
        // Skip extra dims if n_dims > 4 (shouldn't happen)
        for (uint32_t d = 4; d < info.n_dims; d++) {
            reader.read_u64();
        }
        // Fill remaining dims with 1
        for (uint32_t d = info.n_dims; d < 4; d++) {
            info.dims[d] = 1;
        }
        info.type = static_cast<GGMLType>(reader.read_u32());
        info.offset = reader.read_u64();
        tensor_infos.push_back(std::move(info));
    }

    // 5. Compute tensor data start offset (aligned)
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    auto it_align = metadata.find("general.alignment");
    if (it_align != metadata.end()) {
        alignment = static_cast<size_t>(val_uint(it_align->second));
        if (alignment == 0) alignment = GGUF_DEFAULT_ALIGNMENT;
    }

    reader.align(alignment);
    size_t tensor_data_start = reader.pos();

    IMP_LOG_DEBUG("Tensor data starts at offset %zu (alignment=%zu)", tensor_data_start, alignment);

    // 6. Extract model config from metadata
    auto model = std::make_unique<Model>();
    model->mmap_base_ = mmap_base;
    model->mmap_size_ = file_size;

    ModelConfig& cfg = model->config_;

    auto it_arch = metadata.find("general.architecture");
    std::string arch_str = (it_arch != metadata.end()) ? it_arch->second.str_val : "llama";
    cfg.arch = parse_model_arch(arch_str);

    IMP_LOG_INFO("Architecture: %s -> %s", arch_str.c_str(), model_arch_name(cfg.arch));

    // Helper lambdas for metadata lookup with arch prefix
    auto get_uint = [&](const std::string& key, uint64_t def = 0) -> uint64_t {
        auto it = metadata.find(arch_str + "." + key);
        if (it != metadata.end()) return val_uint(it->second);
        it = metadata.find(key);
        if (it != metadata.end()) return val_uint(it->second);
        return def;
    };

    auto get_float = [&](const std::string& key, double def = 0.0) -> double {
        auto it = metadata.find(arch_str + "." + key);
        if (it != metadata.end()) return val_float(it->second);
        it = metadata.find(key);
        if (it != metadata.end()) return val_float(it->second);
        return def;
    };

    cfg.n_layers     = static_cast<int>(get_uint("block_count"));
    cfg.d_model      = static_cast<int>(get_uint("embedding_length"));
    cfg.d_ff         = static_cast<int>(get_uint("feed_forward_length"));
    cfg.n_heads      = static_cast<int>(get_uint("attention.head_count"));
    cfg.n_kv_heads   = static_cast<int>(get_uint("attention.head_count_kv", cfg.n_heads));
    cfg.head_dim     = static_cast<int>(get_uint("attention.key_length", 0));
    if (cfg.head_dim == 0 && cfg.n_heads > 0) {
        cfg.head_dim = cfg.d_model / cfg.n_heads;
    }
    cfg.max_seq_len  = static_cast<int>(get_uint("context_length", 4096));
    cfg.vocab_size   = static_cast<int>(get_uint("vocab_size", 0));
    cfg.rope_theta   = static_cast<float>(get_float("rope.freq_base", 10000.0));
    cfg.rms_norm_eps = static_cast<float>(get_float("attention.layer_norm_rms_epsilon", 1e-5));

    cfg.n_experts        = static_cast<int>(get_uint("expert_count", 0));
    cfg.n_experts_active = static_cast<int>(get_uint("expert_used_count", 0));
    cfg.expert_d_ff      = static_cast<int>(get_uint("expert_feed_forward_length", cfg.d_ff));

    // Per-layer arrays (Nemotron hybrid: head_count_kv and feed_forward_length are arrays)
    {
        auto get_int_array = [&](const std::string& key) -> std::vector<int> {
            auto it = metadata.find(arch_str + "." + key);
            if (it == metadata.end()) it = metadata.find(key);
            if (it == metadata.end() || it->second.int_array.empty()) return {};
            std::vector<int> result;
            result.reserve(it->second.int_array.size());
            for (auto v : it->second.int_array)
                result.push_back(static_cast<int>(v));
            return result;
        };

        cfg.n_kv_heads_per_layer = get_int_array("attention.head_count_kv");
        cfg.d_ff_per_layer = get_int_array("feed_forward_length");

        // If we got per-layer arrays, set the scalar config to max values (for buffer sizing)
        if (!cfg.n_kv_heads_per_layer.empty()) {
            int max_kv = 0;
            for (int v : cfg.n_kv_heads_per_layer) max_kv = std::max(max_kv, v);
            cfg.n_kv_heads = max_kv;
            IMP_LOG_INFO("Per-layer KV heads: %zu layers, max=%d",
                         cfg.n_kv_heads_per_layer.size(), max_kv);
        }
        if (!cfg.d_ff_per_layer.empty()) {
            int max_ff = 0;
            for (int v : cfg.d_ff_per_layer) max_ff = std::max(max_ff, v);
            cfg.d_ff = max_ff;
            IMP_LOG_INFO("Per-layer d_ff: %zu layers, max=%d",
                         cfg.d_ff_per_layer.size(), max_ff);
        }
    }

    // Mamba2 SSM config
    cfg.ssm_conv_kernel = static_cast<int>(get_uint("ssm.conv_kernel", 0));
    cfg.ssm_state_size  = static_cast<int>(get_uint("ssm.state_size", 0));
    cfg.ssm_group_count = static_cast<int>(get_uint("ssm.group_count", 0));
    cfg.ssm_inner_size  = static_cast<int>(get_uint("ssm.inner_size", 0));
    cfg.ssm_dt_rank     = static_cast<int>(get_uint("ssm.time_step_rank", 0));

    // Partial RoPE
    cfg.rope_dim = static_cast<int>(get_uint("rope.dimension_count", 0));

    // Extended MoE config
    cfg.n_experts_shared     = static_cast<int>(get_uint("expert_shared_count", 0));
    cfg.expert_shared_d_ff   = static_cast<int>(get_uint("expert_shared_feed_forward_length", 0));
    cfg.expert_weights_scale = static_cast<float>(get_float("expert_weights_scale", 1.0));
    cfg.expert_weights_norm  = (get_uint("expert_weights_norm", 0) != 0);
    // Apply arch-specific config defaults (e.g. sigmoid gating for Nemotron)
    apply_arch_defaults(cfg);

    IMP_LOG_INFO("Config: layers=%d d_model=%d d_ff=%d heads=%d kv_heads=%d head_dim=%d vocab=%d ctx=%d",
                 cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv_heads,
                 cfg.head_dim, cfg.vocab_size, cfg.max_seq_len);

    if (cfg.n_experts > 0) {
        IMP_LOG_INFO("MoE: %d experts, %d active, expert_d_ff=%d",
                     cfg.n_experts, cfg.n_experts_active, cfg.expert_d_ff);
    }

    if (cfg.ssm_inner_size > 0) {
        IMP_LOG_INFO("SSM: conv_kernel=%d state_size=%d groups=%d inner=%d dt_rank=%d",
                     cfg.ssm_conv_kernel, cfg.ssm_state_size, cfg.ssm_group_count,
                     cfg.ssm_inner_size, cfg.ssm_dt_rank);
    }

    if (cfg.rope_dim > 0) {
        IMP_LOG_INFO("Partial RoPE: rope_dim=%d (full head_dim=%d)", cfg.rope_dim, cfg.head_dim);
    }

    // 7. Allocate layers and assign weights
    model->layers_.resize(cfg.n_layers);

    if (cfg.n_experts > 0) {
        for (auto& layer : model->layers_) {
            layer.expert_w_gate.resize(cfg.n_experts);
            layer.expert_w_up.resize(cfg.n_experts);
            layer.expert_w_down.resize(cfg.n_experts);
        }
    }

    int assigned = 0, skipped = 0;

    for (const auto& info : tensor_infos) {
        // Compute pointer into mmap'd data
        auto* tensor_data = const_cast<void*>(
            static_cast<const void*>(data + tensor_data_start + info.offset));

        // Build tensor descriptor
        // GGUF stores dims as ne[0]=innermost. We reverse for shape[0]=outermost.
        int ndim = static_cast<int>(info.n_dims);
        int64_t shape[4] = {1, 1, 1, 1};
        for (int d = 0; d < ndim; d++) {
            shape[d] = info.dims[ndim - 1 - d];
        }

        Tensor t(tensor_data, ggml_type_to_dtype(info.type), ndim, shape, /*on_device=*/false);

        if (assign_tensor(*model, info.name, t, info.type)) {
            assigned++;
        } else {
            IMP_LOG_DEBUG("Unassigned tensor: %s [%s] shape=[%ld,%ld,%ld,%ld]",
                          info.name.c_str(), ggml_type_name(info.type),
                          (long)info.dims[0], (long)info.dims[1],
                          (long)info.dims[2], (long)info.dims[3]);
            skipped++;
        }
    }

    // Infer vocab_size from token_embd if not in metadata
    if (cfg.vocab_size == 0 && model->tok_emb_.data != nullptr) {
        cfg.vocab_size = static_cast<int>(model->tok_emb_.shape[0]);
    }

    // Weight tying: if no output.weight, share token_embd
    if (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr) {
        model->out_proj_ = model->tok_emb_;
        model->out_proj_qtype_ = model->tok_emb_qtype_;
        IMP_LOG_INFO("Weight tying: output projection shares token embedding");
    }

    IMP_LOG_INFO("Weights: %d assigned, %d skipped", assigned, skipped);

    // 8. Extract tokenizer from GGUF metadata
    auto tokenizer = std::make_unique<Tokenizer>();

    // Detect tokenizer type (default: SentencePiece)
    auto it_tok_model = metadata.find("tokenizer.ggml.model");
    std::string tok_type = "spm";
    if (it_tok_model != metadata.end()) {
        const std::string& tm = it_tok_model->second.str_val;
        if (tm == "gpt2") tok_type = "gpt2";
    }
    tokenizer->set_type(tok_type);

    // add_bos_token flag (Qwen3: 0, LLaMA: 1)
    auto it_add_bos = metadata.find("tokenizer.ggml.add_bos_token");
    if (it_add_bos != metadata.end()) {
        tokenizer->set_add_bos(val_uint(it_add_bos->second) != 0);
    }

    auto it_tokens = metadata.find("tokenizer.ggml.tokens");
    if (it_tokens != metadata.end() && !it_tokens->second.str_array.empty()) {
        const auto& tokens = it_tokens->second.str_array;

        // Scores (optional, used for SentencePiece BPE merge priority)
        std::vector<float> scores;
        auto it_scores = metadata.find("tokenizer.ggml.scores");
        if (it_scores != metadata.end()) {
            scores = it_scores->second.float_array;
        }
        scores.resize(tokens.size(), 0.0f);

        // Special token IDs
        int bos_id = 1, eos_id = 2;
        auto it_bos = metadata.find("tokenizer.ggml.bos_token_id");
        if (it_bos != metadata.end()) bos_id = static_cast<int>(val_uint(it_bos->second));
        auto it_eos = metadata.find("tokenizer.ggml.eos_token_id");
        if (it_eos != metadata.end()) eos_id = static_cast<int>(val_uint(it_eos->second));

        tokenizer->load_vocab(tokens, scores, bos_id, eos_id);

        // Load BPE merge rules (for GPT2-style tokenizers)
        if (tok_type == "gpt2") {
            auto it_merges = metadata.find("tokenizer.ggml.merges");
            if (it_merges != metadata.end() && !it_merges->second.str_array.empty()) {
                tokenizer->load_merges(it_merges->second.str_array);
                IMP_LOG_INFO("Tokenizer: loaded %zu BPE merge rules",
                             it_merges->second.str_array.size());
            }
        }

        // Extract chat template string (Jinja2) for template family detection
        auto it_tpl = metadata.find("tokenizer.chat_template");
        if (it_tpl != metadata.end() && !it_tpl->second.str_val.empty()) {
            tokenizer->set_chat_template_str(it_tpl->second.str_val);
            IMP_LOG_INFO("Chat template: %zu chars", it_tpl->second.str_val.size());
        }

        IMP_LOG_INFO("Tokenizer: type=%s, %d tokens, bos=%d, eos=%d, add_bos=%d",
                     tok_type.c_str(), tokenizer->vocab_size(), bos_id, eos_id,
                     tokenizer->add_bos() ? 1 : 0);
    } else {
        IMP_LOG_WARN("No tokenizer data found in GGUF metadata");
    }

    model->set_tokenizer(std::move(tokenizer));

    IMP_LOG_INFO("GGUF model loaded successfully from %s", path.c_str());
    return model;
}

} // namespace imp
