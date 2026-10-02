#include "gguf_stub.h"

#include "model/safetensors_writer.h"
#include "runtime/config.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace imp {
namespace test {

// ---- GGUF constants ----

static constexpr uint32_t STUB_GGUF_MAGIC = 0x46554747;  // "GGUF" LE
static constexpr uint32_t STUB_GGUF_VERSION = 3;
static constexpr uint32_t STUB_ALIGNMENT = 32;

// GGUF value types
static constexpr uint32_t GGUF_TYPE_UINT32 = 4;
static constexpr uint32_t GGUF_TYPE_INT32 = 5;
static constexpr uint32_t GGUF_TYPE_FLOAT32 = 6;
static constexpr uint32_t GGUF_TYPE_STRING = 8;
static constexpr uint32_t GGUF_TYPE_ARRAY = 9;

// GGML tensor types
static constexpr uint32_t GGML_TYPE_F32 = 0;
static constexpr uint32_t GGML_TYPE_F16 = 1;

// ---- Stub model dimensions ----

static constexpr int VOCAB = 256;
static constexpr int D_MODEL = 64;
static constexpr int N_HEADS = 2;
static constexpr int HEAD_DIM = 32;  // D_MODEL / N_HEADS
static constexpr int D_FF = 128;
static constexpr int N_LAYERS = 1;
static constexpr int CTX_LEN = 512;

// ---- Binary writer ----

class BinaryWriter {
    std::vector<uint8_t> buf_;

public:
    void write_u32(uint32_t v) {
        size_t pos = buf_.size();
        buf_.resize(pos + 4);
        memcpy(buf_.data() + pos, &v, 4);
    }
    void write_u64(uint64_t v) {
        size_t pos = buf_.size();
        buf_.resize(pos + 8);
        memcpy(buf_.data() + pos, &v, 8);
    }
    void write_i32(int32_t v) {
        size_t pos = buf_.size();
        buf_.resize(pos + 4);
        memcpy(buf_.data() + pos, &v, 4);
    }
    void write_f32(float v) {
        size_t pos = buf_.size();
        buf_.resize(pos + 4);
        memcpy(buf_.data() + pos, &v, 4);
    }
    void write_bytes(const void* data, size_t n) {
        size_t pos = buf_.size();
        buf_.resize(pos + n);
        memcpy(buf_.data() + pos, data, n);
    }
    void write_string(const std::string& s) {
        write_u64(s.size());
        write_bytes(s.data(), s.size());
    }

    // KV pair writers
    void write_kv_string(const std::string& key, const std::string& val) {
        write_string(key);
        write_u32(GGUF_TYPE_STRING);
        write_string(val);
    }
    void write_kv_u32(const std::string& key, uint32_t val) {
        write_string(key);
        write_u32(GGUF_TYPE_UINT32);
        write_u32(val);
    }
    void write_kv_f32(const std::string& key, float val) {
        write_string(key);
        write_u32(GGUF_TYPE_FLOAT32);
        write_f32(val);
    }
    void write_kv_string_array(const std::string& key, const std::vector<std::string>& arr) {
        write_string(key);
        write_u32(GGUF_TYPE_ARRAY);
        write_u32(static_cast<uint32_t>(GGUF_TYPE_STRING));  // element type
        write_u64(arr.size());
        for (const auto& s : arr)
            write_string(s);
    }
    void write_kv_i32_array(const std::string& key, const std::vector<int32_t>& arr) {
        write_string(key);
        write_u32(GGUF_TYPE_ARRAY);
        write_u32(GGUF_TYPE_INT32);
        write_u64(arr.size());
        for (int32_t v : arr)
            write_i32(v);
    }
    void write_kv_f32_array(const std::string& key, const std::vector<float>& arr) {
        write_string(key);
        write_u32(GGUF_TYPE_ARRAY);
        write_u32(GGUF_TYPE_FLOAT32);
        write_u64(arr.size());
        for (float v : arr)
            write_f32(v);
    }

    // Tensor info entry (no data, just metadata)
    // dims are in GGUF order: dims[0] = innermost (fastest-changing)
    void write_tensor_info(const std::string& name, uint32_t n_dims, const uint64_t* dims, uint32_t type,
                           uint64_t offset) {
        write_string(name);
        write_u32(n_dims);
        for (uint32_t d = 0; d < n_dims; d++)
            write_u64(dims[d]);
        write_u32(type);
        write_u64(offset);
    }

    void pad_to(size_t alignment) {
        while (buf_.size() % alignment)
            buf_.push_back(0);
    }

    size_t size() const { return buf_.size(); }
    const uint8_t* data() const { return buf_.data(); }

    bool write_file(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f)
            return false;
        size_t written = fwrite(buf_.data(), 1, buf_.size(), f);
        fclose(f);
        return written == buf_.size();
    }
};

// ---- Tensor descriptor for offset computation ----

struct TensorDesc {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];  // GGUF order (innermost first)
    uint32_t type;     // GGML_TYPE_F16 or GGML_TYPE_F32
    size_t byte_size;
    std::vector<float> f32_values;  // explicit F32 payload; empty = fill with 1.0
};

static size_t tensor_bytes(uint32_t type, const uint64_t* dims, uint32_t n_dims) {
    uint64_t n_elements = 1;
    for (uint32_t d = 0; d < n_dims; d++)
        n_elements *= dims[d];
    if (type == GGML_TYPE_F16)
        return n_elements * 2;
    if (type == GGML_TYPE_F32)
        return n_elements * 4;
    return n_elements * 4;  // fallback
}

std::string generate_gguf_stub(const std::string& arch) { return generate_gguf_stub(arch, {}); }

std::string generate_gguf_stub(const std::string& arch, const std::vector<float>& rope_freqs,
                               const std::string& pre) {
    GgufStubSpec spec;
    spec.arch = arch;
    spec.rope_freqs = rope_freqs;
    spec.pre = pre;
    return generate_gguf_stub(spec);
}

std::string generate_gguf_stub(const GgufStubSpec& spec) {
    const std::string& arch = spec.arch;
    const std::vector<float>& rope_freqs = spec.rope_freqs;
    const std::string& pre = spec.pre;
    const int n_layers = spec.n_layers > 0 ? spec.n_layers : N_LAYERS;
    // ---- 1. Build tensor list ----
    // GGUF dims are stored innermost-first. For a 2D weight [rows, cols] in our
    // convention, GGUF stores ne[0]=cols, ne[1]=rows.

    std::vector<TensorDesc> tensors;

    auto add_2d = [&](const std::string& name, int rows, int cols, uint32_t type) {
        TensorDesc td;
        td.name = name;
        td.n_dims = 2;
        td.dims[0] = static_cast<uint64_t>(cols);  // innermost
        td.dims[1] = static_cast<uint64_t>(rows);  // outermost
        td.dims[2] = 1;
        td.dims[3] = 1;
        td.type = type;
        td.byte_size = tensor_bytes(type, td.dims, td.n_dims);
        tensors.push_back(td);
    };

    auto add_1d = [&](const std::string& name, int size, uint32_t type) {
        TensorDesc td;
        td.name = name;
        td.n_dims = 1;
        td.dims[0] = static_cast<uint64_t>(size);
        td.dims[1] = 1;
        td.dims[2] = 1;
        td.dims[3] = 1;
        td.type = type;
        td.byte_size = tensor_bytes(type, td.dims, td.n_dims);
        tensors.push_back(td);
    };

    // token_embd.weight [VOCAB, D_MODEL] FP16
    add_2d("token_embd.weight", VOCAB, D_MODEL, GGML_TYPE_F16);

    for (int l = 0; l < n_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        // attention; attn_q: [n_heads * head_dim, d_model] = [64, 64] for our config
        add_1d(blk + "attn_norm.weight", D_MODEL, GGML_TYPE_F32);
        add_2d(blk + "attn_q.weight", N_HEADS * HEAD_DIM, D_MODEL, GGML_TYPE_F16);
        add_2d(blk + "attn_k.weight", N_HEADS * HEAD_DIM, D_MODEL, GGML_TYPE_F16);
        add_2d(blk + "attn_v.weight", N_HEADS * HEAD_DIM, D_MODEL, GGML_TYPE_F16);
        // attn_output: [d_model, n_heads * head_dim]
        add_2d(blk + "attn_output.weight", D_MODEL, N_HEADS * HEAD_DIM, GGML_TYPE_F16);

        // FFN
        add_1d(blk + "ffn_norm.weight", D_MODEL, GGML_TYPE_F32);
        add_2d(blk + "ffn_gate.weight", D_FF, D_MODEL, GGML_TYPE_F16);
        add_2d(blk + "ffn_up.weight", D_FF, D_MODEL, GGML_TYPE_F16);
        add_2d(blk + "ffn_down.weight", D_MODEL, D_FF, GGML_TYPE_F16);
    }

    // output norm + output projection
    add_1d("output_norm.weight", D_MODEL, GGML_TYPE_F32);
    add_2d("output.weight", VOCAB, D_MODEL, GGML_TYPE_F16);
    if (!rope_freqs.empty()) {
        add_1d("rope_freqs.weight", static_cast<int>(rope_freqs.size()), GGML_TYPE_F32);
        tensors.back().f32_values = rope_freqs;
    }

    // ---- 2. Compute tensor data offsets (relative to data section start) ----
    // Each tensor's data aligned to STUB_ALIGNMENT within the data section.
    std::vector<uint64_t> offsets(tensors.size());
    size_t data_offset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        size_t rem = data_offset % STUB_ALIGNMENT;
        if (rem != 0)
            data_offset += STUB_ALIGNMENT - rem;
        offsets[i] = data_offset;
        data_offset += tensors[i].byte_size;
    }

    // ---- 3. Build tokenizer data ----
    // 256 single-byte tokens: "<0x00>", "<0x01>", ..., "<0xFF>"
    std::vector<std::string> token_strings(VOCAB);
    for (int i = 0; i < VOCAB; i++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "<0x%02X>", i);
        token_strings[i] = buf;
    }

    std::vector<int32_t> token_types(VOCAB, 1);  // all type=1 (normal)
    std::vector<float> token_scores(VOCAB, 0.0f);

    // ---- 4. Count metadata KV pairs ----
    // Architecture + name + context_length + embedding_length + block_count +
    // feed_forward_length + head_count + head_count_kv + rope.dimension_count +
    // layer_norm_rms_epsilon + tokenizer.ggml.model + tokens + token_type +
    // scores + bos_token_id + eos_token_id = 16
    uint64_t n_kv = 15 + (spec.rope_dimension_count ? 1 : 0) + (pre.empty() ? 0 : 1) + spec.u32.size() +
                    spec.f32.size() + spec.i32_arrays.size();

    // ---- 5. Write GGUF file ----
    BinaryWriter w;

    // Header
    w.write_u32(STUB_GGUF_MAGIC);
    w.write_u32(STUB_GGUF_VERSION);
    w.write_u64(static_cast<uint64_t>(tensors.size()));
    w.write_u64(n_kv);

    // Metadata KV pairs
    w.write_kv_string("general.architecture", arch);
    w.write_kv_string("general.name", "stub");
    w.write_kv_u32(arch + ".context_length", CTX_LEN);
    w.write_kv_u32(arch + ".embedding_length", D_MODEL);
    w.write_kv_u32(arch + ".block_count", static_cast<uint32_t>(n_layers));
    w.write_kv_u32(arch + ".feed_forward_length", D_FF);
    w.write_kv_u32(arch + ".attention.head_count", N_HEADS);
    w.write_kv_u32(arch + ".attention.head_count_kv", N_HEADS);
    if (spec.rope_dimension_count)
        w.write_kv_u32(arch + ".rope.dimension_count", HEAD_DIM);
    w.write_kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    for (const auto& [k, v] : spec.u32)
        w.write_kv_u32(arch + "." + k, v);
    for (const auto& [k, v] : spec.f32)
        w.write_kv_f32(arch + "." + k, v);
    for (const auto& [k, v] : spec.i32_arrays)
        w.write_kv_i32_array(arch + "." + k, v);
    w.write_kv_string("tokenizer.ggml.model", "gpt2");
    if (!pre.empty())
        w.write_kv_string("tokenizer.ggml.pre", pre);
    w.write_kv_string_array("tokenizer.ggml.tokens", token_strings);
    w.write_kv_i32_array("tokenizer.ggml.token_type", token_types);
    w.write_kv_f32_array("tokenizer.ggml.scores", token_scores);
    w.write_kv_u32("tokenizer.ggml.bos_token_id", 1);
    w.write_kv_u32("tokenizer.ggml.eos_token_id", 2);

    // Tensor info entries
    for (size_t i = 0; i < tensors.size(); i++) {
        w.write_tensor_info(tensors[i].name, tensors[i].n_dims, tensors[i].dims, tensors[i].type, offsets[i]);
    }

    // Pad to alignment before tensor data
    w.pad_to(STUB_ALIGNMENT);

    // ---- 6. Write tensor data ----
    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

    for (size_t i = 0; i < tensors.size(); i++) {
        // Pad to alignment for this tensor
        w.pad_to(STUB_ALIGNMENT);

        const auto& td = tensors[i];
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < td.n_dims; d++)
            n_elements *= td.dims[d];

        if (td.type == GGML_TYPE_F32) {
            // Norm weights: fill with 1.0
            for (uint64_t j = 0; j < n_elements; j++) {
                float v = td.f32_values.empty() ? 1.0f : td.f32_values[j];
                w.write_f32(v);
            }
        } else {
            // FP16 weights: small random values
            // Write as uint16 (IEEE 754 half-precision)
            for (uint64_t j = 0; j < n_elements; j++) {
                float fv = dist(rng);
                // Convert float to FP16 (simple truncation via bit manipulation)
                // Use a union-based approach for correct IEEE 754 conversion
                uint32_t fbits;
                memcpy(&fbits, &fv, 4);
                uint32_t sign = (fbits >> 16) & 0x8000;
                int exp = ((fbits >> 23) & 0xFF) - 127;
                uint32_t mantissa = fbits & 0x007FFFFF;

                uint16_t h;
                if (exp > 15) {
                    h = static_cast<uint16_t>(sign | 0x7C00);  // inf
                } else if (exp < -14) {
                    h = static_cast<uint16_t>(sign);  // zero/denorm
                } else {
                    h = static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mantissa >> 13));
                }
                uint8_t bytes[2];
                memcpy(bytes, &h, 2);
                w.write_bytes(bytes, 2);
            }
        }
    }

    // ---- 7. Write to temp file ----
    // One private dir per stub: holds the model and its warm-cache scratch dir (#2192).
    char dir_tmpl[] = "/tmp/imp_stub_XXXXXX";
    if (!mkdtemp(dir_tmpl))
        return "";
    const std::string path = std::string(dir_tmpl) + "/model.gguf";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        remove_gguf_stub(path);
        return "";
    }

    ssize_t written = write(fd, w.data(), w.size());
    close(fd);

    if (written < 0 || static_cast<size_t>(written) != w.size()) {
        remove_gguf_stub(path);
        return "";
    }

    return path;
}

std::string generate_hf_stub(const std::string& config_json, int n_layers) {
    struct Desc {
        std::string name;
        std::vector<int64_t> shape;
    };
    std::vector<Desc> descs = {{"model.embed_tokens.weight", {VOCAB, D_MODEL}}};
    for (int l = 0; l < n_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        descs.push_back({p + "input_layernorm.weight", {D_MODEL}});
        descs.push_back({p + "self_attn.q_proj.weight", {N_HEADS * HEAD_DIM, D_MODEL}});
        descs.push_back({p + "self_attn.k_proj.weight", {N_HEADS * HEAD_DIM, D_MODEL}});
        descs.push_back({p + "self_attn.v_proj.weight", {N_HEADS * HEAD_DIM, D_MODEL}});
        descs.push_back({p + "self_attn.o_proj.weight", {D_MODEL, N_HEADS * HEAD_DIM}});
        descs.push_back({p + "post_attention_layernorm.weight", {D_MODEL}});
        descs.push_back({p + "mlp.gate_proj.weight", {D_FF, D_MODEL}});
        descs.push_back({p + "mlp.up_proj.weight", {D_FF, D_MODEL}});
        descs.push_back({p + "mlp.down_proj.weight", {D_MODEL, D_FF}});
    }
    descs.push_back({"model.norm.weight", {D_MODEL}});
    descs.push_back({"lm_head.weight", {VOCAB, D_MODEL}});

    size_t max_elems = 0;
    for (const auto& d : descs) {
        size_t n = 1;
        for (int64_t s : d.shape)
            n *= static_cast<size_t>(s);
        max_elems = std::max(max_elems, n);
    }
    const std::vector<uint16_t> zeros(max_elems, 0);  // F16 zeros: RoPE parity reads no weights
    std::vector<SafeTensorsOut> out;
    for (const auto& d : descs) {
        size_t n = 1;
        for (int64_t s : d.shape)
            n *= static_cast<size_t>(s);
        out.push_back({d.name, "F16", d.shape, zeros.data(), n * sizeof(uint16_t)});
    }

    char dir_tmpl[] = "/tmp/imp_stub_XXXXXX";
    if (!mkdtemp(dir_tmpl))
        return "";
    const std::string dir = std::string(dir_tmpl) + "/hf";
    std::error_code ec;
    std::filesystem::create_directory(dir, ec);
    FILE* f = ec ? nullptr : fopen((dir + "/config.json").c_str(), "wb");
    const bool cfg_ok = f && fwrite(config_json.data(), 1, config_json.size(), f) == config_json.size();
    if (f)
        fclose(f);
    if (!cfg_ok || !write_safetensors(dir + "/model.safetensors", out).empty()) {
        remove_gguf_stub(dir);
        return "";
    }
    return dir;
}

std::string stub_cache_dir(const std::string& stub_path) {
    return stub_path.substr(0, stub_path.rfind('/')) + "/warm";
}

void arm_stub_warm_cache(const std::string& stub_path) {
    imp::RuntimeConfig rc;
    rc.warm_cache.enabled = true;
    rc.warm_cache.dir = stub_cache_dir(stub_path);
    imp::set_pending_runtime_config(rc);
}

void remove_gguf_stub(const std::string& stub_path) {
    const size_t slash = stub_path.rfind('/');
    if (slash == std::string::npos)
        return;
    const std::string dir = stub_path.substr(0, slash);
    if (dir.rfind("/tmp/imp_stub_", 0) != 0)  // never recurse into a foreign dir
        return;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

}  // namespace test
}  // namespace imp
