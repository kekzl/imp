#include "gguf_stub.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
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

std::string generate_gguf_stub(const std::string& arch) {
    // ---- 1. Build tensor list ----
    // GGUF dims are stored innermost-first. For a 2D weight [rows, cols] in our
    // convention, GGUF stores ne[0]=cols, ne[1]=rows.

    std::vector<TensorDesc> tensors;

    auto add_2d = [&](const char* name, int rows, int cols, uint32_t type) {
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

    auto add_1d = [&](const char* name, int size, uint32_t type) {
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

    // blk.0 attention
    add_1d("blk.0.attn_norm.weight", D_MODEL, GGML_TYPE_F32);
    // attn_q: [n_heads * head_dim, d_model] = [64, 64] for our config
    add_2d("blk.0.attn_q.weight", N_HEADS * HEAD_DIM, D_MODEL, GGML_TYPE_F16);
    add_2d("blk.0.attn_k.weight", N_HEADS * HEAD_DIM, D_MODEL, GGML_TYPE_F16);
    add_2d("blk.0.attn_v.weight", N_HEADS * HEAD_DIM, D_MODEL, GGML_TYPE_F16);
    // attn_output: [d_model, n_heads * head_dim]
    add_2d("blk.0.attn_output.weight", D_MODEL, N_HEADS * HEAD_DIM, GGML_TYPE_F16);

    // blk.0 FFN
    add_1d("blk.0.ffn_norm.weight", D_MODEL, GGML_TYPE_F32);
    add_2d("blk.0.ffn_gate.weight", D_FF, D_MODEL, GGML_TYPE_F16);
    add_2d("blk.0.ffn_up.weight", D_FF, D_MODEL, GGML_TYPE_F16);
    add_2d("blk.0.ffn_down.weight", D_MODEL, D_FF, GGML_TYPE_F16);

    // output norm + output projection
    add_1d("output_norm.weight", D_MODEL, GGML_TYPE_F32);
    add_2d("output.weight", VOCAB, D_MODEL, GGML_TYPE_F16);

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
    uint64_t n_kv = 16;

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
    w.write_kv_u32(arch + ".block_count", N_LAYERS);
    w.write_kv_u32(arch + ".feed_forward_length", D_FF);
    w.write_kv_u32(arch + ".attention.head_count", N_HEADS);
    w.write_kv_u32(arch + ".attention.head_count_kv", N_HEADS);
    w.write_kv_u32(arch + ".rope.dimension_count", HEAD_DIM);
    w.write_kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    w.write_kv_string("tokenizer.ggml.model", "gpt2");
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
                float v = 1.0f;
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
    char path[] = "/tmp/imp_stub_XXXXXX.gguf";
    int fd = mkstemps(path, 5);
    if (fd < 0)
        return "";

    ssize_t written = write(fd, w.data(), w.size());
    close(fd);

    if (written < 0 || static_cast<size_t>(written) != w.size()) {
        unlink(path);
        return "";
    }

    return std::string(path);
}

}  // namespace test
}  // namespace imp
