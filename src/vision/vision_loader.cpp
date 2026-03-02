#include "vision/vision_loader.h"
#include "model/gguf_loader.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---- Binary reader (duplicated from gguf_loader.cpp since it's file-local) ----

namespace {

class BinaryReader {
public:
    BinaryReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    size_t pos() const { return pos_; }
    bool failed() const { return failed_; }
    bool check(size_t n) const { return pos_ + n <= size_; }

    void skip(size_t n) {
        if (!check(n)) { failed_ = true; return; }
        pos_ += n;
    }

    void align(size_t alignment) {
        size_t rem = pos_ % alignment;
        if (rem != 0) {
            size_t pad = alignment - rem;
            if (!check(pad)) { failed_ = true; return; }
            pos_ += pad;
        }
    }

    template<typename T>
    T read() {
        if (!check(sizeof(T))) { failed_ = true; return T{}; }
        T val;
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    uint32_t read_u32() { return read<uint32_t>(); }
    uint64_t read_u64() { return read<uint64_t>(); }
    int32_t  read_i32() { return read<int32_t>(); }
    float    read_f32() { return read<float>(); }
    double   read_f64() { return read<double>(); }
    uint8_t  read_u8()  { return read<uint8_t>(); }
    int8_t   read_i8()  { return read<int8_t>(); }
    uint16_t read_u16() { return read<uint16_t>(); }
    int16_t  read_i16() { return read<int16_t>(); }
    int64_t  read_i64() { return read<int64_t>(); }

    std::string read_string() {
        uint64_t len = read_u64();
        if (!check(static_cast<size_t>(len))) { failed_ = true; return ""; }
        std::string s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += static_cast<size_t>(len);
        return s;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
    bool failed_ = false;
};

// ---- GGUF value types (same as gguf_loader.cpp) ----

enum class VGGUFValueType : uint32_t {
    UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5,
    FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12,
};

struct VGGUFValue {
    VGGUFValueType type = VGGUFValueType::UINT32;
    uint64_t uint_val = 0;
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string str_val;
    std::vector<float> float_array;
    std::vector<int32_t> int_array;
};

VGGUFValue read_value(BinaryReader& r, VGGUFValueType type) {
    VGGUFValue v;
    v.type = type;
    switch (type) {
        case VGGUFValueType::UINT8:   v.uint_val = r.read_u8(); break;
        case VGGUFValueType::INT8:    v.int_val  = r.read_i8(); break;
        case VGGUFValueType::UINT16:  v.uint_val = r.read_u16(); break;
        case VGGUFValueType::INT16:   v.int_val  = r.read_i16(); break;
        case VGGUFValueType::UINT32:  v.uint_val = r.read_u32(); break;
        case VGGUFValueType::INT32:   v.int_val  = r.read_i32(); break;
        case VGGUFValueType::FLOAT32: v.float_val = r.read_f32(); break;
        case VGGUFValueType::BOOL:    v.uint_val = r.read_u8(); break;
        case VGGUFValueType::STRING:  v.str_val = r.read_string(); break;
        case VGGUFValueType::UINT64:  v.uint_val = r.read_u64(); break;
        case VGGUFValueType::INT64:   v.int_val  = r.read_i64(); break;
        case VGGUFValueType::FLOAT64: v.float_val = r.read_f64(); break;
        case VGGUFValueType::ARRAY: {
            auto arr_type = static_cast<VGGUFValueType>(r.read_u32());
            uint64_t count = r.read_u64();
            if (arr_type == VGGUFValueType::FLOAT32) {
                v.float_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++)
                    v.float_array.push_back(r.read_f32());
            } else if (arr_type == VGGUFValueType::INT32) {
                v.int_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++)
                    v.int_array.push_back(r.read_i32());
            } else if (arr_type == VGGUFValueType::UINT32) {
                v.int_array.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++)
                    v.int_array.push_back(static_cast<int32_t>(r.read_u32()));
            } else {
                for (uint64_t i = 0; i < count; i++)
                    read_value(r, arr_type);
            }
            break;
        }
    }
    return v;
}

uint64_t val_uint(const VGGUFValue& v) {
    switch (v.type) {
        case VGGUFValueType::UINT8: case VGGUFValueType::UINT16:
        case VGGUFValueType::UINT32: case VGGUFValueType::UINT64:
        case VGGUFValueType::BOOL:
            return v.uint_val;
        case VGGUFValueType::INT8: case VGGUFValueType::INT16:
        case VGGUFValueType::INT32: case VGGUFValueType::INT64:
            return static_cast<uint64_t>(v.int_val);
        case VGGUFValueType::FLOAT32: case VGGUFValueType::FLOAT64:
            return static_cast<uint64_t>(v.float_val);
        default: return 0;
    }
}

double val_float(const VGGUFValue& v) {
    switch (v.type) {
        case VGGUFValueType::FLOAT32: case VGGUFValueType::FLOAT64:
            return v.float_val;
        case VGGUFValueType::UINT8: case VGGUFValueType::UINT16:
        case VGGUFValueType::UINT32: case VGGUFValueType::UINT64:
            return static_cast<double>(v.uint_val);
        case VGGUFValueType::INT8: case VGGUFValueType::INT16:
        case VGGUFValueType::INT32: case VGGUFValueType::INT64:
            return static_cast<double>(v.int_val);
        default: return 0.0;
    }
}

// Upload raw tensor data to GPU as FP16.
// Handles F32 -> FP16 conversion and F16 passthrough.
bool upload_tensor_fp16(const void* src, GGMLType type,
                        int64_t n_elements, void** d_out,
                        std::vector<void*>& gpu_allocs) {
    size_t fp16_bytes = static_cast<size_t>(n_elements) * sizeof(half);
    half* d_ptr = nullptr;
    if (cudaMalloc(&d_ptr, fp16_bytes) != cudaSuccess) {
        IMP_LOG_ERROR("Vision: cudaMalloc failed for %zu bytes", fp16_bytes);
        return false;
    }
    gpu_allocs.push_back(d_ptr);

    if (type == GGMLType::F16) {
        cudaMemcpy(d_ptr, src, fp16_bytes, cudaMemcpyHostToDevice);
    } else if (type == GGMLType::F32) {
        // Convert F32 -> F16 on host, then upload
        const float* f32 = static_cast<const float*>(src);
        std::vector<half> h_fp16(static_cast<size_t>(n_elements));
        for (int64_t i = 0; i < n_elements; i++) {
            h_fp16[i] = __float2half(f32[i]);
        }
        cudaMemcpy(d_ptr, h_fp16.data(), fp16_bytes, cudaMemcpyHostToDevice);
    } else if (type == GGMLType::BF16) {
        // Convert BF16 -> FP16 on host
        const uint16_t* bf16 = static_cast<const uint16_t*>(src);
        std::vector<half> h_fp16(static_cast<size_t>(n_elements));
        for (int64_t i = 0; i < n_elements; i++) {
            // BF16 -> F32 -> FP16
            uint32_t bits = static_cast<uint32_t>(bf16[i]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            h_fp16[i] = __float2half(f);
        }
        cudaMemcpy(d_ptr, h_fp16.data(), fp16_bytes, cudaMemcpyHostToDevice);
    } else {
        IMP_LOG_ERROR("Vision: unsupported GGML type %u for tensor upload",
                      static_cast<uint32_t>(type));
        return false;
    }

    *d_out = d_ptr;
    return true;
}

} // anonymous namespace

std::unique_ptr<VisionModel> load_vision_gguf(const std::string& path) {
    // 1. Open and mmap
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("Vision: failed to open %s", path.c_str());
        return nullptr;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return nullptr;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_base == MAP_FAILED) {
        IMP_LOG_ERROR("Vision: failed to mmap %s", path.c_str());
        return nullptr;
    }
    madvise(mmap_base, file_size, MADV_SEQUENTIAL);

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);
    BinaryReader reader(data, file_size);

    // 2. Parse header
    uint32_t magic = reader.read_u32();
    if (magic != 0x46554747) { // "GGUF"
        IMP_LOG_ERROR("Vision: invalid GGUF magic in %s", path.c_str());
        munmap(mmap_base, file_size);
        return nullptr;
    }

    uint32_t version = reader.read_u32();
    if (version < 2 || version > 3) {
        IMP_LOG_ERROR("Vision: unsupported GGUF version %u", version);
        munmap(mmap_base, file_size);
        return nullptr;
    }

    uint64_t tensor_count = reader.read_u64();
    uint64_t kv_count = reader.read_u64();

    IMP_LOG_INFO("Vision GGUF: %lu tensors, %lu metadata",
                 (unsigned long)tensor_count, (unsigned long)kv_count);

    // 3. Parse metadata
    std::unordered_map<std::string, VGGUFValue> metadata;
    for (uint64_t i = 0; i < kv_count && !reader.failed(); i++) {
        std::string key = reader.read_string();
        auto vtype = static_cast<VGGUFValueType>(reader.read_u32());
        VGGUFValue value = read_value(reader, vtype);
        metadata.emplace(std::move(key), std::move(value));
    }

    if (reader.failed()) {
        IMP_LOG_ERROR("Vision: metadata truncated");
        munmap(mmap_base, file_size);
        return nullptr;
    }

    // 4. Parse tensor infos
    struct TensorInfo {
        std::string name;
        uint32_t n_dims;
        int64_t dims[4];
        GGMLType type;
        uint64_t offset;
    };

    std::vector<TensorInfo> tensor_infos;
    tensor_infos.reserve(static_cast<size_t>(tensor_count));

    for (uint64_t i = 0; i < tensor_count && !reader.failed(); i++) {
        TensorInfo info;
        info.name = reader.read_string();
        info.n_dims = reader.read_u32();
        for (uint32_t d = 0; d < info.n_dims && d < 4; d++)
            info.dims[d] = static_cast<int64_t>(reader.read_u64());
        for (uint32_t d = 4; d < info.n_dims; d++)
            reader.read_u64();
        for (uint32_t d = info.n_dims; d < 4; d++)
            info.dims[d] = 1;
        info.type = static_cast<GGMLType>(reader.read_u32());
        info.offset = reader.read_u64();
        tensor_infos.push_back(std::move(info));
    }

    if (reader.failed()) {
        IMP_LOG_ERROR("Vision: tensor info truncated");
        munmap(mmap_base, file_size);
        return nullptr;
    }

    // 5. Align to tensor data
    size_t alignment = 32;
    {
        auto it = metadata.find("general.alignment");
        if (it != metadata.end())
            alignment = static_cast<size_t>(val_uint(it->second));
        if (alignment == 0) alignment = 32;
    }
    reader.align(alignment);
    size_t tensor_data_start = reader.pos();

    // 6. Extract vision config from metadata
    auto model = std::make_unique<VisionModel>();
    auto& cfg = model->config;

    auto get_uint = [&](const std::string& key, uint64_t def = 0) -> uint64_t {
        auto it = metadata.find(key);
        return (it != metadata.end()) ? val_uint(it->second) : def;
    };
    auto get_float = [&](const std::string& key, double def = 0.0) -> double {
        auto it = metadata.find(key);
        return (it != metadata.end()) ? val_float(it->second) : def;
    };

    cfg.image_size  = static_cast<int>(get_uint("clip.vision.image_size", 896));
    cfg.patch_size  = static_cast<int>(get_uint("clip.vision.patch_size", 14));
    cfg.hidden_size = static_cast<int>(get_uint("clip.vision.embedding_length", 1152));
    cfg.num_heads   = static_cast<int>(get_uint("clip.vision.attention.head_count", 16));
    cfg.intermediate_size = static_cast<int>(get_uint("clip.vision.feed_forward_length", 4304));
    cfg.num_layers  = static_cast<int>(get_uint("clip.vision.block_count", 27));

    if (cfg.num_heads > 0)
        cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.num_patches = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);

    // Image normalization parameters
    {
        auto it = metadata.find("clip.vision.image_mean");
        if (it != metadata.end() && it->second.float_array.size() >= 3) {
            for (int c = 0; c < 3; c++)
                cfg.image_mean[c] = it->second.float_array[c];
        }
    }
    {
        auto it = metadata.find("clip.vision.image_std");
        if (it != metadata.end() && it->second.float_array.size() >= 3) {
            for (int c = 0; c < 3; c++)
                cfg.image_std[c] = it->second.float_array[c];
        }
    }

    // mm_tokens_per_image for verification
    int mm_tokens = static_cast<int>(get_uint("clip.vision.mm_tokens_per_image", 256));
    cfg.num_image_tokens = mm_tokens;

    IMP_LOG_INFO("Vision config: image=%d, patch=%d, hidden=%d, heads=%d, "
                 "layers=%d, patches=%d, tokens=%d",
                 cfg.image_size, cfg.patch_size, cfg.hidden_size, cfg.num_heads,
                 cfg.num_layers, cfg.num_patches, cfg.num_image_tokens);

    // 7. Allocate layers
    model->layers.resize(cfg.num_layers);

    // 8. Assign tensors
    int assigned = 0;
    for (const auto& info : tensor_infos) {
        // Compute total elements and data pointer
        int64_t n_elements = 1;
        for (uint32_t d = 0; d < info.n_dims; d++)
            n_elements *= info.dims[d];

        const void* tensor_data = data + tensor_data_start + info.offset;

        // Reverse dims for our convention: shape[0] = outermost
        int64_t shape[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < info.n_dims; d++)
            shape[d] = info.dims[info.n_dims - 1 - d];

        // Upload to GPU as FP16
        void* d_ptr = nullptr;
        if (!upload_tensor_fp16(tensor_data, info.type, n_elements,
                                &d_ptr, model->gpu_allocs)) {
            IMP_LOG_ERROR("Vision: failed to upload tensor %s", info.name.c_str());
            munmap(mmap_base, file_size);
            return nullptr;
        }

        // Create Tensor (on_device = true)
        Tensor t(d_ptr, DType::FP16, static_cast<int>(info.n_dims), shape, true);

        const auto& name = info.name;

        // Patch embedding
        if (name == "v.patch_embd.weight")        { model->patch_embd_w = t; assigned++; }
        else if (name == "v.patch_embd.bias")      { model->patch_embd_b = t; assigned++; }
        else if (name == "v.position_embd.weight") { model->position_embd = t; assigned++; }
        // Post LayerNorm
        else if (name == "v.post_ln.weight")  { model->post_norm_w = t; assigned++; }
        else if (name == "v.post_ln.bias")    { model->post_norm_b = t; assigned++; }
        // Multimodal projector
        else if (name == "mm.0.weight")       { model->mm_proj_w = t; assigned++; }
        else if (name == "mm.0.bias")         { model->mm_proj_b = t; assigned++; }
        else if (name == "mm.pre_norm.weight")  { model->mm_pre_norm_w = t; assigned++; }
        else if (name == "mm.post_norm.weight") { model->mm_post_norm_w = t; assigned++; }
        // Layer weights: "v.blk.{i}.{field}"
        else if (name.substr(0, 6) == "v.blk.") {
            // Parse layer index
            size_t dot2 = name.find('.', 6);
            if (dot2 == std::string::npos) continue;
            int layer_idx = 0;
            try { layer_idx = std::stoi(name.substr(6, dot2 - 6)); }
            catch (...) { continue; }
            if (layer_idx < 0 || layer_idx >= cfg.num_layers) continue;

            auto& layer = model->layers[layer_idx];
            std::string field = name.substr(dot2 + 1);

            if      (field == "ln1.weight")       layer.ln1_w = t;
            else if (field == "ln1.bias")         layer.ln1_b = t;
            else if (field == "ln2.weight")       layer.ln2_w = t;
            else if (field == "ln2.bias")         layer.ln2_b = t;
            else if (field == "attn_q.weight")    layer.wq = t;
            else if (field == "attn_q.bias")      layer.bq = t;
            else if (field == "attn_k.weight")    layer.wk = t;
            else if (field == "attn_k.bias")      layer.bk = t;
            else if (field == "attn_v.weight")    layer.wv = t;
            else if (field == "attn_v.bias")      layer.bv = t;
            else if (field == "attn_out.weight")  layer.wo = t;
            else if (field == "attn_out.bias")    layer.bo = t;
            else if (field == "ffn_up.weight")    layer.ffn_up_w = t;
            else if (field == "ffn_up.bias")      layer.ffn_up_b = t;
            else if (field == "ffn_down.weight")  layer.ffn_down_w = t;
            else if (field == "ffn_down.bias")    layer.ffn_down_b = t;
            else {
                IMP_LOG_DEBUG("Vision: unrecognized layer tensor: %s", name.c_str());
                continue;
            }
            assigned++;
        } else {
            IMP_LOG_DEBUG("Vision: unrecognized tensor: %s", name.c_str());
        }
    }

    // Determine LLM d_model from mm_proj output dimension
    if (model->mm_proj_w.data) {
        model->lm_d_model = static_cast<int>(model->mm_proj_w.shape[0]);
        IMP_LOG_INFO("Vision: LLM d_model = %d (from mm_proj)", model->lm_d_model);
    }

    IMP_LOG_INFO("Vision: assigned %d / %lu tensors",
                 assigned, (unsigned long)tensor_count);

    // Cleanup mmap — data is on GPU now
    munmap(mmap_base, file_size);

    return model;
}

} // namespace imp
