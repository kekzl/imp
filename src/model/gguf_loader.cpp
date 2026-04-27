#include "model/gguf_loader.h"
#include "model/loader_assign.h"
#include "model/model_arch.h"
#include "model/tensor_kind_matcher.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// ---- GGML type tables ----

int gguf_blck_size(GgufWireType type) {
    switch (type) {
        case GgufWireType::F32:     return 1;
        case GgufWireType::F16:     return 1;
        case GgufWireType::BF16:    return 1;
        case GgufWireType::F64:     return 1;
        case GgufWireType::I8:      return 1;
        case GgufWireType::I16:     return 1;
        case GgufWireType::I32:     return 1;
        case GgufWireType::I64:     return 1;
        case GgufWireType::Q4_0:    return 32;
        case GgufWireType::Q4_1:    return 32;
        case GgufWireType::Q5_0:    return 32;
        case GgufWireType::Q5_1:    return 32;
        case GgufWireType::Q8_0:    return 32;
        case GgufWireType::Q8_1:    return 32;
        case GgufWireType::IQ4_NL:  return 32;
        case GgufWireType::Q2_K:    return 256;
        case GgufWireType::Q3_K:    return 256;
        case GgufWireType::Q4_K:    return 256;
        case GgufWireType::Q5_K:    return 256;
        case GgufWireType::Q6_K:    return 256;
        case GgufWireType::Q8_K:    return 256;
        case GgufWireType::IQ2_XXS: return 256;
        case GgufWireType::IQ2_XS:  return 256;
        case GgufWireType::IQ2_S:   return 256;
        case GgufWireType::IQ3_XXS: return 256;
        case GgufWireType::IQ3_S:   return 256;
        case GgufWireType::IQ1_S:   return 256;
        case GgufWireType::IQ1_M:   return 256;
        case GgufWireType::IQ4_XS:  return 256;
        case GgufWireType::MXFP4:   return 32;
        default: return 0;
    }
}

size_t gguf_type_size(GgufWireType type) {
    switch (type) {
        case GgufWireType::F32:     return 4;
        case GgufWireType::F16:     return 2;
        case GgufWireType::BF16:    return 2;
        case GgufWireType::F64:     return 8;
        case GgufWireType::I8:      return 1;
        case GgufWireType::I16:     return 2;
        case GgufWireType::I32:     return 4;
        case GgufWireType::I64:     return 8;
        case GgufWireType::Q4_0:    return 18;   // 32*4/8 + 2 (fp16 scale)
        case GgufWireType::Q4_1:    return 20;   // 32*4/8 + 2 + 2 (scale + min)
        case GgufWireType::Q5_0:    return 22;   // 32*5/8 + 4 (high bits) + 2
        case GgufWireType::Q5_1:    return 24;   // 32*5/8 + 4 + 2 + 2
        case GgufWireType::Q8_0:    return 34;   // 32*1 + 2
        case GgufWireType::Q8_1:    return 36;   // 32*1 + 2 + 2
        case GgufWireType::Q2_K:    return 84;
        case GgufWireType::Q3_K:    return 110;
        case GgufWireType::Q4_K:    return 144;
        case GgufWireType::Q5_K:    return 176;
        case GgufWireType::Q6_K:    return 210;
        case GgufWireType::Q8_K:    return 292;
        case GgufWireType::IQ2_XXS: return 66;
        case GgufWireType::IQ2_XS:  return 74;
        case GgufWireType::IQ2_S:   return 82;
        case GgufWireType::IQ3_XXS: return 98;
        case GgufWireType::IQ3_S:   return 110;
        case GgufWireType::IQ1_S:   return 50;
        case GgufWireType::IQ1_M:   return 56;
        case GgufWireType::IQ4_NL:  return 18;
        case GgufWireType::IQ4_XS:  return 136;
        case GgufWireType::MXFP4:   return 17;   // 32*4/8 + 1 (UE8M0 scale)
        default: return 0;
    }
}

size_t gguf_row_size(GgufWireType type, int64_t n_elements) {
    int bs = gguf_blck_size(type);
    if (bs == 0) return 0;
    return static_cast<size_t>((n_elements + bs - 1) / bs) * gguf_type_size(type);
}

QType gguf_type_to_qtype(GgufWireType type) {
    // Wire-stable values 0..31 in QType match the GGUF on-disk numbering,
    // so the cast is exact for every supported block-quant type. Anything
    // outside the 0..31 range falls through to NONE.
    switch (type) {
        case GgufWireType::F32:   return QType::F32;
        case GgufWireType::F16:   return QType::F16;
        case GgufWireType::BF16:  return QType::BF16;
        case GgufWireType::Q4_0:  return QType::Q4_0;
        case GgufWireType::Q4_1:  return QType::Q4_1;
        case GgufWireType::Q5_0:  return QType::Q5_0;
        case GgufWireType::Q5_1:  return QType::Q5_1;
        case GgufWireType::Q8_0:  return QType::Q8_0;
        case GgufWireType::Q8_1:  return QType::Q8_1;
        case GgufWireType::Q2_K:  return QType::Q2_K;
        case GgufWireType::Q3_K:  return QType::Q3_K;
        case GgufWireType::Q4_K:  return QType::Q4_K;
        case GgufWireType::Q5_K:  return QType::Q5_K;
        case GgufWireType::Q6_K:  return QType::Q6_K;
        case GgufWireType::Q8_K:  return QType::Q8_K;
        case GgufWireType::MXFP4: return QType::MXFP4;
        case GgufWireType::I8:    return QType::INT8;
        case GgufWireType::I32:   return QType::INT32;
        default:
            // IQ4_NL/IQ4_XS/etc. — no native QType yet; mark unsupported.
            return QType::NONE;
    }
}

const char* gguf_type_name(GgufWireType type) {
    switch (type) {
        case GgufWireType::F32:     return "F32";
        case GgufWireType::F16:     return "F16";
        case GgufWireType::BF16:    return "BF16";
        case GgufWireType::F64:     return "F64";
        case GgufWireType::I8:      return "I8";
        case GgufWireType::I16:     return "I16";
        case GgufWireType::I32:     return "I32";
        case GgufWireType::I64:     return "I64";
        case GgufWireType::Q4_0:    return "Q4_0";
        case GgufWireType::Q4_1:    return "Q4_1";
        case GgufWireType::Q5_0:    return "Q5_0";
        case GgufWireType::Q5_1:    return "Q5_1";
        case GgufWireType::Q8_0:    return "Q8_0";
        case GgufWireType::Q8_1:    return "Q8_1";
        case GgufWireType::Q2_K:    return "Q2_K";
        case GgufWireType::Q3_K:    return "Q3_K";
        case GgufWireType::Q4_K:    return "Q4_K";
        case GgufWireType::Q5_K:    return "Q5_K";
        case GgufWireType::Q6_K:    return "Q6_K";
        case GgufWireType::Q8_K:    return "Q8_K";
        case GgufWireType::IQ2_XXS: return "IQ2_XXS";
        case GgufWireType::IQ2_XS:  return "IQ2_XS";
        case GgufWireType::IQ2_S:   return "IQ2_S";
        case GgufWireType::IQ3_XXS: return "IQ3_XXS";
        case GgufWireType::IQ3_S:   return "IQ3_S";
        case GgufWireType::IQ1_S:   return "IQ1_S";
        case GgufWireType::IQ1_M:   return "IQ1_M";
        case GgufWireType::IQ4_NL:  return "IQ4_NL";
        case GgufWireType::IQ4_XS:  return "IQ4_XS";
        case GgufWireType::MXFP4:   return "MXFP4";
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
        if (!check(sizeof(T))) {
            failed_ = true;
            return T{};
        }
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
    bool failed_ = false;
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

// ---- Read array elements by type into a GGUFValue ----

template<typename T, typename ReadFn>
static void read_array_elements(BinaryReader& r, uint64_t count,
                                 std::vector<T>& out, ReadFn read_fn,
                                 size_t element_size) {
    size_t safe = std::min(static_cast<size_t>(count), r.remaining() / element_size);
    out.reserve(safe);
    for (uint64_t i = 0; i < count && !r.failed(); i++) {
        out.push_back(read_fn(r));
    }
}

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
                // Each string is at least 8 bytes (u64 length prefix)
                read_array_elements(r, count, v.str_array,
                    [](BinaryReader& br) { return br.read_string(); }, 8);
            } else if (arr_type == GGUFValueType::FLOAT32) {
                read_array_elements(r, count, v.float_array,
                    [](BinaryReader& br) { return br.read_f32(); }, 4);
            } else if (arr_type == GGUFValueType::INT32) {
                read_array_elements(r, count, v.int_array,
                    [](BinaryReader& br) { return br.read_i32(); }, 4);
            } else if (arr_type == GGUFValueType::UINT32) {
                read_array_elements(r, count, v.int_array,
                    [](BinaryReader& br) { return static_cast<int32_t>(br.read_u32()); }, 4);
            } else if (arr_type == GGUFValueType::BOOL ||
                       arr_type == GGUFValueType::UINT8 ||
                       arr_type == GGUFValueType::INT8) {
                read_array_elements(r, count, v.int_array,
                    [](BinaryReader& br) { return static_cast<int32_t>(br.read_u8()); }, 1);
            } else {
                // Skip unknown array element types
                for (uint64_t i = 0; i < count && !r.failed(); i++)
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
                           const Tensor& tensor, GgufWireType gtype) {
    auto qtype = static_cast<QType>(static_cast<uint32_t>(gtype));
    if (name == "token_embd.weight") {
        assign_quant(model.tok_emb_, model.tok_emb_qtype_, tensor);
        return true;
    }
    if (name == "output_norm.weight") {
        assign_quant(model.out_norm_, model.out_norm_qtype_, tensor);
        return true;
    }
    if (name == "rope_freqs.weight") {
        model.layers_[0].rope_freqs = tensor;
        return true;
    }
    if (name == "output.weight") {
        assign_quant(model.out_proj_, model.out_proj_qtype_, tensor);
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
        const auto& suffix = parts[3];  // "weight" or "bias"

        // Attention projections: distinguish weight vs bias
        if (field == "attn_q") {
            if (suffix == "bias")       layer.q_bias = tensor;
            else                        assign_quant(layer.wq, layer.wq_qtype, tensor);
        }
        else if (field == "attn_k") {
            if (suffix == "bias")       layer.k_bias = tensor;
            else                        assign_quant(layer.wk, layer.wk_qtype, tensor);
        }
        else if (field == "attn_v") {
            if (suffix == "bias")       layer.v_bias = tensor;
            else                        assign_quant(layer.wv, layer.wv_qtype, tensor);
        }
        else if (field == "attn_output") assign_quant(layer.wo, layer.wo_qtype, tensor);
        else if (field == "attn_norm")   layer.attn_norm = tensor;
        else if (field == "attn_q_norm") layer.attn_q_norm = tensor;
        else if (field == "attn_k_norm") layer.attn_k_norm = tensor;
        // Fused QKV: either standard attention (Phi-4) or GDN (Qwen3.5)
        else if (field == "attn_qkv") {
            const auto& cfg = model.config();
            int64_t total_rows = tensor.shape[0];  // outermost dim after reversal
            int64_t d_model = tensor.shape[1];     // inner dim

            // Check if this is a GDN layer (total rows match SSM conv_channels)
            int ssm_conv_channels = cfg.ssm_inner_size +
                                    2 * cfg.ssm_group_count * cfg.ssm_state_size;
            if (cfg.ssm_inner_size > 0 && total_rows == ssm_conv_channels) {
                // GDN layer: treat attn_qkv as ssm_in (fused projection → conv1d input)
                assign_quant(layer.ssm_in, layer.ssm_in_qtype, tensor);
            } else {
                // Standard fused QKV: split into separate Q, K, V
                // For Qwen3.5 attention: Q has 2× output (Q + gate interleaved),
                // so q_rows = total_rows - k_rows - v_rows (not just n_heads * head_dim).
                int q_rows = cfg.n_heads * cfg.head_dim;
                int k_rows = cfg.n_kv_heads * cfg.head_dim;
                size_t row_bytes = qtype_row_bytes(qtype, d_model);

                uint8_t* base = static_cast<uint8_t*>(tensor.data);
                int64_t q_shape[4] = {q_rows, d_model, 1, 1};
                int64_t kv_shape[4] = {k_rows, d_model, 1, 1};

                Tensor q_t(base, tensor.qtype, 2, q_shape, tensor.on_device);
                Tensor k_t(base + static_cast<size_t>(q_rows) * row_bytes,
                           tensor.qtype, 2, kv_shape, tensor.on_device);
                Tensor v_t(base + static_cast<size_t>(q_rows + k_rows) * row_bytes,
                           tensor.qtype, 2, kv_shape, tensor.on_device);
                assign_quant(layer.wq, layer.wq_qtype, q_t);
                assign_quant(layer.wk, layer.wk_qtype, k_t);
                assign_quant(layer.wv, layer.wv_qtype, v_t);
            }
        }
        // Post-layer norms (Gemma-3)
        else if (field == "post_attention_norm") layer.post_attn_norm = tensor;
        else if (field == "post_ffw_norm")       layer.post_ffn_norm = tensor;
        // Gemma 4: parallel shared MLP + MoE expert branch norms
        else if (field == "pre_ffw_norm_2")      layer.ffn_pre_norm_2 = tensor;
        else if (field == "post_ffw_norm_1")     layer.ffn_post_norm_1 = tensor;
        else if (field == "post_ffw_norm_2")     layer.ffn_post_norm_2 = tensor;
        else if (field == "layer_output_scale")  layer.layer_out_scale = tensor;
        else if (field == "rope_freqs")          layer.rope_freqs = tensor;
        // Gemma 4: fused gate+up experts: [n_experts, n_ff_exp*2, d_model]
        // We keep it packed; the MoE executor handles de-interleaving at dispatch.
        else if (field == "ffn_gate_up_exps") {
            // Reuses gate packed slot with full fused tensor.
            // Mark fused by leaving expert_up_packed null — executor detects this.
            assign_quant(layer.expert_gate_packed, layer.expert_gate_qtype, tensor);
        }
        // FFN
        else if (field == "ffn_gate")    assign_quant(layer.w_gate, layer.w_gate_qtype, tensor);
        else if (field == "ffn_up")      assign_quant(layer.w_up,   layer.w_up_qtype,   tensor);
        else if (field == "ffn_down")    assign_quant(layer.w_down, layer.w_down_qtype, tensor);
        else if (field == "ffn_norm")    layer.ffn_norm = tensor;
        else if (field == "ffn_gate_inp") {
            // Distinguish .weight (the gate matrix) from .scale (per-channel multiplier).
            // Gemma 4 stores `blk.X.ffn_gate_inp.scale` as a 4-part tensor name; without
            // this branch the scale would be silently misassigned to layer.moe_gate.
            if (suffix == "scale") layer.ffn_gate_inp_scale = tensor;
            else                   layer.moe_gate = tensor;
        }
        // Packed expert tensors: 3D [n_experts, rows, cols]
        else if (field == "ffn_gate_exps") assign_quant(layer.expert_gate_packed, layer.expert_gate_qtype, tensor);
        else if (field == "ffn_up_exps")   assign_quant(layer.expert_up_packed,   layer.expert_up_qtype,   tensor);
        else if (field == "ffn_down_exps") {
            // Distinguish .weight (the per-expert FFN down weights) from .scale
            // (per-expert output multiplier, shape [n_expert]). Same 4-part-name
            // bug as ffn_gate_inp.scale: the scale tensor would otherwise overwrite
            // expert_down_packed.
            if (suffix == "scale") layer.expert_down_scale = tensor;
            else                   assign_quant(layer.expert_down_packed, layer.expert_down_qtype, tensor);
        }
        // Shared expert (always-active, e.g. Nemotron/DeepSeek)
        else if (field == "ffn_gate_shexp") assign_quant(layer.w_gate_shared, layer.w_gate_shared_qtype, tensor);
        else if (field == "ffn_up_shexp")   assign_quant(layer.w_up_shared,   layer.w_up_shared_qtype,   tensor);
        else if (field == "ffn_down_shexp") assign_quant(layer.w_down_shared, layer.w_down_shared_qtype, tensor);
        // Qwen3-Next / Qwen3.6 per-token sigmoid gate on the shared expert
        // output. 1D [d_model] FP32 projection; sigmoid(cur @ W) yields [M, 1].
        else if (field == "ffn_gate_inp_shexp") layer.shared_expert_gate_inp = tensor;
        // SSM weights (Mamba2)
        else if (field == "ssm_in")   assign_quant(layer.ssm_in,  layer.ssm_in_qtype,  tensor);
        else if (field == "ssm_out")  assign_quant(layer.ssm_out, layer.ssm_out_qtype, tensor);
        else if (field == "ssm_dt") {
            // Some converters (Qwen3.5-27B-mxfp4) emit A_log under the name
            // "ssm_dt.weight" — a 1D vector of shape [n_heads]. Differentiate
            // bias vs weight: bias → ssm_dt_b (per-head dt bias),
            // weight → ssm_a (per-head A_log). Without this branch the weight
            // silently overwrites the bias and ssm_a stays null, causing the
            // GDN scan kernel to NULL-deref A_log[h] on first launch.
            if (suffix == "bias")        layer.ssm_dt_b = tensor;
            else if (suffix == "weight") layer.ssm_a = tensor;
            else return false;
        }
        else if (field == "ssm_norm")   layer.ssm_norm_w = tensor;
        // SSM conv1d: "blk.{i}.ssm_conv1d.weight" / "blk.{i}.ssm_conv1d.bias"
        else if (field == "ssm_conv1d") {
            if (suffix == "weight")     layer.ssm_conv1d_w = tensor;
            else if (suffix == "bias")  layer.ssm_conv1d_b = tensor;
            else return false;
        }
        // Gated DeltaNet (GDN) weights (Qwen3.5)
        else if (field == "attn_gate")  assign_quant(layer.gdn_gate,  layer.gdn_gate_qtype,  tensor);
        else if (field == "ssm_alpha")  assign_quant(layer.gdn_alpha, layer.gdn_alpha_qtype, tensor);
        else if (field == "ssm_beta")   assign_quant(layer.gdn_beta,  layer.gdn_beta_qtype,  tensor);
        // Router bias (Nemotron MoE)
        else if (field == "exp_probs_b") layer.moe_router_bias = tensor;
        else return false;
        return true;
    }

    // 5-part: "blk.{i}.ffn_*.{expert_idx}.weight" — MoE per-expert weights
    //    or: "blk.{i}.ffn_gate_inp.scale.weight" / "blk.{i}.ffn_down_exps.scale.weight" (Gemma 4)
    if (parts.size() == 5) {
        const auto& field = parts[2];
        const auto& subfield = parts[3];

        // Gemma 4 scale tensors
        if (subfield == "scale") {
            if (field == "ffn_gate_inp") {
                layer.ffn_gate_inp_scale = tensor;
                return true;
            }
            if (field == "ffn_down_exps") {
                // Per-expert output scale. Not yet consumed by the executor — store as
                // router bias slot so weight_upload at least preserves it.
                layer.moe_router_bias = tensor;
                return true;
            }
            return false;
        }

        // MoE expert weights: "blk.{i}.ffn_*.{expert_idx}.weight"
        int expert_idx = 0;
        try { expert_idx = std::stoi(parts[3]); }
        catch (...) { return false; }

        int n_experts = model.config().n_experts;
        if (expert_idx < 0 || expert_idx >= n_experts) return false;

        // Per-expert vectors: assign to slot N. The layer-wide qtype mirror is
        // populated only on the first expert (all experts share the same qtype).
        if (field == "ffn_gate") {
            layer.expert_w_gate[expert_idx] = tensor;
            if (expert_idx == 0) layer.expert_gate_qtype = qtype;
        }
        else if (field == "ffn_up") {
            layer.expert_w_up[expert_idx] = tensor;
            if (expert_idx == 0) layer.expert_up_qtype = qtype;
        }
        else if (field == "ffn_down") {
            layer.expert_w_down[expert_idx] = tensor;
            if (expert_idx == 0) layer.expert_down_qtype = qtype;
        }
        else return false;
        return true;
    }

    return false;
}

// ---- Parse tensor info entries from a BinaryReader ----

static void parse_tensor_infos(BinaryReader& reader, uint64_t tensor_count,
                                std::vector<GGUFTensorInfo>& out) {
    for (uint64_t i = 0; i < tensor_count && !reader.failed(); i++) {
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
        info.type = static_cast<GgufWireType>(reader.read_u32());
        info.offset = reader.read_u64();
        out.push_back(std::move(info));
    }
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

    if (reader.failed()) {
        IMP_LOG_ERROR("GGUF header truncated: %s", path.c_str());
        munmap(mmap_base, file_size);
        return nullptr;
    }

    IMP_LOG_INFO("GGUF v%u: %lu tensors, %lu metadata KVs",
                 version, (unsigned long)tensor_count, (unsigned long)kv_count);

    // 3. Parse metadata key-value pairs
    std::unordered_map<std::string, GGUFValue> metadata;
    metadata.reserve(static_cast<size_t>(kv_count));

    for (uint64_t i = 0; i < kv_count && !reader.failed(); i++) {
        std::string key = reader.read_string();
        auto vtype = static_cast<GGUFValueType>(reader.read_u32());
        GGUFValue value = read_gguf_value(reader, vtype);
        metadata.emplace(std::move(key), std::move(value));
    }

    if (reader.failed()) {
        IMP_LOG_ERROR("GGUF metadata truncated: %s", path.c_str());
        munmap(mmap_base, file_size);
        return nullptr;
    }

    // 4. Parse tensor info entries
    std::vector<GGUFTensorInfo> tensor_infos;
    tensor_infos.reserve(static_cast<size_t>(tensor_count));
    parse_tensor_infos(reader, tensor_count, tensor_infos);

    if (reader.failed()) {
        IMP_LOG_ERROR("GGUF tensor info truncated: %s", path.c_str());
        munmap(mmap_base, file_size);
        return nullptr;
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

    // Set data_base for primary shard tensors
    for (auto& info : tensor_infos) {
        info.data_base = data + tensor_data_start;
    }

    // 5b. Handle split GGUF files (multiple shards)
    auto it_split = metadata.find("split.count");
    int split_count = (it_split != metadata.end()) ? static_cast<int>(val_uint(it_split->second)) : 1;

    // Store extra shard mmaps for cleanup (primary shard stored separately in model)
    std::vector<std::pair<void*, size_t>> extra_mmaps;

    if (split_count > 1) {
        IMP_LOG_INFO("Split GGUF: %d shards", split_count);

        // Derive shard filenames: path ends with -00001-of-NNNNN.gguf
        // Replace the shard number for each additional shard
        std::string base_path = path;
        auto dash_pos = base_path.rfind("-00001-of-");
        if (dash_pos == std::string::npos) {
            // Try without the -00001 suffix (user passed the base name)
            IMP_LOG_WARN("Split GGUF: cannot derive shard paths from '%s'", path.c_str());
        } else {
            for (int shard = 2; shard <= split_count; shard++) {
                char shard_path[4096];
                snprintf(shard_path, sizeof(shard_path), "%.*s-%05d-of-%05d.gguf",
                         static_cast<int>(dash_pos), base_path.c_str(), shard, split_count);

                int sfd = open(shard_path, O_RDONLY);
                if (sfd < 0) {
                    IMP_LOG_ERROR("Failed to open shard %d: %s", shard, shard_path);
                    for (auto& [p, s] : extra_mmaps) munmap(p, s);
                    munmap(mmap_base, file_size);
                    return nullptr;
                }

                struct stat sst;
                fstat(sfd, &sst);
                size_t shard_size = static_cast<size_t>(sst.st_size);
                void* shard_mmap = mmap(nullptr, shard_size, PROT_READ, MAP_PRIVATE, sfd, 0);
                close(sfd);

                if (shard_mmap == MAP_FAILED) {
                    IMP_LOG_ERROR("Failed to mmap shard %d: %s", shard, shard_path);
                    for (auto& [p, s] : extra_mmaps) munmap(p, s);
                    munmap(mmap_base, file_size);
                    return nullptr;
                }
                madvise(shard_mmap, shard_size, MADV_SEQUENTIAL);
                extra_mmaps.emplace_back(shard_mmap, shard_size);

                // Parse shard header to get tensor infos
                auto* sdata = reinterpret_cast<const uint8_t*>(shard_mmap);
                BinaryReader sreader(sdata, shard_size);
                uint32_t smagic = sreader.read_u32();
                sreader.read_u32(); // sversion (unused)
                uint64_t stensor_count = sreader.read_u64();
                uint64_t skv_count = sreader.read_u64();

                if (smagic != GGUF_MAGIC || sreader.failed()) {
                    IMP_LOG_ERROR("Invalid shard %d header", shard);
                    for (auto& [p, s] : extra_mmaps) munmap(p, s);
                    munmap(mmap_base, file_size);
                    return nullptr;
                }

                // Skip shard metadata (we already have it from shard 0)
                for (uint64_t i = 0; i < skv_count && !sreader.failed(); i++) {
                    sreader.read_string();
                    auto vtype = static_cast<GGUFValueType>(sreader.read_u32());
                    read_gguf_value(sreader, vtype);
                }

                // Parse shard tensor infos
                parse_tensor_infos(sreader, stensor_count, tensor_infos);

                // Compute shard tensor data start
                size_t salign = alignment;  // use same alignment as primary
                sreader.align(salign);
                size_t shard_data_start = sreader.pos();

                // Set data_base for this shard's tensors
                size_t shard_tensor_start = tensor_infos.size() - static_cast<size_t>(stensor_count);
                for (size_t ti = shard_tensor_start; ti < tensor_infos.size(); ti++) {
                    tensor_infos[ti].data_base = sdata + shard_data_start;
                }

                IMP_LOG_INFO("  Shard %d: %lu tensors, %.1f MiB",
                             shard, (unsigned long)stensor_count,
                             shard_size / (1024.0 * 1024.0));
            }
        }

        tensor_count = tensor_infos.size();
    }

    // 6. Extract model config from metadata
    auto model = std::make_unique<Model>();
    model->mmap_base_ = mmap_base;
    model->mmap_size_ = file_size;
    model->split_mmaps_ = std::move(extra_mmaps);

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

    // RoPE frequency scaling (linear: divide frequencies by factor)
    cfg.rope_freq_scale = static_cast<float>(get_float("rope.scaling.factor", 1.0));
    // Fallback: try legacy key
    if (cfg.rope_freq_scale == 1.0f) {
        float legacy_scale = static_cast<float>(get_float("rope.scale_linear", 0.0));
        if (legacy_scale > 0.0f) cfg.rope_freq_scale = legacy_scale;
    }

    // YaRN / Dynamic NTK RoPE scaling
    {
        std::string rope_type_str;
        auto it = metadata.find(arch_str + ".rope.scaling.type");
        if (it == metadata.end()) it = metadata.find("rope.scaling.type");
        if (it != metadata.end() && it->second.type == GGUFValueType::STRING)
            rope_type_str = it->second.str_val;

        cfg.rope_n_ctx_orig = static_cast<int>(get_uint("rope.scaling.original_context_length", 0));
        cfg.yarn_beta_fast  = static_cast<float>(get_float("rope.scaling.yarn_beta_fast", 32.0));
        cfg.yarn_beta_slow  = static_cast<float>(get_float("rope.scaling.yarn_beta_slow", 1.0));
        cfg.yarn_attn_factor = static_cast<float>(get_float("rope.scaling.yarn_attn_factor", 1.0));
        // Also try the generic attn_factor key
        if (cfg.yarn_attn_factor == 1.0f)
            cfg.yarn_attn_factor = static_cast<float>(get_float("rope.scaling.attn_factor", 1.0));

        float yarn_ext = static_cast<float>(get_float("rope.scaling.yarn_ext_factor", -1.0));
        if (rope_type_str == "yarn") {
            cfg.yarn_ext_factor = (yarn_ext < 0.0f) ? 1.0f : yarn_ext;
        } else {
            cfg.yarn_ext_factor = (yarn_ext < 0.0f) ? 0.0f : yarn_ext;
        }

        // LongRoPE per-dimension frequency scaling (Phi-4)
        if (rope_type_str == "longrope") {
            auto sf = metadata.find(arch_str + ".rope.scaling.short_factor");
            if (sf == metadata.end()) sf = metadata.find("rope.scaling.short_factor");
            if (sf != metadata.end()) cfg.rope_short_factor = sf->second.float_array;

            auto lf = metadata.find(arch_str + ".rope.scaling.long_factor");
            if (lf == metadata.end()) lf = metadata.find("rope.scaling.long_factor");
            if (lf != metadata.end()) cfg.rope_long_factor = lf->second.float_array;

            cfg.rope_scaling_orig_max_pos = static_cast<int>(
                get_uint("rope.scaling.original_max_position_embeddings", 0));

            IMP_LOG_INFO("LongRoPE: short_factor[%zu], long_factor[%zu], orig_max_pos=%d",
                         cfg.rope_short_factor.size(), cfg.rope_long_factor.size(),
                         cfg.rope_scaling_orig_max_pos);
        }

        // Compute mscale compensation (same as llama.cpp)
        if (cfg.yarn_ext_factor != 0.0f && cfg.rope_freq_scale > 1.0f) {
            float factor = cfg.rope_freq_scale;  // scaling factor
            float mscale = 1.0f + 0.1f * logf(factor);
            // Pre-compensate for the internal mscale that rope_yarn() also applies
            cfg.yarn_attn_factor *= mscale / (1.0f + 0.1f * logf(factor));
        }
    }

    // Gemma-specific: per-layer sliding window and local RoPE (metadata-dependent)
    // Note: embed_scale, ffn_activation, norm_placement are set by apply_arch_defaults().
    if (arch_str == "gemma" || arch_str == "gemma2" || arch_str == "gemma3") {
        // Per-layer sliding window pattern: every Nth layer is global (no window)
        // Gemma-3 uses pattern=6 (5 local + 1 global)
        cfg.sliding_window_pattern = static_cast<int>(get_uint("attention.sliding_window_pattern", 0));
        if (cfg.sliding_window_pattern == 0 && arch_str == "gemma3") {
            cfg.sliding_window_pattern = 6;  // Gemma-3 default: 5 local + 1 global
        }

        // Local RoPE theta (used for sliding window layers; global layers use rope_theta)
        cfg.rope_local_theta = static_cast<float>(get_float("rope.local.freq_base", 0.0));
        if (cfg.rope_local_theta == 0.0f && cfg.sliding_window_pattern > 0) {
            cfg.rope_local_theta = 10000.0f;  // Gemma-3 default local theta
        }
    }

    // Gemma 4: per-layer SWA pattern (array), SWA-specific head dims, RoPE base.
    if (arch_str == "gemma4") {
        // Per-layer SWA bool array: 1 = sliding-window attention, 0 = full/global attention.
        {
            auto it = metadata.find("gemma4.attention.sliding_window_pattern");
            if (it == metadata.end())
                it = metadata.find("attention.sliding_window_pattern");
            if (it != metadata.end() && !it->second.int_array.empty()) {
                cfg.swa_layers.reserve(it->second.int_array.size());
                for (auto v : it->second.int_array)
                    cfg.swa_layers.push_back(v ? 1 : 0);
            }
        }
        // Default: 5:1 SWA:full pattern (matches google/gemma-4-26B-A4B-it)
        if (cfg.swa_layers.empty()) {
            cfg.swa_layers.resize(cfg.n_layers, 0);
            for (int i = 0; i < cfg.n_layers; i++)
                cfg.swa_layers[i] = ((i % 6) == 5) ? 0 : 1;  // every 6th is full
        }

        // SWA-specific attention dims (full attention uses key_length/value_length)
        int key_len      = static_cast<int>(get_uint("attention.key_length", 0));
        int val_len      = static_cast<int>(get_uint("attention.value_length", 0));
        int key_len_swa  = static_cast<int>(get_uint("attention.key_length_swa", key_len));
        int val_len_swa  = static_cast<int>(get_uint("attention.value_length_swa", val_len));
        (void)val_len; (void)val_len_swa;  // V head_dim assumed == K head_dim

        cfg.sliding_window = static_cast<int>(get_uint("attention.sliding_window", 0));
        cfg.rope_local_theta = static_cast<float>(get_float("rope.freq_base_swa", 0.0));
        if (cfg.rope_local_theta == 0.0f) cfg.rope_local_theta = 10000.0f;
        cfg.rope_theta_swa = cfg.rope_local_theta;

        // Build per-layer head_dim and n_kv_heads from swa_layers.
        // The GGUF may already supply per-layer arrays for head_count_kv; if not,
        // we derive from swa_layers using key_length / key_length_swa.
        if (cfg.head_dim_per_layer.empty() && key_len > 0 && key_len_swa > 0) {
            cfg.head_dim_per_layer.resize(cfg.n_layers);
            for (int i = 0; i < cfg.n_layers; i++)
                cfg.head_dim_per_layer[i] = cfg.swa_layers[i] ? key_len_swa : key_len;
        }
        // scalar head_dim = max for buffer sizing
        if (!cfg.head_dim_per_layer.empty()) {
            int max_hd = 0;
            for (int v : cfg.head_dim_per_layer) max_hd = std::max(max_hd, v);
            cfg.head_dim = max_hd;
            IMP_LOG_INFO("Gemma 4 per-layer head_dim: max=%d", max_hd);
        }

        IMP_LOG_INFO("Gemma 4: SWA layers=%zu (of %d), rope_theta_swa=%.0f, key_len=%d, key_len_swa=%d",
                     std::count(cfg.swa_layers.begin(), cfg.swa_layers.end(), uint8_t(1)),
                     cfg.n_layers, cfg.rope_theta_swa, key_len, key_len_swa);
        // Per-layer head_dim/n_kv_heads detection happens at runtime in run_attention
        // by reading wq.shape[0] / hd and wk.shape[0] / hd. Authoritative source =
        // the loaded tensor shapes, not GGUF metadata.
    }

    // Attention logit softcapping (Gemma-2/3: tanh(score/cap)*cap)
    cfg.attn_logit_softcap  = static_cast<float>(get_float("attn_logit_softcapping", 0.0));
    if (cfg.attn_logit_softcap == 0.0f)
        cfg.attn_logit_softcap = static_cast<float>(get_float("attention.logit_softcapping", 0.0));
    cfg.final_logit_softcap = static_cast<float>(get_float("final_logit_softcapping", 0.0));

    // MXFP4 Hadamard rotation metadata
    cfg.mxfp4_hadamard_attn = static_cast<int>(get_uint("mxfp4.hadamard_block_size_attn", 0));
    cfg.mxfp4_hadamard_ffn  = static_cast<int>(get_uint("mxfp4.hadamard_block_size_ffn", 0));
    if (cfg.mxfp4_hadamard_attn > 0 || cfg.mxfp4_hadamard_ffn > 0)
        IMP_LOG_INFO("MXFP4 Hadamard: attn_bs=%d ffn_bs=%d", cfg.mxfp4_hadamard_attn, cfg.mxfp4_hadamard_ffn);

    cfg.sliding_window   = static_cast<int>(get_uint("attention.sliding_window", 0));

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
    IMP_LOG_INFO("RoPE: theta=%.1f, rope_dim=%d, neox=%d, freq_scale=%.1f, eps=%.2e",
                 cfg.rope_theta, cfg.rope_dim, cfg.rope_neox ? 1 : 0,
                 cfg.rope_freq_scale, cfg.rms_norm_eps);
    if (cfg.yarn_ext_factor > 0.0f)
        IMP_LOG_INFO("YaRN: ext_factor=%.1f, attn_factor=%.3f, beta_fast=%.1f, beta_slow=%.1f, n_ctx_orig=%d",
                     cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                     cfg.yarn_beta_fast, cfg.yarn_beta_slow, cfg.rope_n_ctx_orig);
    if (cfg.embed_scale > 0.0f)
        IMP_LOG_INFO("Embedding scale: %.2f (sqrt(d_model))", cfg.embed_scale);

    if (cfg.sliding_window > 0) {
        IMP_LOG_INFO("Sliding window attention: %d tokens", cfg.sliding_window);
        if (cfg.sliding_window_pattern > 0) {
            IMP_LOG_INFO("Sliding window pattern: every %dth layer is global, local_theta=%.1f",
                         cfg.sliding_window_pattern, cfg.rope_local_theta);
        }
    }
    if (cfg.ffn_activation != FFNActivation::SWIGLU) {
        const char* act_name = (cfg.ffn_activation == FFNActivation::GEGLU) ? "GeGLU" : "ReLU²";
        IMP_LOG_INFO("FFN activation: %s", act_name);
    }
    if (cfg.norm_placement == NormPlacement::POST_NORM)
        IMP_LOG_INFO("Norm placement: post-norm (residual after norm)");
    if (cfg.attn_logit_softcap > 0.0f)
        IMP_LOG_INFO("Attention logit softcap: %.1f", cfg.attn_logit_softcap);
    if (cfg.final_logit_softcap > 0.0f)
        IMP_LOG_INFO("Final logit softcap: %.1f", cfg.final_logit_softcap);

    if (cfg.n_experts > 0) {
        IMP_LOG_INFO("MoE: %d experts, %d active, expert_d_ff=%d, shared=%d (shared_d_ff=%d), "
                     "norm_weights=%d",
                     cfg.n_experts, cfg.n_experts_active, cfg.expert_d_ff,
                     cfg.n_experts_shared, cfg.expert_shared_d_ff,
                     cfg.expert_weights_norm ? 1 : 0);
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
        // Compute pointer into mmap'd data (supports split GGUF via per-tensor data_base)
        auto* tensor_data = const_cast<void*>(
            static_cast<const void*>(info.data_base + info.offset));

        // Build tensor descriptor
        // GGUF stores dims as ne[0]=innermost. We reverse for shape[0]=outermost.
        int ndim = static_cast<int>(info.n_dims);
        int64_t shape[4] = {1, 1, 1, 1};
        for (int d = 0; d < ndim; d++) {
            shape[d] = info.dims[ndim - 1 - d];
        }

        Tensor t(tensor_data, gguf_type_to_qtype(info.type), ndim, shape, /*on_device=*/false);
        t.kind = match_tensor_kind(info.name);

        if (assign_tensor(*model, info.name, t, info.type)) {
            assigned++;
        } else {
            IMP_LOG_DEBUG("Unassigned tensor: %s [%s] shape=[%ld,%ld,%ld,%ld]",
                          info.name.c_str(), gguf_type_name(info.type),
                          (long)info.dims[0], (long)info.dims[1],
                          (long)info.dims[2], (long)info.dims[3]);
            skipped++;
        }
    }

    // Infer vocab_size from token_embd if not in metadata
    if (cfg.vocab_size == 0 && model->tok_emb_.data != nullptr) {
        cfg.vocab_size = static_cast<int>(model->tok_emb_.shape[0]);
        IMP_LOG_INFO("Inferred vocab_size=%d from token_embd.weight", cfg.vocab_size);
    }

    // Weight tying: if no output.weight, share token_embd
    if (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr) {
        model->out_proj_ = model->tok_emb_;
        model->out_proj_qtype_ = model->tok_emb_qtype_;
        IMP_LOG_INFO("Weight tying: output projection shares token embedding");
    }

    // Split fused gate+up FFN (Phi-4/phi3): ffn_up contains gate||up concatenated
    // Detected when: w_gate is null, w_up exists, and w_up.shape[0] == 2 * d_ff
    if (cfg.d_ff > 0) {
        int fused_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            auto& ly = model->layers_[i];
            if (ly.w_gate.data == nullptr && ly.w_up.data != nullptr &&
                ly.w_up.shape[0] == 2 * cfg.d_ff) {
                int64_t d_model = ly.w_up.shape[1];
                int64_t d_ff = cfg.d_ff;
                size_t row_bytes = qtype_row_bytes(ly.w_up_qtype, d_model);

                uint8_t* base = static_cast<uint8_t*>(ly.w_up.data);
                int64_t half_shape[4] = {d_ff, d_model, 1, 1};

                ly.w_gate = Tensor(base, ly.w_up.qtype, 2, half_shape, ly.w_up.on_device);
                ly.w_gate_qtype = ly.w_up_qtype;
                ly.w_up = Tensor(base + static_cast<size_t>(d_ff) * row_bytes,
                                  ly.w_up.qtype, 2, half_shape, ly.w_up.on_device);
                // w_up_qtype unchanged
                fused_count++;
            }
        }
        if (fused_count > 0) {
            IMP_LOG_INFO("Split fused gate+up FFN in %d layers (d_ff=%d)", fused_count, cfg.d_ff);
        }
    }

    IMP_LOG_INFO("Weights: %d assigned, %d skipped", assigned, skipped);

    // 7b. Tensor validation and shared expert detection
    //     Inspect actual loaded tensors to detect capabilities, remap shared
    //     experts stored as regular FFN tensors, and warn about mismatches.
    {
        int n_attn = 0, n_moe = 0, n_dense_ffn = 0, n_shared_exp = 0;
        int n_qk_norm = 0, n_ssm = 0, n_gdn = 0, n_remapped = 0;

        for (int i = 0; i < cfg.n_layers; i++) {
            auto& ly = model->layers_[i];
            bool has_moe   = (ly.moe_gate.data != nullptr);
            bool has_dense = (ly.w_up.data != nullptr);
            bool has_shared = (ly.w_up_shared.data != nullptr);

            if (ly.wq.data != nullptr) n_attn++;
            if (has_moe) n_moe++;
            if (ly.attn_q_norm.data != nullptr) n_qk_norm++;
            if (ly.ssm_in.data != nullptr) n_ssm++;
            if (ly.gdn_gate.data != nullptr) n_gdn++;

            // Detect shared expert: MoE layer with dense FFN tensors loaded
            // alongside expert tensors → remap dense FFN to shared expert.
            // Some GGUF converters output shared experts as ffn_gate/ffn_up/ffn_down.
            if (has_moe && has_dense && !has_shared) {
                ly.w_gate_shared = ly.w_gate;   ly.w_gate_shared_qtype = ly.w_gate_qtype;
                ly.w_up_shared   = ly.w_up;     ly.w_up_shared_qtype   = ly.w_up_qtype;
                ly.w_down_shared = ly.w_down;   ly.w_down_shared_qtype = ly.w_down_qtype;
                ly.w_gate = Tensor();  ly.w_gate_qtype = QType::NONE;
                ly.w_up   = Tensor();  ly.w_up_qtype   = QType::NONE;
                ly.w_down = Tensor();  ly.w_down_qtype = QType::NONE;
                n_remapped++;
                has_shared = true;
            }

            if (has_shared) n_shared_exp++;
            if (has_dense && !has_moe) n_dense_ffn++;
        }

        IMP_LOG_INFO("Layer census: %d attn, %d GDN, %d MoE, %d dense FFN, %d shared expert, "
                     "%d QK-norm, %d SSM  (of %d layers)",
                     n_attn, n_gdn, n_moe, n_dense_ffn, n_shared_exp, n_qk_norm, n_ssm,
                     cfg.n_layers);

        if (n_remapped > 0) {
            IMP_LOG_INFO("Remapped %d layers: dense FFN tensors -> shared expert", n_remapped);
        }
        // Gemma 4: verify MoE-specific norms and router scale loaded
        if (cfg.arch == ModelArch::GEMMA4) {
            int n_pre2 = 0, n_post1 = 0, n_post2 = 0, n_gscale = 0, n_dscale = 0;
            for (int i = 0; i < cfg.n_layers; ++i) {
                if (model->layers_[i].ffn_pre_norm_2.data) n_pre2++;
                if (model->layers_[i].ffn_post_norm_1.data) n_post1++;
                if (model->layers_[i].ffn_post_norm_2.data) n_post2++;
                if (model->layers_[i].ffn_gate_inp_scale.data) n_gscale++;
                if (model->layers_[i].expert_down_scale.data) n_dscale++;
            }
            IMP_LOG_INFO("Gemma 4 MoE norms: pre_ffw_norm_2=%d, post_ffw_norm_1=%d, "
                         "post_ffw_norm_2=%d, gate_inp_scale=%d, down_exps_scale=%d (of %d layers)",
                         n_pre2, n_post1, n_post2, n_gscale, n_dscale, cfg.n_layers);
        }

        // Qwen3.5: GGUF converter adds +1 to non-GDN norm weights.
        // imp's RMSNorm expects raw weights (w, not w+1). Subtract 1 back.
        // Qwen3.5: GGUF converter adds +1 to norm weights (same as Gemma).
        // imp's rmsnorm uses weight directly, which is correct since the +1 is
        // already baked into the stored weights. No adjustment needed.

        // Update config from actual tensor presence
        if (n_shared_exp > 0 && cfg.n_experts_shared == 0) {
            cfg.n_experts_shared = 1;
            for (int i = 0; i < cfg.n_layers; i++) {
                if (model->layers_[i].w_up_shared.data != nullptr) {
                    cfg.expert_shared_d_ff = static_cast<int>(model->layers_[i].w_up_shared.shape[0]);
                    break;
                }
            }
            IMP_LOG_INFO("Inferred shared expert config: n_shared=%d, shared_d_ff=%d",
                         cfg.n_experts_shared, cfg.expert_shared_d_ff);
        }

        // Gemma 4: convert top-level rope_freqs (a freq DIVISOR table for global
        // layers) into pre-computed effective per-pair frequencies, then fan out
        // to every global layer. The kernel's `longrope_inv_freqs` parameter
        // expects ready-to-use freq values, so do the math on the host.
        if (cfg.arch == ModelArch::GEMMA4 && !cfg.swa_layers.empty() &&
            model->layers_[0].rope_freqs.data != nullptr &&
            model->layers_[0].rope_freqs.qtype == QType::F32) {
            const Tensor& src = model->layers_[0].rope_freqs;
            int n_pairs = static_cast<int>(src.shape[0]);  // hd/2 for global layer
            int hd_global = n_pairs * 2;
            const float* divisors = static_cast<const float*>(src.data);
            float theta_global = cfg.rope_theta;  // 1e6 for Gemma 4

            // Pre-compute effective per-pair frequencies = theta^(-2*pair/hd)/divisor[pair]
            // and present them via the layer's rope_freqs slot. The kernel reads
            // these directly as the freq value (longrope_inv_freqs path), no further
            // theta math. Memory is leaked deliberately (4 KB total, model-lifetime).
            float* effective = new float[n_pairs];
            for (int p = 0; p < n_pairs; ++p) {
                float exp_p   = -2.0f * static_cast<float>(p) / static_cast<float>(hd_global);
                float base_freq = std::pow(theta_global, exp_p);
                effective[p] = base_freq / divisors[p];
            }
            int64_t shape[4] = {n_pairs, 0, 0, 0};
            Tensor eff_tensor(effective, QType::F32, 1, shape, /*on_device=*/false);
            int n_global = 0;
            for (int i = 0; i < cfg.n_layers; ++i) {
                bool is_swa = (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i]);
                if (!is_swa) {
                    model->layers_[i].rope_freqs = eff_tensor;
                    n_global++;
                }
            }
            if (cfg.swa_layers[0]) {
                model->layers_[0].rope_freqs = Tensor();
            }
            IMP_LOG_INFO("Gemma 4: rope_freqs → %d effective freqs, %d global layers",
                         n_pairs, n_global);
        }

        // Warn about config/tensor mismatches
        if (cfg.n_experts_shared > 0 && n_shared_exp == 0) {
            IMP_LOG_WARN("Config declares %d shared expert(s) but no shared expert "
                         "tensors found — GGUF may be incomplete", cfg.n_experts_shared);
        }

        if (cfg.n_experts > 0 && n_moe == 0) {
            IMP_LOG_WARN("Config declares %d experts but no MoE gate tensors found",
                         cfg.n_experts);
        }

        if (n_moe > 0 && n_moe < cfg.n_layers && n_dense_ffn == 0 && n_ssm == 0) {
            IMP_LOG_WARN("Only %d/%d layers have MoE, remaining layers have no FFN",
                         n_moe, cfg.n_layers);
        }
    }

    // 8. Extract tokenizer from GGUF metadata
    auto tokenizer = std::make_unique<Tokenizer>();

    // Detect tokenizer type (default: SentencePiece)
    auto it_tok_model = metadata.find("tokenizer.ggml.model");
    std::string tok_type = "spm";
    if (it_tok_model != metadata.end()) {
        const std::string& tm = it_tok_model->second.str_val;
        if (tm == "gpt2") tok_type = "gpt2";
        // Gemma-4 uses SPM-style BPE: ▁ for spaces + BPE merge ranks.
        else if (tm == "gemma4") tok_type = "gemma4";
    }
    tokenizer->set_type(tok_type);

    // Pre-tokenizer type (e.g. "default", "llama3", "deepseek-llm", "qwen2")
    auto it_pre = metadata.find("tokenizer.ggml.pre");
    if (it_pre != metadata.end() && !it_pre->second.str_val.empty()) {
        tokenizer->set_pre_tokenizer(it_pre->second.str_val);
        IMP_LOG_INFO("Tokenizer pre-tokenizer: %s", it_pre->second.str_val.c_str());
    }

    // add_bos_token flag (Qwen3: 0, LLaMA: 1)
    auto it_add_bos = metadata.find("tokenizer.ggml.add_bos_token");
    if (it_add_bos != metadata.end()) {
        tokenizer->set_add_bos(val_uint(it_add_bos->second) != 0);
    } else if (tok_type == "gpt2") {
        // GPT2/BPE tokenizers (Qwen, etc.) typically don't use BOS.
        // Default to false when metadata is absent.
        tokenizer->set_add_bos(false);
    }

    // Gemma-4: always add BOS regardless of GGUF metadata.
    // Some GGUF converters (ggml-org) set add_bos=false incorrectly.
    // llama.cpp forces add_bos=true for Gemma-4 (see llama-vocab.cpp "override").
    if (tok_type == "gemma4") {
        tokenizer->set_add_bos(true);
    }

    // add_space_prefix flag (Gemma: false, LLaMA: true/default)
    auto it_add_sp = metadata.find("tokenizer.ggml.add_space_prefix");
    if (it_add_sp != metadata.end()) {
        tokenizer->set_add_space_prefix(val_uint(it_add_sp->second) != 0);
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

        // Load BPE merge rules (for GPT2-style tokenizers and gemma4)
        if (tok_type == "gpt2" || tok_type == "gemma4") {
            auto it_merges = metadata.find("tokenizer.ggml.merges");
            if (it_merges != metadata.end() && !it_merges->second.str_array.empty()) {
                tokenizer->load_merges(it_merges->second.str_array);
                IMP_LOG_INFO("Tokenizer: loaded %zu BPE merge rules",
                             it_merges->second.str_array.size());
            }
        }

        // Load per-token type metadata (NORMAL=1, CONTROL=3, etc.)
        auto it_types = metadata.find("tokenizer.ggml.token_type");
        if (it_types != metadata.end() && !it_types->second.int_array.empty()) {
            tokenizer->load_token_types(it_types->second.int_array);
        }

        // Extract chat template string (Jinja2) for template family detection
        auto it_tpl = metadata.find("tokenizer.chat_template");
        if (it_tpl != metadata.end() && !it_tpl->second.str_val.empty()) {
            tokenizer->set_chat_template_str(it_tpl->second.str_val);
            IMP_LOG_INFO("Chat template: %zu chars", it_tpl->second.str_val.size());
        }

        // Load additional EOS-like token IDs (EOT, end-of-generation, etc.)
        // Some models define multiple stop tokens beyond the primary eos_token_id.
        for (const char* key : {"tokenizer.ggml.eot_token_id",
                                 "tokenizer.ggml.eog_token_id"}) {
            auto it_extra = metadata.find(key);
            if (it_extra != metadata.end()) {
                int32_t extra_id = static_cast<int32_t>(val_uint(it_extra->second));
                if (extra_id >= 0) {
                    tokenizer->add_eos_id(extra_id);
                    IMP_LOG_INFO("Tokenizer: additional EOS from %s: %d", key, extra_id);
                }
            }
        }

        IMP_LOG_INFO("Tokenizer: type=%s, %d tokens, bos=%d, eos=%d (%zu total), add_bos=%d",
                     tok_type.c_str(), tokenizer->vocab_size(), bos_id, eos_id,
                     tokenizer->eos_ids().size(),
                     tokenizer->add_bos() ? 1 : 0);
    } else {
        IMP_LOG_WARN("No tokenizer data found in GGUF metadata");
    }

    model->set_tokenizer(std::move(tokenizer));

    IMP_LOG_INFO("GGUF model loaded successfully from %s", path.c_str());
    return model;
}

} // namespace imp
