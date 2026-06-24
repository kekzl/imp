// ============================================================================
// GGUF parsing: GGML type tables, binary-value decoding, tensor-info parsing,
// and tensor bounds checks. Split out of gguf_loader.cpp to bound recompile
// blast radius (see tools/check_filesize.py). Top-level load orchestration
// stays in gguf_loader.cpp.
// ============================================================================

#include "model/gguf_loader.h"
#include "model/gguf_loader_internal.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace imp {

// ---- GGML type tables ----

int gguf_blck_size(GgufWireType type) {
    switch (type) {
        case GgufWireType::F32:
            return 1;
        case GgufWireType::F16:
            return 1;
        case GgufWireType::BF16:
            return 1;
        case GgufWireType::F64:
            return 1;
        case GgufWireType::I8:
            return 1;
        case GgufWireType::I16:
            return 1;
        case GgufWireType::I32:
            return 1;
        case GgufWireType::I64:
            return 1;
        case GgufWireType::Q4_0:
            return 32;
        case GgufWireType::Q4_1:
            return 32;
        case GgufWireType::Q5_0:
            return 32;
        case GgufWireType::Q5_1:
            return 32;
        case GgufWireType::Q8_0:
            return 32;
        case GgufWireType::Q8_1:
            return 32;
        case GgufWireType::IQ4_NL:
            return 32;
        case GgufWireType::Q2_K:
            return 256;
        case GgufWireType::Q3_K:
            return 256;
        case GgufWireType::Q4_K:
            return 256;
        case GgufWireType::Q5_K:
            return 256;
        case GgufWireType::Q6_K:
            return 256;
        case GgufWireType::Q8_K:
            return 256;
        case GgufWireType::IQ2_XXS:
            return 256;
        case GgufWireType::IQ2_XS:
            return 256;
        case GgufWireType::IQ2_S:
            return 256;
        case GgufWireType::IQ3_XXS:
            return 256;
        case GgufWireType::IQ3_S:
            return 256;
        case GgufWireType::IQ1_S:
            return 256;
        case GgufWireType::IQ1_M:
            return 256;
        case GgufWireType::IQ4_XS:
            return 256;
        case GgufWireType::MXFP4:
        case GgufWireType::MXFP4_V2:
            return 32;
        default:
            return 0;
    }
}

size_t gguf_type_size(GgufWireType type) {
    switch (type) {
        case GgufWireType::F32:
            return 4;
        case GgufWireType::F16:
            return 2;
        case GgufWireType::BF16:
            return 2;
        case GgufWireType::F64:
            return 8;
        case GgufWireType::I8:
            return 1;
        case GgufWireType::I16:
            return 2;
        case GgufWireType::I32:
            return 4;
        case GgufWireType::I64:
            return 8;
        case GgufWireType::Q4_0:
            return 18;  // 32*4/8 + 2 (fp16 scale)
        case GgufWireType::Q4_1:
            return 20;  // 32*4/8 + 2 + 2 (scale + min)
        case GgufWireType::Q5_0:
            return 22;  // 32*5/8 + 4 (high bits) + 2
        case GgufWireType::Q5_1:
            return 24;  // 32*5/8 + 4 + 2 + 2
        case GgufWireType::Q8_0:
            return 34;  // 32*1 + 2
        case GgufWireType::Q8_1:
            return 36;  // 32*1 + 2 + 2
        case GgufWireType::Q2_K:
            return 84;
        case GgufWireType::Q3_K:
            return 110;
        case GgufWireType::Q4_K:
            return 144;
        case GgufWireType::Q5_K:
            return 176;
        case GgufWireType::Q6_K:
            return 210;
        case GgufWireType::Q8_K:
            return 292;
        case GgufWireType::IQ2_XXS:
            return 66;
        case GgufWireType::IQ2_XS:
            return 74;
        case GgufWireType::IQ2_S:
            return 82;
        case GgufWireType::IQ3_XXS:
            return 98;
        case GgufWireType::IQ3_S:
            return 110;
        case GgufWireType::IQ1_S:
            return 50;
        case GgufWireType::IQ1_M:
            return 56;
        case GgufWireType::IQ4_NL:
            return 18;
        case GgufWireType::IQ4_XS:
            return 136;
        case GgufWireType::MXFP4:
        case GgufWireType::MXFP4_V2:
            return 17;  // 32*4/8 + 1 (UE8M0 scale)
        default:
            return 0;
    }
}

size_t gguf_row_size(GgufWireType type, int64_t n_elements) {
    int bs = gguf_blck_size(type);
    if (bs == 0)
        return 0;
    return static_cast<size_t>((n_elements + bs - 1) / bs) * gguf_type_size(type);
}

QType gguf_type_to_qtype(GgufWireType type) {
    // Wire-stable values 0..31 in QType match the GGUF on-disk numbering,
    // so the cast is exact for every supported block-quant type. Anything
    // outside the 0..31 range falls through to NONE.
    switch (type) {
        case GgufWireType::F32:
            return QType::F32;
        case GgufWireType::F16:
            return QType::F16;
        case GgufWireType::BF16:
            return QType::BF16;
        case GgufWireType::Q4_0:
            return QType::Q4_0;
        case GgufWireType::Q4_1:
            return QType::Q4_1;
        case GgufWireType::Q5_0:
            return QType::Q5_0;
        case GgufWireType::Q5_1:
            return QType::Q5_1;
        case GgufWireType::Q8_0:
            return QType::Q8_0;
        case GgufWireType::Q8_1:
            return QType::Q8_1;
        case GgufWireType::Q2_K:
            return QType::Q2_K;
        case GgufWireType::Q3_K:
            return QType::Q3_K;
        case GgufWireType::Q4_K:
            return QType::Q4_K;
        case GgufWireType::Q5_K:
            return QType::Q5_K;
        case GgufWireType::Q6_K:
            return QType::Q6_K;
        case GgufWireType::Q8_K:
            return QType::Q8_K;
        case GgufWireType::MXFP4:
        case GgufWireType::MXFP4_V2:
            return QType::MXFP4;
        case GgufWireType::I8:
            return QType::INT8;
        case GgufWireType::I32:
            return QType::INT32;
        case GgufWireType::IQ4_NL:
            return QType::IQ4_NL;
        case GgufWireType::IQ4_XS:
            return QType::IQ4_XS;
        default:
            // IQ1/IQ2/IQ3 i-quants — no native QType yet; mark unsupported.
            return QType::NONE;
    }
}

const char* gguf_type_name(GgufWireType type) {
    switch (type) {
        case GgufWireType::F32:
            return "F32";
        case GgufWireType::F16:
            return "F16";
        case GgufWireType::BF16:
            return "BF16";
        case GgufWireType::F64:
            return "F64";
        case GgufWireType::I8:
            return "I8";
        case GgufWireType::I16:
            return "I16";
        case GgufWireType::I32:
            return "I32";
        case GgufWireType::I64:
            return "I64";
        case GgufWireType::Q4_0:
            return "Q4_0";
        case GgufWireType::Q4_1:
            return "Q4_1";
        case GgufWireType::Q5_0:
            return "Q5_0";
        case GgufWireType::Q5_1:
            return "Q5_1";
        case GgufWireType::Q8_0:
            return "Q8_0";
        case GgufWireType::Q8_1:
            return "Q8_1";
        case GgufWireType::Q2_K:
            return "Q2_K";
        case GgufWireType::Q3_K:
            return "Q3_K";
        case GgufWireType::Q4_K:
            return "Q4_K";
        case GgufWireType::Q5_K:
            return "Q5_K";
        case GgufWireType::Q6_K:
            return "Q6_K";
        case GgufWireType::Q8_K:
            return "Q8_K";
        case GgufWireType::IQ2_XXS:
            return "IQ2_XXS";
        case GgufWireType::IQ2_XS:
            return "IQ2_XS";
        case GgufWireType::IQ2_S:
            return "IQ2_S";
        case GgufWireType::IQ3_XXS:
            return "IQ3_XXS";
        case GgufWireType::IQ3_S:
            return "IQ3_S";
        case GgufWireType::IQ1_S:
            return "IQ1_S";
        case GgufWireType::IQ1_M:
            return "IQ1_M";
        case GgufWireType::IQ4_NL:
            return "IQ4_NL";
        case GgufWireType::IQ4_XS:
            return "IQ4_XS";
        case GgufWireType::MXFP4:
        case GgufWireType::MXFP4_V2:
            return "MXFP4";
        default:
            return "UNKNOWN";
    }
}

// ---- Read array elements by type into a GGUFValue ----

template <typename T, typename ReadFn>
static void read_array_elements(BinaryReader& r, uint64_t count, std::vector<T>& out, ReadFn read_fn,
                                size_t element_size) {
    size_t safe = std::min(static_cast<size_t>(count), r.remaining() / element_size);
    out.reserve(safe);
    for (uint64_t i = 0; i < count && !r.failed(); i++) {
        out.push_back(read_fn(r));
    }
}

GGUFValue read_gguf_value(BinaryReader& r, GGUFValueType type) {
    GGUFValue v;
    v.type = type;
    switch (type) {
        case GGUFValueType::UINT8:
            v.uint_val = r.read_u8();
            break;
        case GGUFValueType::INT8:
            v.int_val = r.read_i8();
            break;
        case GGUFValueType::UINT16:
            v.uint_val = r.read_u16();
            break;
        case GGUFValueType::INT16:
            v.int_val = r.read_i16();
            break;
        case GGUFValueType::UINT32:
            v.uint_val = r.read_u32();
            break;
        case GGUFValueType::INT32:
            v.int_val = r.read_i32();
            break;
        case GGUFValueType::FLOAT32:
            v.float_val = r.read_f32();
            break;
        case GGUFValueType::BOOL:
            v.uint_val = r.read_u8();
            break;
        case GGUFValueType::STRING:
            v.str_val = r.read_string();
            break;
        case GGUFValueType::UINT64:
            v.uint_val = r.read_u64();
            break;
        case GGUFValueType::INT64:
            v.int_val = r.read_i64();
            break;
        case GGUFValueType::FLOAT64:
            v.float_val = r.read_f64();
            break;
        case GGUFValueType::ARRAY: {
            auto arr_type = static_cast<GGUFValueType>(r.read_u32());
            uint64_t count = r.read_u64();
            if (arr_type == GGUFValueType::STRING) {
                // Each string is at least 8 bytes (u64 length prefix)
                read_array_elements(
                    r, count, v.str_array, [](BinaryReader& br) { return br.read_string(); }, 8);
            } else if (arr_type == GGUFValueType::FLOAT32) {
                read_array_elements(
                    r, count, v.float_array, [](BinaryReader& br) { return br.read_f32(); }, 4);
            } else if (arr_type == GGUFValueType::INT32) {
                read_array_elements(r, count, v.int_array, [](BinaryReader& br) { return br.read_i32(); }, 4);
            } else if (arr_type == GGUFValueType::UINT32) {
                read_array_elements(
                    r, count, v.int_array,
                    [](BinaryReader& br) { return static_cast<int32_t>(br.read_u32()); }, 4);
            } else if (arr_type == GGUFValueType::BOOL || arr_type == GGUFValueType::UINT8 ||
                       arr_type == GGUFValueType::INT8) {
                read_array_elements(
                    r, count, v.int_array,
                    [](BinaryReader& br) { return static_cast<int32_t>(br.read_u8()); }, 1);
            } else {
                // Unknown/unsupported array element type. read_gguf_value()'s
                // switch has no default, so it would consume zero bytes per
                // element — a `count` of 2^60 would then spin ~forever without
                // ever tripping the EOF guard. Treat it as a parse error.
                r.fail();
            }
            break;
        }
    }
    return v;
}

uint64_t val_uint(const GGUFValue& v) {
    switch (v.type) {
        case GGUFValueType::UINT8:
        case GGUFValueType::UINT16:
        case GGUFValueType::UINT32:
        case GGUFValueType::UINT64:
        case GGUFValueType::BOOL:
            return v.uint_val;
        case GGUFValueType::INT8:
        case GGUFValueType::INT16:
        case GGUFValueType::INT32:
        case GGUFValueType::INT64:
            return static_cast<uint64_t>(v.int_val);
        case GGUFValueType::FLOAT32:
        case GGUFValueType::FLOAT64:
            return static_cast<uint64_t>(v.float_val);
        default:
            return 0;
    }
}

double val_float(const GGUFValue& v) {
    switch (v.type) {
        case GGUFValueType::FLOAT32:
        case GGUFValueType::FLOAT64:
            return v.float_val;
        case GGUFValueType::UINT8:
        case GGUFValueType::UINT16:
        case GGUFValueType::UINT32:
        case GGUFValueType::UINT64:
            return static_cast<double>(v.uint_val);
        case GGUFValueType::INT8:
        case GGUFValueType::INT16:
        case GGUFValueType::INT32:
        case GGUFValueType::INT64:
            return static_cast<double>(v.int_val);
        default:
            return 0.0;
    }
}

// ---- Parse tensor info entries from a BinaryReader ----

void parse_tensor_infos(BinaryReader& reader, uint64_t tensor_count,
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

// ---- Tensor on-disk byte span ----

// Total bytes a tensor occupies in the file, with saturating arithmetic so a
// crafted dim product (e.g. ne[0]*ne[1] overflowing int64) can never wrap to a
// small value that then passes the bounds check. Returns SIZE_MAX on overflow,
// which makes the caller reject the tensor.
static size_t gguf_tensor_byte_size(const GGUFTensorInfo& info) {
    uint64_t n_elements = 1;
    for (uint32_t d = 0; d < info.n_dims && d < 4; d++) {
        int64_t dim = info.dims[d];
        if (dim < 0)
            return SIZE_MAX;  // negative/huge dim — reject
        uint64_t ud = static_cast<uint64_t>(dim);
        if (ud != 0 && n_elements > UINT64_MAX / ud)
            return SIZE_MAX;  // multiply would overflow
        n_elements *= ud;
    }
    int bs = gguf_blck_size(info.type);
    size_t ts = gguf_type_size(info.type);
    if (bs <= 0 || ts == 0)
        return SIZE_MAX;  // unknown / unsupported quant type
    uint64_t n_blocks = (n_elements + static_cast<uint64_t>(bs) - 1) / static_cast<uint64_t>(bs);
    if (n_blocks != 0 && ts > UINT64_MAX / n_blocks)
        return SIZE_MAX;
    return static_cast<size_t>(n_blocks * ts);
}

// True iff the tensor's [offset, offset+size) window lies fully inside its
// shard's data region (data_limit bytes from data_base).
bool gguf_tensor_in_bounds(const GGUFTensorInfo& info) {
    size_t size = gguf_tensor_byte_size(info);
    if (size == SIZE_MAX)
        return false;
    if (info.offset > info.data_limit)
        return false;
    return size <= info.data_limit - info.offset;
}

}  // namespace imp
