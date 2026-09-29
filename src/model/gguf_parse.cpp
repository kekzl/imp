// GGUF parsing: type tables, binary-value decoding, tensor-info parsing, bounds checks.
// Split out of gguf_loader.cpp to bound recompile blast radius (tools/check_filesize.py).

#include "model/gguf_loader.h"
#include "model/gguf_loader_internal.h"
#include "core/enum_table.h"
#include "core/logging.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace imp {

// ---- GGML type tables ----

namespace {

struct WireTypeInfo {
    GgufWireType id;
    const char* name;
    int block_size;    // elements per block
    size_t type_size;  // bytes per block (quantized) or per element
    QType qtype;       // NONE: no engine path for this wire type
};

// One row per GgufWireType enumerator; the single source for the four gguf_type lookups.
constexpr auto kWireRows = std::to_array<WireTypeInfo>({
    {GgufWireType::F32, "F32", 1, 4, QType::F32},
    {GgufWireType::F16, "F16", 1, 2, QType::F16},
    {GgufWireType::Q4_0, "Q4_0", 32, 18, QType::Q4_0},  // 32*4/8 + 2 (fp16 scale)
    {GgufWireType::Q4_1, "Q4_1", 32, 20, QType::Q4_1},  // 32*4/8 + 2 + 2 (scale + min)
    {GgufWireType::Q5_0, "Q5_0", 32, 22, QType::Q5_0},  // 32*5/8 + 4 (high bits) + 2
    {GgufWireType::Q5_1, "Q5_1", 32, 24, QType::Q5_1},  // 32*5/8 + 4 + 2 + 2
    {GgufWireType::Q8_0, "Q8_0", 32, 34, QType::Q8_0},  // 32*1 + 2
    // Q8_1 maps to NONE on purpose: parse_tensor_infos refuses the file (#1917).
    {GgufWireType::Q8_1, "Q8_1", 32, 36, QType::NONE},  // 32*1 + 2 + 2
    {GgufWireType::Q2_K, "Q2_K", 256, 84, QType::Q2_K},
    {GgufWireType::Q3_K, "Q3_K", 256, 110, QType::Q3_K},
    {GgufWireType::Q4_K, "Q4_K", 256, 144, QType::Q4_K},
    {GgufWireType::Q5_K, "Q5_K", 256, 176, QType::Q5_K},
    {GgufWireType::Q6_K, "Q6_K", 256, 210, QType::Q6_K},
    {GgufWireType::Q8_K, "Q8_K", 256, 292, QType::Q8_K},
    // IQ1/IQ2/IQ3 i-quants: no native QType yet.
    {GgufWireType::IQ2_XXS, "IQ2_XXS", 256, 66, QType::NONE},
    {GgufWireType::IQ2_XS, "IQ2_XS", 256, 74, QType::NONE},
    {GgufWireType::IQ3_XXS, "IQ3_XXS", 256, 98, QType::NONE},
    {GgufWireType::IQ1_S, "IQ1_S", 256, 50, QType::NONE},
    {GgufWireType::IQ4_NL, "IQ4_NL", 32, 18, QType::IQ4_NL},
    {GgufWireType::IQ3_S, "IQ3_S", 256, 110, QType::NONE},
    {GgufWireType::IQ2_S, "IQ2_S", 256, 82, QType::NONE},
    {GgufWireType::IQ4_XS, "IQ4_XS", 256, 136, QType::IQ4_XS},
    {GgufWireType::I8, "I8", 1, 1, QType::INT8},
    {GgufWireType::I16, "I16", 1, 2, QType::NONE},
    {GgufWireType::I32, "I32", 1, 4, QType::INT32},
    {GgufWireType::I64, "I64", 1, 8, QType::NONE},
    {GgufWireType::F64, "F64", 1, 8, QType::NONE},
    {GgufWireType::IQ1_M, "IQ1_M", 256, 56, QType::NONE},
    {GgufWireType::BF16, "BF16", 1, 2, QType::BF16},
    {GgufWireType::MXFP4, "MXFP4", 32, 17, QType::MXFP4},     // 32*4/8 + 1 (UE8M0 scale)
    {GgufWireType::MXFP4_V2, "MXFP4", 32, 17, QType::MXFP4},  // same block as MXFP4
});

constexpr WireTypeInfo kUnknownWire{GgufWireType::F32, "UNKNOWN", 0, 0, QType::NONE};
static_assert(enum_table::rows_cover_enumerators<GgufWireType, 256>(kWireRows),
              "kWireRows must have exactly one row per GgufWireType enumerator");

// Unused ids (4, 5, 32..38) and ids past the last row resolve to kUnknownWire.
constexpr auto kWireIndex = enum_table::index_rows<enum_table::index_size(kWireRows)>(kWireRows);

const WireTypeInfo& wire_info(GgufWireType type) {
    return enum_table::lookup(kWireIndex, type, kUnknownWire);
}

}  // namespace

int gguf_blck_size(GgufWireType type) { return wire_info(type).block_size; }

size_t gguf_type_size(GgufWireType type) { return wire_info(type).type_size; }

size_t gguf_row_size(GgufWireType type, int64_t n_elements) {
    int bs = gguf_blck_size(type);
    if (bs == 0)
        return 0;
    return static_cast<size_t>((n_elements + bs - 1) / bs) * gguf_type_size(type);
}

QType gguf_type_to_qtype(GgufWireType type) { return wire_info(type).qtype; }

const char* gguf_type_name(GgufWireType type) { return wire_info(type).name; }

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
                // read_gguf_value()'s switch has no default, so an unknown array element type would consume
                // zero bytes per element; a count of 2^60 would spin without tripping the EOF guard.
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

void parse_tensor_infos(BinaryReader& reader, uint64_t tensor_count, std::vector<GGUFTensorInfo>& out) {
    for (uint64_t i = 0; i < tensor_count && !reader.failed(); i++) {
        GGUFTensorInfo info;
        info.name = reader.read_string();
        info.n_dims = reader.read_u32();
        // GGML_MAX_DIMS is 4 and `dims` has 4 slots. This used to skip the
        // extra words and keep going, and the loader then wrote `n_dims`
        // entries into a 4-element stack array (AUDIT_arch_2026 F1-1).
        if (info.n_dims > 4) {
            reader.fail();
            break;
        }
        for (uint32_t d = 0; d < info.n_dims; d++) {
            info.dims[d] = static_cast<int64_t>(reader.read_u64());
        }
        // Fill remaining dims with 1
        for (uint32_t d = info.n_dims; d < 4; d++) {
            info.dims[d] = 1;
        }
        info.type = static_cast<GgufWireType>(reader.read_u32());
        // Q8_1 (wire type 9) is llama.cpp's activation format, not a weight storage type: imp has
        // no dequant/kernel/registry entry for it (AUDIT_arch_2026 G-8). Refused like n_dims>4 above.
        if (info.type == GgufWireType::Q8_1) {
            IMP_LOG_ERROR(
                "GGUF tensor '%s' is stored as Q8_1 (an activation format, "
                "no weight path) - refusing the file",
                info.name.c_str());
            reader.fail();
            break;
        }
        info.offset = reader.read_u64();
        out.push_back(std::move(info));
    }
}

// ---- Tensor on-disk byte span ----

// Total tensor bytes with saturating arithmetic: a crafted dim product (ne[0]*ne[1]
// overflow) can never wrap to a small value that passes the bounds check. Returns SIZE_MAX
// on overflow, which makes the caller reject the tensor.
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
