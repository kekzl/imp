#include "core/qtype.h"
#include "core/enum_table.h"

#include <array>

namespace imp {

namespace {

struct QTypeInfo {
    QType id;
    const char* name;
    size_t elem_bytes;  // 0 for block-quant types; INT4/FP4_E2M1 pack 2 elems/byte
};

// One row per QType enumerator; the single source for qtype_name and qtype_elem_bytes.
constexpr auto kQTypeRows = std::to_array<QTypeInfo>({
    {QType::F32, "F32", 4},           {QType::F16, "F16", 2},           {QType::Q4_0, "Q4_0", 0},
    {QType::Q4_1, "Q4_1", 0},         {QType::Q5_0, "Q5_0", 0},         {QType::Q5_1, "Q5_1", 0},
    {QType::Q8_0, "Q8_0", 0},         {QType::Q8_1, "Q8_1", 0},         {QType::Q2_K, "Q2_K", 0},
    {QType::Q3_K, "Q3_K", 0},         {QType::Q4_K, "Q4_K", 0},         {QType::Q5_K, "Q5_K", 0},
    {QType::Q6_K, "Q6_K", 0},         {QType::Q8_K, "Q8_K", 0},         {QType::IQ4_NL, "IQ4_NL", 0},
    {QType::IQ4_XS, "IQ4_XS", 0},     {QType::BF16, "BF16", 2},         {QType::MXFP4, "MXFP4", 0},
    {QType::NONE, "NONE", 0},         {QType::FP8_E4M3, "FP8_E4M3", 1}, {QType::FP8_E5M2, "FP8_E5M2", 1},
    {QType::INT8, "INT8", 1},         {QType::INT4, "INT4", 1},         {QType::INT32, "INT32", 4},
    {QType::FP4_E2M1, "FP4_E2M1", 1}, {QType::NVFP4, "NVFP4", 0},       {QType::MXFP4_KV, "MXFP4_KV", 0},
});

constexpr QTypeInfo kUnknownQType{QType::NONE, "UNKNOWN", 0};
static_assert(enum_table::rows_cover_enumerators<QType, 256>(kQTypeRows),
              "kQTypeRows must have exactly one row per QType enumerator");

// Gaps and values past the last row resolve to kUnknownQType.
constexpr auto kQTypeIndex = enum_table::index_rows<enum_table::index_size(kQTypeRows)>(kQTypeRows);

const QTypeInfo& qtype_info(QType q) { return enum_table::lookup(kQTypeIndex, q, kUnknownQType); }

}  // namespace

size_t qtype_elem_bytes(QType q) { return qtype_info(q).elem_bytes; }

size_t qtype_row_bytes(QType q, int64_t cols) {
    switch (q) {
        case QType::Q6_K:
            return static_cast<size_t>(cols / 256) * 210;
        case QType::Q8_0:
            return static_cast<size_t>(cols / 32) * 34;
        case QType::Q4_0:
            return static_cast<size_t>(cols / 32) * 18;
        case QType::Q8_1:
            return static_cast<size_t>(cols / 32) * 36;
        case QType::Q4_1:
            return static_cast<size_t>(cols / 32) * 20;
        case QType::Q5_0:
            return static_cast<size_t>(cols / 32) * 22;
        case QType::Q5_1:
            return static_cast<size_t>(cols / 32) * 24;
        case QType::Q2_K:
            return static_cast<size_t>(cols / 256) * 84;
        case QType::Q3_K:
            return static_cast<size_t>(cols / 256) * 110;
        case QType::Q4_K:
            return static_cast<size_t>(cols / 256) * 144;
        case QType::Q5_K:
            return static_cast<size_t>(cols / 256) * 176;
        case QType::Q8_K:
            return static_cast<size_t>(cols / 256) * 292;
        case QType::IQ4_NL:
            return static_cast<size_t>(cols / 32) * 18;
        case QType::IQ4_XS:
            return static_cast<size_t>(cols / 256) * 136;
        case QType::F16:
        case QType::BF16:
            return static_cast<size_t>(cols) * 2;
        case QType::F32:
            return static_cast<size_t>(cols) * 4;
        case QType::INT4:
        case QType::FP4_E2M1:
        case QType::MXFP4:
            return static_cast<size_t>((cols + 1) / 2);
        case QType::NVFP4:
        case QType::MXFP4_KV:
            return static_cast<size_t>((cols + 1) / 2);  // packed; scales separate
        case QType::FP8_E4M3:
        case QType::FP8_E5M2:
        case QType::INT8:
            return static_cast<size_t>(cols);
        case QType::INT32:
            return static_cast<size_t>(cols) * 4;
        default:
            return static_cast<size_t>(cols) * 2;  // safe fallback
    }
}

const char* qtype_name(QType q) { return qtype_info(q).name; }

}  // namespace imp
