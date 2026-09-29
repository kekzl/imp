// Golden outputs of the GgufWireType and QType property lookups (#2212 item 1).
// Every input 0..65535 plus the uint32 edge is checked; ids not listed must give the default row.
// Captured from the switch implementation; the table implementation must reproduce it unchanged.

#include "core/qtype.h"
#include "model/gguf_loader.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>

using namespace imp;

namespace {

struct WireGolden {
    int blck;
    size_t type_size;
    int qtype;
    const char* name;
};

// Wire id -> {gguf_blck_size, gguf_type_size, gguf_type_to_qtype, gguf_type_name}.
const std::map<uint32_t, WireGolden> kWireGolden = {
    {0, {1, 4, 0, "F32"}},          {1, {1, 2, 1, "F16"}},          {2, {32, 18, 2, "Q4_0"}},
    {3, {32, 20, 3, "Q4_1"}},       {6, {32, 22, 6, "Q5_0"}},       {7, {32, 24, 7, "Q5_1"}},
    {8, {32, 34, 8, "Q8_0"}},       {9, {32, 36, 64, "Q8_1"}},      {10, {256, 84, 10, "Q2_K"}},
    {11, {256, 110, 11, "Q3_K"}},   {12, {256, 144, 12, "Q4_K"}},   {13, {256, 176, 13, "Q5_K"}},
    {14, {256, 210, 14, "Q6_K"}},   {15, {256, 292, 15, "Q8_K"}},   {16, {256, 66, 64, "IQ2_XXS"}},
    {17, {256, 74, 64, "IQ2_XS"}},  {18, {256, 98, 64, "IQ3_XXS"}}, {19, {256, 50, 64, "IQ1_S"}},
    {20, {32, 18, 20, "IQ4_NL"}},   {21, {256, 110, 64, "IQ3_S"}},  {22, {256, 82, 64, "IQ2_S"}},
    {23, {256, 136, 23, "IQ4_XS"}}, {24, {1, 1, 67, "I8"}},         {25, {1, 2, 64, "I16"}},
    {26, {1, 4, 69, "I32"}},        {27, {1, 8, 64, "I64"}},        {28, {1, 8, 64, "F64"}},
    {29, {256, 56, 64, "IQ1_M"}},   {30, {1, 2, 30, "BF16"}},       {31, {32, 17, 31, "MXFP4"}},
    {39, {32, 17, 31, "MXFP4"}},
};
constexpr WireGolden kWireDefault{0, 0, 64, "UNKNOWN"};

struct QTypeGolden {
    size_t elem_bytes;
    const char* name;
};

// QType value -> {qtype_elem_bytes, qtype_name}.
const std::map<uint16_t, QTypeGolden> kQTypeGolden = {
    {0, {4, "F32"}},       {1, {2, "F16"}},    {2, {0, "Q4_0"}},      {3, {0, "Q4_1"}},
    {6, {0, "Q5_0"}},      {7, {0, "Q5_1"}},   {8, {0, "Q8_0"}},      {9, {0, "Q8_1"}},
    {10, {0, "Q2_K"}},     {11, {0, "Q3_K"}},  {12, {0, "Q4_K"}},     {13, {0, "Q5_K"}},
    {14, {0, "Q6_K"}},     {15, {0, "Q8_K"}},  {20, {0, "IQ4_NL"}},   {23, {0, "IQ4_XS"}},
    {30, {2, "BF16"}},     {31, {0, "MXFP4"}}, {64, {0, "NONE"}},     {65, {1, "FP8_E4M3"}},
    {66, {1, "FP8_E5M2"}}, {67, {1, "INT8"}},  {68, {1, "INT4"}},     {69, {4, "INT32"}},
    {70, {1, "FP4_E2M1"}}, {71, {0, "NVFP4"}}, {72, {0, "MXFP4_KV"}},
};
constexpr QTypeGolden kQTypeDefault{0, "UNKNOWN"};

void check_wire(uint32_t v) {
    auto it = kWireGolden.find(v);
    const WireGolden& g = it == kWireGolden.end() ? kWireDefault : it->second;
    auto t = static_cast<GgufWireType>(v);
    EXPECT_EQ(gguf_blck_size(t), g.blck) << "wire " << v;
    EXPECT_EQ(gguf_type_size(t), g.type_size) << "wire " << v;
    EXPECT_EQ(static_cast<int>(gguf_type_to_qtype(t)), g.qtype) << "wire " << v;
    EXPECT_STREQ(gguf_type_name(t), g.name) << "wire " << v;
}

}  // namespace

TEST(TypeTablesGolden, WireTypeEveryIdMatchesGolden) {
    for (uint32_t v = 0; v <= 0xFFFFu; ++v)
        check_wire(v);
    for (uint32_t v : {0x10000u, 0x7FFFFFFFu, 0xFFFFFFFFu})
        check_wire(v);
}

TEST(TypeTablesGolden, QTypeEveryValueMatchesGolden) {
    for (uint32_t v = 0; v <= 0xFFFFu; ++v) {
        auto it = kQTypeGolden.find(static_cast<uint16_t>(v));
        const QTypeGolden& g = it == kQTypeGolden.end() ? kQTypeDefault : it->second;
        auto q = static_cast<QType>(v);
        EXPECT_EQ(qtype_elem_bytes(q), g.elem_bytes) << "qtype " << v;
        EXPECT_STREQ(qtype_name(q), g.name) << "qtype " << v;
    }
}
