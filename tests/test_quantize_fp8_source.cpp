// Reading an FP8 checkpoint as a quantization source.
//
// The two things here that cannot be checked by looking at a converted
// checkpoint: the E4M3 bit layout, and the block-scale stride. Both produce
// numbers of a plausible magnitude when wrong, so a model built on a wrong
// exponent bias or a transposed scale grid loads, generates, and is simply a
// worse model than it should be.

#include "../tools/imp-quantize/fp8_source.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace imp::quantize {
namespace {

float fp16_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0)
        bits = m ? sign | ((127u - 15u + 1u) << 23) | (m << 13) : sign;
    else if (e == 0x1Fu)
        bits = sign | 0x7F800000u | (m << 13);
    else
        bits = sign | ((e + 127u - 15u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

RawTensor make(const std::string& dtype, std::vector<int64_t> shape, const void* data) {
    RawTensor t;
    t.name = "w";
    t.dtype = dtype;
    t.shape = std::move(shape);
    t.data = const_cast<void*>(data);
    return t;
}

// --- E4M3 bit layout -------------------------------------------------------

TEST(Fp8Source, DecodesTheE4M3Grid) {
    EXPECT_EQ(e4m3_to_float(0x00), 0.0f);
    EXPECT_EQ(e4m3_to_float(0x38), 1.0f);   // exp bias 7: e=7, m=0
    EXPECT_EQ(e4m3_to_float(0xB8), -1.0f);  // same with the sign bit
    EXPECT_EQ(e4m3_to_float(0x3C), 1.5f);   // e=7, m=4 -> 1 + 4/8
    EXPECT_EQ(e4m3_to_float(0x40), 2.0f);   // e=8, m=0
}

TEST(Fp8Source, HandlesTheRangeEndsThatDistinguishE4M3) {
    // 448 is E4M3's largest finite value. Getting the bias wrong by one moves
    // this to 224 or 896 and nothing else in a checkpoint would look odd.
    EXPECT_EQ(e4m3_to_float(0x7E), 448.0f);
    // E4M3 has NO infinity: that encoding is spent on NaN.
    EXPECT_TRUE(std::isnan(e4m3_to_float(0x7F)));
    // Subnormals use a fixed 2^-6 scale with no implicit leading one.
    EXPECT_FLOAT_EQ(e4m3_to_float(0x01), std::ldexp(1.0f / 8.0f, -6));
    EXPECT_FLOAT_EQ(e4m3_to_float(0x07), std::ldexp(7.0f / 8.0f, -6));
}

// --- block geometry --------------------------------------------------------

TEST(Fp8Source, DerivesTheBlockEdgeFromBothDimensions) {
    // Qwen3.8-27B-FP8: weight [10240, 5120], scale [80, 40].
    EXPECT_EQ(derive_block_edge(10240, 5120, 80, 40), 128);
    // DeepSeek-V3: weight [1536, 7168], scale [12, 56].
    EXPECT_EQ(derive_block_edge(1536, 7168, 12, 56), 128);
}

TEST(Fp8Source, RefusesAScaleGridNoSingleBlockExplains) {
    // A transposed grid. Deriving the edge from the rows alone would accept
    // this and then read the scales with a wrong stride.
    EXPECT_EQ(derive_block_edge(10240, 5120, 40, 80), 0);
    // Right shape, wrong count.
    EXPECT_EQ(derive_block_edge(1536, 7168, 12, 55), 0);
    EXPECT_EQ(derive_block_edge(0, 5120, 80, 40), 0);
}

TEST(Fp8Source, DerivesAnEdgeThatDoesNotDivideEvenly) {
    // 300 rows in blocks of 128 is 3 scale rows, and the last one is partial.
    EXPECT_EQ(derive_block_edge(300, 256, 3, 2), 128);
}

// --- end to end: each block gets its own scale -----------------------------

TEST(Fp8Source, AppliesThePerBlockScaleToTheRightTile) {
    // 2x2 blocks of edge 1, so every element has its own scale. That makes a
    // row/column mix-up visible, which a uniform scale would hide.
    const uint8_t w[4] = {0x38, 0x38, 0x38, 0x38};  // all 1.0
    const float s[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto out = fp8_block_scaled_to_fp16(make("F8_E4M3", {2, 2}, w), make("F32", {2, 2}, s));
    ASSERT_TRUE(out) << out.error();
    ASSERT_EQ(out->size(), 4u);
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[0]), 1.0f);
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[1]), 2.0f);
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[2]), 3.0f);  // row 1 must take s[2], not s[1]
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[3]), 4.0f);
}

TEST(Fp8Source, ReadsABf16ScaleGridAsBf16) {
    // Qwen ships the grid in BF16, DeepSeek in F32. Reading one as the other
    // is a large silent factor, so both paths are pinned.
    const uint8_t w[1] = {0x38};                      // 1.0
    const uint16_t s_bf16[1] = {0x4040};              // 3.0 in BF16
    const auto out = fp8_block_scaled_to_fp16(make("F8_E4M3", {1, 1}, w), make("BF16", {1, 1}, s_bf16));
    ASSERT_TRUE(out) << out.error();
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[0]), 3.0f);
}

TEST(Fp8Source, OneScaleCoversItsWholeTile) {
    // 2x2 weight under a single scale: the 1x1 grid means block edge 2.
    const uint8_t w[4] = {0x38, 0x40, 0x38, 0x40};  // 1, 2, 1, 2
    const float s[1] = {0.5f};
    const auto out = fp8_block_scaled_to_fp16(make("F8_E4M3", {2, 2}, w), make("F32", {1, 1}, s));
    ASSERT_TRUE(out) << out.error();
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[0]), 0.5f);
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[1]), 1.0f);
    EXPECT_FLOAT_EQ(fp16_to_float((*out)[3]), 1.0f);
}

// "must not hand back a partial buffer" used to be a property of the code that
// a test had to check, by passing a pre-filled vector and asserting it survived.
// It is now a property of the signature: a refusal carries no vector at all.
TEST(Fp8Source, RefusesWithAReasonAndNoBuffer) {
    const uint8_t w[4] = {0x38, 0x38, 0x38, 0x38};
    const float s[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    // Wrong weight dtype.
    const auto bad_weight = fp8_block_scaled_to_fp16(make("BF16", {2, 2}, w), make("F32", {2, 2}, s));
    ASSERT_FALSE(bad_weight.has_value());
    EXPECT_FALSE(bad_weight.error().empty());
    // Unreadable scale dtype.
    EXPECT_FALSE(fp8_block_scaled_to_fp16(make("F8_E4M3", {2, 2}, w), make("U8", {2, 2}, s)).has_value());
    // Scale grid that no block size explains.
    EXPECT_FALSE(fp8_block_scaled_to_fp16(make("F8_E4M3", {2, 2}, w), make("F32", {3, 1}, s)).has_value());
}

TEST(Fp8Source, AcceptsTheDtypeSpellingsExportersUse) {
    EXPECT_TRUE(is_fp8_e4m3_dtype("F8_E4M3"));
    EXPECT_TRUE(is_fp8_e4m3_dtype("F8_E4M3FN"));
    // E5M2 has a different exponent bias; decoding it as E4M3 would be silent.
    EXPECT_FALSE(is_fp8_e4m3_dtype("F8_E5M2"));
    EXPECT_FALSE(is_fp8_e4m3_dtype("BF16"));
}

}  // namespace
}  // namespace imp::quantize
