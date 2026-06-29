// CPU unit tests for the host half/bf16 <-> float helpers used by the gpt-oss
// GGUF 2^-4 residual rescale. These lock in the PR #808 fix: the rescale MUST be
// a float-domain multiply (correct for denormals/underflow), not the old
// exponent-bit-subtract that left denormal scales 16x too large and flushed small
// exponents to zero — which corrupted Q8_0 gpt-oss weights into garbage output.

#include "model/gguf_half.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

namespace imp {

TEST(GgufHalf, KnownValues) {
    EXPECT_EQ(gguf_float_to_half(1.0f), 0x3C00u);
    EXPECT_EQ(gguf_float_to_half(0.0f), 0x0000u);
    EXPECT_EQ(gguf_float_to_half(2.0f), 0x4000u);
    EXPECT_EQ(gguf_float_to_half(0.5f), 0x3800u);
    EXPECT_FLOAT_EQ(gguf_half_to_float(0x3C00u), 1.0f);
    EXPECT_FLOAT_EQ(gguf_half_to_float(0x4000u), 2.0f);
    EXPECT_FLOAT_EQ(gguf_half_to_float(0xBC00u), -1.0f);  // sign bit
    // 1.0 * 2^-4 must be exactly 0.0625 (this is the rescale factor).
    EXPECT_FLOAT_EQ(gguf_half_to_float(gguf_float_to_half(1.0f * 0.0625f)), 0.0625f);
}

TEST(GgufHalf, RoundTripFinite) {
    for (uint32_t bits = 0; bits <= 0xFFFFu; ++bits) {
        uint16_t h = static_cast<uint16_t>(bits);
        uint32_t exp = (h >> 10) & 0x1Fu;
        if (exp == 0x1Fu)
            continue;  // skip inf/nan
        float v = gguf_half_to_float(h);
        // Re-encoding an exact fp16 value must reproduce the same bits (zeros may
        // be either +0/-0 → compare values, not bits, for the zero case).
        uint16_t re = gguf_float_to_half(v);
        if (v == 0.0f)
            EXPECT_EQ(gguf_half_to_float(re), 0.0f) << "bits=" << bits;
        else
            EXPECT_EQ(re, h) << "round-trip changed bits: " << bits << " -> " << re;
    }
}

// THE REGRESSION GUARD: scaling by 0.0625 must be correct across the whole fp16
// range — including denormals and small exponents, which the old bit-shift hack
// got wrong (denormal -> unchanged = 16x too big; exp 1..4 -> zero).
TEST(GgufHalf, RescaleByOneSixteenthIsCorrect) {
    for (uint32_t bits = 0; bits <= 0xFFFFu; ++bits) {
        uint16_t h = static_cast<uint16_t>(bits);
        uint32_t exp = (h >> 10) & 0x1Fu;
        if (exp == 0x1Fu)
            continue;  // skip inf/nan
        float v = gguf_half_to_float(h);
        double ref = static_cast<double>(v) * 0.0625;  // the correct scaled value
        float got = gguf_half_to_float(gguf_float_to_half(v * 0.0625f));
        double tol = std::max(1e-7, std::fabs(ref) * 1e-3);  // ~fp16 rounding
        EXPECT_NEAR(static_cast<double>(got), ref, tol)
            << "bits=" << bits << " v=" << v << " (the old bit-shift bug fails here)";
    }
}

TEST(GgufHalf, RescaleDenormalNotLeftUnscaled) {
    // fp16 denormal (biased exp 0): the old code left it UNCHANGED (16x too big).
    uint16_t denorm = 0x0200u;  // 0.5 * 2^-14 = 2^-15
    float v = gguf_half_to_float(denorm);
    ASSERT_GT(v, 0.0f);
    float scaled = gguf_half_to_float(gguf_float_to_half(v * 0.0625f));
    EXPECT_LT(scaled, v);  // must shrink, not stay equal
    EXPECT_NEAR(static_cast<double>(scaled), static_cast<double>(v) * 0.0625, 1e-7);
}

TEST(GgufHalf, RescaleSmallExponentNotZeroed) {
    // fp16 with biased exp 3 (~2^-12): the old code flushed exp<=4 to zero.
    uint16_t small = static_cast<uint16_t>(3u << 10);  // 2^-12
    float v = gguf_half_to_float(small);
    ASSERT_GT(v, 0.0f);
    float scaled = gguf_half_to_float(gguf_float_to_half(v * 0.0625f));
    EXPECT_GT(scaled, 0.0f);  // must NOT be zeroed
    EXPECT_NEAR(static_cast<double>(scaled), static_cast<double>(v) * 0.0625, 1e-7);
}

TEST(GgufBf16, RoundTripAndRescale) {
    EXPECT_FLOAT_EQ(gguf_bf16_to_float(gguf_float_to_bf16(1.0f)), 1.0f);
    EXPECT_FLOAT_EQ(gguf_bf16_to_float(gguf_float_to_bf16(-3.5f)), -3.5f);
    // bf16 has 8 exponent bits, so normal weights scale exactly by 2^-4.
    for (float v : {1.0f, 7.0f, 0.013f, -2.5f, 100.0f}) {
        float got = gguf_bf16_to_float(gguf_float_to_bf16(v * 0.0625f));
        EXPECT_NEAR(static_cast<double>(got), static_cast<double>(v) * 0.0625,
                    std::fabs(v) * 0.01 + 1e-6);
    }
}

}  // namespace imp
