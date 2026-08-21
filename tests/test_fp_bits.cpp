// CPU unit tests for core/fp_bits.h, the tree's single copy of the host
// half/bf16 <-> float conversions.
//
// Two regressions live here. PR #808: the gpt-oss GGUF 2^-4 residual rescale
// MUST be a float-domain multiply (correct for denormals/underflow), not the
// old exponent-bit-subtract that left denormal scales 16x too large and
// flushed small exponents to zero, which corrupted Q8_0 gpt-oss weights into
// garbage. And the subnormal widening imp-quantize's FP8 reader used to do by
// hand, which read all 2046 subnormal halves up to 1025x too large.
//
// The loops walk all 65536 patterns of a 16-bit format, so "for every fp16
// value" is literal here, not a sample.

#include "core/fp_bits.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <ranges>

namespace imp {

TEST(FpBits, HalfKnownValues) {
    EXPECT_EQ(float_to_half(1.0f), 0x3C00u);
    EXPECT_EQ(float_to_half(0.0f), 0x0000u);
    EXPECT_EQ(float_to_half(2.0f), 0x4000u);
    EXPECT_EQ(float_to_half(0.5f), 0x3800u);
    EXPECT_FLOAT_EQ(half_to_float(0x3C00u), 1.0f);
    EXPECT_FLOAT_EQ(half_to_float(0x4000u), 2.0f);
    EXPECT_FLOAT_EQ(half_to_float(0xBC00u), -1.0f);  // sign bit
    // 1.0 * 2^-4 must be exactly 0.0625 (this is the rescale factor).
    EXPECT_FLOAT_EQ(half_to_float(float_to_half(1.0f * 0.0625f)), 0.0625f);
}

TEST(FpBits, HalfRoundTripFinite) {
    for (uint32_t bits : std::views::iota(0u, 0x10000u)) {
        uint16_t h = static_cast<uint16_t>(bits);
        uint32_t exp = (h >> 10) & 0x1Fu;
        if (exp == 0x1Fu)
            continue;  // skip inf/nan
        float v = half_to_float(h);
        // Re-encoding an exact fp16 value must reproduce the same bits (zeros may
        // be either +0/-0 → compare values, not bits, for the zero case).
        uint16_t re = float_to_half(v);
        if (v == 0.0f)
            EXPECT_EQ(half_to_float(re), 0.0f) << "bits=" << bits;
        else
            EXPECT_EQ(re, h) << "round-trip changed bits: " << bits << " -> " << re;
    }
}

// THE REGRESSION GUARD: scaling by 0.0625 must be correct across the whole fp16
// range — including denormals and small exponents, which the old bit-shift hack
// got wrong (denormal -> unchanged = 16x too big; exp 1..4 -> zero).
TEST(FpBits, RescaleByOneSixteenthIsCorrect) {
    for (uint32_t bits : std::views::iota(0u, 0x10000u)) {
        uint16_t h = static_cast<uint16_t>(bits);
        uint32_t exp = (h >> 10) & 0x1Fu;
        if (exp == 0x1Fu)
            continue;  // skip inf/nan
        float v = half_to_float(h);
        double ref = static_cast<double>(v) * 0.0625;  // the correct scaled value
        float got = half_to_float(float_to_half(v * 0.0625f));
        double tol = std::max(1e-7, std::fabs(ref) * 1e-3);  // ~fp16 rounding
        EXPECT_NEAR(static_cast<double>(got), ref, tol)
            << "bits=" << bits << " v=" << v << " (the old bit-shift bug fails here)";
    }
}

TEST(FpBits, RescaleDenormalNotLeftUnscaled) {
    // fp16 denormal (biased exp 0): the old code left it UNCHANGED (16x too big).
    uint16_t denorm = 0x0200u;  // 0.5 * 2^-14 = 2^-15
    float v = half_to_float(denorm);
    ASSERT_GT(v, 0.0f);
    float scaled = half_to_float(float_to_half(v * 0.0625f));
    EXPECT_LT(scaled, v);  // must shrink, not stay equal
    EXPECT_NEAR(static_cast<double>(scaled), static_cast<double>(v) * 0.0625, 1e-7);
}

TEST(FpBits, RescaleSmallExponentNotZeroed) {
    // fp16 with biased exp 3 (~2^-12): the old code flushed exp<=4 to zero.
    uint16_t small = static_cast<uint16_t>(3u << 10);  // 2^-12
    float v = half_to_float(small);
    ASSERT_GT(v, 0.0f);
    float scaled = half_to_float(float_to_half(v * 0.0625f));
    EXPECT_GT(scaled, 0.0f);  // must NOT be zeroed
    EXPECT_NEAR(static_cast<double>(scaled), static_cast<double>(v) * 0.0625, 1e-7);
}

TEST(FpBits, Bf16RoundTripAndRescale) {
    EXPECT_FLOAT_EQ(bf16_to_float(float_to_bf16(1.0f)), 1.0f);
    EXPECT_FLOAT_EQ(bf16_to_float(float_to_bf16(-3.5f)), -3.5f);
    // bf16 has 8 exponent bits, so normal weights scale exactly by 2^-4.
    for (float v : {1.0f, 7.0f, 0.013f, -2.5f, 100.0f}) {
        float got = bf16_to_float(float_to_bf16(v * 0.0625f));
        EXPECT_NEAR(static_cast<double>(got), static_cast<double>(v) * 0.0625, std::fabs(v) * 0.01 + 1e-6);
    }
}

// Subnormal halves must renormalise. imp-quantize's FP8 scale reader pasted the
// subnormal mantissa under a normal exponent instead, so 0x0001 came out as
// 6.1e-05 where the value is 5.96e-08, a factor of 1025, on every one of the
// 2046 subnormal patterns, silently rescaling whole weight blocks.
TEST(FpBits, HalfSubnormalsAreExact) {
    int checked = 0;
    for (uint32_t m : std::views::iota(1u, 0x400u)) {
        const uint16_t h = static_cast<uint16_t>(m);  // biased exponent 0
        const double ref = static_cast<double>(m) * 0x1p-24;
        EXPECT_DOUBLE_EQ(static_cast<double>(half_to_float(h)), ref) << "mantissa=" << m;
        EXPECT_DOUBLE_EQ(static_cast<double>(half_to_float(static_cast<uint16_t>(h | 0x8000u))), -ref);
        ++checked;
    }
    EXPECT_EQ(checked, 1023);  // 2046 patterns counting both signs
}

// Narrowing must round to nearest even, not truncate. The round-trip test below
// CANNOT see this: a float that came from a bf16 has 16 zero low bits, so both
// rules agree on it. These inputs have the low bits set, which is the only way
// the rounding rule is reachable at all. (Written after a truncating mutant
// left the round-trip test green.)
TEST(FpBits, Bf16NarrowingRoundsToNearestEven) {
    struct Case {
        uint32_t f_bits;
        uint16_t expect;
        const char* what;
    };
    constexpr Case cases[] = {
        {0x3F818000u, 0x3F82u, "exact tie, odd -> up (truncation gives 3F81)"},
        {0x3F808000u, 0x3F80u, "exact tie, even -> stay"},
        {0x3F80C000u, 0x3F81u, "above the tie -> up (truncation gives 3F80)"},
        {0x3F804000u, 0x3F80u, "below the tie -> stay"},
    };
    for (const auto& c : cases)
        EXPECT_EQ(float_to_bf16(std::bit_cast<float>(c.f_bits)), c.expect) << c.what;
}

// Every bf16 pattern is the top half of a float pattern, so widening then
// narrowing must be the identity on all 65536 of them.
TEST(FpBits, Bf16RoundTripIsExactForEveryPattern) {
    for (uint32_t bits : std::views::iota(0u, 0x10000u)) {
        const uint16_t b = static_cast<uint16_t>(bits);
        const float v = bf16_to_float(b);
        if (std::isnan(v))
            continue;  // narrowing a NaN may pick a different payload
        EXPECT_EQ(float_to_bf16(v), b) << "bf16 round-trip changed bits: " << bits;
    }
}

}  // namespace imp
