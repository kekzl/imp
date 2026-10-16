// H-011: each host fp16/bf16/E4M3 copy merged into core/fp_bits.h vs its pre-merge body, bit for bit.
// Inputs: all 2^16 fp16, 2^16 bf16, 2^8 E4M3, 2^16 MXFP8 pairs, 2^24 ngram (e4m3, scale) pairs;
// encoders: every float with exponent field 101..143 plus every 251st pattern outside it.

#include "../tools/imp-quantize/fp8_source.h"
#include "core/fp_bits.h"
#include "model/mxfp8_widen.h"
#include "model/ngram_table.h"
#include "model/weight_upload_traits.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

// ---- pre-merge bodies, verbatim (origin/main 7e75eb8c) ----

// src/model/weight_upload_traits.h wupload::fp16_to_float
float old_wupload_fp16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 1;
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t man = h & 0x3FF;
    float result;
    if (exp == 0) {
        if (man == 0) {
            result = 0.0f;
        } else {
            result = std::ldexp(static_cast<float>(man) / 1024.0f, -14);
        }
    } else if (exp == 31) {
        result = 0.0f;
    } else {
        result = std::ldexp(1.0f + static_cast<float>(man) / 1024.0f, exp - 15);
    }
    return sign ? -result : result;
}

// src/model/weight_upload_traits.h wupload::float_to_fp16
uint16_t old_wupload_float_to_fp16(float val) {
    uint32_t fbits;
    std::memcpy(&fbits, &val, 4);
    uint32_t f_sign = (fbits >> 31) & 1;
    int f_exp = static_cast<int>((fbits >> 23) & 0xFF) - 127;
    uint32_t f_man = fbits & 0x7FFFFF;
    if ((fbits & 0x7FFFFFFF) == 0) {
        return static_cast<uint16_t>(f_sign << 15);
    }
    if (f_exp > 15) {
        return static_cast<uint16_t>((f_sign << 15) | 0x7C00);
    }
    if (f_exp < -24) {
        return static_cast<uint16_t>(f_sign << 15);
    }
    if (f_exp < -14) {
        int shift = -14 - f_exp;
        uint32_t subnormal_man = (0x800000 | f_man) >> (shift + 13);
        return static_cast<uint16_t>((f_sign << 15) | (subnormal_man & 0x3FF));
    }
    uint16_t h_exp = static_cast<uint16_t>(f_exp + 15);
    uint32_t round_bit = (f_man >> 12) & 1;
    uint32_t sticky = f_man & 0xFFF;
    uint16_t h_man = static_cast<uint16_t>(f_man >> 13);
    if (round_bit && (sticky || (h_man & 1))) {
        h_man++;
        if (h_man > 0x3FF) {
            h_man = 0;
            h_exp++;
            if (h_exp > 30) {
                return static_cast<uint16_t>((f_sign << 15) | 0x7C00);
            }
        }
    }
    return static_cast<uint16_t>((f_sign << 15) | (h_exp << 10) | h_man);
}

// src/model/ngram_table.cpp f32_to_f16
uint16_t old_ngram_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu)
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1f)
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)
            return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1u)))
            half_mant++;
        return static_cast<uint16_t>(sign | half_mant);
    }
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
        half++;
    return static_cast<uint16_t>(half);
}

// src/model/ngram_table.cpp fp8_e4m3_to_f32
float old_ngram_fp8_e4m3_to_f32(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 0xf;
    const int mant = b & 7;
    float v;
    if (exp == 0)
        v = std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    else if (exp == 15 && mant == 7)
        v = std::nanf("");
    else
        v = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
    return sign ? -v : v;
}

// src/model/ngram_table.cpp bf16_to_f32; same body as the inline widen loops in
// weight_upload.cpp (MTP FP8 scales, SSM F32 tensors, host experts) and wupload::stage_bf16.
float old_bf16_widen(uint16_t b) {
    const uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// src/model/mxfp8_widen.cpp e4m3_value + mxfp8_to_bf16
float old_mxfp8_e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0xF && m == 0x7)
        return NAN;
    const float mag = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                             : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
    return (b & 0x80) ? -mag : mag;
}
uint16_t old_mxfp8_to_bf16(uint8_t e4m3, uint8_t e8m0) {
    const float v = std::ldexp(old_mxfp8_e4m3_value(e4m3), static_cast<int>(e8m0) - 127);
    return static_cast<uint16_t>(std::bit_cast<uint32_t>(v) >> 16);
}

// tools/imp-quantize/fp8_source.cpp e4m3_to_float
float old_quantize_e4m3_to_float(uint8_t bits) {
    const uint32_t s = (bits >> 7) & 0x1u;
    const uint32_t e = (bits >> 3) & 0xFu;
    const uint32_t m = bits & 0x7u;
    float v;
    if (e == 0) {
        v = std::ldexp(static_cast<float>(m) / 8.0f, -6);
    } else if (e == 0xFu && m == 0x7u) {
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        v = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
    }
    return s ? -v : v;
}

uint32_t fbits(float f) { return std::bit_cast<uint32_t>(f); }

// Calls fn(float) for every float with exponent field in [101, 143] (fp16 flush, subnormal, normal and
// overflow edges, both signs: 43 * 2^24), every 251st pattern outside it, and the 0/Inf/NaN bit patterns.
template <class Fn>
uint64_t for_each_float(Fn&& fn) {
    uint64_t n = 0;
    for (uint32_t sign : {0u, 0x80000000u}) {
        for (uint32_t b = 101u << 23; b < (144u << 23); ++b, ++n)
            fn(std::bit_cast<float>(sign | b));
    }
    for (uint64_t u = 0; u <= 0xFFFFFFFFull; u += 251) {
        const auto b = static_cast<uint32_t>(u);
        const uint32_t ue = (b >> 23) & 0xFFu;
        if (ue >= 101 && ue <= 143)
            continue;
        fn(std::bit_cast<float>(b));
        ++n;
    }
    for (uint32_t b : {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u,
                       0x7F800001u, 0xFFFFFFFFu, 0x00000001u, 0x807FFFFFu}) {
        fn(std::bit_cast<float>(b));
        ++n;
    }
    return n;
}

}  // namespace

namespace imp {

TEST(FpBitsLegacyParity, Fp16DecodeAll65536) {
    uint32_t checked = 0, wup_mismatch = 0, finite_mismatch = 0;
    for (uint32_t i = 0; i < 0x10000u; ++i) {
        const auto h = static_cast<uint16_t>(i);
        ++checked;
        if (fbits(wupload::fp16_to_float(h)) != fbits(old_wupload_fp16_to_float(h)))
            ++wup_mismatch;
        if ((h & 0x7C00u) != 0x7C00u && fbits(half_to_float(h)) != fbits(old_wupload_fp16_to_float(h)))
            ++finite_mismatch;
    }
    std::printf("[ parity ] fp16 decode: %u patterns, wupload %u mismatches, finite %u mismatches\n", checked,
                wup_mismatch, finite_mismatch);
    EXPECT_EQ(checked, 65536u);
    EXPECT_EQ(wup_mismatch, 0u);
    EXPECT_EQ(finite_mismatch, 0u);
    // The Inf/NaN -> signed zero policy is the wrapper's, not fp_bits'.
    EXPECT_EQ(fbits(wupload::fp16_to_float(0x7C00)), 0x00000000u);
    EXPECT_EQ(fbits(wupload::fp16_to_float(0xFE01)), 0x80000000u);
}

TEST(FpBitsLegacyParity, Bf16DecodeAll65536) {
    uint32_t checked = 0, mismatch = 0;
    for (uint32_t i = 0; i < 0x10000u; ++i) {
        const auto b = static_cast<uint16_t>(i);
        ++checked;
        if (fbits(bf16_to_float(b)) != fbits(old_bf16_widen(b)))
            ++mismatch;
    }
    std::printf("[ parity ] bf16 decode: %u patterns, %u mismatches\n", checked, mismatch);
    EXPECT_EQ(checked, 65536u);
    EXPECT_EQ(mismatch, 0u);
}

TEST(FpBitsLegacyParity, E4m3DecodeAll256) {
    uint32_t checked = 0, ngram = 0, mxfp8 = 0, quantize = 0;
    for (uint32_t i = 0; i < 0x100u; ++i) {
        const auto b = static_cast<uint8_t>(i);
        ++checked;
        if (fbits(e4m3_to_float<E4m3Nan::kSigned>(b)) != fbits(old_ngram_fp8_e4m3_to_f32(b)))
            ++ngram;
        if (fbits(e4m3_to_float(b)) != fbits(old_mxfp8_e4m3_value(b)))
            ++mxfp8;
        if (fbits(quantize::e4m3_to_float(b)) != fbits(old_quantize_e4m3_to_float(b)))
            ++quantize;
    }
    std::printf("[ parity ] e4m3 decode: %u patterns, ngram %u, mxfp8 %u, imp-quantize %u mismatches\n",
                checked, ngram, mxfp8, quantize);
    EXPECT_EQ(checked, 256u);
    EXPECT_EQ(ngram, 0u);
    EXPECT_EQ(mxfp8, 0u);
    EXPECT_EQ(quantize, 0u);
    // NaN sign: ngram decoded 0xFF to -qNaN; mxfp8 and imp-quantize to +qNaN.
    EXPECT_EQ(fbits(old_ngram_fp8_e4m3_to_f32(0xFF)), 0xFFC00000u);
    EXPECT_EQ(fbits(old_mxfp8_e4m3_value(0xFF)), 0x7FC00000u);
    EXPECT_EQ(fbits(quantize::e4m3_to_float(0xFF)), 0x7FC00000u);
}

TEST(FpBitsLegacyParity, Mxfp8ToBf16All65536) {
    uint32_t checked = 0, mismatch = 0;
    for (uint32_t q = 0; q < 0x100u; ++q) {
        for (uint32_t s = 0; s < 0x100u; ++s) {
            ++checked;
            if (mxfp8_to_bf16(static_cast<uint8_t>(q), static_cast<uint8_t>(s)) !=
                old_mxfp8_to_bf16(static_cast<uint8_t>(q), static_cast<uint8_t>(s)))
                ++mismatch;
        }
    }
    std::printf("[ parity ] mxfp8_to_bf16: %u pairs, %u mismatches\n", checked, mismatch);
    EXPECT_EQ(checked, 65536u);
    EXPECT_EQ(mismatch, 0u);
}

// The ngram gather element over its whole input domain: 256 E4M3 bytes x 65536 BF16 table scales.
// NaN byte x NaN scale (2 x 254 pairs) skipped: x86 returns the first NaN operand, the compiler picks it.
TEST(FpBitsLegacyParity, NgramGatherElementAll16M) {
    uint32_t checked = 0, skipped = 0, mismatch = 0;
    for (uint32_t s = 0; s < 0x10000u; ++s) {
        const float scale = old_bf16_widen(static_cast<uint16_t>(s));
        for (uint32_t q = 0; q < 0x100u; ++q) {
            const auto b = static_cast<uint8_t>(q);
            if (std::isnan(scale) && (b & 0x7Fu) == 0x7Fu) {
                ++skipped;
                continue;
            }
            ++checked;
            if (ngram_fp8_to_half(b, bf16_to_float(static_cast<uint16_t>(s))) !=
                old_ngram_f32_to_f16(old_ngram_fp8_e4m3_to_f32(b) * scale))
                ++mismatch;
        }
    }
    std::printf(
        "[ parity ] ngram gather element: %u (e4m3, bf16) pairs, %u NaN x NaN skipped, %u mismatches\n",
        checked, skipped, mismatch);
    EXPECT_EQ(checked + skipped, 1u << 24);
    EXPECT_EQ(skipped, 2u * 254u);
    EXPECT_EQ(mismatch, 0u);
}

TEST(FpBitsLegacyParity, Fp16EncodeFloatSweep) {
    uint64_t ngram = 0, wup = 0;
    const uint64_t checked = for_each_float([&](float f) {
        if (float_to_half_as<HalfSubnormal::kNearestEven>(f) != old_ngram_f32_to_f16(f))
            ++ngram;
        if (wupload::float_to_fp16(f) != old_wupload_float_to_fp16(f))
            ++wup;
    });
    std::printf("[ parity ] fp16 encode: %llu floats, ngram %llu, wupload %llu mismatches\n",
                static_cast<unsigned long long>(checked), static_cast<unsigned long long>(ngram),
                static_cast<unsigned long long>(wup));
    EXPECT_GT(checked, 43ull << 24);  // the [101, 143] exponent window alone, both signs
    EXPECT_EQ(ngram, 0u);
    EXPECT_EQ(wup, 0u);
    // NaN -> signed Inf is the wrapper's policy; float_to_half keeps NaN.
    EXPECT_EQ(wupload::float_to_fp16(-std::numeric_limits<float>::quiet_NaN()), 0xFC00u);
}

}  // namespace imp
