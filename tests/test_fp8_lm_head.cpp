// FP8 E4M3 per-row LM head (gemm.nvfp4_lm_head=fp8, #2156): the shared quantizer
// (quant/fp8_row_quant.h, also run by quantize_fp8_rows_kernel) against an independent
// bit-field E4M3 reference with round-to-nearest-even, and the config value.

#include "core/config/lm_head_mode.h"
#include "quant/fp8_row_quant.h"
#include "runtime/config.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace imp {
namespace {

// Reference decode from the E4M3 bit fields: bias 7, subnormal 2^-6 * m/8, 0x7F/0xFF = NaN.
double ref_decode(uint8_t c) {
    const int s = c >> 7, e = (c >> 3) & 0xF, m = c & 7;
    if (e == 15 && m == 7)
        return std::numeric_limits<double>::quiet_NaN();
    const double mag = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return s ? -mag : mag;
}

// Reference encode: nearest finite code, ties to the even mantissa, saturate at 448.
uint8_t ref_encode(double v) {
    const uint8_t sign = std::signbit(v) ? 0x80 : 0x00;
    const double a = std::fabs(v);
    uint8_t best = 0;
    double best_err = std::numeric_limits<double>::infinity();
    for (int c = 0; c <= 0x7E; ++c) {
        const double err = std::fabs(ref_decode(static_cast<uint8_t>(c)) - a);
        if (err < best_err || (err == best_err && (c & 1) == 0 && (best & 1) != 0)) {
            best = static_cast<uint8_t>(c);
            best_err = err;
        }
    }
    return static_cast<uint8_t>(sign | best);
}

bool same_code(uint8_t a, uint8_t b) {
    return a == b || ((a & 0x7F) == 0 && (b & 0x7F) == 0);  // +0 / -0
}

uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

TEST(Fp8LmHead, DecodeMatchesBitFieldReferenceForAllCodes) {
    for (int c = 0; c < 256; ++c) {
        const double ref = ref_decode(static_cast<uint8_t>(c));
        const float got = fp8_rows::decode(static_cast<uint8_t>(c));
        if (std::isnan(ref)) {
            EXPECT_TRUE(std::isnan(got)) << "code " << c;
            continue;
        }
        EXPECT_EQ(static_cast<double>(got), ref) << "code " << c;
    }
    EXPECT_EQ(fp8_rows::decode(0x7E), 448.0f);
    EXPECT_EQ(fp8_rows::decode(0x01), std::ldexp(1.0f, -9));
}

TEST(Fp8LmHead, EncodeIsRoundToNearestEvenWithSaturation) {
    // Every exact midpoint between adjacent finite codes: ties must go to the even code.
    int ties = 0;
    for (int c = 0; c < 0x7E; ++c) {
        const double mid = 0.5 *
                           (ref_decode(static_cast<uint8_t>(c)) + ref_decode(static_cast<uint8_t>(c + 1)));
        const uint8_t even = (c & 1) == 0 ? static_cast<uint8_t>(c) : static_cast<uint8_t>(c + 1);
        EXPECT_EQ(fp8_rows::encode(static_cast<float>(mid), 1.0f), even) << "tie above code " << c;
        EXPECT_EQ(fp8_rows::encode(static_cast<float>(-mid), 1.0f), even | 0x80) << "tie above code " << c;
        ++ties;
    }
    EXPECT_EQ(ties, 126);
    // Log-uniform samples over [2^-12, 2^10] (subnormal, normal and saturating range).
    uint32_t s = 7u;
    for (int i = 0; i < 200000; ++i) {
        const double mag = std::exp2(-12.0 + 22.0 * (lcg(s) / 16777216.0));
        const float v = static_cast<float>((lcg(s) & 1) ? -mag : mag);
        const uint8_t got = fp8_rows::encode(v, 1.0f);
        const uint8_t ref = ref_encode(v);
        ASSERT_TRUE(same_code(got, ref)) << "v=" << v << " got=" << int(got) << " ref=" << int(ref);
    }
    EXPECT_EQ(fp8_rows::encode(1.0e6f, 1.0f), 0x7E) << "saturates to 448, never NaN";
    EXPECT_EQ(fp8_rows::encode(-1.0e6f, 1.0f), 0xFE);
}

// Synthetic head: vocab rows with 5 decades of row magnitude, per-row scale = absmax / 448.
TEST(Fp8LmHead, PerRowRoundTripOnSyntheticHead) {
    constexpr int kRows = 96, kCols = 512;
    std::vector<float> w(static_cast<size_t>(kRows) * kCols);
    uint32_t s = 17u;
    for (int r = 0; r < kRows; ++r) {
        const float row_mag = std::pow(10.0f, -3.0f + 5.0f * r / (kRows - 1));
        for (int k = 0; k < kCols; ++k)
            w[static_cast<size_t>(r) * kCols + k] = row_mag * (lcg(s) / 16777216.0f - 0.5f);
    }
    for (int k = 0; k < kCols; ++k)
        w[static_cast<size_t>(kRows - 1) * kCols + k] = 0.0f;  // all-zero row

    for (int r = 0; r < kRows; ++r) {
        const float* row = &w[static_cast<size_t>(r) * kCols];
        float amax = 0.0f;
        for (int k = 0; k < kCols; ++k)
            amax = std::fmax(amax, std::fabs(row[k]));
        const float scale = fp8_rows::row_scale(amax);
        const float inv = 1.0f / scale;
        if (amax == 0.0f) {
            EXPECT_EQ(scale, 1.0f);
        }
        for (int k = 0; k < kCols; ++k) {
            const uint8_t c = fp8_rows::encode(row[k], inv);
            ASSERT_TRUE(same_code(c, ref_encode(static_cast<double>(row[k] * inv))))
                << "row " << r << " k " << k;
            const double back = static_cast<double>(fp8_rows::decode(c)) * scale;
            // Half an E4M3 step: 2^-4 relative for normals, 2^-10 * scale absolute in the subnormal band.
            const double bound = std::fmax(std::fabs(row[k]) * 0.0625,
                                           std::ldexp(static_cast<double>(scale), -10));
            EXPECT_LE(std::fabs(back - row[k]), bound * (1.0 + 1e-6)) << "row " << r << " k " << k;
            if (std::fabs(row[k]) == amax && amax > 0.0f) {
                EXPECT_EQ(c & 0x7F, 0x7E) << "row absmax must land on 448, row " << r;
            }
            // Re-quantizing the dequantized value with the stored scale is exact.
            EXPECT_TRUE(same_code(fp8_rows::encode(fp8_rows::decode(c) * scale, inv), c)) << "row " << r;
        }
    }
}

TEST(Fp8LmHead, ConfigValueSelectsFp8AndKeepsTheOtherSpellings) {
    RuntimeConfig cfg;
    EXPECT_EQ(cfg.gemm.nvfp4_lm_head, "auto");
    EXPECT_EQ(lm_head_mode(cfg.gemm.nvfp4_lm_head), LmHeadMode::Auto);
    // #2166: auto builds the FP8 head from a 16-bit source; on = NVFP4.
    EXPECT_TRUE(lm_head_mode_fp8(LmHeadMode::Auto, QType::F16));
    EXPECT_TRUE(lm_head_mode_fp8(LmHeadMode::Fp8, QType::F16));
    EXPECT_FALSE(lm_head_mode_fp8(LmHeadMode::Nvfp4, QType::F16));
    EXPECT_FALSE(lm_head_mode_fp8(LmHeadMode::Source, QType::F16));
    ASSERT_TRUE(cfg.apply_overrides({"gemm.nvfp4_lm_head=fp8"}).empty());
    EXPECT_EQ(cfg.gemm.nvfp4_lm_head, "fp8");
    EXPECT_EQ(lm_head_mode(cfg.gemm.nvfp4_lm_head), LmHeadMode::Fp8);
    const std::pair<const char*, LmHeadMode> cases[] = {
        {"auto", LmHeadMode::Auto},  {"on", LmHeadMode::Nvfp4},     {"off", LmHeadMode::Source},
        {"true", LmHeadMode::Nvfp4}, {"false", LmHeadMode::Source}, {"1", LmHeadMode::Nvfp4},
        {"0", LmHeadMode::Source},   {"yes", LmHeadMode::Nvfp4},    {"no", LmHeadMode::Source},
    };
    for (const auto& [v, mode] : cases) {
        RuntimeConfig c;
        ASSERT_TRUE(c.apply_overrides({std::string("gemm.nvfp4_lm_head=") + v}).empty()) << v;
        EXPECT_EQ(lm_head_mode(c.gemm.nvfp4_lm_head), mode) << v;
    }
    RuntimeConfig bad;
    EXPECT_EQ(bad.apply_overrides({"gemm.nvfp4_lm_head=fp16"}).size(), 1u) << "unknown value is rejected";
}

// #2224: auto takes the FP8 head only from a 16-bit source; an 8-bit quantized head keeps checkpoint
// precision (E4M3 flipped Qwen3-8B-Q8_0's top-1); narrower heads keep the #982 NVFP4 rule.
TEST(Fp8LmHead, AutoPicksTheHeadBySourceWidth) {
    for (QType q : {QType::F16, QType::BF16, QType::F32}) {
        EXPECT_TRUE(lm_head_mode_fp8(LmHeadMode::Auto, q)) << qtype_name(q);
        EXPECT_FALSE(lm_head_auto_keeps_source(LmHeadMode::Auto, q)) << qtype_name(q);
    }
    for (QType q : {QType::Q8_0, QType::Q8_1, QType::Q8_K, QType::FP8_E4M3, QType::FP8_E5M2}) {
        EXPECT_FALSE(lm_head_mode_fp8(LmHeadMode::Auto, q)) << qtype_name(q);
        EXPECT_TRUE(lm_head_auto_keeps_source(LmHeadMode::Auto, q)) << qtype_name(q);
        EXPECT_TRUE(lm_head_mode_fp8(LmHeadMode::Fp8, q))
            << "explicit fp8 still builds it, " << qtype_name(q);
        EXPECT_FALSE(lm_head_auto_keeps_source(LmHeadMode::Nvfp4, q)) << "on still forces NVFP4";
    }
    for (QType q : {QType::Q4_K, QType::Q6_K, QType::Q4_0, QType::IQ4_XS, QType::MXFP4}) {
        EXPECT_FALSE(lm_head_mode_fp8(LmHeadMode::Auto, q)) << qtype_name(q);
        EXPECT_FALSE(lm_head_auto_keeps_source(LmHeadMode::Auto, q))
            << "#982 rule decides, " << qtype_name(q);
    }
}

}  // namespace
}  // namespace imp
