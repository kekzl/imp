// kv_cache.fp8_scale selection (src/core/kv_fp8_scale.h) and checkpoint scale tensors
// (src/model/kv_scale_tensor.h). GPU round trip: tests/test_fp8_kv_static_scale.cu.

#include "core/kv_fp8_scale.h"
#include "model/kv_scale_tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace imp {
namespace {

std::vector<std::string> split_dots(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, '.'))
        out.push_back(part);
    return out;
}

TEST(KvFp8Scale, ParseMode) {
    EXPECT_EQ(parse_fp8_kv_scale_mode("legacy"), Fp8KvScaleMode::Legacy);
    EXPECT_EQ(parse_fp8_kv_scale_mode("checkpoint"), Fp8KvScaleMode::Checkpoint);
    EXPECT_EQ(parse_fp8_kv_scale_mode("calibrate"), Fp8KvScaleMode::Calibrate);
    EXPECT_FALSE(parse_fp8_kv_scale_mode("").has_value());
    EXPECT_FALSE(parse_fp8_kv_scale_mode("Legacy").has_value());
    EXPECT_FALSE(parse_fp8_kv_scale_mode("auto").has_value());
}

TEST(KvFp8Scale, Pow2Ceil) {
    EXPECT_EQ(fp8_kv_pow2_ceil(1.0f), 1.0f);
    EXPECT_EQ(fp8_kv_pow2_ceil(0.75f), 1.0f);
    EXPECT_EQ(fp8_kv_pow2_ceil(1.0001f), 2.0f);
    EXPECT_EQ(fp8_kv_pow2_ceil(0.5f), 0.5f);
    EXPECT_EQ(fp8_kv_pow2_ceil(0.098f), 0.125f);
    EXPECT_EQ(fp8_kv_pow2_ceil(0.0f), 1.0f);
    EXPECT_EQ(fp8_kv_pow2_ceil(-3.0f), 1.0f);
    EXPECT_EQ(fp8_kv_pow2_ceil(std::numeric_limits<float>::quiet_NaN()), 1.0f);
    EXPECT_EQ(fp8_kv_pow2_ceil(std::numeric_limits<float>::infinity()), 1.0f);
    EXPECT_EQ(fp8_kv_pow2_ceil(1e-9f), std::ldexp(1.0f, kFp8KvScaleMinExp));
    EXPECT_EQ(fp8_kv_pow2_ceil(1e9f), std::ldexp(1.0f, kFp8KvScaleMaxExp));
}

TEST(KvFp8Scale, FromCheckpoint) {
    // Every local ModelOpt FP8-KV export ships k_scale = v_scale = 1.0 (Qwen3-8B-FP8, Qwen3-14B-NVFP4).
    EXPECT_EQ(fp8_kv_scale_from_checkpoint(1.0f, 1.0f), 1.0f);
    EXPECT_EQ(fp8_kv_scale_from_checkpoint(0.0f, 0.0f), 1.0f);
    EXPECT_EQ(fp8_kv_scale_from_checkpoint(0.3f, 0.1f), 0.5f);
    EXPECT_EQ(fp8_kv_scale_from_checkpoint(0.0f, 0.2f), 0.25f);
    EXPECT_EQ(fp8_kv_scale_from_checkpoint(1.5f, 0.2f), 2.0f);
}

TEST(KvFp8Scale, FromAmax) {
    // Qwen3-1.7B per-layer ranges (docs/DETERMINISM.md): K 22..394, V 1.2..398.
    EXPECT_EQ(fp8_kv_scale_from_amax(22.0f, 1.2f, 2.0f), 0.125f);
    EXPECT_EQ(fp8_kv_scale_from_amax(394.0f, 398.0f, 2.0f), 2.0f);
    EXPECT_EQ(fp8_kv_scale_from_amax(1.2f, 1.2f, 2.0f), 0.0078125f);
    EXPECT_EQ(fp8_kv_scale_from_amax(0.0f, 0.0f, 2.0f), 1.0f);
    EXPECT_EQ(fp8_kv_scale_from_amax(224.0f, 0.0f, 1.0f), 0.5f);
    EXPECT_EQ(fp8_kv_scale_from_amax(448.0f, 0.0f, 0.5f), 1.0f);  // headroom < 1 acts as 1
    EXPECT_EQ(fp8_kv_scale_from_amax(448.0f, 0.0f, std::numeric_limits<float>::quiet_NaN()), 1.0f);
}

// No saturation inside the headroom, and never more than 2x looser than needed (unclamped range).
TEST(KvFp8Scale, FromAmaxBracketsTheHeadroom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> log_amax(-6.0f, 12.0f);
    for (int i = 0; i < 2000; ++i) {
        const float amax = std::exp2(log_amax(rng));
        for (float h : {1.0f, 2.0f, 4.0f}) {
            const float s = fp8_kv_scale_from_amax(amax, 0.0f, h);
            const double need = static_cast<double>(amax) * h / kFp8KvE4M3Max;
            ASSERT_GE(static_cast<double>(s), need) << "amax " << amax << " h " << h;
            ASSERT_LT(static_cast<double>(s), 2.0 * need) << "amax " << amax << " h " << h;
            int e = 0;
            ASSERT_EQ(std::frexp(s, &e), 0.5f) << "not a power of two: " << s;
        }
    }
}

// kv_cache.fp8_scale_split: Gemma-4-26B calibration K amax 0.94-1.78, V 15.2-22.0 (docs/DETERMINISM.md).
TEST(KvFp8Scale, SplitFromAmax) {
    const Fp8KvScalePair shared = fp8_kv_scales_from_amax(1.6f, 16.0f, 2.0f, false);
    EXPECT_EQ(shared.k, 0.125f);  // 16 * 2 / 448 -> 2^-3 for both
    EXPECT_EQ(shared.v, 0.125f);
    const Fp8KvScalePair split = fp8_kv_scales_from_amax(1.6f, 16.0f, 2.0f, true);
    EXPECT_EQ(split.k, 0.0078125f);  // 1.6 * 2 / 448 -> 2^-7
    EXPECT_EQ(split.v, 0.125f);
    const Fp8KvScalePair k_only = fp8_kv_scales_from_amax(1.6f, 0.0f, 2.0f, true);
    EXPECT_EQ(k_only.v, 1.0f);  // unwritten side keeps 1.0
    EXPECT_EQ(fp8_kv_scales_from_amax(22.0f, 1.2f, 2.0f, false), (Fp8KvScalePair{0.125f, 0.125f}));
    EXPECT_EQ(fp8_kv_scales_from_amax(22.0f, 1.2f, 2.0f, true), (Fp8KvScalePair{0.125f, 0.0078125f}));
}

TEST(KvFp8Scale, SplitFromCheckpoint) {
    EXPECT_EQ(fp8_kv_scales_from_checkpoint(1.0f, 1.0f, true), (Fp8KvScalePair{1.0f, 1.0f}));
    EXPECT_EQ(fp8_kv_scales_from_checkpoint(0.3f, 0.1f, false), (Fp8KvScalePair{0.5f, 0.5f}));
    EXPECT_EQ(fp8_kv_scales_from_checkpoint(0.3f, 0.1f, true), (Fp8KvScalePair{0.5f, 0.125f}));
    EXPECT_EQ(fp8_kv_scales_from_checkpoint(0.0f, 0.2f, true), (Fp8KvScalePair{1.0f, 0.25f}));
    EXPECT_EQ(fp8_kv_scales_from_checkpoint(0.0f, 0.0f, true), (Fp8KvScalePair{1.0f, 1.0f}));
}

// The decode kernels form logits with softmax_scale * kv_scale and weight V with kv_scale: the fold must
// reproduce softmax_scale * k and v exactly, and leave a shared pair untouched (legacy bit-equal).
TEST(KvFp8Scale, DecodeFoldIsExact) {
    for (float sm : {0.0883883476f, 0.0625f, 1.0f, 0.0441941738f}) {
        for (int ek = -15; ek <= 15; ++ek) {
            for (int ev = -15; ev <= 15; ++ev) {
                const Fp8KvScalePair s{std::ldexp(1.0f, ek), std::ldexp(1.0f, ev)};
                const Fp8KvDecodeScales d = fp8_kv_decode_scales(sm, s);
                ASSERT_EQ(d.kv_scale, s.v);
                ASSERT_EQ(d.softmax_scale * d.kv_scale, sm * s.k) << "ek " << ek << " ev " << ev;
            }
        }
        const Fp8KvDecodeScales same = fp8_kv_decode_scales(sm, {0.3f, 0.3f});
        EXPECT_EQ(same.softmax_scale, sm);
        EXPECT_EQ(same.kv_scale, 0.3f);
    }
}

float half_bits_to_float(uint16_t b) {
    const int sign = (b >> 15) ? -1 : 1;
    const int e = (b >> 10) & 0x1F;
    const int m = b & 0x3FF;
    if (e == 0)
        return static_cast<float>(sign) * std::ldexp(static_cast<float>(m), -24);
    if (e == 0x1F)
        return m ? std::numeric_limits<float>::quiet_NaN()
                 : static_cast<float>(sign) * std::numeric_limits<float>::infinity();
    return static_cast<float>(sign) * std::ldexp(1.0f + (static_cast<float>(m) / 1024.0f), e - 15);
}

TEST(KvFp8Scale, HistBinIsFloorLog2OfEveryHalf) {
    EXPECT_EQ(fp8_kv_hist_bin(0x0000), 0);
    EXPECT_EQ(fp8_kv_hist_bin(0x8000), 0);   // -0
    EXPECT_EQ(fp8_kv_hist_bin(0x0001), 1);   // 2^-24
    EXPECT_EQ(fp8_kv_hist_bin(0x3C00), 25);  // 1.0
    EXPECT_EQ(fp8_kv_hist_bin(0xBC00), 25);  // -1.0
    EXPECT_EQ(fp8_kv_hist_bin(0x7BFF), 40);  // 65504
    EXPECT_EQ(fp8_kv_hist_bin(0x7C00), kFp8KvHistBins - 1);
    for (uint32_t b = 0; b < 0x10000u; ++b) {
        const float x = half_bits_to_float(static_cast<uint16_t>(b));
        if (x == 0.0f || !std::isfinite(x))
            continue;
        ASSERT_EQ(fp8_kv_hist_bin(static_cast<uint16_t>(b)) - 25, std::ilogb(x)) << std::hex << b;
    }
}

// Histogram edges against a direct count of |x / s| over random values and every power-of-two scale.
TEST(KvFp8Scale, EdgeCountsMatchDirectCount) {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> bits_dist(0, 0xFFFF);
    std::vector<uint16_t> xs(20000);
    for (auto& b : xs)
        b = static_cast<uint16_t>(bits_dist(rng));
    xs[0] = 0x0000;
    xs[1] = 0x7C00;  // inf: saturates
    std::vector<uint32_t> hist(kFp8KvHistBins, 0);
    for (uint16_t b : xs)
        if (((b >> 10) & 0x1F) != 0x1F || (b & 0x3FF) == 0)  // NaN excluded from both sides
            ++hist[fp8_kv_hist_bin(b)];
    for (int j = kFp8KvScaleMinExp; j <= kFp8KvScaleMaxExp; ++j) {
        const float s = std::ldexp(1.0f, j);
        Fp8KvEdgeCounts want;
        for (uint16_t b : xs) {
            const double x = std::fabs(static_cast<double>(half_bits_to_float(b)));
            if (std::isnan(x) || x == 0.0)
                continue;
            const double r = x / s;
            ++want.nonzero;
            want.flushed += r < std::ldexp(1.0, -10) ? 1 : 0;
            want.subnormal += (r >= std::ldexp(1.0, -10) && r < std::ldexp(1.0, -6)) ? 1 : 0;
            want.saturated += r >= 512.0 ? 1 : 0;
        }
        const Fp8KvEdgeCounts got = fp8_kv_edge_counts(hist.data(), s);
        ASSERT_EQ(got.nonzero, want.nonzero) << "j " << j;
        ASSERT_EQ(got.flushed, want.flushed) << "j " << j;
        ASSERT_EQ(got.subnormal, want.subnormal) << "j " << j;
        ASSERT_EQ(got.saturated, want.saturated) << "j " << j;
    }
}

TEST(KvScaleTensor, ParseNames) {
    auto p = [](const std::string& n) { return parse_kv_scale_tensor_name(split_dots(n)); };
    EXPECT_EQ(p("model.layers.3.self_attn.k_proj.k_scale"), KvScaleSlot::K);  // ModelOpt
    EXPECT_EQ(p("model.layers.3.self_attn.v_proj.v_scale"), KvScaleSlot::V);
    EXPECT_EQ(p("model.layers.0.self_attn.k_scale"), KvScaleSlot::K);  // compressed-tensors
    EXPECT_EQ(p("model.layers.0.self_attn.v_scale"), KvScaleSlot::V);
    EXPECT_EQ(p("model.layers.7.self_attn.attn.v_scale"), KvScaleSlot::V);
    EXPECT_EQ(p("model.layers.7.self_attn.kv_scale"), KvScaleSlot::KV);
    EXPECT_EQ(p("model.layers.7.self_attn.qkv_proj.kv_scale"), KvScaleSlot::KV);
    EXPECT_FALSE(p("model.layers.3.self_attn.k_proj.weight_scale").has_value());
    EXPECT_FALSE(p("model.layers.3.self_attn.k_proj.input_scale").has_value());
    EXPECT_FALSE(p("model.layers.3.self_attn.o_proj.k_scale").has_value());
    EXPECT_FALSE(p("model.layers.3.mlp.k_scale").has_value());
    EXPECT_FALSE(p("model.layers.3.self_attn.k_proj.k_scale.extra").has_value());
    EXPECT_FALSE(p("lm_head.k_scale").has_value());
}

template <typename T>
Tensor host_scalar(T* v, QType q, int64_t n = 1, bool on_device = false) {
    const int64_t shape[1] = {n};
    return Tensor(v, q, 1, shape, on_device);
}

TEST(KvScaleTensor, ReadHostScalar) {
    float f = 0.25f;
    EXPECT_EQ(read_host_positive_scalar(host_scalar(&f, QType::F32)), 0.25f);
    const int64_t no_shape[1] = {0};
    EXPECT_EQ(read_host_positive_scalar(Tensor(&f, QType::F32, 0, no_shape, false)), 0.25f);  // shape []
    uint16_t bf = 0x3E80;                                                                     // 0.25
    EXPECT_EQ(read_host_positive_scalar(host_scalar(&bf, QType::BF16)), 0.25f);
    uint16_t h = 0x3400;  // 0.25
    EXPECT_EQ(read_host_positive_scalar(host_scalar(&h, QType::F16)), 0.25f);
    uint16_t h_sub = 0x0001;  // 2^-24
    EXPECT_EQ(read_host_positive_scalar(host_scalar(&h_sub, QType::F16)), std::ldexp(1.0f, -24));
    uint16_t h_big = 0x5C00;  // 256
    EXPECT_EQ(read_host_positive_scalar(host_scalar(&h_big, QType::F16)), 256.0f);

    float neg = -1.0f;
    float zero = 0.0f;
    float nan = std::numeric_limits<float>::quiet_NaN();
    float two[2] = {1.0f, 1.0f};
    uint8_t fp8 = 0x38;
    EXPECT_FALSE(read_host_positive_scalar(host_scalar(&neg, QType::F32)).has_value());
    EXPECT_FALSE(read_host_positive_scalar(host_scalar(&zero, QType::F32)).has_value());
    EXPECT_FALSE(read_host_positive_scalar(host_scalar(&nan, QType::F32)).has_value());
    EXPECT_FALSE(read_host_positive_scalar(host_scalar(two, QType::F32, 2)).has_value());
    EXPECT_FALSE(read_host_positive_scalar(host_scalar(&f, QType::F32, 1, true)).has_value());
    EXPECT_FALSE(read_host_positive_scalar(host_scalar(&fp8, QType::FP8_E4M3)).has_value());
    EXPECT_FALSE(read_host_positive_scalar(Tensor{}).has_value());
}

TEST(KvScaleTensor, MatchedStepRoutesKAndV) {
    float val = 0.5f;
    const Tensor t = host_scalar(&val, QType::F32);
    float k = 0.0f;
    float v = 0.0f;
    const std::string kn = "model.layers.1.self_attn.k_proj.k_scale";
    EXPECT_TRUE(kv_scale_matched(true, split_dots(kn), t, kn, k, v));  // already matched: untouched
    EXPECT_EQ(k, 0.0f);
    EXPECT_TRUE(kv_scale_matched(false, split_dots(kn), t, kn, k, v));
    EXPECT_EQ(k, 0.5f);
    EXPECT_EQ(v, 0.0f);
    const std::string both = "model.layers.1.self_attn.kv_scale";
    val = 0.25f;
    EXPECT_TRUE(kv_scale_matched(false, split_dots(both), t, both, k, v));
    EXPECT_EQ(k, 0.25f);
    EXPECT_EQ(v, 0.25f);
    const std::string other = "model.layers.1.self_attn.k_proj.weight";
    EXPECT_FALSE(kv_scale_matched(false, split_dots(other), t, other, k, v));
    float bad = -2.0f;
    const std::string vn = "model.layers.1.self_attn.v_scale";
    EXPECT_TRUE(kv_scale_matched(false, split_dots(vn), host_scalar(&bad, QType::F32), vn, k, v));
    EXPECT_EQ(v, 0.25f);  // unreadable: dropped, previous value kept
}

}  // namespace
}  // namespace imp
