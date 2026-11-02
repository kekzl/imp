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
