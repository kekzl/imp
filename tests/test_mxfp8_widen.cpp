// MXFP8 weights widened to BF16 at load (#2475).
#include "model/mxfp8_widen.h"

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <bit>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

float bf16_to_float(uint16_t b) { return std::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

}  // namespace

TEST(Mxfp8Widen, ElementDecodeIsExact) {
    EXPECT_EQ(bf16_to_float(imp::mxfp8_to_bf16(0x38, 127)), 1.0f);     // e=7 m=0
    EXPECT_EQ(bf16_to_float(imp::mxfp8_to_bf16(0x7E, 127)), 448.0f);   // max normal
    EXPECT_EQ(bf16_to_float(imp::mxfp8_to_bf16(0x01, 127)), 0x1p-9f);  // min subnormal
    EXPECT_EQ(bf16_to_float(imp::mxfp8_to_bf16(0xBC, 130)), -12.0f);   // -1.5 * 2^3
    EXPECT_EQ(bf16_to_float(imp::mxfp8_to_bf16(0x38, 117)), 0x1p-10f);
}

TEST(Mxfp8Widen, PairBecomesBf16AndTheScaleIsConsumed) {
    std::vector<uint8_t> q(2 * 64, 0x38), s = {127, 128, 126, 127};
    q[33] = 0xC0;  // -2.0 in row 0, block 1 (scale 2^1)
    const int64_t ws[2] = {2, 64}, ss[2] = {2, 2};
    std::unordered_map<std::string, imp::Tensor> map;
    map["l.out_proj.weight"] = imp::Tensor(q.data(), imp::QType::FP8_E4M3, 2, ws, false);
    map["l.out_proj.weight_scale"] = imp::Tensor(s.data(), imp::QType::INT8, 2, ss, false);
    std::vector<void*> owned;
    EXPECT_EQ(imp::widen_mxfp8_pairs(map, owned), 1);
    ASSERT_EQ(owned.size(), 1u);
    EXPECT_EQ(map.count("l.out_proj.weight_scale"), 0u);
    const auto& w = map.at("l.out_proj.weight");
    EXPECT_EQ(w.qtype, imp::QType::BF16);
    const auto* d = static_cast<const uint16_t*>(w.data);
    EXPECT_EQ(bf16_to_float(d[0]), 1.0f);
    EXPECT_EQ(bf16_to_float(d[32]), 2.0f);
    EXPECT_EQ(bf16_to_float(d[33]), -4.0f);
    EXPECT_EQ(bf16_to_float(d[64]), 0.5f);
    EXPECT_EQ(bf16_to_float(d[127]), 1.0f);
    for (void* p : owned)
        std::free(p);
}

TEST(Mxfp8Widen, NvFp4AndNanScalesAreNotWidened) {
    // NVFP4 weight_scale: F8_E4M3 [N, K/16], not an E8M0 grid.
    std::vector<uint8_t> q(64, 0x38), s4(4, 0x38), sn = {255, 127};
    const int64_t ws[2] = {1, 64}, s4s[2] = {1, 4}, sns[2] = {1, 2};
    std::unordered_map<std::string, imp::Tensor> map;
    map["a.weight"] = imp::Tensor(q.data(), imp::QType::FP8_E4M3, 2, ws, false);
    map["a.weight_scale"] = imp::Tensor(s4.data(), imp::QType::FP8_E4M3, 2, s4s, false);
    std::vector<void*> owned;
    EXPECT_EQ(imp::widen_mxfp8_pairs(map, owned), 0);
    map["a.weight_scale"] = imp::Tensor(sn.data(), imp::QType::INT8, 2, sns, false);
    EXPECT_THROW(imp::widen_mxfp8_pairs(map, owned), std::runtime_error);
    for (void* p : owned)
        std::free(p);
}

TEST(Mxfp8Widen, QuantizeRoundTripStaysWithinE4M3Precision) {
    // Two rows x 64: a wide-range block, a tiny block, an all-zero block, a block at 448 * 2^4.
    std::vector<float> f(128);
    for (int i = 0; i < 32; ++i)
        f[i] = (i % 2 ? -1.0f : 1.0f) * 0.01f * static_cast<float>(i * i + 1);
    for (int i = 32; i < 64; ++i)
        f[i] = 1e-4f * static_cast<float>(i - 31);
    for (int i = 96; i < 128; ++i)
        f[i] = 7168.0f * static_cast<float>(i - 95) / 32.0f;
    std::vector<uint16_t> h(f.size());
    for (size_t i = 0; i < f.size(); ++i)
        h[i] = __half_as_ushort(__float2half(f[i]));
    std::vector<uint8_t> q(f.size()), s(4);
    imp::mxfp8_quantize(h.data(), 2, 64, q.data(), s.data());
    EXPECT_EQ(s[2], 127);  // all-zero block: scale 1
    EXPECT_EQ(s[3], 131);  // amax 7168 = 448 * 2^4
    for (size_t i = 0; i < f.size(); ++i) {
        const float ref = __half2float(__ushort_as_half(h[i]));
        const float got = bf16_to_float(imp::mxfp8_to_bf16(q[i], s[i / 32]));
        EXPECT_LE(std::fabs(got - ref), std::fabs(ref) * 0.0625f + 1e-9f) << i;
    }
}
