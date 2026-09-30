// AWQ GEMM 4-bit dequant (#2205): host reference vs hand-packed tensors, bit-exact FP16.

#include "quant/dequant_awq.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace imp {
namespace {

uint16_t bits_of(__half h) {
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

__half h(float f) { return __float2half_rn(f); }

// AutoAWQ packer (awq/modules/linear/gemm.py from_linear): nibble s of pack j holds column
// j*8 + AWQ_ORDER[s]. Written from the table, not from nibble_shift, so the two can disagree.
std::vector<int32_t> awq_pack(const std::vector<int>& vals, int rows, int cols) {
    static const int kAwqOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<int32_t> out(static_cast<size_t>(rows) * (cols / 8), 0);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < cols / 8; ++j) {
            uint32_t p = 0;
            for (int s = 0; s < 8; ++s)
                p |= static_cast<uint32_t>(vals[r * cols + j * 8 + kAwqOrder[s]] & 0xF) << (4 * s);
            out[r * (cols / 8) + j] = static_cast<int32_t>(p);
        }
    return out;
}

// Literal: 0x76543210 holds nibble s = s, so column c reads [0,4,1,5,2,6,3,7][c].
TEST(DequantAwq, LiteralPackDecodesInAwqOrder) {
    const int32_t qweight[1] = {0x76543210};
    const int32_t qzeros_0[1] = {0};
    const int32_t qzeros_8[1] = {static_cast<int32_t>(0x88888888u)};
    std::vector<__half> s1(8, h(1.0f)), s05(8, h(0.5f));
    const float want_s1[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const float want_s05[8] = {-4.0f, -2.0f, -3.5f, -1.5f, -3.0f, -1.0f, -2.5f, -0.5f};
    __half out[8];

    awq::dequant4_host(out, qweight, qzeros_0, s1.data(), /*N=*/8, /*K=*/1, /*group_size=*/1);
    for (int n = 0; n < 8; ++n)
        EXPECT_EQ(bits_of(out[n]), bits_of(h(want_s1[n]))) << "column " << n;

    awq::dequant4_host(out, qweight, qzeros_8, s05.data(), 8, 1, 1);
    for (int n = 0; n < 8; ++n)
        EXPECT_EQ(bits_of(out[n]), bits_of(h(want_s05[n]))) << "column " << n;

    // Sensitivity: natural nibble order would read column c as c, which differs on 6 of 8.
    int differs = 0;
    for (int n = 0; n < 8; ++n)
        differs += (want_s1[n] != static_cast<float>(n));
    EXPECT_EQ(differs, 6);
}

// K=256 N=32 group 128: every (q, z, s) through the packer, out [N, K] bit-equal to FP16 (q-z)*s.
TEST(DequantAwq, PackedTensorBitExactPerGroupAndColumn) {
    const int K = 256, N = 32, G = 128, groups = K / G;
    std::vector<int> q(K * N), z(groups * N);
    std::vector<__half> scales(groups * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            q[k * N + n] = (k * 7 + n * 3 + k / 5) & 0xF;
    for (int g = 0; g < groups; ++g)
        for (int n = 0; n < N; ++n) {
            z[g * N + n] = (g * 5 + n) & 0xF;
            scales[g * N + n] = h(0.001f * static_cast<float>(1 + g * N + n) + 0.0137f);
        }
    const auto qweight = awq_pack(q, K, N);
    const auto qzeros = awq_pack(z, groups, N);
    std::vector<__half> out(static_cast<size_t>(N) * K);
    awq::dequant4_host(out.data(), qweight.data(), qzeros.data(), scales.data(), N, K, G);

    int mismatches = 0;
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const int g = k / G;
            const __half want =
                __hmul(__int2half_rn(q[k * N + n] - z[g * N + n]), scales[g * N + n]);
            mismatches += bits_of(out[n * K + k]) != bits_of(want);
        }
    EXPECT_EQ(mismatches, 0);
}

TEST(DequantAwq, CheckShapesAcceptsGemmLayoutAndNamesEveryMismatch) {
    const int64_t qw[2] = {256, 4}, qz[2] = {2, 4}, sc[2] = {2, 32};
    int K = 0, N = 0;
    std::string err;
    ASSERT_TRUE(awq::check_shapes(qw, qz, sc, 128, &K, &N, &err)) << err;
    EXPECT_EQ(K, 256);
    EXPECT_EQ(N, 32);

    const int64_t qz_bad[2] = {4, 2}, sc_bad[2] = {2, 16}, qw_bad_k[2] = {200, 4};
    EXPECT_FALSE(awq::check_shapes(qw, qz_bad, sc, 128, &K, &N, &err));
    EXPECT_NE(err.find("qzeros"), std::string::npos) << err;
    EXPECT_FALSE(awq::check_shapes(qw, qz, sc_bad, 128, &K, &N, &err));
    EXPECT_NE(err.find("scales"), std::string::npos) << err;
    EXPECT_FALSE(awq::check_shapes(qw_bad_k, qz, sc, 128, &K, &N, &err));
    EXPECT_NE(err.find("multiple of group_size"), std::string::npos) << err;
    EXPECT_FALSE(awq::check_shapes(qw, qz, sc, 0, &K, &N, &err));
}

}  // namespace
}  // namespace imp
