// GPTQ 4-bit dequant (#2249): host reference vs hand-packed tensors, bit-exact FP16.

#include "quant/dequant_gptq.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace imp {
namespace {

using gptq::ZeroFormat;

uint16_t bits_of(__half h) {
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

__half h(float f) { return __float2half_rn(f); }

// AutoGPTQ qlinear_cuda_old.py pack(): qweight row r, nibble j holds input k = r*8 + j (packed along K).
std::vector<int32_t> pack_qweight(const std::vector<int>& q, int K, int N) {
    std::vector<int32_t> out(static_cast<size_t>(K / 8) * N, 0);
    for (int r = 0; r < K / 8; ++r)
        for (int n = 0; n < N; ++n) {
            uint32_t p = 0;
            for (int j = 0; j < 8; ++j)
                p |= static_cast<uint32_t>(q[(r * 8 + j) * N + n] & 0xF) << (4 * j);
            out[r * N + n] = static_cast<int32_t>(p);
        }
    return out;
}

// qzeros column c, nibble j holds output n = c*8 + j (packed along N); v1 stores z - 1 (line 166).
std::vector<int32_t> pack_qzeros(const std::vector<int>& z, int groups, int N, ZeroFormat fmt) {
    const int sub = fmt == ZeroFormat::V1 ? 1 : 0;
    std::vector<int32_t> out(static_cast<size_t>(groups) * (N / 8), 0);
    for (int g = 0; g < groups; ++g)
        for (int c = 0; c < N / 8; ++c) {
            uint32_t p = 0;
            for (int j = 0; j < 8; ++j)
                p |= static_cast<uint32_t>((z[g * N + c * 8 + j] - sub) & 0xF) << (4 * j);
            out[g * (N / 8) + c] = static_cast<int32_t>(p);
        }
    return out;
}

// Pre-#2249 kernel rule: qzeros read as [groups/8, N] packed along groups, no +1.
__half old_layout_elem(const int32_t* qweight, const int32_t* qzeros, const __half* scales, int N, int G,
                       int k, int n) {
    const int g = k / G;
    const int q = (qweight[(k / 8) * N + n] >> ((k % 8) * 4)) & 0xF;
    const int zp = (qzeros[(g / 8) * N + n] >> ((g % 8) * 4)) & 0xF;
    return __float2half_rn(__half2float(scales[g * N + n]) * static_cast<float>(q - zp));
}

// Literal: qweight 0x76543210 gives q = k; qzeros 0x76543210 gives stored nibble n for output n.
TEST(DequantGptq, LiteralPackDecodesAlongKAndZerosAlongNWithOffset) {
    const int K = 8, N = 8;
    std::vector<int32_t> qweight(N, 0x76543210);
    const int32_t qzeros[1] = {0x76543210};
    const int32_t qzeros_f[1] = {static_cast<int32_t>(0xFFFFFFFFu)};
    std::vector<__half> s1(N, h(1.0f));
    std::vector<__half> out(N * K);

    gptq::dequant4_host(out.data(), qweight.data(), qzeros, s1.data(), nullptr, N, K, K, ZeroFormat::V1);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            EXPECT_EQ(bits_of(out[n * K + k]), bits_of(h(static_cast<float>(k - (n + 1))))) << n << "," << k;

    gptq::dequant4_host(out.data(), qweight.data(), qzeros, s1.data(), nullptr, N, K, K, ZeroFormat::V2);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            EXPECT_EQ(bits_of(out[n * K + k]), bits_of(h(static_cast<float>(k - n)))) << n << "," << k;

    // v1 nibble 0xF: +1 then mask to 4 bits gives z = 0 (qlinear_cuda_old.py:301-304); v2 keeps 15.
    gptq::dequant4_host(out.data(), qweight.data(), qzeros_f, s1.data(), nullptr, N, K, K, ZeroFormat::V1);
    for (int k = 0; k < K; ++k)
        EXPECT_EQ(bits_of(out[3 * K + k]), bits_of(h(static_cast<float>(k))));
    gptq::dequant4_host(out.data(), qweight.data(), qzeros_f, s1.data(), nullptr, N, K, K, ZeroFormat::V2);
    for (int k = 0; k < K; ++k)
        EXPECT_EQ(bits_of(out[3 * K + k]), bits_of(h(static_cast<float>(k - 15))));
}

struct Case {
    int K, N, G, groups;
    std::vector<int> q, z;
    std::vector<__half> scales;
};

// K=256 N=32 group 64: 4 groups, asymmetric zeros that differ per output column and per group.
Case make_case(int zmin) {
    Case c{256, 32, 64, 4, {}, {}, {}};
    c.q.resize(c.K * c.N);
    c.z.resize(c.groups * c.N);
    c.scales.resize(c.groups * c.N);
    for (int k = 0; k < c.K; ++k)
        for (int n = 0; n < c.N; ++n)
            c.q[k * c.N + n] = (k * 7 + n * 3 + k / 5) & 0xF;
    for (int g = 0; g < c.groups; ++g)
        for (int n = 0; n < c.N; ++n) {
            c.z[g * c.N + n] = zmin + (g * 5 + n * 3) % (16 - zmin);
            c.scales[g * c.N + n] = h(0.001f * static_cast<float>(1 + g * c.N + n) + 0.0137f);
        }
    return c;
}

int count_mismatches(const Case& c, const std::vector<__half>& out, const int32_t* g_idx) {
    int mismatches = 0;
    for (int n = 0; n < c.N; ++n)
        for (int k = 0; k < c.K; ++k) {
            const int g = g_idx ? g_idx[k] : k / c.G;
            const __half want = __hmul(__int2half_rn(c.q[k * c.N + n] - c.z[g * c.N + n]),
                                       c.scales[g * c.N + n]);
            mismatches += bits_of(out[n * c.K + k]) != bits_of(want);
        }
    return mismatches;
}

// v1 zeros in [1, 15] (z - 1 fits a nibble), v2 zeros in [0, 15]; out [N, K] bit-equal to FP16 (q - z) * s.
TEST(DequantGptq, PackedTensorBitExactAsymZerosV1AndV2) {
    for (ZeroFormat fmt : {ZeroFormat::V1, ZeroFormat::V2}) {
        const Case c = make_case(fmt == ZeroFormat::V1 ? 1 : 0);
        const auto qweight = pack_qweight(c.q, c.K, c.N);
        const auto qzeros = pack_qzeros(c.z, c.groups, c.N, fmt);
        std::vector<__half> out(static_cast<size_t>(c.N) * c.K);
        gptq::dequant4_host(out.data(), qweight.data(), qzeros.data(), c.scales.data(), nullptr, c.N, c.K,
                            c.G, fmt);
        EXPECT_EQ(count_mismatches(c, out, nullptr), 0) << "fmt " << static_cast<int>(fmt);
    }
}

// desc_act: g_idx maps k to a non-sequential group; zeros and scales follow g_idx, not k / G.
TEST(DequantGptq, GIdxSelectsGroupPerInput) {
    const Case c = make_case(1);
    std::vector<int32_t> g_idx(c.K);
    for (int k = 0; k < c.K; ++k)
        g_idx[k] = (k * 3 + k / 7) % c.groups;
    const auto qweight = pack_qweight(c.q, c.K, c.N);
    const auto qzeros = pack_qzeros(c.z, c.groups, c.N, ZeroFormat::V1);
    std::vector<__half> out(static_cast<size_t>(c.N) * c.K);
    gptq::dequant4_host(out.data(), qweight.data(), qzeros.data(), c.scales.data(), g_idx.data(), c.N, c.K,
                        c.G, ZeroFormat::V1);
    EXPECT_EQ(count_mismatches(c, out, g_idx.data()), 0);
    EXPECT_GT(count_mismatches(c, out, nullptr), 0);
}

// Sensitivity: the pre-#2249 rule on the same v1 tensors is wrong on most elements.
TEST(DequantGptq, OldLayoutRuleDisagreesOnPackedTensor) {
    const Case c = make_case(1);
    const auto qweight = pack_qweight(c.q, c.K, c.N);
    auto qzeros = pack_qzeros(c.z, c.groups, c.N, ZeroFormat::V1);
    qzeros.resize(static_cast<size_t>(c.N) * ((c.groups + 7) / 8), 0);  // old rule reads [groups/8, N]
    std::vector<__half> out(static_cast<size_t>(c.N) * c.K);
    for (int n = 0; n < c.N; ++n)
        for (int k = 0; k < c.K; ++k)
            out[n * c.K + k] = old_layout_elem(qweight.data(), qzeros.data(), c.scales.data(), c.N, c.G, k,
                                               n);
    EXPECT_GT(count_mismatches(c, out, nullptr), c.N * c.K / 2);
}

TEST(DequantGptq, ParseZeroFormatAcceptsV1AndV2Only) {
    ZeroFormat f = ZeroFormat::V2;
    ASSERT_TRUE(gptq::parse_zero_format("", &f));
    EXPECT_EQ(f, ZeroFormat::V1);
    ASSERT_TRUE(gptq::parse_zero_format("GPTQ", &f));
    EXPECT_EQ(f, ZeroFormat::V1);
    ASSERT_TRUE(gptq::parse_zero_format("gptq_v2", &f));
    EXPECT_EQ(f, ZeroFormat::V2);
    for (const char* bad : {"marlin", "bitblas", "gptq_p", "awq_gemm", "qqq"})
        EXPECT_FALSE(gptq::parse_zero_format(bad, &f)) << bad;
}

TEST(DequantGptq, CheckShapesAcceptsAutoGptqLayoutAndNamesEveryMismatch) {
    const int64_t qw[2] = {32, 32}, qz[2] = {4, 4}, sc[2] = {4, 32};
    gptq::Dims d;
    std::string err;
    ASSERT_TRUE(gptq::check_shapes(qw, qz, sc, 64, &d, &err)) << err;
    EXPECT_EQ(d.K, 256);
    EXPECT_EQ(d.N, 32);
    EXPECT_EQ(d.groups, 4);
    EXPECT_EQ(d.group_size, 64);

    // The pre-#2249 qzeros layout [groups/8, N] is refused.
    const int64_t qz_old[2] = {1, 32}, sc_bad[2] = {4, 16}, qw_n7[2] = {32, 28};
    EXPECT_FALSE(gptq::check_shapes(qw, qz_old, sc, 64, &d, &err));
    EXPECT_NE(err.find("qzeros"), std::string::npos) << err;
    EXPECT_FALSE(gptq::check_shapes(qw, qz, sc_bad, 64, &d, &err));
    EXPECT_NE(err.find("scales"), std::string::npos) << err;
    EXPECT_FALSE(gptq::check_shapes(qw_n7, qz, sc, 64, &d, &err));
    EXPECT_NE(err.find("multiple of 8"), std::string::npos) << err;
    EXPECT_FALSE(gptq::check_shapes(qw, qz, sc, 0, &d, &err));

    // group_size -1: one group spanning K.
    const int64_t qz1[2] = {1, 4}, sc1[2] = {1, 32};
    ASSERT_TRUE(gptq::check_shapes(qw, qz1, sc1, -1, &d, &err)) << err;
    EXPECT_EQ(d.groups, 1);
    EXPECT_EQ(d.group_size, 256);

    // K not a multiple of group_size: ceil(K / group_size) groups (AutoGPTQ math.ceil).
    const int64_t qz3[2] = {3, 4}, sc3[2] = {3, 32};
    ASSERT_TRUE(gptq::check_shapes(qw, qz3, sc3, 100, &d, &err)) << err;
    EXPECT_EQ(d.groups, 3);
}

TEST(DequantGptq, CheckGIdxRefusesWrongLengthAndOutOfRange) {
    gptq::Dims d;
    d.K = 16;
    d.groups = 2;
    std::vector<int32_t> g(16, 1);
    std::string err;
    EXPECT_TRUE(gptq::check_g_idx(g.data(), 16, d, &err)) << err;
    EXPECT_FALSE(gptq::check_g_idx(g.data(), 8, d, &err));
    g[5] = 2;
    EXPECT_FALSE(gptq::check_g_idx(g.data(), 16, d, &err));
    EXPECT_NE(err.find("g_idx[5]"), std::string::npos) << err;
}

}  // namespace
}  // namespace imp
