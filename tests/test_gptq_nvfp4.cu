// GPTQ for NVFP4 (roadmap row 101, tools/imp-quantize/gptq.cu) against synthetic Hessians.

#include "gptq.h"

#include "core/fp_bits.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace imp {
namespace {

constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// sum over rows of e H e^T, e = W - dequant(packed, micro).
double decoded_loss(const std::vector<float>& w, const gptq::Result& r, const std::vector<float>& H, int N,
                    int K, float tensor_scale) {
    double total = 0.0;
    std::vector<double> e(static_cast<size_t>(K));
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            const uint8_t byte = r.packed[static_cast<size_t>(n) * K / 2 + k / 2];
            const uint8_t code = (k % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            const float s = e4m3_to_float(r.micro[static_cast<size_t>(n) * K / 16 + k / 16]) * tensor_scale;
            const float q = ((code & 8) ? -kE2M1[code & 7] : kE2M1[code & 7]) * s;
            e[k] = w[static_cast<size_t>(n) * K + k] - q;
        }
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                total += e[i] * H[static_cast<size_t>(i) * K + j] * e[j];
    }
    return total;
}

struct Case {
    int N = 48, K = 256;
    std::vector<float> w;
    std::vector<uint16_t> w16;
    float tensor_scale = 0.0f;
    explicit Case(unsigned seed) {
        std::mt19937 g(seed);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        w.resize(static_cast<size_t>(N) * K);
        w16.resize(w.size());
        float amax = 0.0f;
        for (size_t i = 0; i < w.size(); i++) {
            w16[i] = float_to_half(nd(g));
            w[i] = half_to_float(w16[i]);
            amax = std::max(amax, std::fabs(w[i]));
        }
        tensor_scale = amax / (6.0f * 448.0f);
    }
};

// Correlated inputs with a few loud channels, like a post-norm residual.
std::vector<float> correlated_hessian(int K, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int rows = 4 * K;
    std::vector<float> x(static_cast<size_t>(rows) * K);
    for (int r = 0; r < rows; r++) {
        float shared = nd(g);
        for (int k = 0; k < K; k++)
            x[static_cast<size_t>(r) * K + k] = 0.6f * shared + nd(g) * (k % 37 == 0 ? 8.0f : 1.0f);
    }
    std::vector<float> H(static_cast<size_t>(K) * K, 0.0f);
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                H[static_cast<size_t>(i) * K + j] += x[static_cast<size_t>(r) * K + i] *
                                                     x[static_cast<size_t>(r) * K + j] / rows;
    return H;
}

TEST(GptqNvfp4, BeatsRoundToNearestOnCorrelatedInputs) {
    Case c(7);
    const auto H = correlated_hessian(c.K, 11);
    // Damp 0.01: the in-sample fit the algorithm owes; the export default 10 trades it for PPL.
    auto r = gptq::quantize_nvfp4(c.w16.data(), c.N, c.K, H, c.tensor_scale, 0.01f);
    ASSERT_TRUE(r.has_value()) << r.error();
    EXPECT_LT(r->loss_gptq, 0.8 * r->loss_rtn) << "gptq " << r->loss_gptq << " rtn " << r->loss_rtn;
    // The bytes written are the codes the loss was computed from.
    EXPECT_NEAR(decoded_loss(c.w, *r, H, c.N, c.K, c.tensor_scale), r->loss_gptq, 1e-3 * r->loss_gptq);
}

TEST(GptqNvfp4, IdentityHessianIsRoundToNearest) {
    Case c(3);
    std::vector<float> H(static_cast<size_t>(c.K) * c.K, 0.0f);
    for (int i = 0; i < c.K; i++)
        H[static_cast<size_t>(i) * c.K + i] = 1.0f;
    auto r = gptq::quantize_nvfp4(c.w16.data(), c.N, c.K, H, c.tensor_scale);
    ASSERT_TRUE(r.has_value()) << r.error();
    EXPECT_NEAR(r->loss_gptq, r->loss_rtn, 1e-6 * r->loss_rtn) << "no correlation, no error feedback";
}

TEST(GptqNvfp4, LayerOfName) {
    EXPECT_EQ(gptq::layer_of("model.layers.12.mlp.down_proj.weight"), 12);
    EXPECT_EQ(gptq::layer_of("model.language_model.layers.3.self_attn.q_proj.weight"), 3);
    EXPECT_EQ(gptq::layer_of("lm_head.weight"), -1);
}

}  // namespace
}  // namespace imp
