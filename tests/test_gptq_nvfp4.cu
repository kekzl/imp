// GPTQ for NVFP4 (roadmap row 101, tools/imp-quantize/gptq.cu) against synthetic Hessians.

#include "gptq.h"

#include "core/fp_bits.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
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

// U H U^T = I (H^-1 = U^T U), U upper with a positive diagonal. K 300 / 513: partial last 128-block.
TEST(GptqNvfp4, UpperCholeskyOfInverse) {
    for (const int K : {300, 513}) {
        auto H = correlated_hessian(K, 5);
        double mean_diag = 0.0;
        for (int i = 0; i < K; i++)
            mean_diag += H[static_cast<size_t>(i) * K + i] / K;
        for (int i = 0; i < K; i++)
            H[static_cast<size_t>(i) * K + i] += static_cast<float>(0.01 * mean_diag);
        const size_t kk = static_cast<size_t>(K) * K;
        float* d = nullptr;
        ASSERT_EQ(cudaMalloc(&d, kk * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d, H.data(), kk * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        const auto r = gptq::upper_cholesky_of_inverse(d, K);
        std::vector<float> a(kk);
        ASSERT_EQ(cudaMemcpy(a.data(), d, kk * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaFree(d), cudaSuccess);
        ASSERT_TRUE(r.has_value()) << r.error();
        auto U = [&](int i, int j) {
            return i <= j ? static_cast<double>(a[static_cast<size_t>(j) * K + i]) : 0.0;
        };
        for (int i = 0; i < K; i++)
            ASSERT_GT(U(i, i), 0.0) << "K " << K << " i " << i;
        // UH = U * H, then (UH) U^T; H is symmetric, row-major = column-major.
        std::vector<double> uh(kk, 0.0);
        for (int i = 0; i < K; i++)
            for (int k = i; k < K; k++) {
                const double u = U(i, k);
                for (int j = 0; j < K; j++)
                    uh[static_cast<size_t>(i) * K + j] += u * H[static_cast<size_t>(k) * K + j];
            }
        double worst = 0.0;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                double s = 0.0;
                for (int k = j; k < K; k++)
                    s += uh[static_cast<size_t>(i) * K + k] * U(j, k);
                worst = std::max(worst, std::fabs(s - (i == j ? 1.0 : 0.0)));
            }
        EXPECT_LT(worst, 1e-3) << "K " << K;
    }
}

std::vector<float> dequant(const gptq::Result& r, int N, int K, float tensor_scale) {
    std::vector<float> q(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
            const uint8_t byte = r.packed[static_cast<size_t>(n) * K / 2 + k / 2];
            const uint8_t code = (k % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            const float s = e4m3_to_float(r.micro[static_cast<size_t>(n) * K / 16 + k / 16]) * tensor_scale;
            q[static_cast<size_t>(n) * K + k] = ((code & 8) ? -kE2M1[code & 7] : kE2M1[code & 7]) * s;
        }
    return q;
}

// GPTAQ (arXiv 2504.02692): inputs x, a coarse-grid copy xq standing in for upstream quantization
// error. Target sum_t |W x_t - Wq xq_t|^2: the asymmetric term must beat GPTQ on H = E[xq xq^T].
TEST(GptqNvfp4, AsymmetricTermTracksTheFullPrecisionOutput) {
    Case c(5);
    const int K = c.K, N = c.N, rows = 4 * K;
    std::mt19937 g(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(rows) * K), xq(x.size());
    for (int r = 0; r < rows; r++) {
        const float shared = nd(g);
        for (int k = 0; k < K; k++) {
            const size_t p = static_cast<size_t>(r) * K + k;
            x[p] = 0.6f * shared + nd(g) * (k % 37 == 0 ? 8.0f : 1.0f);
            xq[p] = 0.75f * std::round(x[p] / 0.75f);
        }
    }
    const size_t kk = static_cast<size_t>(K) * K;
    std::vector<double> H(kk, 0.0), C(kk, 0.0);
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                H[static_cast<size_t>(i) * K + j] += xq[static_cast<size_t>(r) * K + i] *
                                                     xq[static_cast<size_t>(r) * K + j];
                C[static_cast<size_t>(i) * K + j] += x[static_cast<size_t>(r) * K + i] *
                                                     xq[static_cast<size_t>(r) * K + j];
            }
    std::vector<float> Hf(kk), dxx(kk);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) {
            const size_t ij = static_cast<size_t>(i) * K + j;
            Hf[ij] = static_cast<float>(H[ij] / rows);
            dxx[static_cast<size_t>(j) * K + i] = static_cast<float>((C[ij] - H[ij]) / rows);
        }
    auto out_err = [&](const gptq::Result& r) {
        const auto q = dequant(r, N, K, c.tensor_scale);
        double total = 0.0;
        for (int n = 0; n < N; n++)
            for (int t = 0; t < rows; t++) {
                double y = 0.0, yq = 0.0;
                for (int k = 0; k < K; k++) {
                    y += static_cast<double>(c.w[static_cast<size_t>(n) * K + k]) *
                         x[static_cast<size_t>(t) * K + k];
                    yq += static_cast<double>(q[static_cast<size_t>(n) * K + k]) *
                          xq[static_cast<size_t>(t) * K + k];
                }
                total += (y - yq) * (y - yq);
            }
        return total;
    };
    auto gq = gptq::quantize_nvfp4(c.w16.data(), N, K, Hf, c.tensor_scale, 0.01f);
    ASSERT_TRUE(gq.has_value()) << gq.error();
    auto ga0 = gptq::quantize_nvfp4(c.w16.data(), N, K, Hf, c.tensor_scale, 0.01f, dxx, 0.0f);
    ASSERT_TRUE(ga0.has_value()) << ga0.error();
    EXPECT_EQ(ga0->packed, gq->packed) << "alpha 0 is GPTQ";
    auto ga = gptq::quantize_nvfp4(c.w16.data(), N, K, Hf, c.tensor_scale, 0.01f, dxx, 1.0f);
    ASSERT_TRUE(ga.has_value()) << ga.error();
    const double e_gptq = out_err(*gq), e_gptaq = out_err(*ga);
    EXPECT_LT(e_gptaq, 0.9 * e_gptq) << "gptaq " << e_gptaq << " gptq " << e_gptq;
}

TEST(GptqNvfp4, LayerOfName) {
    EXPECT_EQ(gptq::layer_of("model.layers.12.mlp.down_proj.weight"), 12);
    EXPECT_EQ(gptq::layer_of("model.language_model.layers.3.self_attn.q_proj.weight"), 3);
    EXPECT_EQ(gptq::layer_of("lm_head.weight"), -1);
}

}  // namespace
}  // namespace imp
