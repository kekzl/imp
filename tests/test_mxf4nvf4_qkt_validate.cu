#include <gtest/gtest.h>
#include "compute/mxf4nvf4_qkt_validate.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>

namespace imp {
namespace {

// E2M1 LUT for FP32 reference. Matches the quant in qkt_mxf4nvf4_kernel.
static float e2m1_ref(float v) {
    float a = fabsf(v);
    int mag = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f)
            + (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
    const float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (v < 0.0f ? -1.0f : 1.0f) * magnitudes[mag];
}

// Helper shared by tests.
static void run_and_compare(const std::vector<half>& h_Q,
                             const std::vector<half>& h_K,
                             std::vector<float>& h_D_out) {
    constexpr int M = 16, K = 64, N = 8;
    half* d_Q = nullptr;
    half* d_K = nullptr;
    float* d_D = nullptr;
    ASSERT_EQ(cudaMalloc(&d_Q, M*K*sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_K, N*K*sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_D, M*N*sizeof(float)), cudaSuccess);
    cudaMemcpy(d_Q, h_Q.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), N*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_D, 0, M*N*sizeof(float));

    bool ok = qkt_mxf4nvf4_validate(d_Q, d_K, d_D, 0);
    cudaDeviceSynchronize();
    ASSERT_TRUE(ok);

    h_D_out.resize(M*N);
    cudaMemcpy(h_D_out.data(), d_D, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_D);
}

TEST(Mxf4nvf4QkTest, UniformInputs_AllOnes) {
    // Simplest test: Q = K = 1.0 everywhere.
    // E2M1(1.0) = 1.0 exactly, so quant is lossless.
    // Expected D[m][n] = sum_k 1.0 * 1.0 = 64 (K=64).
    constexpr int M = 16, K = 64, N = 8;
    std::vector<half> h_Q(M * K, __float2half(1.0f));
    std::vector<half> h_K(N * K, __float2half(1.0f));

    half* d_Q = nullptr;
    half* d_K = nullptr;
    float* d_D = nullptr;
    ASSERT_EQ(cudaMalloc(&d_Q, M*K*sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_K, N*K*sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_D, M*N*sizeof(float)), cudaSuccess);
    cudaMemcpy(d_Q, h_Q.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), N*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_D, 0, M*N*sizeof(float));

    bool ok = qkt_mxf4nvf4_validate(d_Q, d_K, d_D, 0);
    cudaDeviceSynchronize();
    ASSERT_TRUE(ok);

    std::vector<float> h_D(M*N);
    cudaMemcpy(h_D.data(), d_D, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_D);

    // Expected 64.0 everywhere (16 rows × 8 cols). Report deviation.
    int correct = 0, total = M*N;
    float max_err = 0.0f;
    for (int i = 0; i < total; ++i) {
        float err = std::fabs(h_D[i] - 64.0f);
        if (err < 0.5f) correct++;
        max_err = std::max(max_err, err);
    }
    std::printf("[  INFO    ] D[16x8] expected all 64.0; correct=%d/%d max_err=%.3f\n",
                correct, total, max_err);
    std::printf("[  INFO    ] D[0][0..7] = ");
    for (int n = 0; n < N; ++n) std::printf("%.2f ", h_D[n]);
    std::printf("\n");

    EXPECT_EQ(correct, total);
    EXPECT_LT(max_err, 1e-3f);
}

// Row-indicator test: Q[m][:] = m (so row m has values all equal to m).
// K[n][:] = 1. Expected: D[m][n] = m * 64 (K-dim=64).
// This cross-checks row identity is preserved in the A operand layout.
TEST(Mxf4nvf4QkTest, RowIndicator) {
    constexpr int M = 16, K = 64, N = 8;
    std::vector<half> h_Q(M * K), h_K(N * K, __float2half(1.0f));
    for (int m = 0; m < M; ++m) {
        // Values in E2M1 representable range ({0, 0.5, 1, 1.5, 2, 3, 4, 6}).
        // Use m=0..6 (one per magnitude), wrap with scale factor for 7..15.
        // Simpler: map m to a sequence that's E2M1-exact via our quant.
        float v = 0.0f;
        switch (m) {
            case 0: v = 0.0f; break;
            case 1: v = 0.5f; break;
            case 2: v = 1.0f; break;
            case 3: v = 1.5f; break;
            case 4: v = 2.0f; break;
            case 5: v = 3.0f; break;
            case 6: v = 4.0f; break;
            case 7: v = 6.0f; break;
            // Negative half for 8..15
            case 8: v = -0.5f; break;
            case 9: v = -1.0f; break;
            case 10: v = -1.5f; break;
            case 11: v = -2.0f; break;
            case 12: v = -3.0f; break;
            case 13: v = -4.0f; break;
            case 14: v = -6.0f; break;
            case 15: v = 2.0f; break;  // reuse
        }
        for (int k = 0; k < K; ++k) h_Q[m * K + k] = __float2half(v);
    }

    std::vector<float> h_D;
    run_and_compare(h_Q, h_K, h_D);

    // D[m][n] = sum_k Q[m][k] * K[n][k] = Q_row_val * 64.
    const float expected_per_row[16] = {
        0*64.0f, 0.5f*64, 1*64.0f, 1.5f*64, 2*64.0f, 3*64.0f, 4*64.0f, 6*64.0f,
        -0.5f*64, -1*64.0f, -1.5f*64, -2*64.0f, -3*64.0f, -4*64.0f, -6*64.0f, 2*64.0f
    };

    int correct = 0;
    float max_err = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float err = std::fabs(h_D[m * N + n] - expected_per_row[m]);
            if (err < 0.5f) correct++;
            max_err = std::max(max_err, err);
        }
    }
    std::printf("[  INFO    ] RowIndicator: correct=%d/%d max_err=%.3f\n",
                correct, M*N, max_err);
    std::printf("[  INFO    ] D[:][0] rows = ");
    for (int m = 0; m < M; ++m) std::printf("%.1f ", h_D[m * N]);
    std::printf("\n");

    EXPECT_EQ(correct, M*N);
    EXPECT_LT(max_err, 1e-3f);
}

// Col-indicator test: K[n][:] = n-magnitude (representable E2M1). Q = 1.
// Expected: D[m][n] = K_col_val * 64.
TEST(Mxf4nvf4QkTest, ColIndicator) {
    constexpr int M = 16, K = 64, N = 8;
    std::vector<half> h_Q(M * K, __float2half(1.0f));
    std::vector<half> h_K(N * K);
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) h_K[n * K + k] = __float2half(mag[n]);
    }

    std::vector<float> h_D;
    run_and_compare(h_Q, h_K, h_D);

    int correct = 0;
    float max_err = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float expected = mag[n] * 64.0f;
            float err = std::fabs(h_D[m * N + n] - expected);
            if (err < 0.5f) correct++;
            max_err = std::max(max_err, err);
        }
    }
    std::printf("[  INFO    ] ColIndicator: correct=%d/%d max_err=%.3f\n",
                correct, M*N, max_err);
    std::printf("[  INFO    ] D[0][:] cols = ");
    for (int n = 0; n < N; ++n) std::printf("%.1f ", h_D[n]);
    std::printf("\n");

    EXPECT_EQ(correct, M*N);
    EXPECT_LT(max_err, 1e-3f);
}

// Random E2M1 inputs. Both Q and K are drawn from the E2M1 magnitude set
// (scaled so that values stay representable). Compute FP32 reference via
// a plain triple loop with the on-device quant applied to each value.
TEST(Mxf4nvf4QkTest, RandomE2M1_Regression) {
    constexpr int M = 16, K = 64, N = 8;
    std::mt19937 rng(20260424);
    std::uniform_int_distribution<int> mag_idx(0, 7);
    std::bernoulli_distribution sign(0.5);
    const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    std::vector<half> h_Q(M * K), h_K(N * K);
    for (auto& q : h_Q) {
        float v = mags[mag_idx(rng)] * (sign(rng) ? 1.0f : -1.0f);
        q = __float2half(v);
    }
    for (auto& k : h_K) {
        float v = mags[mag_idx(rng)] * (sign(rng) ? 1.0f : -1.0f);
        k = __float2half(v);
    }

    std::vector<float> h_D;
    run_and_compare(h_Q, h_K, h_D);

    // FP32 reference: each Q/K element is already E2M1-exact, so the reference
    // is the plain matmul with those values.
    std::vector<float> ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(h_Q[m * K + k]) * __half2float(h_K[n * K + k]);
            }
            ref[m * N + n] = acc;
        }
    }

    int correct = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = std::fabs(h_D[i] - ref[i]);
        max_err = std::max(max_err, err);
        // FP32 accumulator, lossless E2M1 input — exact agreement expected.
        if (err < 1e-3f) correct++;
    }
    std::printf("[  INFO    ] RandomE2M1: correct=%d/%d max_err=%.4f\n",
                correct, M * N, max_err);

    EXPECT_EQ(correct, M * N);
    EXPECT_LT(max_err, 1e-3f);
}

} // namespace
} // namespace imp
