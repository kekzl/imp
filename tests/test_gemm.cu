#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/gemm.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <numeric>

namespace imp {
namespace {

// ---- GPU tensor helpers (same pattern as other test files) ----

Tensor make_gpu_tensor(const float* host_data, DType dtype,
                       std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    if (dtype == DType::FP32) {
        cudaMemcpy(t.data, host_data, t.nbytes(), cudaMemcpyHostToDevice);
    } else if (dtype == DType::FP16) {
        std::vector<half> h(t.numel());
        for (int64_t j = 0; j < t.numel(); j++)
            h[j] = __float2half(host_data[j]);
        cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    }
    return t;
}

Tensor alloc_gpu_tensor(DType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

std::vector<float> read_gpu_fp32(const Tensor& t) {
    std::vector<float> result(t.numel());
    if (t.dtype == DType::FP32) {
        cudaMemcpy(result.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    } else if (t.dtype == DType::FP16) {
        std::vector<half> h(t.numel());
        cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
        for (int64_t j = 0; j < t.numel(); j++)
            result[j] = __half2float(h[j]);
    }
    return result;
}

void free_gpu(Tensor& t) {
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
}

// ---- CPU reference: C = alpha * A @ B^T + beta * C ----
// A [M, K], B [N, K], C [M, N]
void cpu_gemm(const float* A, const float* B, float* C,
              int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k]; // B^T
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// =========================================================================
// Test fixture: init/cleanup cuBLAS once
// =========================================================================
class GemmTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { gemm_init(); }
    static void TearDownTestSuite() { gemm_cleanup(); }
};

// =========================================================================
// FP32 GEMM tests
// =========================================================================

TEST_F(GemmTest, FP32_Square) {
    constexpr int N = 4;
    // A = [[1,2],[3,4],[5,6],[7,8]] but let's use square 4x4
    std::vector<float> h_A(N * N), h_B(N * N);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(i + 1) * 0.1f;
        h_B[i] = static_cast<float>(N * N - i) * 0.1f;
    }

    std::vector<float> h_C(N * N, 0.0f);
    cpu_gemm(h_A.data(), h_B.data(), h_C.data(), N, N, N, 1.0f, 0.0f);

    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP32, {N, N});
    Tensor d_B = make_gpu_tensor(h_B.data(), DType::FP32, {N, N});
    Tensor d_C = alloc_gpu_tensor(DType::FP32, {N, N});

    gemm(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < N * N; i++) {
        EXPECT_NEAR(result[i], h_C[i], 1e-3f) << "FP32 square mismatch at " << i;
    }

    free_gpu(d_A); free_gpu(d_B); free_gpu(d_C);
}

TEST_F(GemmTest, FP32_NonSquare) {
    constexpr int M = 3, N = 5, K = 4;
    std::vector<float> h_A(M * K), h_B(N * K);
    for (int i = 0; i < M * K; i++) h_A[i] = sinf(static_cast<float>(i) * 0.3f);
    for (int i = 0; i < N * K; i++) h_B[i] = cosf(static_cast<float>(i) * 0.2f);

    std::vector<float> h_C(M * N, 0.0f);
    cpu_gemm(h_A.data(), h_B.data(), h_C.data(), M, N, K, 1.0f, 0.0f);

    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP32, {M, K});
    Tensor d_B = make_gpu_tensor(h_B.data(), DType::FP32, {N, K});
    Tensor d_C = alloc_gpu_tensor(DType::FP32, {M, N});

    gemm(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(result[i], h_C[i], 1e-4f) << "FP32 non-square mismatch at " << i;
    }

    free_gpu(d_A); free_gpu(d_B); free_gpu(d_C);
}

TEST_F(GemmTest, FP32_AlphaBeta) {
    constexpr int M = 2, N = 3, K = 4;
    std::vector<float> h_A(M * K), h_B(N * K), h_C_init(M * N);
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < N * K; i++) h_B[i] = static_cast<float>(i + 1) * 0.5f;
    for (int i = 0; i < M * N; i++) h_C_init[i] = 10.0f;

    std::vector<float> h_C = h_C_init;
    cpu_gemm(h_A.data(), h_B.data(), h_C.data(), M, N, K, 2.0f, 0.5f);

    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP32, {M, K});
    Tensor d_B = make_gpu_tensor(h_B.data(), DType::FP32, {N, K});
    Tensor d_C = make_gpu_tensor(h_C_init.data(), DType::FP32, {M, N});

    gemm(d_A, d_B, d_C, 2.0f, 0.5f);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(result[i], h_C[i], 1e-3f) << "Alpha/beta mismatch at " << i;
    }

    free_gpu(d_A); free_gpu(d_B); free_gpu(d_C);
}

TEST_F(GemmTest, FP32_Identity) {
    constexpr int N = 4;
    // Identity matrix as B
    std::vector<float> h_A(N * N), h_I(N * N, 0.0f);
    for (int i = 0; i < N * N; i++) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < N; i++) h_I[i * N + i] = 1.0f;

    // C = A @ I^T = A (identity is symmetric)
    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP32, {N, N});
    Tensor d_I = make_gpu_tensor(h_I.data(), DType::FP32, {N, N});
    Tensor d_C = alloc_gpu_tensor(DType::FP32, {N, N});

    gemm(d_A, d_I, d_C);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < N * N; i++) {
        EXPECT_NEAR(result[i], h_A[i], 1e-5f) << "Identity mismatch at " << i;
    }

    free_gpu(d_A); free_gpu(d_I); free_gpu(d_C);
}

// =========================================================================
// FP16 GEMM tests
// =========================================================================

TEST_F(GemmTest, FP16_Square) {
    constexpr int N = 4;
    std::vector<float> h_A(N * N), h_B(N * N);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(i + 1) * 0.1f;
        h_B[i] = static_cast<float>(N * N - i) * 0.1f;
    }

    std::vector<float> h_C(N * N, 0.0f);
    cpu_gemm(h_A.data(), h_B.data(), h_C.data(), N, N, N, 1.0f, 0.0f);

    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP16, {N, N});
    Tensor d_B = make_gpu_tensor(h_B.data(), DType::FP16, {N, N});
    Tensor d_C = alloc_gpu_tensor(DType::FP16, {N, N});

    gemm(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < N * N; i++) {
        EXPECT_NEAR(result[i], h_C[i], 5e-2f) << "FP16 square mismatch at " << i;
    }

    free_gpu(d_A); free_gpu(d_B); free_gpu(d_C);
}

TEST_F(GemmTest, FP16_Large) {
    constexpr int M = 32, N = 64, K = 128;
    std::vector<float> h_A(M * K), h_B(N * K);
    for (int i = 0; i < M * K; i++) h_A[i] = sinf(static_cast<float>(i) * 0.01f);
    for (int i = 0; i < N * K; i++) h_B[i] = cosf(static_cast<float>(i) * 0.01f);

    std::vector<float> h_C(M * N, 0.0f);
    cpu_gemm(h_A.data(), h_B.data(), h_C.data(), M, N, K, 1.0f, 0.0f);

    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP16, {M, K});
    Tensor d_B = make_gpu_tensor(h_B.data(), DType::FP16, {N, K});
    Tensor d_C = alloc_gpu_tensor(DType::FP16, {M, N});

    gemm(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(result[i], h_C[i], 0.5f) << "FP16 large mismatch at " << i;
    }

    free_gpu(d_A); free_gpu(d_B); free_gpu(d_C);
}

// =========================================================================
// GEMV tests (M=1 fast path)
// =========================================================================

TEST_F(GemmTest, GEMV_FP16) {
    constexpr int N = 8, K = 16;
    std::vector<float> h_W(N * K), h_x(K);
    for (int i = 0; i < N * K; i++) h_W[i] = static_cast<float>(i % 7) * 0.1f;
    for (int i = 0; i < K; i++) h_x[i] = static_cast<float>(i + 1) * 0.1f;

    // CPU reference: y[i] = sum_k W[i][k] * x[k]
    std::vector<float> h_y(N, 0.0f);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < K; k++)
            h_y[i] += h_W[i * K + k] * h_x[k];

    Tensor d_W = make_gpu_tensor(h_W.data(), DType::FP16, {N, K});
    Tensor d_x = make_gpu_tensor(h_x.data(), DType::FP16, {K});
    Tensor d_y = alloc_gpu_tensor(DType::FP16, {N});

    gemv(d_W, d_x, d_y);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_y);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(result[i], h_y[i], 0.1f) << "GEMV FP16 mismatch at " << i;
    }

    free_gpu(d_W); free_gpu(d_x); free_gpu(d_y);
}

TEST_F(GemmTest, GEMV_FP32) {
    constexpr int N = 16, K = 32;
    std::vector<float> h_W(N * K), h_x(K);
    for (int i = 0; i < N * K; i++) h_W[i] = sinf(static_cast<float>(i) * 0.1f);
    for (int i = 0; i < K; i++) h_x[i] = cosf(static_cast<float>(i) * 0.2f);

    std::vector<float> h_y(N, 0.0f);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < K; k++)
            h_y[i] += h_W[i * K + k] * h_x[k];

    Tensor d_W = make_gpu_tensor(h_W.data(), DType::FP32, {N, K});
    Tensor d_x = make_gpu_tensor(h_x.data(), DType::FP32, {K});
    Tensor d_y = alloc_gpu_tensor(DType::FP32, {N});

    gemv(d_W, d_x, d_y);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_y);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(result[i], h_y[i], 1e-4f) << "GEMV FP32 mismatch at " << i;
    }

    free_gpu(d_W); free_gpu(d_x); free_gpu(d_y);
}

// =========================================================================
// GEMM via gemm() dispatching to GEMV when M=1
// =========================================================================

TEST_F(GemmTest, GemmDispatchToGEMV) {
    // When A is [1, K], gemm() should dispatch to GEMV fast path
    constexpr int N = 16, K = 32;
    std::vector<float> h_A(K), h_B(N * K);
    for (int i = 0; i < K; i++) h_A[i] = static_cast<float>(i + 1) * 0.01f;
    for (int i = 0; i < N * K; i++) h_B[i] = static_cast<float>(i % 11) * 0.1f;

    std::vector<float> h_C(N, 0.0f);
    cpu_gemm(h_A.data(), h_B.data(), h_C.data(), 1, N, K, 1.0f, 0.0f);

    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP16, {1, K});
    Tensor d_B = make_gpu_tensor(h_B.data(), DType::FP16, {N, K});
    Tensor d_C = alloc_gpu_tensor(DType::FP16, {1, N});

    gemm(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(result[i], h_C[i], 0.2f) << "Dispatch-to-GEMV mismatch at " << i;
    }

    free_gpu(d_A); free_gpu(d_B); free_gpu(d_C);
}

// =========================================================================
// gemv_gate_fp32: FP16 weights, FP16 input, FP32 output
// =========================================================================

TEST_F(GemmTest, GemvGateFP32) {
    constexpr int M = 8, K = 64;
    std::vector<float> h_W(M * K), h_x(K);
    for (int i = 0; i < M * K; i++) h_W[i] = static_cast<float>(i % 5) * 0.05f;
    for (int i = 0; i < K; i++) h_x[i] = static_cast<float>(i + 1) * 0.01f;

    std::vector<float> h_y(M, 0.0f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            h_y[i] += h_W[i * K + k] * h_x[k];

    // Upload as FP16
    std::vector<half> h_W_fp16(M * K), h_x_fp16(K);
    for (int i = 0; i < M * K; i++) h_W_fp16[i] = __float2half(h_W[i]);
    for (int i = 0; i < K; i++) h_x_fp16[i] = __float2half(h_x[i]);

    half* d_W = nullptr;
    half* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_W, M * K * sizeof(half));
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMemcpy(d_W, h_W_fp16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x_fp16.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(float));

    gemv_gate_fp32(d_W, d_x, d_y, M, K);
    cudaDeviceSynchronize();

    std::vector<float> result(M);
    cudaMemcpy(result.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        EXPECT_NEAR(result[i], h_y[i], 0.1f) << "gemv_gate_fp32 mismatch at " << i;
    }

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y);
}

// =========================================================================
// Zero matrix
// =========================================================================

TEST_F(GemmTest, ZeroMatrix) {
    constexpr int M = 4, N = 4, K = 4;
    std::vector<float> h_A(M * K, 0.0f), h_B(N * K, 1.0f);

    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP32, {M, K});
    Tensor d_B = make_gpu_tensor(h_B.data(), DType::FP32, {N, K});
    Tensor d_C = alloc_gpu_tensor(DType::FP32, {M, N});

    gemm(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_C);
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(result[i], 0.0f, 1e-6f) << "Zero matrix should give zero output at " << i;
    }

    free_gpu(d_A); free_gpu(d_B); free_gpu(d_C);
}

} // namespace
} // namespace imp
