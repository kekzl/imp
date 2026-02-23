#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

namespace imp {
namespace {

class FP8GemmTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_ = nullptr;
};

TEST_F(FP8GemmTest, GemmCublasLtFP16) {
    // Test cuBLASLt GEMM with FP16 operands
    const int M = 32, N = 64, K = 128;
    size_t a_bytes = M * K * sizeof(half);
    size_t b_bytes = N * K * sizeof(half);
    size_t c_bytes = M * N * sizeof(half);

    void* d_a = nullptr; void* d_b = nullptr; void* d_c = nullptr;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    // Initialize with small values
    std::vector<half> h_a(M * K, __float2half(0.01f));
    std::vector<half> h_b(N * K, __float2half(0.01f));
    cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, c_bytes);

    int64_t a_shape[] = {M, K};
    int64_t b_shape[] = {N, K};
    int64_t c_shape[] = {M, N};
    Tensor A(d_a, DType::FP16, 2, a_shape, true);
    Tensor B(d_b, DType::FP16, 2, b_shape, true);
    Tensor C(d_c, DType::FP16, 2, c_shape, true);

    gemm_cublaslt(A, B, C, 1.0f, 0.0f, nullptr, nullptr, stream_);
    cudaStreamSynchronize(stream_);

    // Verify: C = A @ B^T, each element should be K * 0.01 * 0.01 = 0.0128
    std::vector<half> h_c(M * N);
    cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost);
    float expected = K * 0.01f * 0.01f;
    float actual = __half2float(h_c[0]);
    EXPECT_NEAR(actual, expected, 0.01f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

TEST_F(FP8GemmTest, GemvFP8Basic) {
    // Test FP8 GEMV: y = A_fp8 @ x_fp16
    const int M = 64, K = 128;
    float scale = 1.0f;

    void* d_a = nullptr; void* d_x = nullptr; void* d_y = nullptr;
    cudaMalloc(&d_a, M * K);  // FP8: 1 byte per element
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, M * sizeof(half));

    // Initialize A with zeros (FP8 zero = 0x00)
    cudaMemset(d_a, 0, M * K);
    std::vector<half> h_x(K, __float2half(1.0f));
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(half));

    int64_t a_shape[] = {M, K};
    int64_t x_shape[] = {K};
    int64_t y_shape[] = {M};
    Tensor A(d_a, DType::FP8_E4M3, 2, a_shape, true);
    Tensor x(d_x, DType::FP16, 1, x_shape, true);
    Tensor y(d_y, DType::FP16, 1, y_shape, true);

    gemv_fp8(A, x, y, scale, stream_);
    cudaStreamSynchronize(stream_);

    // All zeros in A -> y should be all zeros
    std::vector<half> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(half), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(__half2float(h_y[0]), 0.0f, 0.001f);

    cudaFree(d_a); cudaFree(d_x); cudaFree(d_y);
}

} // namespace
} // namespace imp
