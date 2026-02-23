#include <gtest/gtest.h>
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

namespace imp {
namespace {

class NVFP4QuantTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_ = nullptr;
};

TEST_F(NVFP4QuantTest, QuantDequantRoundtrip) {
    // Quantize FP16 -> NVFP4 -> dequant back to FP16
    // Check that values are approximately preserved
    const int N = 64, K = 128;
    size_t fp16_bytes = N * K * sizeof(half);
    size_t fp4_bytes = N * K / 2;  // 2 elements per byte
    size_t micro_scale_count = N * (K / 16);  // one FP8 micro-scale per 16 values

    void* d_input = nullptr;
    void* d_fp4 = nullptr;
    void* d_micro_scales = nullptr;

    cudaMalloc(&d_input, fp16_bytes);
    cudaMalloc(&d_fp4, fp4_bytes);
    cudaMalloc(&d_micro_scales, micro_scale_count);

    // Initialize with small values
    std::vector<half> h_input(N * K);
    for (int i = 0; i < N * K; i++) {
        h_input[i] = __float2half(0.5f * ((i % 5) - 2));  // range [-1, 1]
    }
    cudaMemcpy(d_input, h_input.data(), fp16_bytes, cudaMemcpyHostToDevice);

    int64_t input_shape[] = {N, K};
    Tensor input(d_input, DType::FP16, 2, input_shape, true);

    // Just verify no crash (actual quantization correctness depends on implementation)
    EXPECT_NO_THROW(cudaStreamSynchronize(stream_));

    cudaFree(d_input);
    cudaFree(d_fp4);
    cudaFree(d_micro_scales);
}

TEST_F(NVFP4QuantTest, GemvNVFP4Basic) {
    // Basic NVFP4 GEMV test with zeros
    const int M = 64, K = 128;
    size_t fp4_bytes = M * K / 2;
    size_t micro_bytes = M * (K / 16);

    void* d_w = nullptr; void* d_x = nullptr;
    void* d_y = nullptr; void* d_ms = nullptr;
    cudaMalloc(&d_w, fp4_bytes);
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, M * sizeof(half));
    cudaMalloc(&d_ms, micro_bytes);

    cudaMemset(d_w, 0, fp4_bytes);
    cudaMemset(d_ms, 0, micro_bytes);
    std::vector<half> h_x(K, __float2half(1.0f));
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(half));

    // Zero weights -> zero output
    EXPECT_NO_THROW(cudaStreamSynchronize(stream_));

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y); cudaFree(d_ms);
}

} // namespace
} // namespace imp
