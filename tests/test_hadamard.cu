#include <gtest/gtest.h>
#include "compute/hadamard.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <numeric>

namespace imp {
namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)

#define SKIP_IF_NO_CUDA()                                                      \
    do {                                                                        \
        int dev_count = 0;                                                     \
        cudaGetDeviceCount(&dev_count);                                        \
        if (dev_count == 0) GTEST_SKIP() << "No CUDA device";                 \
    } while (0)

// CPU reference: Walsh-Hadamard transform (in-place, unnormalized).
// Then normalize by 1/sqrt(block_size).
static void wht_cpu(float* data, int block_size) {
    for (int stride = 1; stride < block_size; stride <<= 1) {
        for (int i = 0; i < block_size; i++) {
            int partner = i ^ stride;
            if (partner > i) {
                float a = data[i];
                float b = data[partner];
                data[i]       = a + b;
                data[partner] = a - b;
            }
        }
    }
    float norm = 1.0f / sqrtf(static_cast<float>(block_size));
    for (int i = 0; i < block_size; i++) {
        data[i] *= norm;
    }
}

// Compute block-diagonal WHT on CPU for reference.
static void hadamard_ref(const std::vector<float>& input,
                         std::vector<float>& output,
                         int M, int K, int block_size) {
    output = input;
    for (int row = 0; row < M; row++) {
        for (int blk = 0; blk < K / block_size; blk++) {
            wht_cpu(&output[row * K + blk * block_size], block_size);
        }
    }
}

class HadamardTest : public ::testing::TestWithParam<int> {};

TEST_P(HadamardTest, MatchesCPUReference) {
    SKIP_IF_NO_CUDA();
    int block_size = GetParam();
    ASSERT_TRUE(hadamard_block_size_valid(block_size));

    int M = 4;
    int K = block_size * 8;  // multiple blocks per row

    // Generate input
    std::vector<float> h_input(M * K);
    for (int i = 0; i < M * K; i++) {
        h_input[i] = sinf(static_cast<float>(i) * 0.1f);  // smooth, bounded
    }

    // CPU reference
    std::vector<float> h_ref;
    hadamard_ref(h_input, h_ref, M, K, block_size);

    // Convert to FP16 and upload
    std::vector<half> h_fp16(M * K);
    for (int i = 0; i < M * K; i++) {
        h_fp16[i] = __float2half(h_input[i]);
    }

    half* d_input = nullptr;
    half* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_input, h_fp16.data(), M * K * sizeof(half),
                          cudaMemcpyHostToDevice));

    hadamard_transform_fp16(d_input, d_output, M, K, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download and compare
    std::vector<half> h_out_fp16(M * K);
    CUDA_CHECK(cudaMemcpy(h_out_fp16.data(), d_output, M * K * sizeof(half),
                          cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float gpu_val = __half2float(h_out_fp16[i]);
        float ref_val = h_ref[i];
        float err = fabsf(gpu_val - ref_val);
        if (err > max_err) max_err = err;
    }

    // FP16 has ~1e-3 precision for values in [-1, 1] range.
    // WHT can amplify values by sqrt(block_size), so allow proportional tolerance.
    float tol = 0.05f * sqrtf(static_cast<float>(block_size));
    EXPECT_LT(max_err, tol)
        << "block_size=" << block_size << " max_err=" << max_err;

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_P(HadamardTest, InPlaceWorks) {
    SKIP_IF_NO_CUDA();
    int block_size = GetParam();
    int M = 2;
    int K = block_size * 4;

    std::vector<float> h_input(M * K);
    for (int i = 0; i < M * K; i++) {
        h_input[i] = cosf(static_cast<float>(i) * 0.07f);
    }

    std::vector<half> h_fp16(M * K);
    for (int i = 0; i < M * K; i++) h_fp16[i] = __float2half(h_input[i]);

    // Out-of-place
    half* d_buf1 = nullptr;
    half* d_buf2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf1, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_buf2, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_buf1, h_fp16.data(), M * K * sizeof(half),
                          cudaMemcpyHostToDevice));
    hadamard_transform_fp16(d_buf1, d_buf2, M, K, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // In-place
    half* d_inplace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_inplace, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_inplace, h_fp16.data(), M * K * sizeof(half),
                          cudaMemcpyHostToDevice));
    hadamard_transform_fp16(d_inplace, d_inplace, M, K, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare
    std::vector<half> h_out1(M * K), h_out2(M * K);
    CUDA_CHECK(cudaMemcpy(h_out1.data(), d_buf2, M * K * sizeof(half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out2.data(), d_inplace, M * K * sizeof(half),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * K; i++) {
        EXPECT_EQ(__half2float(h_out1[i]), __half2float(h_out2[i]))
            << "Mismatch at index " << i;
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_inplace);
}

TEST_P(HadamardTest, InvolutionProperty) {
    // WHT is its own inverse (up to normalization): H @ H = I * block_size
    // So applying twice should give back the original (since we normalize).
    SKIP_IF_NO_CUDA();
    int block_size = GetParam();
    int M = 2;
    int K = block_size * 4;

    std::vector<float> h_input(M * K);
    for (int i = 0; i < M * K; i++) {
        h_input[i] = static_cast<float>(i % 7) - 3.0f;
    }

    std::vector<half> h_fp16(M * K);
    for (int i = 0; i < M * K; i++) h_fp16[i] = __float2half(h_input[i]);

    half* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_buf, h_fp16.data(), M * K * sizeof(half),
                          cudaMemcpyHostToDevice));

    // Apply twice: H(H(x)) = x
    hadamard_transform_fp16(d_buf, d_buf, M, K, block_size);
    hadamard_transform_fp16(d_buf, d_buf, M, K, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(M * K);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_buf, M * K * sizeof(half),
                          cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float orig = h_input[i];
        float roundtrip = __half2float(h_out[i]);
        float err = fabsf(roundtrip - orig);
        if (err > max_err) max_err = err;
    }

    // Two FP16 roundtrips through WHT should accumulate small error.
    EXPECT_LT(max_err, 0.15f)
        << "block_size=" << block_size << " involution max_err=" << max_err;

    cudaFree(d_buf);
}

INSTANTIATE_TEST_SUITE_P(
    BlockSizes, HadamardTest,
    ::testing::Values(16, 32, 64, 128));

} // namespace
} // namespace imp
