#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/reduce.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <algorithm>

namespace imp {
namespace {

// ---- GPU tensor helpers ----

Tensor make_gpu_tensor(const float* host_data, DType dtype,
                       int ndim, const int64_t* shape) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) t.shape[i] = shape[i];
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

Tensor alloc_gpu_tensor(DType dtype, int ndim, const int64_t* shape) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) t.shape[i] = shape[i];
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

std::vector<float> read_gpu_fp32(const Tensor& t) {
    std::vector<float> result(t.numel());
    cudaMemcpy(result.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    return result;
}

void free_gpu(Tensor& t) {
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
}

// =========================================================================
// reduce_sum tests
// =========================================================================

TEST(ReduceTest, SumLastDimFP32) {
    // Input: [3, 4], reduce dim=1 → output: [3]
    constexpr int rows = 3, cols = 4;
    std::vector<float> h_in = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    std::vector<float> expected = {10.0f, 26.0f, 42.0f};

    int64_t in_shape[] = {rows, cols};
    int64_t out_shape[] = {rows};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_sum(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < rows; i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-5f)
            << "SumLastDim mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, SumLastDimFP16) {
    constexpr int rows = 4, cols = 8;
    std::vector<float> h_in(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = static_cast<float>(i + 1) * 0.1f;

    std::vector<float> expected(rows, 0.0f);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            expected[r] += h_in[r * cols + c];

    int64_t in_shape[] = {rows, cols};
    int64_t out_shape[] = {rows};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP16, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_sum(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < rows; i++) {
        EXPECT_NEAR(result[i], expected[i], 0.1f)
            << "SumLastDim FP16 mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, SumFirstDim) {
    // Input: [3, 4], reduce dim=0 → output: [4]
    std::vector<float> h_in = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    // Sum along rows: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
    std::vector<float> expected = {15.0f, 18.0f, 21.0f, 24.0f};

    int64_t in_shape[] = {3, 4};
    int64_t out_shape[] = {4};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_sum(d_in, d_out, 0);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-5f)
            << "SumFirstDim mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, SumSingleElement) {
    std::vector<float> h_in = {42.0f};
    int64_t in_shape[] = {1, 1};
    int64_t out_shape[] = {1};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_sum(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    EXPECT_NEAR(result[0], 42.0f, 1e-6f);

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, SumLargeRow) {
    constexpr int rows = 4, cols = 1024;
    std::vector<float> h_in(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = sinf(static_cast<float>(i) * 0.001f);

    std::vector<float> expected(rows, 0.0f);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            expected[r] += h_in[r * cols + c];

    int64_t in_shape[] = {rows, cols};
    int64_t out_shape[] = {rows};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_sum(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < rows; i++) {
        EXPECT_NEAR(result[i], expected[i], 0.01f)
            << "SumLargeRow mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, SumNegativeDim) {
    // dim=-1 should be equivalent to dim=1 for a 2D tensor
    constexpr int rows = 2, cols = 3;
    std::vector<float> h_in = {1, 2, 3, 4, 5, 6};
    std::vector<float> expected = {6.0f, 15.0f};

    int64_t in_shape[] = {rows, cols};
    int64_t out_shape[] = {rows};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_sum(d_in, d_out, -1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < rows; i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-5f)
            << "SumNegativeDim mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

// =========================================================================
// reduce_sum 3D
// =========================================================================

TEST(ReduceTest, Sum3DMiddleDim) {
    // Input: [2, 3, 4], reduce dim=1 → output: [2, 4]
    constexpr int d0 = 2, d1 = 3, d2 = 4;
    std::vector<float> h_in(d0 * d1 * d2);
    for (int i = 0; i < d0 * d1 * d2; i++)
        h_in[i] = static_cast<float>(i + 1);

    // CPU reference: sum along dim=1
    std::vector<float> expected(d0 * d2, 0.0f);
    for (int i = 0; i < d0; i++)
        for (int j = 0; j < d1; j++)
            for (int k = 0; k < d2; k++)
                expected[i * d2 + k] += h_in[i * d1 * d2 + j * d2 + k];

    int64_t in_shape[] = {d0, d1, d2};
    int64_t out_shape[] = {d0, d2};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 3, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 2, out_shape);

    reduce_sum(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < d0 * d2; i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-4f)
            << "Sum3DMiddleDim mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

// =========================================================================
// reduce_max tests
// =========================================================================

TEST(ReduceTest, MaxLastDimFP32) {
    constexpr int rows = 3, cols = 4;
    std::vector<float> h_in = {
        3, 1, 4, 1,
        5, 9, 2, 6,
        5, 3, 5, 8
    };
    std::vector<float> expected = {4.0f, 9.0f, 8.0f};

    int64_t in_shape[] = {rows, cols};
    int64_t out_shape[] = {rows};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_max(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < rows; i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-5f)
            << "MaxLastDim mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, MaxFirstDim) {
    std::vector<float> h_in = {
        1, 5, 3,
        4, 2, 6,
        7, 8, 0
    };
    // Max along dim=0: [max(1,4,7), max(5,2,8), max(3,6,0)] = [7, 8, 6]
    std::vector<float> expected = {7.0f, 8.0f, 6.0f};

    int64_t in_shape[] = {3, 3};
    int64_t out_shape[] = {3};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_max(d_in, d_out, 0);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-5f)
            << "MaxFirstDim mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, MaxNegativeValues) {
    // All negative — verify max picks least negative
    std::vector<float> h_in = {-5, -3, -7, -1, -9, -2};
    std::vector<float> expected = {-3.0f, -1.0f};

    int64_t in_shape[] = {2, 3};
    int64_t out_shape[] = {2};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_max(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < 2; i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-5f)
            << "MaxNegative mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, MaxFP16) {
    constexpr int rows = 3, cols = 5;
    std::vector<float> h_in(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = sinf(static_cast<float>(i) * 0.5f);

    std::vector<float> expected(rows);
    for (int r = 0; r < rows; r++) {
        float mx = -FLT_MAX;
        for (int c = 0; c < cols; c++)
            mx = std::max(mx, h_in[r * cols + c]);
        expected[r] = mx;
    }

    int64_t in_shape[] = {rows, cols};
    int64_t out_shape[] = {rows};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP16, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_max(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    for (int i = 0; i < rows; i++) {
        EXPECT_NEAR(result[i], expected[i], 0.01f)
            << "MaxFP16 mismatch at " << i;
    }

    free_gpu(d_in); free_gpu(d_out);
}

TEST(ReduceTest, MaxLargeRow) {
    constexpr int rows = 2, cols = 2048;
    std::vector<float> h_in(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = sinf(static_cast<float>(i) * 0.001f);
    // Plant known maxima
    h_in[0 * cols + 1000] = 100.0f;
    h_in[1 * cols + 500] = 200.0f;

    int64_t in_shape[] = {rows, cols};
    int64_t out_shape[] = {rows};
    Tensor d_in = make_gpu_tensor(h_in.data(), DType::FP32, 2, in_shape);
    Tensor d_out = alloc_gpu_tensor(DType::FP32, 1, out_shape);

    reduce_max(d_in, d_out, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp32(d_out);
    EXPECT_NEAR(result[0], 100.0f, 1e-5f);
    EXPECT_NEAR(result[1], 200.0f, 1e-5f);

    free_gpu(d_in); free_gpu(d_out);
}

} // namespace
} // namespace imp
