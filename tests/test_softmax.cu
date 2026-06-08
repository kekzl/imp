#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/softmax.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <limits>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Helpers (same pattern as test_layernorm.cu)
// ---------------------------------------------------------------------------
Tensor make_gpu_tensor(const float* host_data, QType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());

    if (dtype == QType::F32) {
        cudaMemcpy(t.data, host_data, t.nbytes(), cudaMemcpyHostToDevice);
    } else if (dtype == QType::F16) {
        std::vector<half> h(t.numel());
        for (int64_t j = 0; j < t.numel(); j++)
            h[j] = __float2half(host_data[j]);
        cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    }
    return t;
}

Tensor alloc_gpu_tensor(QType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

std::vector<float> read_gpu_tensor(const Tensor& t) {
    std::vector<float> result(t.numel());
    if (t.qtype == QType::F32) {
        cudaMemcpy(result.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    } else if (t.qtype == QType::F16) {
        std::vector<half> h(t.numel());
        cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
        for (int64_t j = 0; j < t.numel(); j++)
            result[j] = __half2float(h[j]);
    }
    return result;
}

void free_gpu_tensor(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// ===========================================================================
// Test 1: FP16 softmax output sums to 1.0
// ===========================================================================
TEST(SoftmaxTest, OutputSumsToOne) {
    SKIP_IF_NO_CUDA();

    constexpr int rows = 2, cols = 64;
    std::vector<float> h_in(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = std::sin(static_cast<float>(i) * 0.3f) * 2.0f;

    Tensor d_in = make_gpu_tensor(h_in.data(), QType::F16, {rows, cols});
    Tensor d_out = alloc_gpu_tensor(QType::F16, {rows, cols});

    softmax(d_in, d_out, nullptr);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++)
            sum += h_out[r * cols + c];
        EXPECT_NEAR(sum, 1.0f, 1e-2f) << "Row " << r << " sum != 1.0";
    }

    free_gpu_tensor(d_in);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 2: FP32 softmax output sums to 1.0
// ===========================================================================
TEST(SoftmaxTest, OutputSumsToOneFP32) {
    SKIP_IF_NO_CUDA();

    constexpr int rows = 3, cols = 128;
    std::vector<float> h_in(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        h_in[i] = std::cos(static_cast<float>(i) * 0.17f) * 5.0f;

    Tensor d_in = make_gpu_tensor(h_in.data(), QType::F32, {rows, cols});
    Tensor d_out = alloc_gpu_tensor(QType::F32, {rows, cols});

    softmax(d_in, d_out, nullptr);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++)
            sum += h_out[r * cols + c];
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Row " << r << " sum != 1.0";
    }

    free_gpu_tensor(d_in);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 3: Uniform input produces uniform output (1/n)
// ===========================================================================
TEST(SoftmaxTest, AllEqualInput) {
    SKIP_IF_NO_CUDA();

    constexpr int rows = 1, cols = 32;
    std::vector<float> h_in(cols, 3.0f);  // all equal

    Tensor d_in = make_gpu_tensor(h_in.data(), QType::F32, {rows, cols});
    Tensor d_out = alloc_gpu_tensor(QType::F32, {rows, cols});

    softmax(d_in, d_out, nullptr);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    float expected = 1.0f / static_cast<float>(cols);
    for (int c = 0; c < cols; c++) {
        EXPECT_NEAR(h_out[c], expected, 1e-6f) << "Index " << c;
    }

    free_gpu_tensor(d_in);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 4: Single element produces output = 1.0
// ===========================================================================
TEST(SoftmaxTest, SingleElement) {
    SKIP_IF_NO_CUDA();

    std::vector<float> h_in = {42.0f};
    Tensor d_in = make_gpu_tensor(h_in.data(), QType::F32, {1, 1});
    Tensor d_out = alloc_gpu_tensor(QType::F32, {1, 1});

    softmax(d_in, d_out, nullptr);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    EXPECT_NEAR(h_out[0], 1.0f, 1e-6f);

    free_gpu_tensor(d_in);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 5: -inf masked logits become 0 in output
// ===========================================================================
TEST(SoftmaxTest, NegativeInfMasking) {
    SKIP_IF_NO_CUDA();

    constexpr int cols = 8;
    float neg_inf = -std::numeric_limits<float>::infinity();
    // Mask positions 0, 2, 4, 6 to -inf; keep 1, 3, 5, 7 with real values
    std::vector<float> h_in = {neg_inf, 1.0f, neg_inf, 2.0f, neg_inf, 3.0f, neg_inf, 0.5f};

    Tensor d_in = make_gpu_tensor(h_in.data(), QType::F32, {1, cols});
    Tensor d_out = alloc_gpu_tensor(QType::F32, {1, cols});

    softmax(d_in, d_out, nullptr);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    // Masked positions should be 0 (or very near 0)
    for (int c : {0, 2, 4, 6}) {
        EXPECT_NEAR(h_out[c], 0.0f, 1e-7f) << "Masked index " << c << " should be 0";
    }
    // Unmasked should be positive and sum to 1
    float sum = 0.0f;
    for (int c : {1, 3, 5, 7}) {
        EXPECT_GT(h_out[c], 0.0f) << "Unmasked index " << c << " should be > 0";
        sum += h_out[c];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    free_gpu_tensor(d_in);
    free_gpu_tensor(d_out);
}

}  // namespace
}  // namespace imp
