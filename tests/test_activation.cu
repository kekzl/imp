#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/activation.h"
#include "compute/softmax.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>

namespace imp {
namespace {

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

std::vector<float> read_gpu_tensor(const Tensor& t) {
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

void free_gpu_tensor(Tensor& t) {
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
}

// ── CPU references ──────────────────────────────────────────────────────────

float cpu_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

float cpu_gelu(float x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return x * 0.5f * (1.0f + std::tanh(inner));
}

// =========================================================================
// SwiGLU tests
// =========================================================================

TEST(ActivationTest, SwiGLUBasicFP32) {
    constexpr int N = 8;
    std::vector<float> h_gate = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_up   = { 1.0f,  2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // CPU reference
    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++)
        h_ref[i] = cpu_silu(h_gate[i]) * h_up[i];

    Tensor d_gate = make_gpu_tensor(h_gate.data(), DType::FP32, {N});
    Tensor d_up   = make_gpu_tensor(h_up.data(), DType::FP32, {N});
    Tensor d_out  = alloc_gpu_tensor(DType::FP32, {N});

    swiglu(d_gate, d_up, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "SwiGLU FP32 mismatch at " << i;
    }

    free_gpu_tensor(d_gate);
    free_gpu_tensor(d_up);
    free_gpu_tensor(d_out);
}

TEST(ActivationTest, SwiGLUZeroGate) {
    // silu(0) = 0, so output should be all zeros
    constexpr int N = 4;
    std::vector<float> h_gate(N, 0.0f);
    std::vector<float> h_up = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor d_gate = make_gpu_tensor(h_gate.data(), DType::FP32, {N});
    Tensor d_up   = make_gpu_tensor(h_up.data(), DType::FP32, {N});
    Tensor d_out  = alloc_gpu_tensor(DType::FP32, {N});

    swiglu(d_gate, d_up, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], 0.0f, 1e-6f);
    }

    free_gpu_tensor(d_gate);
    free_gpu_tensor(d_up);
    free_gpu_tensor(d_out);
}

TEST(ActivationTest, SwiGLUFP16) {
    constexpr int N = 8;
    std::vector<float> h_gate = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    std::vector<float> h_up   = { 0.5f,  1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

    // FP16 roundtrip reference
    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++) {
        float g = __half2float(__float2half(h_gate[i]));
        float u = __half2float(__float2half(h_up[i]));
        h_ref[i] = cpu_silu(g) * u;
    }

    Tensor d_gate = make_gpu_tensor(h_gate.data(), DType::FP16, {N});
    Tensor d_up   = make_gpu_tensor(h_up.data(), DType::FP16, {N});
    Tensor d_out  = alloc_gpu_tensor(DType::FP16, {N});

    swiglu(d_gate, d_up, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-2f)
            << "SwiGLU FP16 mismatch at " << i;
    }

    free_gpu_tensor(d_gate);
    free_gpu_tensor(d_up);
    free_gpu_tensor(d_out);
}

TEST(ActivationTest, SwiGLULargeVector) {
    // Exercise the float4 vectorized path (N divisible by 4)
    constexpr int N = 1024;
    std::vector<float> h_gate(N), h_up(N), h_ref(N);
    for (int i = 0; i < N; i++) {
        h_gate[i] = std::sin(static_cast<float>(i) * 0.1f);
        h_up[i] = std::cos(static_cast<float>(i) * 0.07f);
        h_ref[i] = cpu_silu(h_gate[i]) * h_up[i];
    }

    Tensor d_gate = make_gpu_tensor(h_gate.data(), DType::FP32, {N});
    Tensor d_up   = make_gpu_tensor(h_up.data(), DType::FP32, {N});
    Tensor d_out  = alloc_gpu_tensor(DType::FP32, {N});

    swiglu(d_gate, d_up, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "SwiGLU large mismatch at " << i;
    }

    free_gpu_tensor(d_gate);
    free_gpu_tensor(d_up);
    free_gpu_tensor(d_out);
}

TEST(ActivationTest, SwiGLUNonAligned) {
    // N=5 not divisible by 4 — tests scalar fallback path
    constexpr int N = 5;
    std::vector<float> h_gate = {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> h_up   = { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++)
        h_ref[i] = cpu_silu(h_gate[i]) * h_up[i];

    Tensor d_gate = make_gpu_tensor(h_gate.data(), DType::FP32, {N});
    Tensor d_up   = make_gpu_tensor(h_up.data(), DType::FP32, {N});
    Tensor d_out  = alloc_gpu_tensor(DType::FP32, {N});

    swiglu(d_gate, d_up, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "SwiGLU non-aligned mismatch at " << i;
    }

    free_gpu_tensor(d_gate);
    free_gpu_tensor(d_up);
    free_gpu_tensor(d_out);
}

// =========================================================================
// GELU tests
// =========================================================================

TEST(ActivationTest, GELUBasicFP32) {
    constexpr int N = 8;
    std::vector<float> h_x = {-3.0f, -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};

    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++)
        h_ref[i] = cpu_gelu(h_x[i]);

    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP32, {N});
    Tensor d_out = alloc_gpu_tensor(DType::FP32, {N});

    gelu(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "GELU FP32 mismatch at " << i;
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

TEST(ActivationTest, GELUZero) {
    // gelu(0) = 0
    std::vector<float> h_x = {0.0f};
    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP32, {1});
    Tensor d_out = alloc_gpu_tensor(DType::FP32, {1});

    gelu(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    EXPECT_NEAR(h_out[0], 0.0f, 1e-6f);

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

TEST(ActivationTest, GELUSymmetry) {
    // gelu(-x) should be close to -x * sigmoid(-1.702*x) for the tanh approx
    // More practically: gelu(large_positive) ≈ x, gelu(large_negative) ≈ 0
    std::vector<float> h_x = {10.0f, -10.0f};
    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP32, {2});
    Tensor d_out = alloc_gpu_tensor(DType::FP32, {2});

    gelu(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    EXPECT_NEAR(h_out[0], 10.0f, 1e-3f) << "gelu(10) should ≈ 10";
    EXPECT_NEAR(h_out[1], 0.0f, 1e-3f)  << "gelu(-10) should ≈ 0";

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

TEST(ActivationTest, GELUFP16) {
    constexpr int N = 8;
    std::vector<float> h_x = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};

    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++) {
        float x = __half2float(__float2half(h_x[i]));
        h_ref[i] = cpu_gelu(x);
    }

    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP16, {N});
    Tensor d_out = alloc_gpu_tensor(DType::FP16, {N});

    gelu(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-2f)
            << "GELU FP16 mismatch at " << i;
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

// =========================================================================
// Softmax tests
// =========================================================================

TEST(SoftmaxTest, BasicFP32) {
    constexpr int rows = 2;
    constexpr int cols = 4;

    std::vector<float> h_x = {
        1.0f, 2.0f, 3.0f, 4.0f,  // row 0
        1.0f, 1.0f, 1.0f, 1.0f   // row 1: uniform => all 0.25
    };

    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP32, {rows, cols});
    Tensor d_out = alloc_gpu_tensor(DType::FP32, {rows, cols});

    softmax(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    // Row 0: verify sum = 1 and monotonically increasing
    float sum0 = 0.0f;
    for (int c = 0; c < cols; c++) sum0 += h_out[c];
    EXPECT_NEAR(sum0, 1.0f, 1e-5f);
    for (int c = 1; c < cols; c++) {
        EXPECT_GT(h_out[c], h_out[c - 1]) << "Row 0 should be monotonically increasing";
    }

    // Row 1: all equal logits => uniform distribution
    for (int c = 0; c < cols; c++) {
        EXPECT_NEAR(h_out[rows * 0 + cols + c], 0.25f, 1e-5f)
            << "Row 1 should be uniform at col " << c;
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

TEST(SoftmaxTest, NumericalStability) {
    // Large logits should not cause overflow
    constexpr int N = 4;
    std::vector<float> h_x = {1000.0f, 1001.0f, 1000.5f, 999.0f};

    // CPU reference with numerically stable softmax
    float max_val = *std::max_element(h_x.begin(), h_x.end());
    std::vector<float> h_ref(N);
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_ref[i] = std::exp(h_x[i] - max_val);
        sum += h_ref[i];
    }
    for (int i = 0; i < N; i++) h_ref[i] /= sum;

    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP32, {1, N});
    Tensor d_out = alloc_gpu_tensor(DType::FP32, {1, N});

    softmax(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-5f)
            << "Numerical stability issue at " << i;
        EXPECT_FALSE(std::isnan(h_out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(h_out[i])) << "Inf at index " << i;
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

TEST(SoftmaxTest, SingleElement) {
    std::vector<float> h_x = {5.0f};
    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP32, {1, 1});
    Tensor d_out = alloc_gpu_tensor(DType::FP32, {1, 1});

    softmax(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    EXPECT_NEAR(h_out[0], 1.0f, 1e-6f);

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

TEST(SoftmaxTest, FP16) {
    constexpr int rows = 1;
    constexpr int cols = 8;
    std::vector<float> h_x = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP16, {rows, cols});
    Tensor d_out = alloc_gpu_tensor(DType::FP16, {rows, cols});

    softmax(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    // Verify sum ≈ 1 and all positive
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        EXPECT_GT(h_out[c], 0.0f) << "Softmax output should be positive at " << c;
        sum += h_out[c];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-2f) << "FP16 softmax sum should be ~1.0";

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

TEST(SoftmaxTest, LargeRow) {
    // 4096 columns — exercises warp reduction paths
    constexpr int cols = 4096;
    std::vector<float> h_x(cols);
    for (int i = 0; i < cols; i++)
        h_x[i] = std::sin(static_cast<float>(i) * 0.01f);

    Tensor d_x   = make_gpu_tensor(h_x.data(), DType::FP32, {1, cols});
    Tensor d_out = alloc_gpu_tensor(DType::FP32, {1, cols});

    softmax(d_x, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        EXPECT_GE(h_out[i], 0.0f);
        EXPECT_LE(h_out[i], 1.0f);
        sum += h_out[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4f);

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_out);
}

} // namespace
} // namespace imp
