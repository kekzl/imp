#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/tensor.h"
#include "compute/rope.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)

// Allocate device memory and copy host data to it.  Returns device pointer.
template <typename T>
T* to_device(const T* host, size_t count) {
    T* dev = nullptr;
    cudaMalloc(&dev, count * sizeof(T));
    cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice);
    return dev;
}

// Copy device data back to a host vector.
template <typename T>
std::vector<T> to_host(const T* dev, size_t count) {
    std::vector<T> host(count);
    cudaMemcpy(host.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
}

// Build a contiguous 4-D Tensor descriptor on the device.
Tensor make_device_tensor(void* dev_ptr, DType dtype,
                          int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    int64_t shape[4] = {d0, d1, d2, d3};
    return Tensor(dev_ptr, dtype, 4, shape, /*on_device=*/true);
}

// ---------------------------------------------------------------------------
// CPU reference for RoPE (operates on FP32 arrays)
// ---------------------------------------------------------------------------
void cpu_rope(float* q, float* k,
              const int* positions,
              int batch, int seq_len,
              int n_heads, int n_kv_heads,
              int head_dim,
              float theta, float scaling) {
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int pos = positions[b * seq_len + s];
            // --- Q heads ---
            for (int h = 0; h < n_heads; h++) {
                float* qh = q + (((int64_t)b * seq_len + s) * n_heads + h) * head_dim;
                for (int i = 0; i < head_dim / 2; i++) {
                    float freq = 1.0f / (powf(theta, (2.0f * i) / head_dim) * scaling);
                    float angle = pos * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float q0 = qh[2 * i];
                    float q1 = qh[2 * i + 1];
                    qh[2 * i]     = q0 * cos_a - q1 * sin_a;
                    qh[2 * i + 1] = q0 * sin_a + q1 * cos_a;
                }
            }
            // --- K heads ---
            for (int h = 0; h < n_kv_heads; h++) {
                float* kh = k + (((int64_t)b * seq_len + s) * n_kv_heads + h) * head_dim;
                for (int i = 0; i < head_dim / 2; i++) {
                    float freq = 1.0f / (powf(theta, (2.0f * i) / head_dim) * scaling);
                    float angle = pos * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float k0 = kh[2 * i];
                    float k1 = kh[2 * i + 1];
                    kh[2 * i]     = k0 * cos_a - k1 * sin_a;
                    kh[2 * i + 1] = k0 * sin_a + k1 * cos_a;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random fill (no external dependencies)
// ---------------------------------------------------------------------------
void fill_linear(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); i++) {
        // Values in [-1, 1] that are reproducible
        v[i] = sinf(static_cast<float>(i) * 0.7f + 0.3f);
    }
}

// =========================================================================
// Test 1 -- RopeBasicFP32
//   Small tensor (1 batch, 2 seq, 2 heads, 4 head_dim), FP32
//   Verify GPU output matches CPU reference within FP32 tolerance.
// =========================================================================
TEST(RoPETest, RopeBasicFP32) {
    const int batch    = 1;
    const int seq_len  = 2;
    const int n_heads  = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    const float theta  = 10000.0f;
    const float scaling = 1.0f;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    // Prepare host data
    std::vector<float> q_host(q_count), k_host(k_count);
    fill_linear(q_host);
    fill_linear(k_host);

    // Positions: token 0 -> pos 0, token 1 -> pos 5
    std::vector<int> pos_host = {0, 5};

    // CPU reference
    std::vector<float> q_ref(q_host), k_ref(k_host);
    cpu_rope(q_ref.data(), k_ref.data(), pos_host.data(),
             batch, seq_len, n_heads, n_kv_heads, head_dim, theta, scaling);

    // Upload to GPU
    float* q_dev = to_device(q_host.data(), q_count);
    float* k_dev = to_device(k_host.data(), k_count);
    int*  pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, DType::FP32, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, DType::FP32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, theta, scaling);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    auto q_out = to_host(q_dev, q_count);
    auto k_out = to_host(k_dev, k_count);

    // Compare
    const float tol = 1e-4f;
    for (int64_t i = 0; i < q_count; i++) {
        EXPECT_NEAR(q_out[i], q_ref[i], tol)
            << "Q mismatch at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        EXPECT_NEAR(k_out[i], k_ref[i], tol)
            << "K mismatch at index " << i;
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(pos_dev);
}

// =========================================================================
// Test 2 -- RopeBasicFP16
//   Same small shape but FP16.  Tolerance 1e-2.
// =========================================================================
TEST(RoPETest, RopeBasicFP16) {
    const int batch    = 1;
    const int seq_len  = 2;
    const int n_heads  = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    const float theta  = 10000.0f;
    const float scaling = 1.0f;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    // Prepare host data in FP32 for the reference
    std::vector<float> q_fp32(q_count), k_fp32(k_count);
    fill_linear(q_fp32);
    fill_linear(k_fp32);

    // Convert to half on host
    std::vector<__half> q_half(q_count), k_half(k_count);
    for (int64_t i = 0; i < q_count; i++) q_half[i] = __float2half(q_fp32[i]);
    for (int64_t i = 0; i < k_count; i++) k_half[i] = __float2half(k_fp32[i]);

    // For the CPU reference, use the FP16-rounded values so we compare apples to apples
    std::vector<float> q_ref(q_count), k_ref(k_count);
    for (int64_t i = 0; i < q_count; i++) q_ref[i] = __half2float(q_half[i]);
    for (int64_t i = 0; i < k_count; i++) k_ref[i] = __half2float(k_half[i]);

    std::vector<int> pos_host = {0, 5};
    cpu_rope(q_ref.data(), k_ref.data(), pos_host.data(),
             batch, seq_len, n_heads, n_kv_heads, head_dim, theta, scaling);

    // Upload FP16 data to GPU
    __half* q_dev = to_device(q_half.data(), q_count);
    __half* k_dev = to_device(k_half.data(), k_count);
    int*  pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, DType::FP16, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, DType::FP16, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, theta, scaling);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back FP16 and convert to float for comparison
    auto q_half_out = to_host(q_dev, q_count);
    auto k_half_out = to_host(k_dev, k_count);

    const float tol = 1e-2f;
    for (int64_t i = 0; i < q_count; i++) {
        float val = __half2float(q_half_out[i]);
        EXPECT_NEAR(val, q_ref[i], tol)
            << "Q FP16 mismatch at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        float val = __half2float(k_half_out[i]);
        EXPECT_NEAR(val, k_ref[i], tol)
            << "K FP16 mismatch at index " << i;
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(pos_dev);
}

// =========================================================================
// Test 3 -- RopePositionInvariance
//   At position 0, angle = 0 for every frequency.
//   cos(0)=1, sin(0)=0, so the rotation is identity.
//   Verify that the output equals the input exactly.
// =========================================================================
TEST(RoPETest, RopePositionInvariance) {
    const int batch    = 2;
    const int seq_len  = 3;
    const int n_heads  = 4;
    const int n_kv_heads = 4;
    const int head_dim = 8;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    std::vector<float> q_host(q_count), k_host(k_count);
    fill_linear(q_host);
    fill_linear(k_host);

    // All positions are 0
    std::vector<int> pos_host(batch * seq_len, 0);

    // Keep copies of the original data
    std::vector<float> q_orig(q_host), k_orig(k_host);

    // Upload
    float* q_dev = to_device(q_host.data(), q_count);
    float* k_dev = to_device(k_host.data(), k_count);
    int* pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, DType::FP32, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, DType::FP32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, 10000.0f, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out = to_host(q_dev, q_count);
    auto k_out = to_host(k_dev, k_count);

    // With position 0, output must equal input (identity rotation)
    const float tol = 1e-5f;
    for (int64_t i = 0; i < q_count; i++) {
        EXPECT_NEAR(q_out[i], q_orig[i], tol)
            << "Q position-0 invariance broken at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        EXPECT_NEAR(k_out[i], k_orig[i], tol)
            << "K position-0 invariance broken at index " << i;
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(pos_dev);
}

// =========================================================================
// Test 4 -- RopeThetaScaling
//   Different theta values must produce different rotations for non-zero
//   positions.  We run with theta=10000 and theta=1000000, then confirm
//   the outputs differ.
// =========================================================================
TEST(RoPETest, RopeThetaScaling) {
    const int batch    = 1;
    const int seq_len  = 2;
    const int n_heads  = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    std::vector<float> q_host(q_count), k_host(k_count);
    fill_linear(q_host);
    fill_linear(k_host);

    // Use a non-zero position so the rotation is not identity
    std::vector<int> pos_host = {3, 7};

    // --- Run with theta = 10000 ---
    float* q_dev1 = to_device(q_host.data(), q_count);
    float* k_dev1 = to_device(k_host.data(), k_count);
    int* pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q1 = make_device_tensor(q_dev1, DType::FP32, batch, seq_len, n_heads, head_dim);
    Tensor K1 = make_device_tensor(k_dev1, DType::FP32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q1, K1, pos_dev, head_dim, 10000.0f, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out1 = to_host(q_dev1, q_count);
    auto k_out1 = to_host(k_dev1, k_count);

    // --- Run with theta = 1000000 ---
    float* q_dev2 = to_device(q_host.data(), q_count);
    float* k_dev2 = to_device(k_host.data(), k_count);

    Tensor Q2 = make_device_tensor(q_dev2, DType::FP32, batch, seq_len, n_heads, head_dim);
    Tensor K2 = make_device_tensor(k_dev2, DType::FP32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q2, K2, pos_dev, head_dim, 1000000.0f, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out2 = to_host(q_dev2, q_count);
    auto k_out2 = to_host(k_dev2, k_count);

    // The two outputs must differ for at least some elements.
    // We check that the maximum absolute difference is non-trivial.
    float max_q_diff = 0.0f;
    float max_k_diff = 0.0f;
    for (int64_t i = 0; i < q_count; i++) {
        max_q_diff = std::max(max_q_diff, std::fabs(q_out1[i] - q_out2[i]));
    }
    for (int64_t i = 0; i < k_count; i++) {
        max_k_diff = std::max(max_k_diff, std::fabs(k_out1[i] - k_out2[i]));
    }

    EXPECT_GT(max_q_diff, 1e-4f)
        << "Q outputs should differ for different theta values";
    EXPECT_GT(max_k_diff, 1e-4f)
        << "K outputs should differ for different theta values";

    cudaFree(q_dev1);
    cudaFree(k_dev1);
    cudaFree(q_dev2);
    cudaFree(k_dev2);
    cudaFree(pos_dev);
}

// =========================================================================
// Test 5 -- RopeLargerDim
//   head_dim = 128 (typical for LLMs) to exercise the full kernel with
//   many rotation pairs (64 threads).  Verify against CPU reference.
// =========================================================================
TEST(RoPETest, RopeLargerDim) {
    const int batch    = 2;
    const int seq_len  = 4;
    const int n_heads  = 8;
    const int n_kv_heads = 2;   // GQA-style: fewer KV heads
    const int head_dim = 128;
    const float theta  = 10000.0f;
    const float scaling = 1.0f;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    std::vector<float> q_host(q_count), k_host(k_count);
    fill_linear(q_host);
    fill_linear(k_host);

    // Diverse positions across two batches of 4 tokens each
    std::vector<int> pos_host = {0, 1, 2, 3,   10, 20, 30, 40};

    // CPU reference
    std::vector<float> q_ref(q_host), k_ref(k_host);
    cpu_rope(q_ref.data(), k_ref.data(), pos_host.data(),
             batch, seq_len, n_heads, n_kv_heads, head_dim, theta, scaling);

    // GPU
    float* q_dev = to_device(q_host.data(), q_count);
    float* k_dev = to_device(k_host.data(), k_count);
    int*  pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, DType::FP32, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, DType::FP32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, theta, scaling);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out = to_host(q_dev, q_count);
    auto k_out = to_host(k_dev, k_count);

    // The GPU kernel uses __cosf/__sinf (fast math intrinsics) which have
    // lower precision than the host cosf/sinf, so we relax the tolerance
    // slightly for larger positions and higher-frequency pairs.
    const float tol = 5e-4f;
    for (int64_t i = 0; i < q_count; i++) {
        EXPECT_NEAR(q_out[i], q_ref[i], tol)
            << "Q large-dim mismatch at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        EXPECT_NEAR(k_out[i], k_ref[i], tol)
            << "K large-dim mismatch at index " << i;
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(pos_dev);
}

} // namespace
} // namespace imp
