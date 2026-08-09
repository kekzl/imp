#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/tensor.h"
#include "compute/rope.h"
#include "compute/mtp_forward.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
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
Tensor make_device_tensor(void* dev_ptr, QType dtype, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    int64_t shape[4] = {d0, d1, d2, d3};
    return Tensor(dev_ptr, dtype, 4, shape, /*on_device=*/true);
}

// ---------------------------------------------------------------------------
// CPU reference for RoPE (operates on FP32 arrays)
// ---------------------------------------------------------------------------
void cpu_rope(float* q, float* k, const int* positions, int batch, int seq_len, int n_heads, int n_kv_heads,
              int head_dim, float theta, float scaling) {
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
                    qh[2 * i] = q0 * cos_a - q1 * sin_a;
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
                    kh[2 * i] = k0 * cos_a - k1 * sin_a;
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
    const int batch = 1;
    const int seq_len = 2;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    const float theta = 10000.0f;
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
    cpu_rope(q_ref.data(), k_ref.data(), pos_host.data(), batch, seq_len, n_heads, n_kv_heads, head_dim,
             theta, scaling);

    // Upload to GPU
    float* q_dev = to_device(q_host.data(), q_count);
    float* k_dev = to_device(k_host.data(), k_count);
    int* pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, QType::F32, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, QType::F32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, theta, scaling);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    auto q_out = to_host(q_dev, q_count);
    auto k_out = to_host(k_dev, k_count);

    // Compare
    const float tol = 1e-4f;
    for (int64_t i = 0; i < q_count; i++) {
        EXPECT_NEAR(q_out[i], q_ref[i], tol) << "Q mismatch at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        EXPECT_NEAR(k_out[i], k_ref[i], tol) << "K mismatch at index " << i;
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(pos_dev);
}

// =========================================================================
// Test 2 -- RopeBasicFP16
//   Same small shape but FP16.  Tolerance 1e-2.
// =========================================================================
// =========================================================================
// RoPE across the long-context position range, against DOUBLE truth.
//
// Every other RoPE test here uses positions <= 40 and a float32 CPU reference.
// That was enough to miss #1316: the angle `pos * freq` was formed in float and
// handed to the fast intrinsics, and the resulting drift grew with position --
// 2.3e-4 at 2000, 1.0e-2 at 131071, the trained context limit. It lands on the
// lowest-frequency rotary pair, the one carrying long-range position
// information.
//
// Two things make this test different from the ones above:
//
//   1. The oracle is DOUBLE, not float32. Comparing two float32 computations
//      cannot tell which one drifted; measured against double, the float32 CPU
//      reference is accurate to ~1e-7 while the pre-fix kernel was 1e-2 out.
//   2. It sweeps to 131071, so the range the model actually supports is
//      covered.
//
// Post-fix the kernel reduces the angle in double before the intrinsic and
// tracks double truth to 1.9e-4 at the context limit -- closer than the float32
// CPU reference manages (5.8e-4). kTol is set from that measurement.
// =========================================================================
TEST(RoPETest, LongContextPositionsMatchDoubleReference) {
    const int batch = 1, seq_len = 1, n_heads = 1, n_kv_heads = 1, head_dim = 8;
    const float theta = 10000.0f, scaling = 1.0f;
    constexpr float kTol = 5e-4f;
    const std::vector<int> positions = {0, 40, 500, 1000, 2000, 4000, 8000, 16000, 32768, 131071};

    for (int pos : positions) {
        const int64_t n = (int64_t)batch * seq_len * n_heads * head_dim;
        std::vector<float> q_host(n), k_host(n);
        fill_linear(q_host);
        fill_linear(k_host);

        // Double-precision truth: angle and trig both in double.
        std::vector<float> q_ref(q_host), k_ref(k_host);
        for (int i = 0; i < head_dim / 2; i++) {
            const double freq = 1.0 / std::pow((double)theta, (2.0 * i) / head_dim);
            const double angle = (double)pos * freq;
            const double ca = std::cos(angle), sa = std::sin(angle);
            for (auto* v : {&q_ref, &k_ref}) {
                const float a = (*v)[2 * i], b = (*v)[2 * i + 1];
                (*v)[2 * i] = (float)(a * ca - b * sa);
                (*v)[2 * i + 1] = (float)(a * sa + b * ca);
            }
        }

        std::vector<int> pos_host = {pos};
        float* q_dev = to_device(q_host.data(), n);
        float* k_dev = to_device(k_host.data(), n);
        int* pos_dev = to_device(pos_host.data(), pos_host.size());
        Tensor Q = make_device_tensor(q_dev, QType::F32, batch, seq_len, n_heads, head_dim);
        Tensor K = make_device_tensor(k_dev, QType::F32, batch, seq_len, n_kv_heads, head_dim);
        rope_forward(Q, K, pos_dev, head_dim, theta, scaling);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto q_out = to_host(q_dev, n);
        auto k_out = to_host(k_dev, n);
        for (int64_t i = 0; i < n; i++) {
            EXPECT_NEAR(q_out[i], q_ref[i], kTol) << "Q drifted from double truth at pos=" << pos;
            EXPECT_NEAR(k_out[i], k_ref[i], kTol) << "K drifted from double truth at pos=" << pos;
        }

        cudaFree(q_dev);
        cudaFree(k_dev);
        cudaFree(pos_dev);
    }
}

TEST(RoPETest, RopeBasicFP16) {
    const int batch = 1;
    const int seq_len = 2;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    const float theta = 10000.0f;
    const float scaling = 1.0f;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    // Prepare host data in FP32 for the reference
    std::vector<float> q_fp32(q_count), k_fp32(k_count);
    fill_linear(q_fp32);
    fill_linear(k_fp32);

    // Convert to half on host
    std::vector<__half> q_half(q_count), k_half(k_count);
    for (int64_t i = 0; i < q_count; i++)
        q_half[i] = __float2half(q_fp32[i]);
    for (int64_t i = 0; i < k_count; i++)
        k_half[i] = __float2half(k_fp32[i]);

    // For the CPU reference, use the FP16-rounded values so we compare apples to apples
    std::vector<float> q_ref(q_count), k_ref(k_count);
    for (int64_t i = 0; i < q_count; i++)
        q_ref[i] = __half2float(q_half[i]);
    for (int64_t i = 0; i < k_count; i++)
        k_ref[i] = __half2float(k_half[i]);

    std::vector<int> pos_host = {0, 5};
    cpu_rope(q_ref.data(), k_ref.data(), pos_host.data(), batch, seq_len, n_heads, n_kv_heads, head_dim,
             theta, scaling);

    // Upload FP16 data to GPU
    __half* q_dev = to_device(q_half.data(), q_count);
    __half* k_dev = to_device(k_half.data(), k_count);
    int* pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, QType::F16, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, QType::F16, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, theta, scaling);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back FP16 and convert to float for comparison
    auto q_half_out = to_host(q_dev, q_count);
    auto k_half_out = to_host(k_dev, k_count);

    const float tol = 1e-2f;
    for (int64_t i = 0; i < q_count; i++) {
        float val = __half2float(q_half_out[i]);
        EXPECT_NEAR(val, q_ref[i], tol) << "Q FP16 mismatch at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        float val = __half2float(k_half_out[i]);
        EXPECT_NEAR(val, k_ref[i], tol) << "K FP16 mismatch at index " << i;
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
    const int batch = 2;
    const int seq_len = 3;
    const int n_heads = 4;
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

    Tensor Q = make_device_tensor(q_dev, QType::F32, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, QType::F32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, 10000.0f, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out = to_host(q_dev, q_count);
    auto k_out = to_host(k_dev, k_count);

    // With position 0, output must equal input (identity rotation)
    const float tol = 1e-5f;
    for (int64_t i = 0; i < q_count; i++) {
        EXPECT_NEAR(q_out[i], q_orig[i], tol) << "Q position-0 invariance broken at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        EXPECT_NEAR(k_out[i], k_orig[i], tol) << "K position-0 invariance broken at index " << i;
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
    const int batch = 1;
    const int seq_len = 2;
    const int n_heads = 2;
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

    Tensor Q1 = make_device_tensor(q_dev1, QType::F32, batch, seq_len, n_heads, head_dim);
    Tensor K1 = make_device_tensor(k_dev1, QType::F32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q1, K1, pos_dev, head_dim, 10000.0f, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out1 = to_host(q_dev1, q_count);
    auto k_out1 = to_host(k_dev1, k_count);

    // --- Run with theta = 1000000 ---
    float* q_dev2 = to_device(q_host.data(), q_count);
    float* k_dev2 = to_device(k_host.data(), k_count);

    Tensor Q2 = make_device_tensor(q_dev2, QType::F32, batch, seq_len, n_heads, head_dim);
    Tensor K2 = make_device_tensor(k_dev2, QType::F32, batch, seq_len, n_kv_heads, head_dim);

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

    EXPECT_GT(max_q_diff, 1e-4f) << "Q outputs should differ for different theta values";
    EXPECT_GT(max_k_diff, 1e-4f) << "K outputs should differ for different theta values";

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
    const int batch = 2;
    const int seq_len = 4;
    const int n_heads = 8;
    const int n_kv_heads = 2;  // GQA-style: fewer KV heads
    const int head_dim = 128;
    const float theta = 10000.0f;
    const float scaling = 1.0f;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    std::vector<float> q_host(q_count), k_host(k_count);
    fill_linear(q_host);
    fill_linear(k_host);

    // Diverse positions across two batches of 4 tokens each
    std::vector<int> pos_host = {0, 1, 2, 3, 10, 20, 30, 40};

    // CPU reference
    std::vector<float> q_ref(q_host), k_ref(k_host);
    cpu_rope(q_ref.data(), k_ref.data(), pos_host.data(), batch, seq_len, n_heads, n_kv_heads, head_dim,
             theta, scaling);

    // GPU
    float* q_dev = to_device(q_host.data(), q_count);
    float* k_dev = to_device(k_host.data(), k_count);
    int* pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, QType::F32, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, QType::F32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, theta, scaling);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out = to_host(q_dev, q_count);
    auto k_out = to_host(k_dev, k_count);

    // The GPU kernel uses __cosf/__sinf (fast math intrinsics) which have
    // lower precision than the host cosf/sinf, so we relax the tolerance
    // slightly for larger positions and higher-frequency pairs.
    const float tol = 5e-4f;
    for (int64_t i = 0; i < q_count; i++) {
        EXPECT_NEAR(q_out[i], q_ref[i], tol) << "Q large-dim mismatch at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        EXPECT_NEAR(k_out[i], k_ref[i], tol) << "K large-dim mismatch at index " << i;
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(pos_dev);
}

// =========================================================================
// Test 6 -- PartialRoPE (Qwen3.5 style: rope_dim=64, head_dim=256)
//   Only the first rope_dim dimensions should be rotated.
//   The remaining (head_dim - rope_dim) dimensions must stay unchanged.
// =========================================================================
TEST(RoPETest, PartialRoPE) {
    const int batch = 1;
    const int seq_len = 2;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 256;
    const int rope_dim = 64;
    const float theta = 10000.0f;
    const float scaling = 1.0f;

    const int64_t q_count = (int64_t)batch * seq_len * n_heads * head_dim;
    const int64_t k_count = (int64_t)batch * seq_len * n_kv_heads * head_dim;

    std::vector<float> q_host(q_count), k_host(k_count);
    fill_linear(q_host);
    fill_linear(k_host);

    // Keep originals for comparison of the unrotated portion
    std::vector<float> q_orig(q_host), k_orig(k_host);

    // Non-zero positions to ensure rotation happens
    std::vector<int> pos_host = {3, 7};

    // CPU reference: only rotate first rope_dim dims
    std::vector<float> q_ref(q_host), k_ref(k_host);
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int pos = pos_host[b * seq_len + s];
            for (int h = 0; h < n_heads; h++) {
                float* qh = q_ref.data() + (((int64_t)b * seq_len + s) * n_heads + h) * head_dim;
                for (int i = 0; i < rope_dim / 2; i++) {
                    float freq = 1.0f / (powf(theta, (2.0f * i) / rope_dim) * scaling);
                    float angle = pos * freq;
                    float c = cosf(angle), sn = sinf(angle);
                    float q0 = qh[2 * i], q1 = qh[2 * i + 1];
                    qh[2 * i] = q0 * c - q1 * sn;
                    qh[2 * i + 1] = q0 * sn + q1 * c;
                }
            }
            for (int h = 0; h < n_kv_heads; h++) {
                float* kh = k_ref.data() + (((int64_t)b * seq_len + s) * n_kv_heads + h) * head_dim;
                for (int i = 0; i < rope_dim / 2; i++) {
                    float freq = 1.0f / (powf(theta, (2.0f * i) / rope_dim) * scaling);
                    float angle = pos * freq;
                    float c = cosf(angle), sn = sinf(angle);
                    float k0 = kh[2 * i], k1 = kh[2 * i + 1];
                    kh[2 * i] = k0 * c - k1 * sn;
                    kh[2 * i + 1] = k0 * sn + k1 * c;
                }
            }
        }
    }

    // Upload to GPU
    float* q_dev = to_device(q_host.data(), q_count);
    float* k_dev = to_device(k_host.data(), k_count);
    int* pos_dev = to_device(pos_host.data(), pos_host.size());

    Tensor Q = make_device_tensor(q_dev, QType::F32, batch, seq_len, n_heads, head_dim);
    Tensor K = make_device_tensor(k_dev, QType::F32, batch, seq_len, n_kv_heads, head_dim);

    rope_forward(Q, K, pos_dev, head_dim, theta, scaling, rope_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto q_out = to_host(q_dev, q_count);
    auto k_out = to_host(k_dev, k_count);

    const float tol = 5e-4f;

    // Verify rotated portion matches CPU reference
    for (int64_t i = 0; i < q_count; i++) {
        EXPECT_NEAR(q_out[i], q_ref[i], tol) << "Q partial RoPE mismatch at index " << i;
    }
    for (int64_t i = 0; i < k_count; i++) {
        EXPECT_NEAR(k_out[i], k_ref[i], tol) << "K partial RoPE mismatch at index " << i;
    }

    // Verify unrotated portion (dims >= rope_dim) is unchanged
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < n_heads; h++) {
                int64_t base = (((int64_t)b * seq_len + s) * n_heads + h) * head_dim;
                for (int d = rope_dim; d < head_dim; d++) {
                    EXPECT_NEAR(q_out[base + d], q_orig[base + d], 1e-6f)
                        << "Q dim " << d << " should be unchanged (partial RoPE)";
                }
            }
            for (int h = 0; h < n_kv_heads; h++) {
                int64_t base = (((int64_t)b * seq_len + s) * n_kv_heads + h) * head_dim;
                for (int d = rope_dim; d < head_dim; d++) {
                    EXPECT_NEAR(k_out[base + d], k_orig[base + d], 1e-6f)
                        << "K dim " << d << " should be unchanged (partial RoPE)";
                }
            }
        }
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(pos_dev);
}

// =========================================================================
// MtpMropeMatchesMainYarn (issue #897)
//   The MTP draft head must rotate Q/K identically to the main forward on a
//   rope-scaled (YaRN) model. Runs the shared main rope_forward and the MTP
//   mtp_apply_mrope with the SAME YaRN params at an extended position and
//   asserts they agree — the exact drift the old inline plain-RoPE caused.
// =========================================================================
TEST(RoPETest, MtpMropeMatchesMainYarn) {
    const int n_heads = 2, n_kv_heads = 2;
    const int head_dim = 256, rope_dim = 64;   // Qwen3.x partial rope
    const float theta = 1000000.0f;
    const float freq_scale = 4.0f;             // rope-scaled (YaRN)
    const float ext_factor = 1.0f;             // YaRN on
    const float attn_factor = 1.0f;
    const int pos = 6000;                      // extended position (> n_ctx_orig)

    // YaRN correction dims, exactly as the engine/executor compute them.
    float corr[2] = {0.0f, 0.0f};
    rope_yarn_corr_dims(rope_dim, /*n_ctx_orig=*/2048, theta, /*beta_fast=*/32.0f, /*beta_slow=*/1.0f, corr);

    // Identical FP16 Q/K for both paths (single token).
    const int64_t qn = (int64_t)n_heads * head_dim;
    const int64_t kn = (int64_t)n_kv_heads * head_dim;
    std::vector<float> qf(qn), kf(kn);
    fill_linear(qf);
    fill_linear(kf);
    std::vector<__half> qh(qn), kh(kn);
    for (int64_t i = 0; i < qn; i++) qh[i] = __float2half(qf[i]);
    for (int64_t i = 0; i < kn; i++) kh[i] = __float2half(kf[i]);

    // --- Main forward path (verifier): neox, YaRN ---
    __half* q_main = to_device(qh.data(), qn);
    __half* k_main = to_device(kh.data(), kn);
    int pos_arr = pos;
    int* pos_dev = to_device(&pos_arr, 1);
    Tensor Qm = make_device_tensor(q_main, QType::F16, 1, 1, n_heads, head_dim);
    Tensor Km = make_device_tensor(k_main, QType::F16, 1, 1, n_kv_heads, head_dim);
    rope_forward(Qm, Km, pos_dev, head_dim, theta, freq_scale, rope_dim, /*neox=*/true, ext_factor,
                 attn_factor, corr, /*stream=*/nullptr, /*longrope=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto q_main_out = to_host(q_main, qn);
    auto k_main_out = to_host(k_main, kn);

    // --- MTP draft path: same params, Qwen3.6 mrope section split [11,11,10] ---
    __half* q_mtp = to_device(qh.data(), qn);
    __half* k_mtp = to_device(kh.data(), kn);
    mtp_apply_mrope(q_mtp, n_heads, k_mtp, n_kv_heads, head_dim, rope_dim, theta, /*sec0=*/11,
                    /*sec1=*/11, /*sec2=*/10, pos, /*inv_scaling=*/1.0f / freq_scale, ext_factor,
                    attn_factor, corr[0], corr[1], /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto q_mtp_out = to_host(q_mtp, qn);
    auto k_mtp_out = to_host(k_mtp, kn);

    // --- Parity: MTP must match the verifier within FP16 tolerance ---
    const float tol = 3e-3f;
    for (int64_t i = 0; i < qn; i++) {
        EXPECT_NEAR(__half2float(q_mtp_out[i]), __half2float(q_main_out[i]), tol)
            << "Q mismatch at " << i << " (MTP draft rope drifted from verifier)";
    }
    for (int64_t i = 0; i < kn; i++) {
        EXPECT_NEAR(__half2float(k_mtp_out[i]), __half2float(k_main_out[i]), tol)
            << "K mismatch at " << i;
    }

    // --- Guard: the YaRN scaling must actually change the result, else the
    // parity above could pass trivially (both silently ignoring the params).
    __half* q_plain = to_device(qh.data(), qn);
    __half* k_plain = to_device(kh.data(), kn);
    mtp_apply_mrope(q_plain, n_heads, k_plain, n_kv_heads, head_dim, rope_dim, theta, 11, 11, 10, pos,
                    /*inv_scaling=*/1.0f, /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, 0.0f, 0.0f,
                    /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto q_plain_out = to_host(q_plain, qn);
    float max_diff = 0.0f;
    for (int64_t i = 0; i < qn; i++)
        max_diff = std::fmax(max_diff, std::fabs(__half2float(q_mtp_out[i]) - __half2float(q_plain_out[i])));
    EXPECT_GT(max_diff, 1e-2f) << "YaRN scaling had no effect vs plain RoPE — params ignored?";

    cudaFree(q_main);  cudaFree(k_main);  cudaFree(pos_dev);
    cudaFree(q_mtp);   cudaFree(k_mtp);
    cudaFree(q_plain); cudaFree(k_plain);
}

}  // namespace
}  // namespace imp
