#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "quant/quant_types.h"
#include "quant/quant_gemm.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace imp {
namespace {

// ===========================================================================
// Helper utilities
// ===========================================================================

// Create a GPU tensor from host float data, with optional FP16 conversion.
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

// Allocate a zeroed GPU tensor (output buffer).
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

// Read a GPU tensor back to host as floats.
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

// Free GPU tensor data.
void free_gpu_tensor(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// CPU reference matmul: C[M,N] = A[M,K] @ B_T[N,K]^T (B_T stored [N,K]).
void cpu_matmul(const float* A, const float* B_T, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B_T[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

// ===========================================================================
// Test 1: QuantConfigDefaults -- verify QuantConfig default values
// ===========================================================================
TEST(QuantTest, QuantConfigDefaults) {
    QuantConfig config;
    EXPECT_EQ(config.quant_dtype, QType::F16);
    EXPECT_EQ(config.compute_dtype, QType::F16);
    EXPECT_EQ(config.group_size, 128);
    EXPECT_FALSE(config.has_zero_point);
}

// Fused INT4 dequant+GEMM: C[M,N]=A[M,K]@dequant(B_quant[N,K/2],scales[N,K/group_size]).
// M=4,K=8,N=4,group_size=8 -> num_groups=1, B_quant [4,4] packed, scales [4,1].
TEST(QuantTest, QuantGemmINT4Basic) {
    constexpr int M = 4;
    constexpr int K = 8;
    constexpr int N = 4;
    constexpr int group_size = 8;
    constexpr int num_groups = K / group_size;  // 1
    constexpr int half_K = K / 2;               // 4

    // --- Prepare A[M,K] in float ---
    // Simple known values.
    std::vector<float> h_A = {
        1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // row 0
        0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,  // row 1
        1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // row 2
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,  // row 3
    };

    // Nibble 10 packed for all weights; dequant=(nibble-8)*scale. Channel 0: scale 1.0 -> 2.0;
    // channel 1: nibble 6, scale 2.0 -> -4.0; channel 2: nibble 12, scale 0.5 -> 2.0;
    // channel 3: nibble 8, scale 3.0 -> 0.0.

    std::vector<uint8_t> h_B_quant(N * half_K);

    // Channel 0: nibble=10 (0xA), byte = 0xAA
    for (int j = 0; j < half_K; j++)
        h_B_quant[0 * half_K + j] = 0xAA;
    // Channel 1: nibble=6 (0x6), byte = 0x66
    for (int j = 0; j < half_K; j++)
        h_B_quant[1 * half_K + j] = 0x66;
    // Channel 2: nibble=12 (0xC), byte = 0xCC
    for (int j = 0; j < half_K; j++)
        h_B_quant[2 * half_K + j] = 0xCC;
    // Channel 3: nibble=8 (0x8), byte = 0x88
    for (int j = 0; j < half_K; j++)
        h_B_quant[3 * half_K + j] = 0x88;

    // --- Prepare scales[N, num_groups] = [4, 1] ---
    std::vector<float> h_scales = {1.0f, 2.0f, 0.5f, 3.0f};

    // --- CPU reference: dequantize B, then matmul ---
    // Dequantize each channel's weights [K] from h_B_quant.
    // B_dequant[n][k] = (nibble(n,k) - 8) * scale[n][group]
    std::vector<float> h_B_dequant(N * K);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int byte_idx = n * half_K + k / 2;
            int nibble;
            if (k % 2 == 0)
                nibble = h_B_quant[byte_idx] & 0x0F;
            else
                nibble = (h_B_quant[byte_idx] >> 4) & 0x0F;
            int group_idx = k / group_size;
            float scale = h_scales[n * num_groups + group_idx];
            h_B_dequant[n * K + k] = (float)(nibble - 8) * scale;
        }
    }

    // C_ref[M,N] = A[M,K] @ B_dequant[N,K]^T
    std::vector<float> h_C_ref(M * N, 0.0f);
    cpu_matmul(h_A.data(), h_B_dequant.data(), h_C_ref.data(), M, N, K);

    // --- GPU tensors ---
    Tensor d_A = make_gpu_tensor(h_A.data(), QType::F16, {M, K});

    // B_quant: raw bytes, use INT4 dtype. Shape [N, K/2].
    Tensor d_B;
    d_B.qtype = QType::INT4;
    d_B.ndim = 2;
    d_B.shape[0] = N;
    d_B.shape[1] = half_K;
    d_B.compute_strides();
    d_B.on_device = true;
    cudaMalloc(&d_B.data, N * half_K);
    cudaMemcpy(d_B.data, h_B_quant.data(), N * half_K, cudaMemcpyHostToDevice);

    // Scales: FP16 [N, num_groups]
    Tensor d_scales = make_gpu_tensor(h_scales.data(), QType::F16, {N, num_groups});

    // Output: C [M, N]
    Tensor d_C = alloc_gpu_tensor(QType::F16, {M, N});

    // --- Run fused quant GEMM ---
    quant_gemm_int4(d_A, d_B, d_scales, d_C, nullptr);
    cudaDeviceSynchronize();

    // --- Read back and compare ---
    auto h_C_got = read_gpu_tensor(d_C);

    for (int i = 0; i < M * N; i++) {
        int m = i / N;
        int n = i % N;
        EXPECT_NEAR(h_C_got[i], h_C_ref[i], 1e-1f)
            << "QuantGemmINT4Basic mismatch at C[" << m << "," << n << "]"
            << ": got " << h_C_got[i] << ", expected " << h_C_ref[i];
    }

    free_gpu_tensor(d_A);
    free_gpu_tensor(d_B);
    free_gpu_tensor(d_scales);
    free_gpu_tensor(d_C);
}

}  // namespace
}  // namespace imp
