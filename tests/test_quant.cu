#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "quant/quant_types.h"
#include "quant/dequant_fp16.h"
#include "quant/dequant_int8.h"
#include "quant/fp8_utils.h"
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
Tensor make_gpu_tensor(const float* host_data, DType dtype,
                       std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = dtype;
    t.ndim  = static_cast<int>(shape_list.size());
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

// Allocate a zeroed GPU tensor (output buffer).
Tensor alloc_gpu_tensor(DType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = dtype;
    t.ndim  = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

// Read a GPU tensor back to host as floats.
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

// Free GPU tensor data.
void free_gpu_tensor(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: INT4 dequantization
// ---------------------------------------------------------------------------
void cpu_dequant_int4(const uint8_t* packed, float* output,
                      const float* scales, int n, int group_size) {
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 2;
        int nibble = (i % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                   : ((packed[byte_idx] >> 4) & 0x0F);
        int group = i / group_size;
        output[i] = (float)(nibble - 8) * scales[group];
    }
}

// ---------------------------------------------------------------------------
// CPU reference: INT8 dequantization (per-element scales)
// ---------------------------------------------------------------------------
void cpu_dequant_int8(const int8_t* input, float* output,
                      const float* scales, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = (float)input[i] * scales[i];
    }
}

// ---------------------------------------------------------------------------
// CPU reference: matmul  C[M,N] = A[M,K] @ B_T[N,K]^T
// (B_T is stored in [N,K] layout, so C[m,n] = sum_k A[m,k] * B_T[n,k])
// ---------------------------------------------------------------------------
void cpu_matmul(const float* A, const float* B_T, float* C,
                int M, int N, int K) {
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
    EXPECT_EQ(config.quant_dtype, DType::FP16);
    EXPECT_EQ(config.compute_dtype, DType::FP16);
    EXPECT_EQ(config.group_size, 128);
    EXPECT_FALSE(config.has_zero_point);
}

// ===========================================================================
// Test 2: DequantINT4Basic -- 32 packed INT4 values with a single scale
// ===========================================================================
TEST(QuantTest, DequantINT4Basic) {
    constexpr int n = 32;
    constexpr int group_size = 32;  // all elements in one group
    constexpr int n_bytes = n / 2;  // 16 packed bytes
    constexpr int n_groups = 1;
    constexpr float scale_val = 0.5f;

    // Prepare host data: pack known nibble values into bytes.
    // We use a pattern that exercises the full nibble range [0..15].
    // Byte i packs nibbles: low = (2*i) % 16, high = (2*i+1) % 16
    std::vector<uint8_t> h_packed(n_bytes);
    for (int i = 0; i < n_bytes; i++) {
        uint8_t lo = static_cast<uint8_t>((2 * i) % 16);
        uint8_t hi = static_cast<uint8_t>((2 * i + 1) % 16);
        h_packed[i] = (hi << 4) | lo;
    }

    std::vector<float> h_scales(n_groups, scale_val);

    // CPU reference
    std::vector<float> h_ref(n);
    cpu_dequant_int4(h_packed.data(), h_ref.data(), h_scales.data(), n, group_size);

    // GPU: upload packed data, scales (as FP16), allocate output
    void* d_packed = nullptr;
    void* d_scales = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_packed, n_bytes);
    cudaMalloc(&d_scales, n_groups * sizeof(half));
    cudaMalloc(&d_output, n * sizeof(half));

    cudaMemcpy(d_packed, h_packed.data(), n_bytes, cudaMemcpyHostToDevice);

    // Convert scales to FP16 for upload
    std::vector<half> h_scales_fp16(n_groups);
    for (int i = 0; i < n_groups; i++)
        h_scales_fp16[i] = __float2half(h_scales[i]);
    cudaMemcpy(d_scales, h_scales_fp16.data(), n_groups * sizeof(half),
               cudaMemcpyHostToDevice);

    // Run kernel
    dequant_int4_fp16(d_packed, d_output, d_scales, n, group_size, nullptr);
    cudaDeviceSynchronize();

    // Read back output (FP16 -> float)
    std::vector<half> h_out_fp16(n);
    cudaMemcpy(h_out_fp16.data(), d_output, n * sizeof(half),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float got = __half2float(h_out_fp16[i]);
        EXPECT_NEAR(got, h_ref[i], 1e-3f)
            << "DequantINT4Basic mismatch at index " << i
            << ": got " << got << ", expected " << h_ref[i];
    }

    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_output);
}

// ===========================================================================
// Test 3: DequantINT4GroupSize -- multiple groups with different scales
// ===========================================================================
TEST(QuantTest, DequantINT4GroupSize) {
    constexpr int n = 32;
    constexpr int group_size = 8;
    constexpr int n_bytes = n / 2;           // 16 packed bytes
    constexpr int n_groups = n / group_size;  // 4 groups

    // Pack known nibble values: use constant nibble=5 for easy verification.
    // dequant = (5 - 8) * scale = -3 * scale
    std::vector<uint8_t> h_packed(n_bytes);
    for (int i = 0; i < n_bytes; i++) {
        // Both nibbles = 5: byte = 0x55
        h_packed[i] = 0x55;
    }

    // Different scale per group: 1.0, 2.0, 0.5, 4.0
    std::vector<float> h_scales = {1.0f, 2.0f, 0.5f, 4.0f};

    // CPU reference
    std::vector<float> h_ref(n);
    cpu_dequant_int4(h_packed.data(), h_ref.data(), h_scales.data(), n, group_size);

    // Verify CPU reference makes sense
    // Group 0 (indices 0..7):  (5-8)*1.0 = -3.0
    // Group 1 (indices 8..15): (5-8)*2.0 = -6.0
    // Group 2 (indices 16..23): (5-8)*0.5 = -1.5
    // Group 3 (indices 24..31): (5-8)*4.0 = -12.0
    ASSERT_NEAR(h_ref[0], -3.0f, 1e-6f);
    ASSERT_NEAR(h_ref[8], -6.0f, 1e-6f);
    ASSERT_NEAR(h_ref[16], -1.5f, 1e-6f);
    ASSERT_NEAR(h_ref[24], -12.0f, 1e-6f);

    // GPU
    void* d_packed = nullptr;
    void* d_scales = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_packed, n_bytes);
    cudaMalloc(&d_scales, n_groups * sizeof(half));
    cudaMalloc(&d_output, n * sizeof(half));

    cudaMemcpy(d_packed, h_packed.data(), n_bytes, cudaMemcpyHostToDevice);

    std::vector<half> h_scales_fp16(n_groups);
    for (int i = 0; i < n_groups; i++)
        h_scales_fp16[i] = __float2half(h_scales[i]);
    cudaMemcpy(d_scales, h_scales_fp16.data(), n_groups * sizeof(half),
               cudaMemcpyHostToDevice);

    dequant_int4_fp16(d_packed, d_output, d_scales, n, group_size, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> h_out_fp16(n);
    cudaMemcpy(h_out_fp16.data(), d_output, n * sizeof(half),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float got = __half2float(h_out_fp16[i]);
        EXPECT_NEAR(got, h_ref[i], 1e-3f)
            << "DequantINT4GroupSize mismatch at index " << i
            << " (group " << (i / group_size) << ")"
            << ": got " << got << ", expected " << h_ref[i];
    }

    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_output);
}

// ===========================================================================
// Test 4: DequantINT8Basic -- 32 INT8 values with per-element scales
// ===========================================================================
TEST(QuantTest, DequantINT8Basic) {
    constexpr int n = 32;

    // INT8 values: a mix of positive, negative, and zero
    std::vector<int8_t> h_input(n);
    for (int i = 0; i < n; i++) {
        // Range: -16..15
        h_input[i] = static_cast<int8_t>((i % 32) - 16);
    }

    // Per-element scales: varying values
    std::vector<float> h_scales(n);
    for (int i = 0; i < n; i++) {
        h_scales[i] = 0.1f * (1.0f + (float)(i % 5));
    }

    // CPU reference
    std::vector<float> h_ref(n);
    cpu_dequant_int8(h_input.data(), h_ref.data(), h_scales.data(), n);

    // GPU: upload INT8 data, FP16 scales, allocate FP16 output
    void* d_input = nullptr;
    void* d_scales = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, n * sizeof(int8_t));
    cudaMalloc(&d_scales, n * sizeof(half));
    cudaMalloc(&d_output, n * sizeof(half));

    cudaMemcpy(d_input, h_input.data(), n * sizeof(int8_t),
               cudaMemcpyHostToDevice);

    std::vector<half> h_scales_fp16(n);
    for (int i = 0; i < n; i++)
        h_scales_fp16[i] = __float2half(h_scales[i]);
    cudaMemcpy(d_scales, h_scales_fp16.data(), n * sizeof(half),
               cudaMemcpyHostToDevice);

    dequant_int8_fp16(d_input, d_output, d_scales, n, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> h_out_fp16(n);
    cudaMemcpy(h_out_fp16.data(), d_output, n * sizeof(half),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float got = __half2float(h_out_fp16[i]);
        // Compute reference with FP16-rounded scale (to match GPU precision)
        float scale_fp16 = __half2float(__float2half(h_scales[i]));
        float expected = (float)h_input[i] * scale_fp16;
        EXPECT_NEAR(got, expected, 5e-3f)
            << "DequantINT8Basic mismatch at index " << i
            << ": got " << got << ", expected " << expected
            << " (input=" << (int)h_input[i] << ", scale=" << scale_fp16 << ")";
    }

    cudaFree(d_input);
    cudaFree(d_scales);
    cudaFree(d_output);
}

// ===========================================================================
// Test 5: FP8RoundTrip -- cast FP16 -> FP8 E4M3 -> FP16, check round-trip
// ===========================================================================
TEST(QuantTest, FP8RoundTrip) {
    // Test values within E4M3 representable range.
    // E4M3 normal range: ~2^-6 to 240 (e=14,m=7 = 2^7 * 1.875 = 240).
    // E4M3 min subnormal: 2^-9 = ~0.00195.
    // Note: e=15 encodes NaN in NVIDIA's E4M3 spec, so max safe normal = 240.
    // Values above 240 may saturate to 240 or become NaN depending on impl.
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 0.001953125f,
        -0.5f, -2.0f, 4.0f, 8.0f, 16.0f, 64.0f, 128.0f,
        0.25f, 0.125f, 0.0625f, 240.0f, -240.0f, 32.0f, -32.0f
    };
    const int n = static_cast<int>(test_values.size());

    // Upload as FP16
    std::vector<half> h_input_fp16(n);
    for (int i = 0; i < n; i++)
        h_input_fp16[i] = __float2half(test_values[i]);

    void* d_fp16_in = nullptr;
    void* d_fp8 = nullptr;
    void* d_fp16_out = nullptr;
    cudaMalloc(&d_fp16_in, n * sizeof(half));
    cudaMalloc(&d_fp8, n * sizeof(uint8_t));
    cudaMalloc(&d_fp16_out, n * sizeof(half));

    cudaMemcpy(d_fp16_in, h_input_fp16.data(), n * sizeof(half),
               cudaMemcpyHostToDevice);

    // FP16 -> FP8
    cast_fp16_to_fp8(d_fp16_in, d_fp8, n, nullptr);
    cudaDeviceSynchronize();

    // FP8 -> FP16
    cast_fp8_to_fp16(d_fp8, d_fp16_out, n, nullptr);
    cudaDeviceSynchronize();

    // Read back
    std::vector<half> h_out_fp16(n);
    cudaMemcpy(h_out_fp16.data(), d_fp16_out, n * sizeof(half),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float original = __half2float(h_input_fp16[i]);
        float roundtrip = __half2float(h_out_fp16[i]);
        float abs_original = std::fabs(original);

        if (abs_original == 0.0f) {
            // Zero must round-trip exactly.
            EXPECT_EQ(roundtrip, 0.0f)
                << "FP8RoundTrip: zero did not round-trip at index " << i;
        } else {
            // FP8 E4M3 has 3 mantissa bits, so relative error up to 12.5%.
            // Use relative tolerance for larger values.
            float tol = std::max(0.5f, abs_original * 0.125f);
            EXPECT_NEAR(roundtrip, original, tol)
                << "FP8RoundTrip mismatch at index " << i
                << " (value=" << original << ")"
                << ": got " << roundtrip;

            // Also check sign is preserved.
            if (original > 0.0f) {
                EXPECT_GT(roundtrip, 0.0f)
                    << "FP8RoundTrip: sign flipped for positive value at index "
                    << i;
            } else if (original < 0.0f) {
                EXPECT_LT(roundtrip, 0.0f)
                    << "FP8RoundTrip: sign flipped for negative value at index "
                    << i;
            }
        }
    }

    cudaFree(d_fp16_in);
    cudaFree(d_fp8);
    cudaFree(d_fp16_out);
}

// ===========================================================================
// Test 6: FP8Saturation -- values > 448 saturate; tiny values flush to zero
// ===========================================================================
TEST(QuantTest, FP8Saturation) {
    // Test that overflow values saturate to E4M3 max (240) or become NaN,
    // and that tiny values flush to zero.
    // E4M3 max normal: e=14, m=7 -> 2^7 * 1.875 = 240.
    // (NVIDIA spec: e=15,m=7 = NaN; e=15,m=0..6 may be NaN or extended normals.)
    // Different CUDA versions may saturate to 240 or produce NaN for overflow.

    // Flush-to-zero test values
    std::vector<float> small_values = {
        1e-4f, 5e-5f, 1e-5f,     // very small -> flush to 0
        -1e-4f, -5e-5f,           // small negative -> flush to 0
    };
    const int n_small = static_cast<int>(small_values.size());

    std::vector<half> h_input_fp16(n_small);
    for (int i = 0; i < n_small; i++)
        h_input_fp16[i] = __float2half(small_values[i]);

    void* d_fp16_in = nullptr;
    void* d_fp8 = nullptr;
    void* d_fp16_out = nullptr;
    cudaMalloc(&d_fp16_in, n_small * sizeof(half));
    cudaMalloc(&d_fp8, n_small * sizeof(uint8_t));
    cudaMalloc(&d_fp16_out, n_small * sizeof(half));

    cudaMemcpy(d_fp16_in, h_input_fp16.data(), n_small * sizeof(half),
               cudaMemcpyHostToDevice);

    cast_fp16_to_fp8(d_fp16_in, d_fp8, n_small, nullptr);
    cudaDeviceSynchronize();

    cast_fp8_to_fp16(d_fp8, d_fp16_out, n_small, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> h_out_fp16(n_small);
    cudaMemcpy(h_out_fp16.data(), d_fp16_out, n_small * sizeof(half),
               cudaMemcpyDeviceToHost);

    // Check flush-to-zero: very small positive values -> 0
    for (int i = 0; i < 3; i++) {
        float got = __half2float(h_out_fp16[i]);
        EXPECT_NEAR(got, 0.0f, 0.002f)
            << "FP8Saturation: small positive value " << small_values[i]
            << " should flush to 0, got " << got
            << " (index " << i << ")";
    }

    // Check flush-to-zero: very small negative values -> 0 (or -0)
    for (int i = 3; i < 5; i++) {
        float got = __half2float(h_out_fp16[i]);
        EXPECT_NEAR(got, 0.0f, 0.002f)
            << "FP8Saturation: small negative value " << small_values[i]
            << " should flush to 0, got " << got
            << " (index " << i << ")";
    }

    cudaFree(d_fp16_in);
    cudaFree(d_fp8);
    cudaFree(d_fp16_out);
}

// ===========================================================================
// Test 7: QuantGemmINT4Basic -- small fused INT4 dequant + GEMM
//
//   C[M,N] = A[M,K] @ dequant(B_quant[N,K/2], scales[N, K/group_size])
//
//   M=4, K=8, N=4, group_size=8 -> num_groups=1
//   B_quant shape: [4, 4] (4 output channels, 8 weights packed into 4 bytes)
//   scales shape:  [4, 1] (one group per channel)
// ===========================================================================
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

    // --- Prepare B_quant[N, K/2] = [4, 4] packed bytes ---
    // For each output channel n, we have K=8 nibbles packed into 4 bytes.
    // Let's use nibble value 10 for all weights.
    // dequant = (10 - 8) * scale = 2 * scale
    //
    // Channel 0: all nibbles = 10, scale = 1.0 -> dequant weights = 2.0 each
    // Channel 1: all nibbles = 6,  scale = 2.0 -> dequant weights = (6-8)*2 = -4.0
    // Channel 2: all nibbles = 12, scale = 0.5 -> dequant weights = (12-8)*0.5 = 2.0
    // Channel 3: all nibbles = 8,  scale = 3.0 -> dequant weights = (8-8)*3 = 0.0

    std::vector<uint8_t> h_B_quant(N * half_K);

    // Channel 0: nibble=10 (0xA), byte = 0xAA
    for (int j = 0; j < half_K; j++) h_B_quant[0 * half_K + j] = 0xAA;
    // Channel 1: nibble=6 (0x6), byte = 0x66
    for (int j = 0; j < half_K; j++) h_B_quant[1 * half_K + j] = 0x66;
    // Channel 2: nibble=12 (0xC), byte = 0xCC
    for (int j = 0; j < half_K; j++) h_B_quant[2 * half_K + j] = 0xCC;
    // Channel 3: nibble=8 (0x8), byte = 0x88
    for (int j = 0; j < half_K; j++) h_B_quant[3 * half_K + j] = 0x88;

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
    Tensor d_A = make_gpu_tensor(h_A.data(), DType::FP16, {M, K});

    // B_quant: raw bytes, use INT4 dtype. Shape [N, K/2].
    Tensor d_B;
    d_B.dtype = DType::INT4;
    d_B.ndim = 2;
    d_B.shape[0] = N;
    d_B.shape[1] = half_K;
    d_B.compute_strides();
    d_B.on_device = true;
    cudaMalloc(&d_B.data, N * half_K);
    cudaMemcpy(d_B.data, h_B_quant.data(), N * half_K,
               cudaMemcpyHostToDevice);

    // Scales: FP16 [N, num_groups]
    Tensor d_scales = make_gpu_tensor(h_scales.data(), DType::FP16,
                                       {N, num_groups});

    // Output: C [M, N]
    Tensor d_C = alloc_gpu_tensor(DType::FP16, {M, N});

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

} // namespace
} // namespace imp
