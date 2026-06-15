#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace imp {
namespace {

class FP8GemmTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

// Independent host decode of an OCP/NVIDIA E4M3 byte (1 sign, 4 exp bias-7,
// 3 mantissa). e=15&m=7 = NaN; e=15&m<=6 are normal finite up to 448; e=0 is
// subnormal (2^(1-7)=2^-6). Deliberately NOT __nv_fp8_e4m3 — this is the
// ground-truth oracle the GPU kernel ((float)__nv_fp8_e4m3) is checked against.
double e4m3_decode_ref(uint8_t b, bool& is_nan) {
    is_nan = false;
    int sign = (b >> 7) & 1;
    int e = (b >> 3) & 0xF;
    int m = b & 0x7;
    double s = sign ? -1.0 : 1.0;
    if (e == 15 && m == 7) {
        is_nan = true;
        return 0.0;
    }
    if (e == 0)
        return s * (static_cast<double>(m) / 8.0) * std::ldexp(1.0, -6);  // subnormal
    return s * (1.0 + static_cast<double>(m) / 8.0) * std::ldexp(1.0, e - 7);
}
inline bool e4m3_byte_is_nan(uint8_t b) { return (b & 0x7F) == 0x7F; }

TEST_F(FP8GemmTest, GemmCublasLtFP16) {
    // Test cuBLASLt GEMM with FP16 operands
    const int M = 32, N = 64, K = 128;
    size_t a_bytes = M * K * sizeof(half);
    size_t b_bytes = N * K * sizeof(half);
    size_t c_bytes = M * N * sizeof(half);

    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_c = nullptr;
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
    Tensor A(d_a, QType::F16, 2, a_shape, true);
    Tensor B(d_b, QType::F16, 2, b_shape, true);
    Tensor C(d_c, QType::F16, 2, c_shape, true);

    gemm_cublaslt(A, B, C, 1.0f, 0.0f, nullptr, nullptr, stream_);
    cudaStreamSynchronize(stream_);

    // Verify: C = A @ B^T, each element should be K * 0.01 * 0.01 = 0.0128
    std::vector<half> h_c(M * N);
    cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost);
    float expected = K * 0.01f * 0.01f;
    float actual = __half2float(h_c[0]);
    EXPECT_NEAR(actual, expected, 0.01f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(FP8GemmTest, GemvFP8Basic) {
    // Test FP8 GEMV: y = A_fp8 @ x_fp16
    const int M = 64, K = 128;
    float scale = 1.0f;

    void* d_a = nullptr;
    void* d_x = nullptr;
    void* d_y = nullptr;
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
    Tensor A(d_a, QType::FP8_E4M3, 2, a_shape, true);
    Tensor x(d_x, QType::F16, 1, x_shape, true);
    Tensor y(d_y, QType::F16, 1, y_shape, true);

    gemv_fp8(A, x, y, scale, stream_);
    cudaStreamSynchronize(stream_);

    // All zeros in A -> y should be all zeros
    std::vector<half> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(half), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(__half2float(h_y[0]), 0.0f, 0.001f);

    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
}

// gemv_fp8 with NONZERO weights vs an independent fp64 reference. The all-zero
// GemvFP8Basic above cannot catch a wrong E4M3 decode or scale application; this
// fills A with real (non-NaN) E4M3 bytes and checks y = sum_k decode(A)*scale*x
// against a host fp64 dot using the independent e4m3_decode_ref LUT.
// Tolerance: fp8 values decode exactly (LUT-exact, all E4M3 values fit f16), so
// the only spread is fp32-GPU vs fp64-host accumulation over K + one f16 output
// round = fp16-class. Normalized by rms(ref) (cancellation-robust), asserted 1e-2.
TEST_F(FP8GemmTest, GemvFP8NonzeroMatchesReference) {
    const int M = 96, K = 256;  // M non-round (row-stride bug surfaces); K%16==0
    const float scale = 0.05f;  // realistic per-tensor weight scale

    std::vector<uint8_t> h_a((size_t)M * K);
    uint32_t s = 0xF8F8u;
    auto next = [&]() { s = s * 1664525u + 1013904223u; return s; };
    for (auto& b : h_a) {
        uint8_t v = static_cast<uint8_t>(next() >> 24);
        if (e4m3_byte_is_nan(v))
            v = 0;  // avoid NaN encodings; the kernel/ref agree on finite bytes
        b = v;
    }
    std::vector<half> h_x(K);
    for (int k = 0; k < K; ++k)
        h_x[k] = __float2half(((next() >> 8) * (1.0f / 8388608.0f) - 1.0f) * 2.0f);  // ~[-2,2]

    void *d_a = nullptr, *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_a, (size_t)M * K);
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, M * sizeof(half));
    cudaMemcpy(d_a, h_a.data(), (size_t)M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(half));

    int64_t a_shape[] = {M, K};
    int64_t x_shape[] = {K};
    int64_t y_shape[] = {M};
    Tensor A(d_a, QType::FP8_E4M3, 2, a_shape, true);
    Tensor x(d_x, QType::F16, 1, x_shape, true);
    Tensor y(d_y, QType::F16, 1, y_shape, true);
    gemv_fp8(A, x, y, scale, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(half), cudaMemcpyDeviceToHost);

    // fp64 reference + scaled error metric.
    std::vector<double> yref(M);
    double sum_sq = 0.0;
    for (int r = 0; r < M; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            bool nan = false;
            double w = e4m3_decode_ref(h_a[(size_t)r * K + k], nan);
            acc += w * static_cast<double>(scale) * static_cast<double>(__half2float(h_x[k]));
        }
        yref[r] = acc;
        sum_sq += acc * acc;
    }
    double ref_rms = std::sqrt(sum_sq / M);
    double inv = ref_rms > 1e-9 ? 1.0 / ref_rms : 0.0;
    double max_rel = 0.0;
    int worst = 0;
    bool any_nan_inf = false;
    for (int r = 0; r < M; ++r) {
        float gf = __half2float(h_y[r]);
        if (std::isnan(gf) || std::isinf(gf))
            any_nan_inf = true;
        double rel = std::fabs(static_cast<double>(gf) - yref[r]) * inv;
        if (rel > max_rel) {
            max_rel = rel;
            worst = r;
        }
    }
    printf("[gemv_fp8 nonzero] M=%d K=%d scale=%.3f max_rel=%.3e ref_rms=%.4f (row=%d gpu=%.4f ref=%.4f)\n",
           M, K, scale, max_rel, ref_rms, worst, __half2float(h_y[worst]), yref[worst]);
    EXPECT_FALSE(any_nan_inf) << "gemv_fp8 produced NaN/Inf on finite weights";
    EXPECT_LT(max_rel, 1e-2) << "gemv_fp8 nonzero output diverges from independent fp64 reference";

    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
}

}  // namespace
}  // namespace imp
