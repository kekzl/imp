#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include "quant/dequant_gpu.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// The GGUF branch of the fp8_ssm_proj decode sidecar: a Q8_0-source GDN
// projection is dequanted (dequant_gpu) at init, per-row FP8-quantized
// (quantize_fp8_rows_async), and decoded via gemv_fp8_rowscale. This chains
// those three kernels exactly as pre_dequant_phase2b does and checks the GEMV
// against an fp64 dot over the format-derived Q8_0 dequant reference. The
// only spread vs that reference is the E4M3 re-quantization (per-row scale =
// row_absmax/448) plus accumulation order, so a wrong block layout, scale
// application, or row indexing shows up as O(1) error against the ~1e-2-class
// rounding floor.
// gemm.fp8_ssm_prefill building blocks: per-row weight quant + per-tensor
// act quant + FP8xFP8 cuBLASLt + column rescale must reproduce the FP16
// GEMM within E4M3 tolerance. Rows carry deliberately different magnitudes
// so a per-tensor-only scale (or a dropped column rescale) fails loudly.
void ssm_prefill_fp8_case(cudaStream_t stream_, int M, int N, int K);

TEST_F(FP8GemmTest, SsmPrefillFp8RowscaleMatchesFp16) {
    ssm_prefill_fp8_case(stream_, 48, 64, 128);      // small
    ssm_prefill_fp8_case(stream_, 15, 2048, 4096);   // odd-M short prompt
    ssm_prefill_fp8_case(stream_, 200, 2048, 4096);  // ppl-corpus M, GDN ssm_out shape
}

void ssm_prefill_fp8_case(cudaStream_t stream_, int M, int N, int K) {
    std::vector<half> h_a(M * K), h_w(N * K);
    // LCG-random values: periodic patterns (i % 17 style) repeat the same
    // quantized value, so E4M3 rounding errors add coherently instead of
    // averaging out and the tolerance triples for test-vector reasons.
    uint32_t lcg = 12345;
    auto frand = [&lcg]() {
        lcg = lcg * 1664525u + 1013904223u;
        return (static_cast<float>(lcg >> 8) / 16777216.0f) - 0.5f;
    };
    for (int i = 0; i < M * K; i++)
        h_a[i] = __float2half(0.3f * frand());
    for (int r = 0; r < N; r++) {
        float row_mag = 0.4f * (1.0f + 0.15f * r);  // heterogeneous rows
        for (int k = 0; k < K; k++)
            h_w[r * K + k] = __float2half(row_mag * frand());
    }
    void *d_a, *d_w, *d_w8, *d_a8, *d_c, *d_cref;
    float *d_row_scales, *d_act_scale, *d_unit;
    ASSERT_EQ(cudaMalloc(&d_a, M * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, N * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w8, (size_t)N * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_a8, (size_t)M * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, M * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_cref, M * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_row_scales, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_act_scale, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_unit, sizeof(float)), cudaSuccess);
    float one = 1.0f;
    cudaMemcpy(d_unit, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);

    int64_t a_shape[] = {M, K}, w_shape[] = {N, K}, c_shape[] = {M, N};
    Tensor A(d_a, QType::F16, 2, a_shape, true);
    Tensor W(d_w, QType::F16, 2, w_shape, true);
    Tensor C(d_c, QType::F16, 2, c_shape, true);
    Tensor Cref(d_cref, QType::F16, 2, c_shape, true);

    // FP16 reference
    gemm_cublaslt(A, W, Cref, 1.0f, 0.0f, nullptr, nullptr, stream_);

    // FP8 path: row-quant W, per-tensor act quant, FP8 GEMM, column rescale
    quantize_fp8_rows_async(d_w, d_w8, N, K, d_row_scales, stream_);
    Tensor A8(d_a8, QType::FP8_E4M3, 2, a_shape, true);
    // Use the ASYNC ext-buffer path — the one the engine dispatch uses.
    int max_grid = ((M * K + 3) / 4 + 255) / 256;
    float *d_bm, *d_am;
    ASSERT_EQ(cudaMalloc(&d_bm, (size_t)max_grid * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_am, sizeof(float)), cudaSuccess);
    quantize_fp16_to_fp8_e4m3(A, A8, d_act_scale, stream_, d_bm, d_am, max_grid);
    Tensor W8(d_w8, QType::FP8_E4M3, 2, w_shape, true);
    gemm_cublaslt(A8, W8, C, 1.0f, 0.0f, d_act_scale, d_unit, stream_);
    scale_cols_fp16(d_c, d_row_scales, M, N, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<half> h_c(M * N), h_cref(M * N);
    cudaMemcpy(h_c.data(), d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cref.data(), d_cref, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    // Normalize by the reference RMS (same convention as the rowscale-GEMV
    // test below): dot products cancel to near zero, so per-element relative
    // error is unbounded there while the absolute error stays E4M3-sized.
    double ref_sq = 0.0;
    for (int i = 0; i < M * N; i++) {
        double ref = __half2float(h_cref[i]);
        ref_sq += ref * ref;
    }
    double ref_rms = std::sqrt(ref_sq / (M * N));
    ASSERT_GT(ref_rms, 0.1);  // the reference itself is non-trivial
    double max_err = 0.0, se = 0.0;
    for (int i = 0; i < M * N; i++) {
        double ref = __half2float(h_cref[i]);
        double got = __half2float(h_c[i]);
        double err = std::fabs(got - ref);
        max_err = std::max(max_err, err);
        se += err * err;
        ASSERT_LT(err, 0.35 * ref_rms) << "i=" << i << " ref=" << ref << " got=" << got;
    }
    double rms_rel = std::sqrt(se / ref_sq);
    printf("[fp8 ssm prefill] M=%d N=%d K=%d max_err=%.3e rms_rel=%.3e ref_rms=%.4f\n", M, N, K,
           max_err, rms_rel, ref_rms);
    // Aggregate quality bar: a dropped column rescale or a transposed
    // operand lands far above it (>100% RMS).
    EXPECT_LT(rms_rel, 0.06);
    EXPECT_GT(max_err, 0.0);  // the path ran and produced non-identical FP8 results

    cudaFree(d_a); cudaFree(d_w); cudaFree(d_w8); cudaFree(d_a8);
    cudaFree(d_c); cudaFree(d_cref); cudaFree(d_row_scales);
    cudaFree(d_act_scale); cudaFree(d_unit); cudaFree(d_bm); cudaFree(d_am);
}

// Negative-result pin for the gemm.fp8_ssm_prefill scoping: the SSM_IN arm
// of the FP8 prefill produced uniform logits in the engine (PPL 4.09 ->
// 248320 on Qwen3.6-35B), and the obvious hypothesis was activation channel
// outliers collapsing the per-tensor E4M3 act scale. This test REFUTES that
// hypothesis: a 400x single-channel outlier still reproduces FP16 within
// ~3.5% RMS through the same pipeline. The engine failure has a different,
// unisolated root cause - which is why the feature covers SSM_OUT only.
TEST_F(FP8GemmTest, PerTensorActQuantSurvivesChannelOutlier) {
    const int M = 64, N = 128, K = 1024;
    std::vector<half> h_a(M * K), h_w(N * K);
    uint32_t lcg = 99;
    auto frand = [&lcg]() {
        lcg = lcg * 1664525u + 1013904223u;
        return (static_cast<float>(lcg >> 8) / 16777216.0f) - 0.5f;
    };
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            h_a[m * K + k] = __float2half(0.3f * frand() * (k == 100 ? 400.0f : 1.0f));
    for (int i = 0; i < N * K; i++)
        h_w[i] = __float2half(0.4f * frand());
    void *d_a, *d_w, *d_w8, *d_a8, *d_c, *d_cref;
    float *d_row_scales, *d_act_scale, *d_unit;
    ASSERT_EQ(cudaMalloc(&d_a, M * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, N * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w8, (size_t)N * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_a8, (size_t)M * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, M * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_cref, M * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_row_scales, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_act_scale, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_unit, sizeof(float)), cudaSuccess);
    float one = 1.0f;
    cudaMemcpy(d_unit, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);
    int64_t a_shape[] = {M, K}, w_shape[] = {N, K}, c_shape[] = {M, N};
    Tensor A(d_a, QType::F16, 2, a_shape, true);
    Tensor W(d_w, QType::F16, 2, w_shape, true);
    Tensor C(d_c, QType::F16, 2, c_shape, true);
    Tensor Cref(d_cref, QType::F16, 2, c_shape, true);
    gemm_cublaslt(A, W, Cref, 1.0f, 0.0f, nullptr, nullptr, stream_);
    quantize_fp8_rows_async(d_w, d_w8, N, K, d_row_scales, stream_);
    Tensor A8(d_a8, QType::FP8_E4M3, 2, a_shape, true);
    quantize_fp16_to_fp8_e4m3(A, A8, d_act_scale, stream_);
    Tensor W8(d_w8, QType::FP8_E4M3, 2, w_shape, true);
    gemm_cublaslt(A8, W8, C, 1.0f, 0.0f, d_act_scale, d_unit, stream_);
    scale_cols_fp16(d_c, d_row_scales, M, N, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<half> h_c(M * N), h_cref(M * N);
    cudaMemcpy(h_c.data(), d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cref.data(), d_cref, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    double se = 0.0, ref_sq = 0.0;
    for (int i = 0; i < M * N; i++) {
        double ref = __half2float(h_cref[i]);
        double got = __half2float(h_c[i]);
        se += (got - ref) * (got - ref);
        ref_sq += ref * ref;
    }
    double rms_rel = std::sqrt(se / ref_sq);
    printf("[fp8 act outlier] M=%d N=%d K=%d rms_rel=%.3f (healthy inputs: 0.036)\n", M, N, K, rms_rel);
    EXPECT_LT(rms_rel, 0.06);  // the outlier does NOT break it
    cudaFree(d_a); cudaFree(d_w); cudaFree(d_w8); cudaFree(d_a8);
    cudaFree(d_c); cudaFree(d_cref); cudaFree(d_row_scales);
    cudaFree(d_act_scale); cudaFree(d_unit);
}

TEST_F(FP8GemmTest, RowscaleGemvFromQ8SourceMatchesReference) {
    const int M = 192, K = 2048;  // K % 32 == 0 (Q8_0 blocks), K % 16 == 0 (sidecar gate)
    constexpr int kQ8BlockBytes = 34;  // [ d:f16 | qs:int8[32] ]
    const int blocks_per_row = K / 32;
    const size_t q8_bytes = static_cast<size_t>(M) * blocks_per_row * kQ8BlockBytes;

    // Random Q8_0 blocks (fixed seed) + fp64 dequant reference: val = d * q.
    std::vector<uint8_t> h_q8(q8_bytes);
    std::vector<double> wref(static_cast<size_t>(M) * K);
    std::srand(1234);
    for (int r = 0; r < M; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* blk = h_q8.data() + (static_cast<size_t>(r) * blocks_per_row + b) * kQ8BlockBytes;
            half d = __float2half(0.001f + 0.05f * (std::rand() / (float)RAND_MAX));
            std::memcpy(blk, &d, 2);
            int8_t* qs = reinterpret_cast<int8_t*>(blk + 2);
            double dd = static_cast<double>(__half2float(d));
            for (int j = 0; j < 32; ++j) {
                qs[j] = static_cast<int8_t>(std::rand() % 255 - 127);
                wref[(static_cast<size_t>(r) * K) + b * 32 + j] = dd * qs[j];
            }
        }
    }
    std::vector<half> h_x(K);
    for (int k = 0; k < K; ++k)
        h_x[k] = __float2half(((std::rand() / (float)RAND_MAX) - 0.5f) * 2.0f);

    void *d_q8 = nullptr, *d_fp16 = nullptr, *d_fp8 = nullptr, *d_x = nullptr, *d_y = nullptr;
    float* d_row_scales = nullptr;
    cudaMalloc(&d_q8, q8_bytes);
    cudaMalloc(&d_fp16, static_cast<size_t>(M) * K * sizeof(half));
    cudaMalloc(&d_fp8, static_cast<size_t>(M) * K);
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, M * sizeof(half));
    cudaMalloc(&d_row_scales, M * sizeof(float));
    cudaMemcpy(d_q8, h_q8.data(), q8_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(half));

    // The sidecar chain: Q8_0 → FP16 scratch → per-row FP8 → rowscale GEMV.
    dequant_gpu(d_q8, d_fp16, QType::Q8_0, M, K, stream_);
    quantize_fp8_rows_async(d_fp16, d_fp8, M, K, d_row_scales, stream_);
    int64_t a_shape[] = {M, K};
    int64_t x_shape[] = {K};
    int64_t y_shape[] = {M};
    Tensor A(d_fp8, QType::FP8_E4M3, 2, a_shape, true);
    Tensor x(d_x, QType::F16, 1, x_shape, true);
    Tensor y(d_y, QType::F16, 1, y_shape, true);
    gemv_fp8_rowscale(A, x, y, d_row_scales, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(half), cudaMemcpyDeviceToHost);

    std::vector<double> yref(M);
    double sum_sq = 0.0;
    for (int r = 0; r < M; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += wref[(static_cast<size_t>(r) * K) + k] * static_cast<double>(__half2float(h_x[k]));
        yref[r] = acc;
        sum_sq += acc * acc;
    }
    double ref_rms = std::sqrt(sum_sq / M);
    double inv = ref_rms > 1e-9 ? 1.0 / ref_rms : 0.0;
    double max_rel = 0.0, sum_rel_sq = 0.0;
    int worst = 0;
    bool any_nan_inf = false;
    for (int r = 0; r < M; ++r) {
        float gf = __half2float(h_y[r]);
        if (std::isnan(gf) || std::isinf(gf))
            any_nan_inf = true;
        double rel = std::fabs(static_cast<double>(gf) - yref[r]) * inv;
        sum_rel_sq += rel * rel;
        if (rel > max_rel) {
            max_rel = rel;
            worst = r;
        }
    }
    double rms_rel = std::sqrt(sum_rel_sq / M);
    printf("[sidecar q8→fp8 rowscale] M=%d K=%d max_rel=%.3e rms_rel=%.3e ref_rms=%.4f "
           "(row=%d gpu=%.4f ref=%.4f)\n",
           M, K, max_rel, rms_rel, ref_rms, worst, __half2float(h_y[worst]), yref[worst]);
    EXPECT_FALSE(any_nan_inf) << "sidecar chain produced NaN/Inf on finite weights";
    // E4M3 rounding floor for this input: dot error rms ≈ √K·rms(w·x)·2⁻⁴/√3
    // ≈ 2.0 against ref_rms ≈ 61 (random signs cancel the reference ~25× below
    // Σ|w·x|, which amplifies the normalized error) → expected rms_rel ≈ 3e-2;
    // measured 2.5e-2 / max 6.4e-2. Layout/scale/indexing bugs produce O(1)
    // errors — orders above these gates.
    EXPECT_LT(max_rel, 1.2e-1) << "rowscale GEMV diverges from Q8_0 dequant reference";
    EXPECT_LT(rms_rel, 4e-2) << "rowscale GEMV rms error above the E4M3 rounding floor";

    cudaFree(d_q8);
    cudaFree(d_fp16);
    cudaFree(d_fp8);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row_scales);
}

}  // namespace
}  // namespace imp
