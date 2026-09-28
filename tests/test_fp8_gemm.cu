#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include "quant/dequant_gpu.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
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

// Independent host decode of an OCP/NVIDIA E4M3 byte (1 sign, 4 exp bias-7, 3 mantissa):
// e=15&m=7=NaN, e=15&m<=6 normal finite to 448, e=0 subnormal (2^-6). Deliberately not
// __nv_fp8_e4m3 - this is the ground-truth oracle the GPU kernel is checked against.
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

// gemv_fp8 with NONZERO weights vs an independent fp64 reference (the all-zero GemvFP8Basic
// test can't catch a wrong E4M3 decode or scale application): fills A with real E4M3 bytes,
// checks y=sum_k decode(A)*scale*x against a host fp64 dot using e4m3_decode_ref. fp8
// decodes exactly (LUT-exact, fits f16), so the only spread is fp32-GPU vs fp64-host
// accumulation + one f16 output round; normalized by rms(ref), 1e-2.
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

// GGUF branch of the fp8_ssm_proj decode sidecar: a Q8_0-source GDN projection is
// dequanted, per-row FP8-quantized, and decoded via gemv_fp8_rowscale - chaining the same
// three kernels pre_dequant_phase2b does, checked against an fp64 dot over the
// format-derived Q8_0 dequant reference. Only spread vs that reference is the E4M3
// re-quantization (row_absmax/448) plus accumulation order, so a layout/scale/indexing bug
// shows as O(1) error against the ~1e-2 rounding floor.
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
    // E4M3 rounding floor for this input: dot error rms ~ sqrt(K)*rms(w*x)*2^-4/sqrt(3) ~ 2.0
    // against ref_rms~61 (random signs cancel the reference ~25x, amplifying normalized error) ->
    // expected rms_rel~3e-2, measured 2.5e-2/max 6.4e-2. Layout/scale/indexing bugs are O(1),
    // orders above this gate.
    EXPECT_LT(max_rel, 1.2e-1) << "rowscale GEMV diverges from Q8_0 dequant reference";
    EXPECT_LT(rms_rel, 4e-2) << "rowscale GEMV rms error above the E4M3 rounding floor";

    cudaFree(d_q8);
    cudaFree(d_fp16);
    cudaFree(d_fp8);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row_scales);
}

// Rebuild of a freed ssm_in from its pack's FP8 rows (released_source_gemm_): dequantizing is exact,
// so re-quantizing the rebuilt rows returns the same bytes and row scales.
TEST_F(FP8GemmTest, DequantizeRowsRoundTripsTheSidecar) {
    constexpr int M = 96, K = 2560;
    std::vector<half> src(static_cast<size_t>(M) * K);
    uint32_t s = 17u;
    for (size_t i = 0; i < src.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        const float row_mag = 0.01f * static_cast<float>(1 + (i / K) % 7);  // rows of different magnitude
        src[i] = __float2half(row_mag * (static_cast<float>(s >> 8) / 16777216.0f - 0.5f));
    }
    void *d_src = nullptr, *d_back = nullptr, *q1 = nullptr, *q2 = nullptr;
    float *s1 = nullptr, *s2 = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_back, src.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&q1, src.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&q2, src.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&s1, M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&s2, M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_src, src.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    quantize_fp8_rows_async(d_src, q1, M, K, s1, stream_);
    dequantize_fp8_rows_async(q1, s1, d_back, M, K, stream_);
    quantize_fp8_rows_async(d_back, q2, M, K, s2, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<uint8_t> h1(src.size()), h2(src.size());
    std::vector<float> hs1(M), hs2(M);
    cudaMemcpy(h1.data(), q1, h1.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), q2, h2.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs1.data(), s1, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs2.data(), s2, M * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h1, h2) << "re-quantized rebuild differs from the sidecar";
    for (int r = 0; r < M; ++r)
        EXPECT_NEAR(hs2[r], hs1[r], 1e-3f * hs1[r]) << "row " << r;
    for (void* p : {d_src, d_back, q1, q2})
        cudaFree(p);
    cudaFree(s1);
    cudaFree(s2);
}

// FP8 LM head (gemm.nvfp4_lm_head=fp8): gemv_fp8_rowscale_fp32 vs an fp64 dot over the decoded
// codes, and per-row bits identical for n_rows = 1, 3, 11 (#2152 rule: decode, batch and PPL agree).
TEST_F(FP8GemmTest, RowscaleFp32HeadMatchesReferenceAndIsRowCountInvariant) {
    constexpr int M = 1003, K = 2560, N = 11;
    std::vector<half> w(static_cast<size_t>(M) * K), x(static_cast<size_t>(N) * K);
    uint32_t s = 29u;
    auto rnd = [&s] {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / 16777216.0f - 0.5f;
    };
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2half(0.02f * static_cast<float>(1 + (i / K) % 9) * rnd());
    for (auto& v : x)
        v = __float2half(4.0f * rnd());
    void *d_w = nullptr, *d_q = nullptr, *d_x = nullptr;
    float *d_s = nullptr, *d_all = nullptr, *d_one = nullptr, *d_three = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, w.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_q, w.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_all, static_cast<size_t>(N) * M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_one, static_cast<size_t>(N) * M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_three, static_cast<size_t>(N) * M * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
    quantize_fp8_rows_async(d_w, d_q, M, K, d_s, stream_);
    const half* hx = static_cast<const half*>(d_x);
    ASSERT_TRUE(gemv_fp8_rowscale_fp32(d_q, d_s, hx, d_all, M, K, N, stream_));
    for (int r = 0; r < N; ++r)
        ASSERT_TRUE(gemv_fp8_rowscale_fp32(d_q, d_s, hx + static_cast<size_t>(r) * K,
                                           d_one + static_cast<size_t>(r) * M, M, K, 1, stream_));
    for (int r = 0; r < N; r += 3)
        ASSERT_TRUE(gemv_fp8_rowscale_fp32(d_q, d_s, hx + static_cast<size_t>(r) * K,
                                           d_three + static_cast<size_t>(r) * M, M, K, std::min(3, N - r),
                                           stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<uint8_t> q(w.size());
    std::vector<float> sc(M), all(static_cast<size_t>(N) * M), one(all.size()), three(all.size());
    cudaMemcpy(q.data(), d_q, q.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(sc.data(), d_s, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(all.data(), d_all, all.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(one.data(), d_one, one.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(three.data(), d_three, three.size() * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, std::memcmp(all.data(), one.data(), all.size() * sizeof(float))) << "n=11 vs n=1 bits";
    EXPECT_EQ(0, std::memcmp(all.data(), three.data(), all.size() * sizeof(float))) << "n=11 vs n=3 bits";
    double max_rel = 0.0;
    for (int r = 0; r < N; ++r) {
        for (int m = 0; m < M; ++m) {
            double acc = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                bool nan = false;
                const double wv = e4m3_decode_ref(q[static_cast<size_t>(m) * K + k], nan) * sc[m];
                const double xv = __half2float(x[static_cast<size_t>(r) * K + k]);
                acc += wv * xv;
                mag += std::fabs(wv * xv);
            }
            const double rel = std::fabs(all[static_cast<size_t>(r) * M + m] - acc) / std::max(mag, 1e-30);
            max_rel = std::max(max_rel, rel);
        }
    }
    // FP32 accumulation over K=2560 terms: error <= ~K * 2^-24 of sum|w*x| (1.5e-4); layout bugs are O(1).
    EXPECT_LT(max_rel, 1.5e-4) << "rowscale FP32 GEMV diverges from the fp64 dot over its own codes";
    for (void* p : {d_w, d_q, d_x})
        cudaFree(p);
    for (float* p : {d_s, d_all, d_one, d_three})
        cudaFree(p);
}

}  // namespace
}  // namespace imp
