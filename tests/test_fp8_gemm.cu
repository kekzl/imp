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
// codes; per-row bits identical when the 40 rows run in groups of 1, 3, 11, 32 or 40 (#2152 rule).
TEST_F(FP8GemmTest, RowscaleFp32HeadMatchesReferenceAndIsRowCountInvariant) {
    constexpr int M = 1003, K = 2560, N = 40;
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
    float *d_s = nullptr, *d_y = nullptr;
    const size_t y_elems = static_cast<size_t>(N) * M;
    ASSERT_EQ(cudaMalloc(&d_w, w.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_q, w.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, y_elems * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
    quantize_fp8_rows_async(d_w, d_q, M, K, d_s, stream_);
    const half* hx = static_cast<const half*>(d_x);
    auto run_groups = [&](int group) {
        std::vector<float> out(y_elems);
        cudaMemset(d_y, 0xff, y_elems * sizeof(float));  // NaN fill: an unwritten logit fails below
        for (int r = 0; r < N; r += group)
            EXPECT_TRUE(gemv_fp8_rowscale_fp32(d_q, d_s, hx + static_cast<size_t>(r) * K,
                                               d_y + static_cast<size_t>(r) * M, M, K, std::min(group, N - r),
                                               stream_));
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        cudaMemcpy(out.data(), d_y, y_elems * sizeof(float), cudaMemcpyDeviceToHost);
        return out;
    };
    const std::vector<float> all = run_groups(N);
    for (int group : {1, 3, 11, 32})
        EXPECT_EQ(0, std::memcmp(all.data(), run_groups(group).data(), y_elems * sizeof(float)))
            << "n=" << N << " vs groups of " << group << ": row bits differ";

    std::vector<uint8_t> q(w.size());
    std::vector<float> sc(M);
    cudaMemcpy(q.data(), d_q, q.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(sc.data(), d_s, M * sizeof(float), cudaMemcpyDeviceToHost);
    double max_rel = 0.0;
    bool finite = true;
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
            const float got = all[static_cast<size_t>(r) * M + m];
            finite = finite && std::isfinite(got);
            max_rel = std::max(max_rel, std::fabs(got - acc) / std::max(mag, 1e-30));
        }
    }
    // Every row-tile variant reproduces the default bits (same per-row MMA chain).
    for (int nt : {4}) {
        std::vector<float> out(y_elems);
        cudaMemset(d_y, 0xff, y_elems * sizeof(float));
        for (int r = 0; r < N; r += 32)
            EXPECT_TRUE(gemv_fp8_rowscale_fp32_tuned(d_q, d_s, hx + static_cast<size_t>(r) * K,
                                                     d_y + static_cast<size_t>(r) * M, M, K,
                                                     std::min(32, N - r), nt, stream_));
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        cudaMemcpy(out.data(), d_y, y_elems * sizeof(float), cudaMemcpyDeviceToHost);
        EXPECT_EQ(0, std::memcmp(all.data(), out.data(), y_elems * sizeof(float))) << "nt=" << nt;
    }
    printf("[fp8 head mma] M=%d K=%d N=%d max_rel=%.3e\n", M, K, N, max_rel);
    EXPECT_TRUE(finite) << "unwritten or non-finite logit";
    // FP32 accumulation over K=2560 terms: error <= ~K * 2^-24 of sum|w*x| (1.5e-4); layout bugs are O(1).
    EXPECT_LT(max_rel, 1.5e-4) << "FP8 head GEMM diverges from the fp64 dot over its own codes";
    for (void* p : {d_w, d_q, d_x})
        cudaFree(p);
    cudaFree(d_s);
    cudaFree(d_y);
}

// FP16 twin (GDN projections at 2 <= M <= 32): every output equals the FP32 head's value rounded to
// FP16, for the narrow (kNT 2: 2560 rows), mid (kNT 4) and wide (kNT 8) launch, at 8 and 29 rows.
TEST_F(FP8GemmTest, RowscaleF16MatchesFp32HeadRounded) {
    constexpr int K = 2560, NMAX = 29;
    for (int M : {2560, 12160, 24320}) {
        std::vector<half> w(static_cast<size_t>(M) * K), x(static_cast<size_t>(NMAX) * K);
        uint32_t s = 31u + static_cast<uint32_t>(M);
        auto rnd = [&s] {
            s = s * 1664525u + 1013904223u;
            return static_cast<float>(s >> 8) / 16777216.0f - 0.5f;
        };
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = __float2half(0.02f * static_cast<float>(1 + (i / K) % 9) * rnd());
        for (auto& v : x)
            v = __float2half(4.0f * rnd());
        void *d_w = nullptr, *d_q = nullptr, *d_x = nullptr, *d_h = nullptr;
        float *d_s = nullptr, *d_y = nullptr;
        const size_t y_elems = static_cast<size_t>(NMAX) * M;
        ASSERT_EQ(cudaMalloc(&d_w, w.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_q, w.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, y_elems * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_h, y_elems * sizeof(half)), cudaSuccess);
        cudaMemcpy(d_w, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
        quantize_fp8_rows_async(d_w, d_q, M, K, d_s, stream_);
        const half* hx = static_cast<const half*>(d_x);
        for (int n : {8, NMAX}) {
            cudaMemset(d_h, 0xff, y_elems * sizeof(half));  // NaN fill: an unwritten output fails below
            ASSERT_TRUE(gemv_fp8_rowscale_fp32(d_q, d_s, hx, d_y, M, K, n, stream_));
            ASSERT_TRUE(gemv_fp8_rowscale_f16(d_q, d_s, hx, static_cast<half*>(d_h), M, K, n, stream_));
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            std::vector<float> f32(static_cast<size_t>(n) * M);
            std::vector<half> f16(static_cast<size_t>(n) * M);
            cudaMemcpy(f32.data(), d_y, f32.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(f16.data(), d_h, f16.size() * sizeof(half), cudaMemcpyDeviceToHost);
            size_t bad = 0;
            for (size_t i = 0; i < f32.size(); ++i) {
                const half ref = __float2half(f32[i]);
                bad += std::memcmp(&ref, &f16[i], sizeof(half)) != 0;
            }
            EXPECT_EQ(bad, 0u) << "M=" << M << " n=" << n
                               << ": FP16 outputs differ from the rounded FP32 head";
        }
        for (void* p : {d_w, d_q, d_x, d_h})
            cudaFree(p);
        cudaFree(d_s);
        cudaFree(d_y);
    }
}

// Qwen3-14B FP8 head (151936 x 5120, 778 MB > L2) at 29 rows: us/call per row tile vs 1628 GB/s.
TEST_F(FP8GemmTest, RowscaleFp32HeadRowTileSweep) {
    if (getenv("IMP_FP8_HEAD_SWEEP") == nullptr)
        GTEST_SKIP() << "set IMP_FP8_HEAD_SWEEP=1 to run the head sweep";
    constexpr int M = 151936, K = 5120;
    const int N = getenv("IMP_FP8_HEAD_ROWS") ? atoi(getenv("IMP_FP8_HEAD_ROWS")) : 29;
    void *d_w = nullptr, *d_q = nullptr, *d_x = nullptr;
    float *d_s = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, static_cast<size_t>(M) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_q, static_cast<size_t>(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(N) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(N) * M * sizeof(float)), cudaSuccess);
    cudaMemset(d_w, 0x2a, static_cast<size_t>(M) * K * sizeof(half));
    cudaMemset(d_x, 0x2a, static_cast<size_t>(N) * K * sizeof(half));
    quantize_fp8_rows_async(d_w, d_q, M, K, d_s, stream_);
    cudaFree(d_w);
    const half* hx = static_cast<const half*>(d_x);
    const double floor_us = (static_cast<double>(M) * K + M * 4.0) / 1628e3;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int nt : {4, 8}) {
        for (int i = 0; i < 2000; ++i)  // >1 s clock ramp
            ASSERT_TRUE(gemv_fp8_rowscale_fp32_tuned(d_q, d_s, hx, d_y, M, K, N, nt, stream_));
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            float ms = 0.0f;
            cudaEventRecord(t0, stream_);
            for (int i = 0; i < 200; ++i)
                (void)gemv_fp8_rowscale_fp32_tuned(d_q, d_s, hx, d_y, M, K, N, nt, stream_);  // timing loop
            cudaEventRecord(t1, stream_);
            ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
            cudaEventElapsedTime(&ms, t0, t1);
            best = std::min(best, ms * 1000.0 / 200);
        }
        printf(
            "[ HEAD     ] M=%d K=%d rows=%d nt=%d (%d rows/block): %.1f us/call, %.0f GB/s, floor %.1f us\n",
            M, K, N, nt, nt * 8, best, (static_cast<double>(M) * K + M * 4.0) / best / 1e3, floor_us);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_q);
    cudaFree(d_x);
    cudaFree(d_s);
    cudaFree(d_y);
}

// The device-scale activation quantize (native FP8 prefill) is byte-identical to the host-scale
// kernel, saturates, and declines a length that is not a multiple of 8.
TEST_F(FP8GemmTest, DeviceScaleQuantizeMatchesHostScaleKernel) {
    const int n = 4096 + 8;
    const float scale = 0.0123f;
    std::vector<half> h_x(n);
    srand(7);
    for (int i = 0; i < n; ++i)
        h_x[i] = __float2half(((rand() % 20001) - 10000) / 1000.0f);  // |x| <= 10: x/scale > 448 saturates
    void *d_x = nullptr, *d_a = nullptr, *d_b = nullptr;
    float* d_s = nullptr;
    cudaMalloc(&d_x, n * sizeof(half));
    cudaMalloc(&d_a, n);
    cudaMalloc(&d_b, n);
    cudaMalloc(&d_s, sizeof(float));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, &scale, sizeof(float), cudaMemcpyHostToDevice);
    quantize_fp16_to_fp8_e4m3_scaled(d_x, d_a, n, scale, stream_);
    ASSERT_TRUE(quantize_fp16_to_fp8_e4m3_dscale(d_x, d_b, n, d_s, stream_));
    EXPECT_FALSE(quantize_fp16_to_fp8_e4m3_dscale(d_x, d_b, n - 4, d_s, stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    std::vector<uint8_t> a(n), b(n);
    cudaMemcpy(a.data(), d_a, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, n, cudaMemcpyDeviceToHost);
    int n_sat = 0;
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(a[i], b[i]) << "element " << i;
        n_sat += (b[i] & 0x7F) == 0x7E;  // +-448
    }
    EXPECT_GT(n_sat, 0);
    cudaFree(d_x);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_s);
}

// Native FP8 prefill: FP8 activation x FP8 weight through cuBLASLt with device scales, at an M
// that is not a multiple of 16, against a host product of the decoded E4M3 operands.
TEST_F(FP8GemmTest, NativeFp8PrefillGemmMatchesDecodedReference) {
    const int M = 17, N = 64, K = 256;
    const float a_scale = 0.02f, w_scale = 0.003f;
    std::vector<half> h_x(static_cast<size_t>(M) * K);
    std::vector<uint8_t> h_w(static_cast<size_t>(N) * K);
    srand(11);
    for (auto& v : h_x)
        v = __float2half(((rand() % 2001) - 1000) / 400.0f);
    for (auto& v : h_w) {
        v = static_cast<uint8_t>(rand() & 0xFF);
        if (e4m3_byte_is_nan(v))
            v = 0;
    }
    void *d_x = nullptr, *d_x8 = nullptr, *d_w = nullptr, *d_y = nullptr;
    float* d_sc = nullptr;
    cudaMalloc(&d_x, h_x.size() * sizeof(half));
    cudaMalloc(&d_x8, h_x.size());
    cudaMalloc(&d_w, h_w.size());
    cudaMalloc(&d_y, static_cast<size_t>(M) * N * sizeof(half));
    cudaMalloc(&d_sc, 2 * sizeof(float));
    const float h_sc[2] = {a_scale, w_scale};
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, h_sc, sizeof(h_sc), cudaMemcpyHostToDevice);

    ASSERT_TRUE(quantize_fp16_to_fp8_e4m3_dscale(d_x, d_x8, h_x.size(), d_sc, stream_));
    int64_t xs[2] = {M, K}, ws[2] = {N, K}, ys[2] = {M, N};
    Tensor x8(d_x8, QType::FP8_E4M3, 2, xs, true), w8(d_w, QType::FP8_E4M3, 2, ws, true);
    Tensor y(d_y, QType::F16, 2, ys, true);
    gemm_cublaslt(x8, w8, y, 1.0f, 0.0f, d_sc, d_sc + 1, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<uint8_t> h_x8(h_x.size());
    std::vector<half> h_y(static_cast<size_t>(M) * N);
    cudaMemcpy(h_x8.data(), d_x8, h_x8.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(half), cudaMemcpyDeviceToHost);
    double max_rel = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0, mag = 0.0;
            bool nan = false;
            for (int k = 0; k < K; ++k) {
                const double p = e4m3_decode_ref(h_x8[static_cast<size_t>(m) * K + k], nan) *
                                 e4m3_decode_ref(h_w[static_cast<size_t>(n) * K + k], nan);
                ref += p;
                mag += std::fabs(p);
            }
            ref *= static_cast<double>(a_scale) * w_scale;
            mag *= static_cast<double>(a_scale) * w_scale;
            const double got = __half2float(h_y[static_cast<size_t>(m) * N + n]);
            max_rel = std::max(max_rel, std::fabs(got - ref) / std::max(mag, 1e-6));
        }
    }
    EXPECT_LT(max_rel, 2e-3);  // FP16 output rounding against the summed magnitude
    cudaFree(d_x);
    cudaFree(d_x8);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_sc);
}

// FP8 x FP8 rows do not depend on M (#2152): the first 16 rows are bit-identical across chunk sizes
// (down_proj shape; per-M algo picks differ bitwise).
TEST_F(FP8GemmTest, Fp8GemmRowsInvariantAcrossM) {
    const int N = 4096, K = 12288, R = 16;
    const int Ms[] = {2048, 17, 64, 208, 272, 512, 1000};
    const int Mmax = 2048;
    std::vector<uint8_t> h_x(static_cast<size_t>(Mmax) * K), h_w(static_cast<size_t>(N) * K);
    srand(5);
    for (auto* v : {&h_x, &h_w})
        for (auto& b : *v) {
            b = static_cast<uint8_t>(rand() & 0xFF);
            if (e4m3_byte_is_nan(b))
                b = 0;
        }
    void *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    float* d_sc = nullptr;
    cudaMalloc(&d_x, h_x.size());
    cudaMalloc(&d_w, h_w.size());
    cudaMalloc(&d_y, static_cast<size_t>(Mmax) * N * sizeof(half));
    cudaMalloc(&d_sc, 2 * sizeof(float));
    const float h_sc[2] = {0.01f, 0.002f};
    cudaMemcpy(d_x, h_x.data(), h_x.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, h_sc, sizeof(h_sc), cudaMemcpyHostToDevice);
    std::vector<half> ref(static_cast<size_t>(R) * N), got(ref.size());
    for (int M : Ms) {
        int64_t xs[2] = {M, K}, ws[2] = {N, K}, ys[2] = {M, N};
        Tensor x8(d_x, QType::FP8_E4M3, 2, xs, true), w8(d_w, QType::FP8_E4M3, 2, ws, true);
        Tensor y(d_y, QType::F16, 2, ys, true);
        gemm_cublaslt(x8, w8, y, 1.0f, 0.0f, d_sc, d_sc + 1, stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        auto& dst = (M == Ms[0]) ? ref : got;
        cudaMemcpy(dst.data(), d_y, dst.size() * sizeof(half), cudaMemcpyDeviceToHost);
        if (M == Ms[0])
            continue;
        int diff = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            diff += std::memcmp(&ref[i], &got[i], sizeof(half)) != 0;
        EXPECT_EQ(diff, 0) << "M=" << M;
    }
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_sc);
}

}  // namespace
}  // namespace imp
