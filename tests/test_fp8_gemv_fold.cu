// FP8 GEMVs inside the RMSNorm fold (quant/fp8_gemv_fold.h) at Qwen3.8-27B GDN dims: the fused input
// launch against gemv_fp8 (bit-identical with the fold off), and residual + fold emit followed by the
// folded input launch against a host rmsnorm + fp32 dot.

#include "quant/fp8_gemv_fold.h"
#include "compute/gemm.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace imp {
namespace {

constexpr int kK = 5120;
constexpr int kInRows = 10240;
constexpr int kGateRows = 6144;
constexpr int kAbRows = 48;
constexpr int kInner = 6144;

// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa; exponent 0 is subnormal.
float e4m3(uint8_t b) {
    const float sign = (b & 0x80) != 0 ? -1.0f : 1.0f;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    return sign * (e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + (m / 8.0f), e - 7));
}

class Fp8GemvFold : public ::testing::Test {
protected:
    void SetUp() override {
        if (cudaSetDevice(0) != cudaSuccess)
            GTEST_SKIP() << "No CUDA device";
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        for (void* p : allocs_)
            cudaFree(p);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    // Bytes with exponent 1..10: finite, never the NaN pattern.
    static std::vector<uint8_t> fp8_weight(int N, int K, unsigned seed) {
        std::vector<uint8_t> w(static_cast<size_t>(N) * K);
        unsigned s = seed;
        for (auto& b : w) {
            s = (s * 1664525u) + 1013904223u;
            const unsigned r = s >> 16;
            b = static_cast<uint8_t>(((r & 1u) << 7) | ((1u + ((r >> 1) % 10u)) << 3) | ((r >> 8) & 7u));
        }
        return w;
    }
    static std::vector<half> f16_vec(int n, int seed, float step) {
        std::vector<half> v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            v[i] = __float2half((((i * 23) + seed) % 29 - 14) * step);
        return v;
    }

    template <class T>
    T* upload(const std::vector<T>& h) {
        T* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        allocs_.push_back(d);
        return d;
    }
    template <class T>
    T* zeros(int n) {
        return upload(std::vector<T>(static_cast<size_t>(n)));
    }
    template <class T>
    std::vector<T> fetch(const T* d, int n) {
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        std::vector<T> h(static_cast<size_t>(n));
        EXPECT_EQ(cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    }

    static std::vector<float> ref_fp8(const std::vector<uint8_t>& w, int N, int K, float scale,
                                      const std::vector<float>& x) {
        std::vector<float> ref(static_cast<size_t>(N));
        for (int r = 0; r < N; ++r) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += static_cast<double>(e4m3(w[(static_cast<size_t>(r) * K) + k]) * scale) * x[k];
            ref[r] = static_cast<float>(s);
        }
        return ref;
    }
    static std::vector<float> ref_f16(const std::vector<half>& w, int N, int K, const std::vector<float>& x) {
        std::vector<float> ref(static_cast<size_t>(N));
        for (int r = 0; r < N; ++r) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += static_cast<double>(__half2float(w[(static_cast<size_t>(r) * K) + k])) * x[k];
            ref[r] = static_cast<float>(s);
        }
        return ref;
    }
    static int count_off(const std::vector<half>& got, const std::vector<float>& ref, float rel, float abs) {
        int bad = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const float g = __half2float(got[i]);
            if (std::isnan(g) || std::fabs(g - ref[i]) > abs + (rel * std::fabs(ref[i])))
                ++bad;
        }
        return bad;
    }
    static int count_diff_bits(const std::vector<half>& a, const std::vector<half>& b) {
        int n = 0;
        for (size_t i = 0; i < a.size(); ++i)
            n += __half_as_ushort(a[i]) != __half_as_ushort(b[i]) ? 1 : 0;
        return n;
    }

    cudaStream_t stream_ = nullptr;
    std::vector<void*> allocs_;
};

TEST_F(Fp8GemvFold, FusedInputWithoutFoldMatchesGemvFp8) {
    const float s_in = 0.013f, s_gate = 0.021f;
    const auto h_in = fp8_weight(kInRows, kK, 1), h_gate = fp8_weight(kGateRows, kK, 2);
    const auto h_a = f16_vec(kAbRows * kK, 3, 0.01f), h_b = f16_vec(kAbRows * kK, 4, 0.01f);
    const auto h_x = f16_vec(kK, 5, 0.02f);
    uint8_t *d_in = upload(h_in), *d_gate = upload(h_gate);
    half *d_a = upload(h_a), *d_b = upload(h_b), *d_x = upload(h_x);

    half *r_in = zeros<half>(kInRows), *r_gate = zeros<half>(kGateRows);
    {
        int64_t in_shape[2] = {kInRows, kK}, gate_shape[2] = {kGateRows, kK}, x_shape[1] = {kK};
        int64_t yi_shape[1] = {kInRows}, yg_shape[1] = {kGateRows};
        Tensor wi(d_in, QType::FP8_E4M3, 2, in_shape, true), wg(d_gate, QType::FP8_E4M3, 2, gate_shape, true);
        Tensor x_t(d_x, QType::F16, 1, x_shape, true);
        Tensor yi(r_in, QType::F16, 1, yi_shape, true), yg(r_gate, QType::F16, 1, yg_shape, true);
        gemv_fp8(wi, x_t, yi, s_in, stream_);
        gemv_fp8(wg, x_t, yg, s_gate, stream_);
    }

    Fp8GdnInputArgs a;
    a.w_in = d_in;
    a.w_gate = d_gate;
    a.w_alpha = d_a;
    a.w_beta = d_b;
    a.s_in = s_in;
    a.s_gate = s_gate;
    a.x = d_x;
    a.y_in = zeros<half>(kInRows);
    a.y_gate = zeros<half>(kGateRows);
    a.y_alpha = zeros<half>(kAbRows);
    a.y_beta = zeros<half>(kAbRows);
    a.n_in = kInRows;
    a.n_gate = kGateRows;
    a.n_ab = kAbRows;
    a.K = kK;
    gemv_fp8_gdn_input(a, stream_);

    EXPECT_EQ(count_diff_bits(fetch(a.y_in, kInRows), fetch(r_in, kInRows)), 0)
        << "in rows differ from gemv_fp8";
    EXPECT_EQ(count_diff_bits(fetch(a.y_gate, kGateRows), fetch(r_gate, kGateRows)), 0)
        << "gate rows differ from gemv_fp8";
    std::vector<float> x(kK);
    for (int i = 0; i < kK; ++i)
        x[i] = __half2float(h_x[i]);
    EXPECT_EQ(count_off(fetch(a.y_alpha, kAbRows), ref_f16(h_a, kAbRows, kK, x), 0.002f, 0.002f), 0)
        << "alpha";
    EXPECT_EQ(count_off(fetch(a.y_beta, kAbRows), ref_f16(h_b, kAbRows, kK, x), 0.002f, 0.002f), 0) << "beta";
    // Per-row scales replace the tensor scale: row r scaled by rs[r].
    std::vector<float> rs(kGateRows);
    for (int r = 0; r < kGateRows; ++r)
        rs[r] = 0.01f + (0.00001f * static_cast<float>(r % 97));
    a.rs_gate = upload(rs);
    gemv_fp8_gdn_input(a, stream_);
    const auto g = fetch(a.y_gate, kGateRows);
    auto ref = ref_fp8(h_gate, kGateRows, kK, 1.0f, x);
    for (int r = 0; r < kGateRows; ++r)
        ref[r] *= rs[r];
    EXPECT_EQ(count_off(g, ref, 0.002f, 0.01f), 0) << "gate rows with per-row scales";
}

TEST_F(Fp8GemvFold, ResidualEmitThenFoldedInputMatchesHostNorm) {
    const float s_out = 0.004f, s_in = 0.013f, s_gate = 0.021f, eps = 1e-6f;
    const auto h_out = fp8_weight(kK, kInner, 7), h_in = fp8_weight(kInRows, kK, 8);
    const auto h_gate = fp8_weight(kGateRows, kK, 9);
    const auto h_a = f16_vec(kAbRows * kK, 10, 0.01f), h_b = f16_vec(kAbRows * kK, 11, 0.01f);
    const auto h_y = f16_vec(kInner, 12, 0.03f), h_h0 = f16_vec(kK, 13, 0.25f);
    std::vector<half> h_gamma(kK);
    for (int i = 0; i < kK; ++i)
        h_gamma[i] = __float2half(0.5f + (0.001f * static_cast<float>(i % 400)));
    uint8_t *d_out = upload(h_out), *d_in = upload(h_in), *d_gate = upload(h_gate);
    half *d_a = upload(h_a), *d_b = upload(h_b), *d_y = upload(h_y), *d_h = upload(h_h0);
    half *d_gamma = upload(h_gamma), *d_no = zeros<half>(kK);
    unsigned long long* d_ssq = zeros<unsigned long long>(1);

    const NvFP4NormFoldOut fout{d_gamma, 0.0f, d_no, d_ssq};
    gemv_fp8_residual(d_out, s_out, nullptr, d_y, d_h, kK, kInner, fout, stream_);

    // h = W_out @ y + h0.
    std::vector<float> y(kInner);
    for (int i = 0; i < kInner; ++i)
        y[i] = __half2float(h_y[i]);
    auto h_ref = ref_fp8(h_out, kK, kInner, s_out, y);
    for (int i = 0; i < kK; ++i)
        h_ref[i] += __half2float(h_h0[i]);
    const auto h_got = fetch(d_h, kK);
    EXPECT_EQ(count_off(h_got, h_ref, 0.002f, 0.01f), 0) << "residual output";
    // The emit is not the norm output: prescaled gamma * h, the 1/rms factor is the consumer's.
    double ssq = 0.0;
    for (int i = 0; i < kK; ++i)
        ssq += static_cast<double>(__half2float(h_got[i])) * __half2float(h_got[i]);
    const float inv_rms = 1.0f / std::sqrt(static_cast<float>(ssq / kK) + eps);
    std::vector<float> x_norm(kK);
    for (int i = 0; i < kK; ++i)
        x_norm[i] = __half2float(h_got[i]) * __half2float(h_gamma[i]) * inv_rms;
    EXPECT_GT(count_off(fetch(d_no, kK), x_norm, 0.01f, 0.001f), kK / 2) << "emit already normalised";

    Fp8GdnInputArgs a;
    a.w_in = d_in;
    a.w_gate = d_gate;
    a.w_alpha = d_a;
    a.w_beta = d_b;
    a.s_in = s_in;
    a.s_gate = s_gate;
    a.x = d_no;
    a.y_in = zeros<half>(kInRows);
    a.y_gate = zeros<half>(kGateRows);
    a.y_alpha = zeros<half>(kAbRows);
    a.y_beta = zeros<half>(kAbRows);
    a.n_in = kInRows;
    a.n_gate = kGateRows;
    a.n_ab = kAbRows;
    a.K = kK;
    a.fold.ssq = d_ssq;
    a.fold.d = static_cast<float>(kK);
    a.fold.eps = eps;
    gemv_fp8_gdn_input(a, stream_);

    EXPECT_EQ(count_off(fetch(a.y_in, kInRows), ref_fp8(h_in, kInRows, kK, s_in, x_norm), 0.01f, 0.02f), 0)
        << "in";
    EXPECT_EQ(count_off(fetch(a.y_gate, kGateRows), ref_fp8(h_gate, kGateRows, kK, s_gate, x_norm), 0.01f,
                        0.02f),
              0)
        << "gate";
    EXPECT_EQ(count_off(fetch(a.y_alpha, kAbRows), ref_f16(h_a, kAbRows, kK, x_norm), 0.01f, 0.02f), 0)
        << "alpha";
    EXPECT_EQ(count_off(fetch(a.y_beta, kAbRows), ref_f16(h_b, kAbRows, kK, x_norm), 0.01f, 0.02f), 0)
        << "beta";
}

}  // namespace
}  // namespace imp
