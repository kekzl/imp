#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/ssm_scan_reg.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// Register-resident scan (ssm_scan_reg.cu) vs the legacy per-token global-state kernel:
// bitwise equal y, h_state and snapshot at Nemotron-H geometry (64 heads, hd 64, S 128, 8 groups)
// and hd 128, across FP16/FP32 state, gated/ungated, padded real_n and a snapshot row.
struct ScanCase {
    int n_heads, head_dim, state_size, n_groups, n_tokens, real_n, snap_n;
    bool fp16, gate;
};

struct ScanOut {
    std::vector<uint16_t> y;
    std::vector<uint8_t> h, snap;
};

ScanOut run_scan_case(const ScanCase& c, bool reg, uint32_t seed) {
    const int inner = c.n_heads * c.head_dim, bc = c.n_groups * c.state_size;
    const size_t h_elems = static_cast<size_t>(c.n_heads) * c.state_size * c.head_dim;
    const size_t h_bytes = h_elems * (c.fp16 ? 2 : 4);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto dev_half = [&](size_t n, float scale, float bias) {
        std::vector<half> v(n);
        for (auto& e : v)
            e = __float2half(bias + scale * u(rng));
        half* d;
        cudaMalloc(&d, n * sizeof(half));
        cudaMemcpy(d, v.data(), n * sizeof(half), cudaMemcpyHostToDevice);
        return d;
    };
    auto dev_float = [&](size_t n, float scale, float bias) {
        std::vector<float> v(n);
        for (auto& e : v)
            e = bias + scale * u(rng);
        float* d;
        cudaMalloc(&d, n * sizeof(float));
        cudaMemcpy(d, v.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        return d;
    };
    half* x = dev_half(static_cast<size_t>(c.n_tokens) * inner, 1.0f, 0.0f);
    half* B = dev_half(static_cast<size_t>(c.n_tokens) * bc, 1.0f, 0.0f);
    half* C = dev_half(static_cast<size_t>(c.n_tokens) * bc, 1.0f, 0.0f);
    half* dt = dev_half(static_cast<size_t>(c.n_tokens) * c.n_heads, 2.0f, -1.0f);
    half* z = c.gate ? dev_half(static_cast<size_t>(c.n_tokens) * inner, 2.0f, 0.0f) : nullptr;
    float* A = dev_float(c.n_heads, 1.0f, -1.5f);  // negative: decaying state
    float* D = dev_float(c.n_heads, 1.0f, 0.0f);
    float* dtb = dev_float(c.n_heads, 0.5f, 0.0f);
    std::vector<uint8_t> h0(h_bytes);
    for (size_t i = 0; i < h_elems; ++i) {
        if (c.fp16) {
            half v = __float2half(0.5f * u(rng));
            std::memcpy(&h0[2 * i], &v, 2);
        } else {
            float v = 0.5f * u(rng);
            std::memcpy(&h0[4 * i], &v, 4);
        }
    }
    void *h, *snap;
    cudaMalloc(&h, h_bytes);
    cudaMalloc(&snap, h_bytes);
    cudaMemcpy(h, h0.data(), h_bytes, cudaMemcpyHostToDevice);
    cudaMemset(snap, 0x5a, h_bytes);
    half* y;
    cudaMalloc(&y, static_cast<size_t>(c.n_tokens) * inner * sizeof(half));
    int ns[2] = {c.real_n, c.snap_n};
    int* d_ns;
    cudaMalloc(&d_ns, sizeof(ns));
    cudaMemcpy(d_ns, ns, sizeof(ns), cudaMemcpyHostToDevice);

    const SsmScanArgs a{x,
                        B,
                        C,
                        dt,
                        A,
                        D,
                        dtb,
                        h,
                        y,
                        z,
                        c.n_tokens,
                        c.n_heads,
                        c.head_dim,
                        c.state_size,
                        c.n_groups,
                        c.real_n >= 0 ? d_ns : nullptr,
                        c.snap_n > 0 ? snap : nullptr,
                        c.snap_n > 0 ? d_ns + 1 : nullptr,
                        nullptr};
    if (reg) {
        const int s_tiles = std::min(c.state_size, 1024 / c.head_dim) >= 16 ? 16 : 8;
        EXPECT_TRUE(ssm_scan_reg_launch(a, s_tiles, c.fp16));
    } else {
        ssm_scan_legacy_launch(a, c.fp16);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ScanOut o;
    o.y.resize(static_cast<size_t>(c.n_tokens) * inner);
    o.h.resize(h_bytes);
    o.snap.resize(h_bytes);
    cudaMemcpy(o.y.data(), y, o.y.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.h.data(), h, h_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.snap.data(), snap, h_bytes, cudaMemcpyDeviceToHost);
    for (void* p :
         {static_cast<void*>(x), static_cast<void*>(B), static_cast<void*>(C), static_cast<void*>(dt),
          static_cast<void*>(z), static_cast<void*>(A), static_cast<void*>(D), static_cast<void*>(dtb), h,
          snap, static_cast<void*>(y), static_cast<void*>(d_ns)})
        cudaFree(p);
    return o;
}

TEST(SSMScanTest, RegisterScanBitIdenticalToLegacy) {
    SKIP_IF_NO_CUDA();
    const ScanCase cases[] = {
        {64, 64, 128, 8, 1, -1, 0, true, true},      // decode step
        {64, 64, 128, 8, 777, -1, 0, true, true},    // prefill, M not a multiple of anything
        {64, 64, 128, 8, 300, -1, 0, true, false},   // ungated
        {64, 64, 128, 8, 300, -1, 0, false, true},   // FP32 state
        {64, 64, 128, 8, 40, 29, 1, true, true},     // padded verify chunk + first-row snapshot
        {64, 64, 128, 8, 40, 40, 17, false, false},  // snapshot mid-chunk
        {64, 64, 128, 8, 12, 0, 0, true, true},      // real_n 0: state must not move
        {32, 128, 128, 4, 257, -1, 0, true, true},   // head_dim 128 -> s_tiles 8
    };
    for (const ScanCase& c : cases) {
        const ScanOut ref = run_scan_case(c, false, 1234);
        const ScanOut got = run_scan_case(c, true, 1234);
        SCOPED_TRACE(testing::Message() << "hd " << c.head_dim << " n " << c.n_tokens << " real " << c.real_n
                                        << " snap " << c.snap_n << " fp16 " << c.fp16 << " gate " << c.gate);
        EXPECT_EQ(ref.y, got.y);
        EXPECT_EQ(ref.h, got.h);
        EXPECT_EQ(ref.snap, got.snap);
    }
}

// Unsupported geometry must refuse (caller falls back to the legacy kernel), not launch.
TEST(SSMScanTest, RegisterScanRefusesUnsupportedShape) {
    SKIP_IF_NO_CUDA();
    SsmScanArgs a{};
    a.n_heads = 2;
    a.head_dim_ssm = 4;
    a.state_size = 8;
    a.n_groups = 1;
    a.n_tokens = 3;
    EXPECT_FALSE(ssm_scan_reg_launch(a, 8, false));
}

// Gate 1 of the kernel dispatch: FP32-state scan vs a double-precision host reference.
TEST(SSMScanTest, RegisterScanMatchesDoubleReference) {
    SKIP_IF_NO_CUDA();
    const int H = 8, hd = 64, S = 128, G = 2, T = 64, inner = H * hd, bc = G * S;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<half> x(T * inner), Bv(T * bc), Cv(T * bc), dtv(T * H);
    for (auto& e : x)
        e = __float2half(u(rng));
    for (auto& e : Bv)
        e = __float2half(u(rng));
    for (auto& e : Cv)
        e = __float2half(u(rng));
    for (auto& e : dtv)
        e = __float2half(u(rng));
    std::vector<float> A(H), D(H), dtb(H);
    for (int i = 0; i < H; ++i) {
        A[i] = -1.0f - 0.5f * u(rng);
        D[i] = u(rng);
        dtb[i] = 0.3f * u(rng);
    }
    auto up = [](const void* src, size_t bytes) {
        void* d;
        cudaMalloc(&d, bytes);
        cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice);
        return d;
    };
    half* dx = static_cast<half*>(up(x.data(), x.size() * 2));
    half* dB = static_cast<half*>(up(Bv.data(), Bv.size() * 2));
    half* dC = static_cast<half*>(up(Cv.data(), Cv.size() * 2));
    half* ddt = static_cast<half*>(up(dtv.data(), dtv.size() * 2));
    float* dA = static_cast<float*>(up(A.data(), H * 4));
    float* dD = static_cast<float*>(up(D.data(), H * 4));
    float* ddtb = static_cast<float*>(up(dtb.data(), H * 4));
    float* dh;
    cudaMalloc(&dh, static_cast<size_t>(H) * S * hd * 4);
    cudaMemset(dh, 0, static_cast<size_t>(H) * S * hd * 4);
    half* dy;
    cudaMalloc(&dy, static_cast<size_t>(T) * inner * 2);
    const SsmScanArgs a{dx, dB, dC, ddt, dA, dD,      ddtb,    dh,      dy,     nullptr,
                        T,  H,  hd, S,   G,  nullptr, nullptr, nullptr, nullptr};
    ASSERT_TRUE(ssm_scan_reg_launch(a, 16, false));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> y(static_cast<size_t>(T) * inner);
    cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost);

    std::vector<double> hs(static_cast<size_t>(H) * S * hd, 0.0);
    double max_err = 0.0;
    for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < H; ++hh) {
            const int g = hh / (H / G);
            double dtv_d = static_cast<double>(__half2float(dtv[t * H + hh])) + dtb[hh];
            dtv_d = dtv_d > 20.0 ? dtv_d : std::log1p(std::exp(dtv_d));
            const double a_bar = std::exp(dtv_d * A[hh]);
            for (int d = 0; d < hd; ++d) {
                const double xv = __half2float(x[t * inner + hh * hd + d]);
                double acc = 0.0;
                for (int s = 0; s < S; ++s) {
                    double& st = hs[(static_cast<size_t>(hh) * S + s) * hd + d];
                    st = a_bar * st + dtv_d * xv * __half2float(Bv[t * bc + g * S + s]);
                    acc += st * __half2float(Cv[t * bc + g * S + s]);
                }
                const double ref = acc + D[hh] * xv;
                const double got = __half2float(y[t * inner + hh * hd + d]);
                max_err = std::max(max_err, std::abs(got - ref) / std::max(1.0, std::abs(ref)));
            }
        }
    // FP16 output rounding (2^-11 relative) plus fast-math expf/logf: 4e-3 relative.
    EXPECT_LT(max_err, 4e-3);
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dB), static_cast<void*>(dC),
                    static_cast<void*>(ddt), static_cast<void*>(dA), static_cast<void*>(dD),
                    static_cast<void*>(ddtb), static_cast<void*>(dh), static_cast<void*>(dy)})
        cudaFree(p);
}

}  // namespace
}  // namespace imp
