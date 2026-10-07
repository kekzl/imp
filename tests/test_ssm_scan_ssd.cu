#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/ssm_scan_ssd.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// Chunked SSD scan (ssm_scan_ssd.cu) vs a double-precision sequential reference.
struct SsdCase {
    int n_heads, head_dim, n_groups, n_tokens, real_n;  // real_n < 0: no device length
    bool fp16, gate;
    float bc_scale;  // B/C magnitude; 8 drives the intra-chunk matrix past the fp16 range
};

struct SsdErr {
    double y = 0.0, h = 0.0;
};

SsdErr run_ssd_case(const SsdCase& c, uint32_t seed) {
    constexpr int S = 128;
    const int H = c.n_heads, hd = c.head_dim, G = c.n_groups, T = c.n_tokens;
    const int inner = H * hd, bc = G * S;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto halves = [&](size_t n, float scale, float bias) {
        std::vector<half> v(n);
        for (auto& e : v)
            e = __float2half(bias + scale * u(rng));
        return v;
    };
    std::vector<half> x = halves(static_cast<size_t>(T) * inner, 1.0f, 0.0f);
    std::vector<half> Bv = halves(static_cast<size_t>(T) * bc, c.bc_scale, 0.0f);
    std::vector<half> Cv = halves(static_cast<size_t>(T) * bc, c.bc_scale, 0.0f);
    std::vector<half> dtv = halves(static_cast<size_t>(T) * H, 2.0f, -1.0f);
    std::vector<half> zv = c.gate ? halves(static_cast<size_t>(T) * inner, 2.0f, 0.0f) : std::vector<half>{};
    std::vector<float> A(H), D(H), dtb(H);
    for (int i = 0; i < H; ++i) {
        A[i] = -1.5f + u(rng);
        D[i] = u(rng);
        dtb[i] = 0.3f * u(rng);
    }
    const size_t h_elems = static_cast<size_t>(H) * S * hd;
    std::vector<double> hs(h_elems);
    for (auto& e : hs)
        e = 0.5 * u(rng);

    auto up = [](const void* src, size_t bytes) {
        void* d = nullptr;
        cudaMalloc(&d, bytes);
        cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice);
        return d;
    };
    void* dh;
    cudaMalloc(&dh, h_elems * (c.fp16 ? 2 : 4));
    if (c.fp16) {
        std::vector<half> h0(h_elems);
        for (size_t i = 0; i < h_elems; ++i) {
            h0[i] = __float2half(static_cast<float>(hs[i]));
            hs[i] = __half2float(h0[i]);
        }
        cudaMemcpy(dh, h0.data(), h_elems * 2, cudaMemcpyHostToDevice);
    } else {
        std::vector<float> h0(h_elems);
        for (size_t i = 0; i < h_elems; ++i) {
            h0[i] = static_cast<float>(hs[i]);
            hs[i] = h0[i];
        }
        cudaMemcpy(dh, h0.data(), h_elems * 4, cudaMemcpyHostToDevice);
    }
    half* dx = static_cast<half*>(up(x.data(), x.size() * 2));
    half* dB = static_cast<half*>(up(Bv.data(), Bv.size() * 2));
    half* dC = static_cast<half*>(up(Cv.data(), Cv.size() * 2));
    half* ddt = static_cast<half*>(up(dtv.data(), dtv.size() * 2));
    half* dz = c.gate ? static_cast<half*>(up(zv.data(), zv.size() * 2)) : nullptr;
    float* dA = static_cast<float*>(up(A.data(), H * 4));
    float* dD = static_cast<float*>(up(D.data(), H * 4));
    float* ddtb = static_cast<float*>(up(dtb.data(), H * 4));
    int* dn = static_cast<int*>(up(&c.real_n, sizeof(int)));
    half* dy;
    cudaMalloc(&dy, static_cast<size_t>(T) * inner * 2);
    const SsmScanArgs a{dx,      dB,      dC,     ddt, dA, dD, ddtb, dh,
                        dy,      dz,      T,      H,   hd, S,  G,    c.real_n >= 0 ? dn : nullptr,
                        nullptr, nullptr, nullptr};
    EXPECT_TRUE(ssm_scan_ssd_launch(a, c.fp16));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> y(static_cast<size_t>(T) * inner);
    cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<double> hout(h_elems);
    if (c.fp16) {
        std::vector<half> t(h_elems);
        cudaMemcpy(t.data(), dh, h_elems * 2, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h_elems; ++i)
            hout[i] = __half2float(t[i]);
    } else {
        std::vector<float> t(h_elems);
        cudaMemcpy(t.data(), dh, h_elems * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h_elems; ++i)
            hout[i] = t[i];
    }
    for (void* p :
         {static_cast<void*>(dx), static_cast<void*>(dB), static_cast<void*>(dC), static_cast<void*>(ddt),
          static_cast<void*>(dz), static_cast<void*>(dA), static_cast<void*>(dD), static_cast<void*>(ddtb),
          static_cast<void*>(dn), dh, static_cast<void*>(dy)})
        cudaFree(p);

    // Reference: h = a_bar h + dt x B, y = C.h + D x (gated), rows < real_n advance the state.
    const int real_n = c.real_n >= 0 ? std::min(c.real_n, T) : T;
    SsdErr err;
    double y_scale = 0.0;
    std::vector<double> yref(static_cast<size_t>(real_n) * inner);
    for (int t = 0; t < real_n; ++t)
        for (int hh = 0; hh < H; ++hh) {
            const int g = hh / (H / G);
            double dv = static_cast<double>(__half2float(dtv[static_cast<size_t>(t) * H + hh])) + dtb[hh];
            dv = dv > 20.0 ? dv : std::log1p(std::exp(dv));
            const double a_bar = std::exp(dv * A[hh]);
            for (int d = 0; d < hd; ++d) {
                const double xv = __half2float(x[static_cast<size_t>(t) * inner + hh * hd + d]);
                double acc = 0.0;
                for (int s = 0; s < S; ++s) {
                    double& st = hs[(static_cast<size_t>(hh) * S + s) * hd + d];
                    st = a_bar * st + dv * xv * __half2float(Bv[static_cast<size_t>(t) * bc + g * S + s]);
                    acc += __half2float(Cv[static_cast<size_t>(t) * bc + g * S + s]) * st;
                }
                double yv = acc + D[hh] * xv;
                if (c.gate) {
                    const double zz = __half2float(zv[static_cast<size_t>(t) * inner + hh * hd + d]);
                    yv *= zz / (1.0 + std::exp(-zz));
                }
                yref[static_cast<size_t>(t) * inner + hh * hd + d] = yv;
                y_scale = std::max(y_scale, std::fabs(yv));
            }
        }
    for (size_t i = 0; i < yref.size(); ++i)
        err.y = std::max(err.y, std::fabs(static_cast<double>(__half2float(y[i])) - yref[i]) / y_scale);
    double h_scale = 0.0;
    for (double v : hs)
        h_scale = std::max(h_scale, std::fabs(v));
    for (size_t i = 0; i < h_elems; ++i)
        err.h = std::max(err.h, std::fabs(hout[i] - hs[i]) / h_scale);
    return err;
}

TEST(SSMScanSsdTest, MatchesDoubleReference) {
    SKIP_IF_NO_CUDA();
    const SsdCase cases[] = {
        {8, 64, 2, 128, -1, false, false, 1.0f}, {8, 64, 2, 300, -1, false, true, 1.0f},
        {8, 64, 2, 777, 700, false, true, 1.0f}, {4, 128, 1, 257, -1, true, true, 1.0f},
        {8, 64, 2, 2048, -1, true, true, 1.0f},  {8, 64, 8, 200, -1, false, false, 8.0f},
    };
    uint32_t seed = 11;
    for (const auto& c : cases) {
        const SsdErr e = run_ssd_case(c, seed++);
        // y: fp16 output rounding (2^-11) dominates; state: fp32-equivalent (hi/lo operands),
        // FP16 state pays one final rounding.
        EXPECT_LT(e.y, 2e-3) << "T=" << c.n_tokens << " hd=" << c.head_dim << " fp16=" << c.fp16;
        EXPECT_LT(e.h, c.fp16 ? 1e-3 : 1e-5) << "T=" << c.n_tokens << " hd=" << c.head_dim;
    }
}

TEST(SSMScanSsdTest, RefusesUncoveredShapes) {
    SKIP_IF_NO_CUDA();
    SsmScanArgs a{};
    a.n_tokens = 64;  // below kSsdMinTokens
    a.n_heads = 8;
    a.head_dim_ssm = 64;
    a.state_size = 128;
    a.n_groups = 2;
    EXPECT_FALSE(ssm_scan_ssd_launch(a, false));
    a.n_tokens = 512;
    a.state_size = 64;
    EXPECT_FALSE(ssm_scan_ssd_launch(a, false));
}

}  // namespace
}  // namespace imp
