// fa2_hd512_prefill (split-D FA2 for head_dim 512, ROADMAP row 123) against a CPU reference:
// causal with a q_offset, GQA, softcap on/off, row and KV tails.

#include "compute/attention_fa2_hd512.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace imp {
namespace {

struct Case {
    int n, q_offset, nh, nkv;
    float softcap;
};

class Fa2Hd512Test : public ::testing::TestWithParam<Case> {};

}  // namespace

TEST_P(Fa2Hd512Test, MatchesReference) {
    constexpr int D = 512;
    const Case c = GetParam();
    const int kv_len = c.n + c.q_offset;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::mt19937 gen(5);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half> q(size_t(c.n) * c.nh * D), k(size_t(kv_len) * c.nkv * D), v(k.size());
    for (auto& x : q)
        x = __float2half(dist(gen));
    for (auto& x : k)
        x = __float2half(dist(gen));
    for (auto& x : v)
        x = __float2half(dist(gen));
    half *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, q.size() * 2);
    cudaMalloc(&dk, k.size() * 2);
    cudaMalloc(&dv, v.size() * 2);
    cudaMalloc(&dout, q.size() * 2);
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
    ASSERT_TRUE(
        fa2_hd512_prefill(dq, dk, dv, dout, c.n, kv_len, c.nh, c.nkv, scale, c.softcap, c.q_offset, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> out(q.size());
    cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost);

    double max_err = 0.0, max_ref = 0.0;
    std::vector<double> s(kv_len), acc(D);
    for (int h = 0; h < c.nh; ++h) {
        const int kh = h / (c.nh / c.nkv);
        for (int i = 0; i < c.n; ++i) {
            const int lim = i + c.q_offset;  // causal: kv <= lim
            double mx = -1e30;
            for (int j = 0; j <= lim; ++j) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d)
                    dot += double(__half2float(q[(size_t(i) * c.nh + h) * D + d])) *
                           __half2float(k[(size_t(j) * c.nkv + kh) * D + d]);
                double x = dot * scale;
                if (c.softcap > 0.0f)
                    x = c.softcap * std::tanh(x / c.softcap);
                s[j] = x;
                mx = std::max(mx, x);
            }
            double l = 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j <= lim; ++j) {
                const double p = std::exp(s[j] - mx);
                l += p;
                for (int d = 0; d < D; ++d)
                    acc[d] += p * __half2float(v[(size_t(j) * c.nkv + kh) * D + d]);
            }
            for (int d = 0; d < D; ++d) {
                const double ref = acc[d] / l;
                const double got = __half2float(out[(size_t(i) * c.nh + h) * D + d]);
                max_ref = std::max(max_ref, std::fabs(ref));
                max_err = std::max(max_err, std::fabs(ref - got));
            }
        }
    }
    std::fprintf(stderr, "[fa2 hd512] n=%d off=%d nh=%d nkv=%d cap=%.0f max|ref|=%.4g max|err|=%.4g\n", c.n,
                 c.q_offset, c.nh, c.nkv, c.softcap, max_ref, max_err);
    EXPECT_GT(max_ref, 0.0);
    EXPECT_LE(max_err, 4e-3);  // fp16 P and output rounding
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
}

INSTANTIATE_TEST_SUITE_P(Cases, Fa2Hd512Test,
                         ::testing::Values(Case{64, 0, 2, 1, 0.0f}, Case{100, 0, 4, 2, 0.0f},
                                           Case{70, 37, 2, 2, 0.0f}, Case{130, 16, 4, 1, 30.0f}));

}  // namespace imp
