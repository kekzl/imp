// moe_router_fused: ids / weights vs a double CPU reference (softmax or sigmoid + selection bias,
// top-k with lowest-index ties, renorm) at the Qwen3-30B, Gemma-4 and Nemotron-H router shapes for
// M = 2..2048 (split-K tickets, 16- and 32-row tiles, partial tiles), bitwise repeatability
// (ticket re-arm, fixed split order) and refusals.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/moe_router_fused.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;

namespace {

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

struct Case {
    int K;
    int ne;
    int top_k;
    bool sigmoid;
    bool bias;
    bool normalize;
};

struct Ref {
    std::vector<int> ids;
    std::vector<double> weights;
    std::vector<double> sel;  // [M, ne] selection scores
};

Ref reference(const std::vector<__half>& x, const std::vector<__half>& w, const std::vector<__half>& bias,
              int M, const Case& c) {
    Ref r;
    r.ids.resize(static_cast<size_t>(M) * c.top_k);
    r.weights.resize(r.ids.size());
    r.sel.resize(static_cast<size_t>(M) * c.ne);
    std::vector<double> p(c.ne);
    for (int m = 0; m < M; ++m) {
        for (int e = 0; e < c.ne; ++e) {
            double s = 0.0;
            for (int k = 0; k < c.K; ++k)
                s += static_cast<double>(__half2float(x[static_cast<size_t>(m) * c.K + k])) *
                     static_cast<double>(__half2float(w[static_cast<size_t>(e) * c.K + k]));
            p[e] = s;
        }
        if (c.sigmoid) {
            for (auto& v : p)
                v = 1.0 / (1.0 + std::exp(-v));
        } else {
            const double mx = *std::max_element(p.begin(), p.end());
            double sum = 0.0;
            for (auto& v : p)
                sum += std::exp(v - mx);
            for (auto& v : p)
                v = std::exp(v - mx) / sum;
        }
        double* sel = &r.sel[static_cast<size_t>(m) * c.ne];
        for (int e = 0; e < c.ne; ++e)
            sel[e] = p[e] + (c.bias ? static_cast<double>(__half2float(bias[e])) : 0.0);
        std::vector<char> used(c.ne, 0);
        double norm = 0.0;
        for (int k = 0; k < c.top_k; ++k) {
            int best = -1;
            for (int e = 0; e < c.ne; ++e)
                if (!used[e] && (best < 0 || sel[e] > sel[best]))
                    best = e;
            used[best] = 1;
            r.ids[static_cast<size_t>(m) * c.top_k + k] = best;
            r.weights[static_cast<size_t>(m) * c.top_k + k] = p[best];
            norm += p[best];
        }
        if (c.normalize)
            for (int k = 0; k < c.top_k; ++k)
                r.weights[static_cast<size_t>(m) * c.top_k + k] /= norm;
    }
    return r;
}

struct FusedOut {
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

bool run_fused(const DevBuf& dx, const DevBuf& dw, const DevBuf& db, int M, const Case& c, FusedOut& out) {
    DevBuf ids(static_cast<size_t>(M) * c.top_k * sizeof(int32_t));
    DevBuf wts(static_cast<size_t>(M) * c.top_k * sizeof(float));
    const size_t ws_bytes = moe_router_fused_workspace_bytes(M);
    static DevBuf* ws = nullptr;  // tickets must stay zero between launches: one zeroed ws per process
    static size_t ws_size = 0;
    if (ws == nullptr || ws_size < ws_bytes) {
        delete ws;
        ws = new DevBuf(ws_bytes);
        ws_size = ws_bytes;
        cudaMemset(ws->p, 0, ws_bytes);
    }
    MoeRouterFusedArgs a;
    a.x = static_cast<const half*>(dx.p);
    a.w = static_cast<const half*>(dw.p);
    a.M = M;
    a.K = c.K;
    a.ne = c.ne;
    a.top_k = c.top_k;
    a.use_sigmoid = c.sigmoid;
    a.normalize = c.normalize;
    a.score_bias = c.bias ? static_cast<const half*>(db.p) : nullptr;
    a.ids = static_cast<int32_t*>(ids.p);
    a.weights = static_cast<float*>(wts.p);
    if (!moe_router_fused(a, ws->p, ws_size, nullptr))
        return false;
    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;
    out.ids.resize(static_cast<size_t>(M) * c.top_k);
    out.weights.resize(out.ids.size());
    cudaMemcpy(out.ids.data(), ids.p, out.ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.weights.data(), wts.p, out.weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return true;
}

class MoeRouterFusedTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override {
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
            GTEST_SKIP() << "no CUDA device";
    }
};

TEST_P(MoeRouterFusedTest, MatchesReference) {
    const Case c = GetParam();
    std::mt19937 rng(1234 + c.K + c.ne);
    std::normal_distribution<float> nx(0.0f, 1.0f), nw(0.0f, 0.03f), nb(0.0f, 0.05f);
    for (int M : {2, 5, 16, 33, 512, 2048}) {
        std::vector<__half> x(static_cast<size_t>(M) * c.K), w(static_cast<size_t>(c.ne) * c.K), b(c.ne);
        for (auto& v : x)
            v = __float2half(nx(rng));
        for (auto& v : w)
            v = __float2half(nw(rng));
        for (auto& v : b)
            v = __float2half(nb(rng));
        DevBuf dx(x.size() * 2), dw(w.size() * 2), db(b.size() * 2);
        cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dw.p, w.data(), w.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(db.p, b.data(), b.size() * 2, cudaMemcpyHostToDevice);

        FusedOut got;
        ASSERT_TRUE(run_fused(dx, dw, db, M, c, got)) << "M=" << M;
        const Ref ref = reference(x, w, b, M, c);

        // An id may differ only at a near tie: |sel(ref id) - sel(got id)| below the fp16-partial noise.
        int mismatches = 0, near_ties = 0;
        double max_w_err = 0.0;
        for (int m = 0; m < M; ++m) {
            const double* sel = &ref.sel[static_cast<size_t>(m) * c.ne];
            for (int k = 0; k < c.top_k; ++k) {
                const size_t o = static_cast<size_t>(m) * c.top_k + k;
                const int gi = got.ids[o];
                ASSERT_GE(gi, 0);
                ASSERT_LT(gi, c.ne);
                if (gi != ref.ids[o]) {
                    ++mismatches;
                    if (std::abs(sel[gi] - sel[ref.ids[o]]) < 2e-4)
                        ++near_ties;
                    continue;
                }
                max_w_err = std::max(max_w_err,
                                     std::abs(static_cast<double>(got.weights[o]) - ref.weights[o]));
            }
        }
        EXPECT_EQ(mismatches, near_ties) << "M=" << M << ": id mismatch away from a near tie";
        EXPECT_LE(mismatches, std::max(1, M * c.top_k / 200)) << "M=" << M;
        EXPECT_LT(max_w_err, 2e-3) << "M=" << M;

        FusedOut again;
        ASSERT_TRUE(run_fused(dx, dw, db, M, c, again));
        EXPECT_EQ(0, std::memcmp(got.ids.data(), again.ids.data(), got.ids.size() * sizeof(int32_t)))
            << "M=" << M;
        EXPECT_EQ(0,
                  std::memcmp(got.weights.data(), again.weights.data(), got.weights.size() * sizeof(float)))
            << "M=" << M;
    }
}

INSTANTIATE_TEST_SUITE_P(RouterShapes, MoeRouterFusedTest,
                         ::testing::Values(Case{2048, 128, 8, false, false, true},    // Qwen3-30B-A3B
                                           Case{2816, 128, 8, false, false, true},    // Gemma-4-26B-A4B
                                           Case{2688, 128, 6, true, true, true},      // Nemotron-3-Nano
                                           Case{1024, 64, 4, false, false, false}));  // narrow, no renorm

TEST(MoeRouterFusedRefusal, UnsupportedShapes) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        GTEST_SKIP() << "no CUDA device";
    DevBuf x(4096 * 2), w(256 * 4096 * 2), ids(64), wts(64), ws(moe_router_fused_workspace_bytes(16));
    MoeRouterFusedArgs a;
    a.x = static_cast<const half*>(x.p);
    a.w = static_cast<const half*>(w.p);
    a.M = 1;
    a.K = 2048;
    a.ne = 136;  // > 128
    a.top_k = 8;
    a.ids = static_cast<int32_t*>(ids.p);
    a.weights = static_cast<float*>(wts.p);
    EXPECT_FALSE(moe_router_fused(a, ws.p, moe_router_fused_workspace_bytes(16), nullptr));
    a.ne = 128;
    a.K = 2050;  // % 32 != 0
    EXPECT_FALSE(moe_router_fused(a, ws.p, moe_router_fused_workspace_bytes(16), nullptr));
    a.K = 2048;
    EXPECT_FALSE(moe_router_fused(a, nullptr, 0, nullptr));  // no workspace
}

}  // namespace
