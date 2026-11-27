// gemm_nvfp4_smallm_ws (batched-decode NVFP4 x NVFP4, M 1..32) against a double reference on the decoded
// nibbles and UE4M3 scales, beta 0 / 1, row invariance over M, batch position and tile config, sibling launch
// bit-equal to singles, refusals. Timing cases: tests/bench/test_nvfp4_smallm_ws_bench.cu.

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace imp;

#include "test_nvfp4_smallm_ws_util.cuh"
namespace {

std::vector<half> download(const half* d, size_t n) {
    std::vector<half> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

bool has_device() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Gemma-4 down (K % 128 == 64, N % 64 != 0 for 2112-row siblings), Qwen3-8B q, a 3-CTA shape with a tail
// step.
const int kShapes[][2] = {{2816, 2112}, {2112, 2816}, {4096, 4096}, {48, 192}, {64, 320}};
const int kMs[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32};
// rows x unroll configs: the K order must not depend on them.
const int kCfgs[][2] = {{16, 1}, {16, 2}, {32, 1}, {32, 2}};

}  // namespace

// Every M against the double reference: only fp32 accumulation order and the fp16 output rounding differ.
TEST(Nvfp4SmallMWsTest, MatchesDoubleReferenceEveryM) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    for (const auto& nk : kShapes) {
        if (size_t(nk[0]) * nk[1] > size_t(4096) * 2816)
            continue;  // keep the host reference cheap
        Shape s;
        s.init(nk[0], nk[1], true);
        double max_ref = 0.0;
        for (double v : s.ref)
            max_ref = std::max(max_ref, std::fabs(v));
        for (int M : kMs) {
            ASSERT_TRUE(gemm_nvfp4_smallm_ws(s.W(), s.X(), s.d_y, M, s.N, s.K, 0.0f, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto y = download(s.d_y, size_t(M) * s.N);
            double worst = 0.0;
            for (size_t i = 0; i < y.size(); ++i) {
                const double r = s.ref[i];
                const double err = std::fabs(double(__half2float(y[i])) - r);
                // fp16 output rounding (2^-11 relative) plus fp32 accumulation over K products.
                worst = std::max(worst, err / (std::fabs(r) * 1e-3 + max_ref * 1e-5 + 1e-6));
            }
            EXPECT_LE(worst, 1.0) << "N=" << s.N << " K=" << s.K << " M=" << M;
        }
        s.free_all();
    }
}

// The v2 kernel on the striped shapes ws took over (Gemma-4 gate / up 5 stripes, k 10 stripes), same oracle.
TEST(Nvfp4SmallMWsTest, V2MatchesDoubleReferenceOnStripedShapes) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    const int shapes[][2] = {{2112, 2816}, {1024, 2816}, {2816, 4096}};
    for (const auto& nk : shapes) {
        Shape s;
        s.init(nk[0], nk[1], true);
        double max_ref = 0.0;
        for (double v : s.ref)
            max_ref = std::max(max_ref, std::fabs(v));
        void* d_ws = nullptr;
        ASSERT_EQ(cudaMalloc(&d_ws, gemm_nvfp4_smallm_v2_workspace_bytes(s.N, s.K)), cudaSuccess);
        for (int M : {1, 16, 32}) {
            ASSERT_TRUE(gemm_nvfp4_smallm_v2_a4(s.W(), s.X(), s.d_y, M, s.N, s.K, d_ws, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto y = download(s.d_y, size_t(M) * s.N);
            double worst = 0.0;
            for (size_t i = 0; i < y.size(); ++i) {
                const double r = s.ref[i];
                worst = std::max(worst, std::fabs(double(__half2float(y[i])) - r) /
                                            (std::fabs(r) * 1e-3 + max_ref * 1e-5 + 1e-6));
            }
            EXPECT_LE(worst, 1.0) << "v2 N=" << s.N << " K=" << s.K << " M=" << M << " stripes "
                                  << gemm_nvfp4_smallm_v2_stripes(s.N, s.K);
        }
        cudaFree(d_ws);
        s.free_all();
    }
}

// Row bits do not move with M, the row's batch position or the CTA config (rows per CTA, unroll).
TEST(Nvfp4SmallMWsTest, RowBitsIndependentOfMPositionAndConfig) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    for (const auto& nk : kShapes) {
        Shape s;
        s.init(nk[0], nk[1], false);
        const size_t out = size_t(kMaxM) * s.N;
        ASSERT_TRUE(gemm_nvfp4_smallm_ws_tuned(s.W(), s.X(), s.d_y2, kMaxM, s.N, s.K, 0.0f, nullptr, 16, 1));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto full = download(s.d_y2, out);
        for (const auto& c : kCfgs) {
            if (s.N % c[0] != 0)
                continue;
            for (int M : kMs) {
                ASSERT_TRUE(
                    gemm_nvfp4_smallm_ws_tuned(s.W(), s.X(), s.d_y, M, s.N, s.K, 0.0f, nullptr, c[0], c[1]));
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const auto y = download(s.d_y, size_t(M) * s.N);
                EXPECT_EQ(std::memcmp(y.data(), full.data(), y.size() * sizeof(half)), 0)
                    << "N=" << s.N << " K=" << s.K << " M=" << M << " rows " << c[0] << " unroll " << c[1];
            }
        }
        // Reversed batch: row r lands at position kMaxM-1-r (another lane, another n-tile).
        std::vector<uint8_t> rp(s.xp.size()), rs(s.xs.size());
        for (int m = 0; m < kMaxM; ++m) {
            std::memcpy(&rp[size_t(kMaxM - 1 - m) * s.K / 2], &s.xp[size_t(m) * s.K / 2], s.K / 2);
            std::memcpy(&rs[size_t(kMaxM - 1 - m) * s.K / 16], &s.xs[size_t(m) * s.K / 16], s.K / 16);
        }
        s.upload_x(rp, rs);
        ASSERT_TRUE(gemm_nvfp4_smallm_ws(s.W(), s.X(), s.d_y, kMaxM, s.N, s.K, 0.0f, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto y = download(s.d_y, out);
        for (int m = 0; m < kMaxM; ++m)
            EXPECT_EQ(std::memcmp(&y[size_t(kMaxM - 1 - m) * s.N], &full[size_t(m) * s.N],
                                  s.N * sizeof(half)),
                      0)
                << "N=" << s.N << " row " << m;
        s.free_all();
    }
}

// beta 1 adds onto y: equals the beta-0 result plus the old y, rounded once in fp32 -> fp16.
TEST(Nvfp4SmallMWsTest, BetaAccumulates) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    Shape s;
    s.init(2816, 2112, true);
    for (int M : {1, 7, 16, 32}) {
        std::vector<half> y0(size_t(M) * s.N);
        uint32_t h = 77u + M;
        for (auto& v : y0)
            v = __float2half((static_cast<float>(lcg(h) % 2001) - 1000.0f) * 0.01f);
        ASSERT_EQ(cudaMemcpy(s.d_y, y0.data(), y0.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_TRUE(gemm_nvfp4_smallm_ws(s.W(), s.X(), s.d_y, M, s.N, s.K, 1.0f, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto y = download(s.d_y, y0.size());
        double max_ref = 0.0, worst = 0.0;
        for (size_t i = 0; i < y.size(); ++i)
            max_ref = std::max(max_ref, std::fabs(s.ref[i]));
        for (size_t i = 0; i < y.size(); ++i) {
            const double r = s.ref[i] + double(__half2float(y0[i]));
            worst = std::max(worst, std::fabs(double(__half2float(y[i])) - r) /
                                        (std::fabs(r) * 1e-3 + max_ref * 1e-5));
        }
        EXPECT_LE(worst, 1.0) << "M=" << M;
    }
    s.free_all();
}

// Sibling launch (q|k|v, gate|up): every tensor bit-equal to its own single launch, M 1..32.
TEST(Nvfp4SmallMWsTest, MultiBitEqualToSingle) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    // Gemma-4 sliding q, k, v at K 2816 (distinct weights), gate|up pairs at N 2112.
    Shape q, k, v;
    q.init(4096, 2816, false, 1);
    k.init(2112, 2816, false, 2);
    v.init(2048, 2816, false, 3);
    Shape* sh[3] = {&q, &k, &v};
    const float ts[3] = {kWTs, 0.03125f, 0.5f};
    for (int count : {2, 3})
        for (int M : {1, 8, 9, 16, 17, 32}) {
            SmallMV2Sibling t[3];
            NvFP4QuantResult wq[3];
            for (int i = 0; i < count; ++i) {
                wq[i] = sh[i]->w.q(sh[i]->N, 2816, ts[i]);
                t[i] = {&wq[i], sh[i]->d_y, sh[i]->N};
                ASSERT_TRUE(
                    gemm_nvfp4_smallm_ws(wq[i], q.X(), sh[i]->d_y2, M, sh[i]->N, 2816, 0.0f, nullptr));
                cudaMemset(sh[i]->d_y, 0xff, size_t(kMaxM) * sh[i]->N * sizeof(half));
            }
            ASSERT_TRUE(gemm_nvfp4_smallm_ws_multi(t, count, q.X(), M, 2816, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            for (int i = 0; i < count; ++i) {
                const auto a = download(sh[i]->d_y, size_t(M) * sh[i]->N);
                const auto b = download(sh[i]->d_y2, size_t(M) * sh[i]->N);
                EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(half)), 0)
                    << "count " << count << " tensor " << i << " M=" << M;
            }
        }
    NvFP4QuantResult w0 = q.W(), w1 = k.W();
    SmallMV2Sibling bad[2] = {{&w0, q.d_y, 4096}, {&w1, k.d_y, 1000}};
    EXPECT_FALSE(gemm_nvfp4_smallm_ws_multi(bad, 2, q.X(), 4, 2816, nullptr));
    EXPECT_FALSE(gemm_nvfp4_smallm_ws_multi(bad, 1, q.X(), 4, 2816, nullptr));
    for (Shape* s : sh)
        s->free_all();
}

// Illegal shapes and pointers return false and leave y byte-identical.
TEST(Nvfp4SmallMWsTest, RefusesIllegalShapes) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    Shape s;
    s.init(64, 320, false);
    const size_t bytes = size_t(kMaxM) * s.N * sizeof(half);
    cudaMemset(s.d_y, 0xab, bytes);
    EXPECT_FALSE(gemm_nvfp4_smallm_ws(s.W(), s.X(), s.d_y, 0, s.N, s.K, 0.0f, nullptr));
    EXPECT_FALSE(gemm_nvfp4_smallm_ws(s.W(), s.X(), s.d_y, kMaxM + 1, s.N, s.K, 0.0f, nullptr));
    EXPECT_FALSE(gemm_nvfp4_smallm_ws(s.W(), s.X(), s.d_y, 4, 40, s.K, 0.0f, nullptr));   // N % 16
    EXPECT_FALSE(gemm_nvfp4_smallm_ws(s.W(), s.X(), s.d_y, 4, s.N, 288, 0.0f, nullptr));  // K % 64
    EXPECT_FALSE(
        gemm_nvfp4_smallm_ws_tuned(s.W(), s.X(), s.d_y, 4, 48, s.K, 0.0f, nullptr, 32, 1));  // N % 32
    EXPECT_FALSE(
        gemm_nvfp4_smallm_ws_tuned(s.W(), s.X(), s.d_y, 4, s.N, s.K, 0.0f, nullptr, 64, 1));  // no 64-row cfg
    NvFP4QuantResult nw = s.W();
    nw.packed_data = static_cast<uint8_t*>(nw.packed_data) + 4;  // misaligned nibbles
    EXPECT_FALSE(gemm_nvfp4_smallm_ws(nw, s.X(), s.d_y, 4, s.N, s.K, 0.0f, nullptr));
    nw = s.W();
    nw.micro_scales = nullptr;
    EXPECT_FALSE(gemm_nvfp4_smallm_ws(nw, s.X(), s.d_y, 4, s.N, s.K, 0.0f, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> host(bytes);
    cudaMemcpy(host.data(), s.d_y, bytes, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](uint8_t b) { return b == 0xab; }));
    s.free_all();
}

