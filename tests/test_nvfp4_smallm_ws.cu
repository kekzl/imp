// gemm_nvfp4_smallm_ws (batched-decode NVFP4 x NVFP4, M 1..32) against a double reference on the decoded
// nibbles and UE4M3 scales, beta 0 / 1, row invariance over M, batch position and tile config, sibling launch
// bit-equal to singles, refusals, plus a cold-L2 timing table vs smallm v2 on the dense decode shapes of
// Gemma-4-26B-A4B, Qwen3-8B, Qwen3-14B and Qwen3-30B-A3B (disabled by default).

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

namespace {

constexpr int kMaxM = kNvfp4SmallMWsMaxM;
constexpr float kWTs = 0.0625f;  // weight tensor scale (the activation side is 1, as the dispatch quantizes)

uint32_t lcg(uint32_t& h) {
    h = (h * 1664525u) + 1013904223u;
    return h >> 8;
}

// Random nibbles and UE4M3 scales 2^-2 .. 2^2 (exponent 5..9), never NaN.
void random_nvfp4(std::vector<uint8_t>& packed, std::vector<uint8_t>& sf, int rows, int K, uint32_t seed) {
    packed.resize(size_t(rows) * K / 2);
    sf.resize(size_t(rows) * K / 16);
    uint32_t h = seed;
    for (auto& b : packed)
        b = static_cast<uint8_t>(lcg(h));
    for (auto& s : sf)
        s = static_cast<uint8_t>(((5 + (lcg(h) % 5)) << 3) | (lcg(h) & 7));
}

double fp4(uint8_t nib) {
    static const double lut[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
    return lut[nib & 15];
}

double ue4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + (m / 8.0), e - 7);
}

std::vector<double> decode(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& sf, int rows,
                           int K) {
    std::vector<double> v(size_t(rows) * K);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) {
            const uint8_t byte = packed[(size_t(r) * K / 2) + (k / 2)];
            v[(size_t(r) * K) + k] = fp4((k & 1) ? (byte >> 4) : byte) *
                                     ue4m3(sf[(size_t(r) * K / 16) + (k / 16)]);
        }
    return v;
}

struct Dev {
    void* p = nullptr;
    void* s = nullptr;
    void up(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& sf) {
        ASSERT_EQ(cudaMalloc(&p, packed.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&s, sf.size()), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(p, packed.data(), packed.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(s, sf.data(), sf.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }
    NvFP4QuantResult q(int rows, int K, float ts) const {
        NvFP4QuantResult r;
        r.packed_data = p;
        r.micro_scales = s;
        r.tensor_scale = ts;
        r.N = rows;
        r.K = K;
        r.owned = false;
        return r;
    }
    void free_all() {
        cudaFree(p);
        cudaFree(s);
        p = s = nullptr;
    }
};

// Operands for one (N, K): kMaxM activation rows, the weight, outputs, double reference (beta 0).
struct Shape {
    int N = 0, K = 0;
    std::vector<uint8_t> xp, xs, wp, wsf;
    std::vector<double> ref;
    Dev x, w;
    half* d_y = nullptr;
    half* d_y2 = nullptr;

    void init(int n, int k, bool want_ref, uint32_t seed = 1) {
        N = n;
        K = k;
        random_nvfp4(xp, xs, kMaxM, K, 0x1234u + n + k + seed);
        random_nvfp4(wp, wsf, N, K, 0x9876u + n + seed);
        x.up(xp, xs);
        w.up(wp, wsf);
        ASSERT_EQ(cudaMalloc(&d_y, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y2, size_t(kMaxM) * N * sizeof(half)), cudaSuccess);
        if (!want_ref)
            return;
        const auto xf = decode(xp, xs, kMaxM, K);
        const auto wf = decode(wp, wsf, N, K);
        ref.assign(size_t(kMaxM) * N, 0.0);
        for (int m = 0; m < kMaxM; ++m)
            for (int r = 0; r < N; ++r) {
                double s = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    s += xf[(size_t(m) * K) + kk] * wf[(size_t(r) * K) + kk];
                ref[(size_t(m) * N) + r] = s * kWTs;
            }
    }
    NvFP4QuantResult W() const { return w.q(N, K, kWTs); }
    NvFP4QuantResult X() const { return x.q(kMaxM, K, 1.0f); }
    void upload_x(const std::vector<uint8_t>& p, const std::vector<uint8_t>& s) {
        ASSERT_EQ(cudaMemcpy(x.p, p.data(), p.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(x.s, s.data(), s.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }
    void free_all() {
        x.free_all();
        w.free_all();
        cudaFree(d_y);
        cudaFree(d_y2);
        *this = Shape{};
    }
};

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

// DRAM timing on the dense decode shapes: back-to-back calls over a ring of >= 4 weight copies (>= 400 MB,
// past the 96 MB L2), best of 5 x 200 calls after >= 1.2 s warm-up; the activation stays L2-resident as in
// decode. "v2" = what the dispatch ran before smallm ws: v2 sibling launch when the combined tiles allow one
// stripe, else v2 per tensor (stripes + reduce); "-" = v2 refuses (K % 256, N % 64: tile kernel in situ).
// floor = weight + scale bytes / 1792 GB/s.
// --gtest_also_run_disabled_tests --gtest_filter=*Nvfp4SmallMWsBench*
TEST(Nvfp4SmallMWsBench, DISABLED_VsSmallmV2) {
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    struct Case {
        const char* name;
        int K;
        int N[3];
    };
    const Case cases[] = {
        {"gemma4 qkv sliding", 2816, {4096, 2048, 2048}}, {"gemma4 qk global", 2816, {8192, 1024, 0}},
        {"gemma4 o sliding", 4096, {2816, 0, 0}},         {"gemma4 o global", 8192, {2816, 0, 0}},
        {"gemma4 gate|up", 2816, {2112, 2112, 0}},        {"gemma4 down", 2112, {2816, 0, 0}},
        {"qwen3-8b qkv", 4096, {4096, 1024, 1024}},       {"qwen3-8b o", 4096, {4096, 0, 0}},
        {"qwen3-8b gate|up", 4096, {12288, 12288, 0}},    {"qwen3-8b down", 12288, {4096, 0, 0}},
        {"qwen3-14b qkv", 5120, {5120, 1024, 1024}},      {"qwen3-14b o", 5120, {5120, 0, 0}},
        {"qwen3-14b gate|up", 5120, {17408, 17408, 0}},   {"qwen3-14b down", 17408, {5120, 0, 0}},
        {"qwen3-30b qkv", 2048, {4096, 512, 512}},        {"qwen3-30b o", 4096, {2048, 0, 0}},
    };
    const char* only = std::getenv("IMP_WS_BENCH_ONLY");  // substring filter on the case name
    constexpr int kCfgN = 4;
    const int cfg[kCfgN][2] = {{16, 1}, {16, 2}, {32, 1}, {32, 2}};
    std::fprintf(stderr, "%-20s %3s %7s %7s %7s %7s %7s %7s %7s %6s\n", "shape", "M", "floor", "v2", "ws16u1",
                 "ws16u2", "ws32u1", "ws32u2", "default", "GB/s");
    for (const Case& c : cases) {
        if (only != nullptr && std::strstr(c.name, only) == nullptr)
            continue;
        const int K = c.K;
        int count = 0;
        double bytes = 0.0;
        Shape w[3];
        for (int i = 0; i < 3 && c.N[i] > 0; ++i, ++count) {
            w[i].init(c.N[i], K, false, i + 1);
            bytes += double(c.N[i]) * K * (0.5 + (1.0 / 16.0));
        }
        const int copies = std::max(4, static_cast<int>(400e6 / bytes) + 1);
        std::vector<Dev> ring(size_t(copies) * count);
        for (int r = 0; r < copies; ++r)
            for (int i = 0; i < count; ++i)
                ring[(size_t(r) * count) + i].up(w[i].wp, w[i].wsf);
        void* d_ws = nullptr;
        size_t ws_bytes = 0;
        bool v2_ok = (K % 256) == 0;
        int tiles = 0;
        for (int i = 0; i < count; ++i) {
            ws_bytes = std::max(ws_bytes, gemm_nvfp4_smallm_v2_workspace_bytes(c.N[i], K));
            v2_ok = v2_ok && (c.N[i] % 64) == 0;
            tiles += c.N[i] / 64;
        }
        cudaMalloc(&d_ws, std::max<size_t>(ws_bytes, 256));
        const NvFP4QuantResult xq = w[0].X();
        // One call on ring copy r: kind 0 = v2 dispatch, 1..kCfgN = ws tuned cfg, kCfgN + 1 = ws default.
        auto call = [&](int kind, int r, int M) {
            NvFP4QuantResult wq[3];
            SmallMV2Sibling sib[3];
            for (int i = 0; i < count; ++i) {
                wq[i] = ring[(size_t(r) * count) + i].q(c.N[i], K, kWTs);
                sib[i] = {&wq[i], w[i].d_y, c.N[i]};
            }
            if (kind == 0) {
                if (count > 1 && gemm_nvfp4_smallm_v2_stripes(tiles * 64, K) == 1) {
                    (void)gemm_nvfp4_smallm_v2_multi_a4(sib, count, xq, M, K, st);
                    return;
                }
                for (int i = 0; i < count; ++i)
                    (void)gemm_nvfp4_smallm_v2_a4(wq[i], xq, w[i].d_y, M, c.N[i], K, d_ws, st);
            } else if (kind == kCfgN + 1) {
                if (count > 1)
                    (void)gemm_nvfp4_smallm_ws_multi(sib, count, xq, M, K, st);
                else
                    (void)gemm_nvfp4_smallm_ws(wq[0], xq, w[0].d_y, M, c.N[0], K, 0.0f, st);
            } else if (count > 1) {
                (void)gemm_nvfp4_smallm_ws_multi_tuned(sib, count, xq, M, K, st, cfg[kind - 1][0],
                                                       cfg[kind - 1][1]);
            } else {
                (void)gemm_nvfp4_smallm_ws_tuned(wq[0], xq, w[0].d_y, M, c.N[0], K, 0.0f, st,
                                                 cfg[kind - 1][0], cfg[kind - 1][1]);
            }
        };
        // 200 calls captured into one graph (decode replays graphs; eager launches on WSL2 cost ~7 us each),
        // >= 1.2 s of warm replays, best of 15.
        auto time_us = [&](int kind, int M) {
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ge = nullptr;
            call(kind, 0, M);  // eager first call: lazy cudaFuncSetAttribute fails inside a capture
            EXPECT_EQ(cudaStreamSynchronize(st), cudaSuccess);
            EXPECT_EQ(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), cudaSuccess);
            for (int i = 0; i < 200; ++i)
                call(kind, i % copies, M);
            EXPECT_EQ(cudaStreamEndCapture(st, &g), cudaSuccess);
            EXPECT_EQ(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
            float ms = 0.0f;
            double warm = 0.0;
            while (warm < 1200.0) {
                cudaEventRecord(e0, st);
                cudaGraphLaunch(ge, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                warm += ms;
            }
            double best = 1e30;
            for (int rep = 0; rep < 15; ++rep) {
                cudaEventRecord(e0, st);
                cudaGraphLaunch(ge, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, double(ms));
            }
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
            return best * 1000.0 / 200.0;
        };
        for (int M : {1, 4, 8, 16, 32}) {
            const double t_v2 = v2_ok ? time_us(0, M) : -1.0;
            double t_c[kCfgN];
            for (int k = 0; k < kCfgN; ++k) {
                bool ok = true;
                for (int i = 0; i < count; ++i)
                    ok = ok && c.N[i] % cfg[k][0] == 0;
                t_c[k] = ok ? time_us(k + 1, M) : -1.0;
            }
            const double t_def = time_us(kCfgN + 1, M);
            std::fprintf(stderr, "%-20s %3d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %6.0f\n", c.name, M,
                         bytes / 1.792e6, t_v2, t_c[0], t_c[1], t_c[2], t_c[3], t_def, bytes / (t_def * 1e3));
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_ws);
        for (auto& d : ring)
            d.free_all();
        for (int i = 0; i < count; ++i)
            w[i].free_all();
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(st);
}
