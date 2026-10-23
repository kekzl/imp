// Dense small-M IMMA on GGUF blocks in place (M 2..32): Q8_0 vs its dequant formula, IQ4_NL / IQ4_XS / Q5_K
// vs the independent fp64 reference in gguf_format_ref.h.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "gguf_format_ref.h"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the IMMA activation scratch
namespace {

constexpr int kQ8Bytes = 34;

void gen_q8_0(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    W.assign(static_cast<size_t>(N) * (K / 32) * kQ8Bytes, 0);
    std::mt19937 rng(seed);
    for (size_t b = 0; b < W.size() / kQ8Bytes; ++b) {
        uint8_t* bp = W.data() + (b * kQ8Bytes);
        const __half dh = __float2half(0.0005f + (0.0003f * static_cast<float>(rng() % 100)));
        std::memcpy(bp, &dh, 2);
        for (int i = 0; i < 32; ++i)
            bp[2 + i] = static_cast<uint8_t>(static_cast<int8_t>((rng() % 255) - 127));
    }
}

struct Q8Raw {
    int N, K;
    std::vector<uint8_t> W;
    uint8_t* d_w = nullptr;
    __half* d_x = nullptr;
    __half* d_out = nullptr;
    std::vector<__half> x;
    Q8Raw(int n, int k, int max_m, int w_off = 0) : N(n), K(k) {
        gen_q8_0(W, N, K, 411);
        x.resize(static_cast<size_t>(max_m) * K);
        std::mt19937 rng(412);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : x)
            v = __float2half(nd(rng));
        EXPECT_EQ(cudaMalloc(&d_w, W.size() + 16), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out, static_cast<size_t>(max_m) * N * 2), cudaSuccess);
        d_w += w_off;
        cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        w_base_off = w_off;
    }
    ~Q8Raw() {
        cudaFree(d_w - w_base_off);
        cudaFree(d_x);
        cudaFree(d_out);
        mmq_q8_imma_release_all();
    }
    int w_base_off = 0;

    // out[M][N] after one call; resid is the beta=1 base (zeros for beta 0).
    bool run(int M, bool beta1, const std::vector<__half>& resid, std::vector<__half>& out) {
        cudaMemcpy(d_out, resid.data(), resid.size() * 2, cudaMemcpyHostToDevice);
        const bool ok = mmq_gguf_imma_smallm_gemm(d_w, QType::Q8_0, d_x, d_out, M, N, K, nullptr,
                                                  beta1 ? 1.0f : 0.0f);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        out.resize(static_cast<size_t>(M) * N);
        cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);
        return ok;
    }

    double nrmse(int M, bool beta1) {
        std::vector<__half> resid(static_cast<size_t>(M) * N, __float2half(0.0f));
        std::mt19937 rng(413);
        std::normal_distribution<float> nd(0.0f, 4.0f);
        if (beta1)
            for (auto& v : resid)
                v = __float2half(nd(rng));
        std::vector<__half> out;
        EXPECT_TRUE(run(M, beta1, resid, out));
        const int bpr = K / 32;
        double err2 = 0, ref2 = 0;
        for (int n = 0; n < N; ++n)
            for (int m = 0; m < M; ++m) {
                double ref = 0;
                for (int b = 0; b < bpr; ++b) {
                    const uint8_t* bp = W.data() + ((static_cast<size_t>(n) * bpr + b) * kQ8Bytes);
                    __half dh;
                    std::memcpy(&dh, bp, 2);
                    const double d = __half2float(dh);
                    for (int i = 0; i < 32; ++i)
                        ref += static_cast<double>(
                                   __half2float(x[(static_cast<size_t>(m) * K) + (b * 32) + i])) *
                               d * static_cast<int8_t>(bp[2 + i]);
                }
                const size_t o = (static_cast<size_t>(m) * N) + n;
                const double got = static_cast<double>(__half2float(out[o])) -
                                   static_cast<double>(__half2float(resid[o]));
                err2 += (got - ref) * (got - ref);
                ref2 += ref * ref;
            }
        return std::sqrt(err2 / std::max(ref2, 1e-30));
    }
};

}  // namespace

// K=2816: 11 steps (odd split tail); N=384: 3 tiles; beta 1 = residual add.
TEST(MmqQ8Imma, Q8RawDenseSmallMNRMSE) {
    Q8Raw t(384, 2816, 32);
    for (auto [M, beta1] : {std::pair{2, false}, {9, true}, {16, false}, {17, true}, {32, false}})
        EXPECT_LT(t.nrmse(M, beta1), 1e-2) << "M=" << M;
}

// N=200: partial last tile; N=48: one tile, 48 live rows (GDN alpha/beta shape).
TEST(MmqQ8Imma, Q8RawDensePartialTiles) {
    Q8Raw a(200, 512, 32);
    EXPECT_LT(a.nrmse(5, false), 1e-2);
    Q8Raw b(48, 5120, 32);
    EXPECT_LT(b.nrmse(32, true), 1e-2);
    Q8Raw c(200, 1024, 161);  // above 32 rows: 32-row blocks on gridDim.x, no K-split
    EXPECT_LT(c.nrmse(100, false), 1e-2);
    EXPECT_LT(c.nrmse(161, true), 1e-2);
}

// Splits come from N and K only: the first rows of M=3 and M=32 are bit-identical.
TEST(MmqQ8Imma, Q8RawDenseRowsInvariantInM) {
    Q8Raw t(384, 5120, 32);
    std::vector<__half> z3(3 * 384, __float2half(0.0f)), z32(32 * 384, __float2half(0.0f)), o3, o32;
    ASSERT_TRUE(t.run(3, false, z3, o3));
    ASSERT_TRUE(t.run(32, false, z32, o32));
    EXPECT_EQ(std::memcmp(o3.data(), o32.data(), o3.size() * 2), 0);
}

// Declines: M 1, K % 256, a weight base not 16-B aligned.
TEST(MmqQ8Imma, Q8RawDenseDeclines) {
    Q8Raw t(128, 512, 33);
    Q8Raw u(128, 512, 4, 2);
    std::vector<__half> out;
    EXPECT_FALSE(t.run(1, false, std::vector<__half>(128, __float2half(0.0f)), out));
    EXPECT_FALSE(mmq_gguf_imma_smallm_gemm(t.d_w, QType::Q8_0, t.d_x, t.d_out, 4, 128, 480, nullptr, 0.0f));
    EXPECT_FALSE(mmq_gguf_imma_smallm_gemm(t.d_w, QType::Q4_K, t.d_x, t.d_out, 4, 128, 512, nullptr, 0.0f));
    EXPECT_FALSE(u.run(4, false, std::vector<__half>(4 * 128, __float2half(0.0f)), out));
}

namespace {

struct RefFormat {
    QType qt;
    int block_bytes, block_elems;
    void (*build)(std::vector<uint8_t>&, int, int, gguf_ref::Lcg&, gguf_ref::ScaleMode);
    void (*deq)(const uint8_t*, double*);
};

const RefFormat kIq4Nl{QType::IQ4_NL, 18, 32, gguf_ref::build_iq4_nl, gguf_ref::ref_dequant_iq4_nl};
const RefFormat kIq4Xs{QType::IQ4_XS, 136, 256, gguf_ref::build_iq4_xs, gguf_ref::ref_dequant_iq4_xs};
const RefFormat kQ5K{QType::Q5_K, 176, 256, gguf_ref::build_q5_k, gguf_ref::ref_dequant_q5_k};
const RefFormat kQ8{QType::Q8_0, 34, 32, gguf_ref::build_q8_0, gguf_ref::ref_dequant_q8_0};

// nrmse of one call vs the fp64 dequant of the same bytes; beta 1 adds onto a random base.
double ref_nrmse(const RefFormat& f, int N, int K, int M, bool beta1) {
    gguf_ref::Lcg g(0x51A0u + static_cast<uint32_t>(f.qt));
    std::vector<uint8_t> W;
    f.build(W, N, K, g, gguf_ref::NORMAL);
    std::vector<double> wd(static_cast<size_t>(N) * K);
    const size_t blocks = wd.size() / f.block_elems;
    for (size_t b = 0; b < blocks; ++b)
        f.deq(W.data() + (b * f.block_bytes), wd.data() + (b * f.block_elems));
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__half> x(static_cast<size_t>(M) * K), base(static_cast<size_t>(M) * N, __float2half(0.0f));
    for (auto& v : x)
        v = __float2half(nd(rng));
    if (beta1)
        for (auto& v : base)
            v = __float2half(4.0f * nd(rng));
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_o = nullptr;
    EXPECT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_o, base.size() * 2), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, base.data(), base.size() * 2, cudaMemcpyHostToDevice);
    EXPECT_TRUE(mmq_gguf_imma_smallm_gemm(d_w, f.qt, d_x, d_o, M, N, K, nullptr, beta1 ? 1.0f : 0.0f));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(base.size());
    cudaMemcpy(out.data(), d_o, out.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_o);
    mmq_q8_imma_release_all();
    double err2 = 0, ref2 = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double ref = 0;
            for (int k = 0; k < K; ++k)
                ref += static_cast<double>(__half2float(x[(static_cast<size_t>(m) * K) + k])) *
                       wd[(static_cast<size_t>(n) * K) + k];
            const size_t o = (static_cast<size_t>(m) * N) + n;
            const double got = static_cast<double>(__half2float(out[o])) - __half2float(base[o]);
            err2 += (got - ref) * (got - ref);
            ref2 += ref * ref;
        }
    return std::sqrt(err2 / std::max(ref2, 1e-30));
}

}  // namespace

// N=200: partial last tile; K=2816: 11 steps, odd K-split tail; M 2 / 17 / 32: one or both 16-row halves.
TEST(MmqQ8Imma, GgufSmallMIq4NlVsReference) {
    for (auto [M, beta1] : {std::pair{2, false}, {17, true}, {32, false}})
        EXPECT_LT(ref_nrmse(kIq4Nl, 200, 2816, M, beta1), 1e-2) << "M=" << M;
}

TEST(MmqQ8Imma, GgufSmallMIq4XsVsReference) {
    for (auto [M, beta1] : {std::pair{2, false}, {17, true}, {32, false}})
        EXPECT_LT(ref_nrmse(kIq4Xs, 200, 2816, M, beta1), 1e-2) << "M=" << M;
}

TEST(MmqQ8Imma, GgufSmallMQ5KVsReference) {
    for (auto [M, beta1] : {std::pair{2, false}, {17, true}, {32, false}})
        EXPECT_LT(ref_nrmse(kQ5K, 200, 2816, M, beta1), 1e-2) << "M=" << M;
}

// Above 32 rows: IQ4 on 32-row blocks (gridDim.x), Q5_K on the BM=128 kernel; M=100 leaves a partial block.
TEST(MmqQ8Imma, GgufLargeMVsReference) {
    for (const RefFormat* f : {&kIq4Nl, &kIq4Xs, &kQ5K})
        for (auto [M, beta1] : {std::pair{100, false}, {161, true}})
            EXPECT_LT(ref_nrmse(*f, 200, 1024, M, beta1), 1e-2) << qtype_name(f->qt) << " M=" << M;
}

// BM=128 grid with a short last wave (100 rows, n_sms + n_sms / 8 column tiles): the trailing column tiles
// run K-split (imma_wave_tail); the whole output stays on the reference.
TEST(MmqQ8Imma, GgufLargeMWaveTailVsReference) {
    int n_sms = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, 0), cudaSuccess);
    const int N = (n_sms + (n_sms / 8)) * 128;
    EXPECT_LT(ref_nrmse(kIq4Xs, N, 256, 100, false), 1e-2) << "IQ4_XS";
    EXPECT_LT(ref_nrmse(kQ5K, N, 256, 100, true), 1e-2) << "Q5_K";
    EXPECT_LT(ref_nrmse(kQ8, N, 256, 100, true), 1e-2) << "Q8_0";
}

}  // namespace imp
