// mmq_q8_imma_gemm (Q8_0 INT8 IMMA prefill GEMM): two-level check against CPU reference.
// Exact-model mirrors device math (per-32-block amax/127, rn-round, +-127 clamp) but is not
// bit-exact: Release --use_fast_math can flip a near-tie LSB; unquantized sanity bounds
// end-to-end activation-quant error.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "compute/gemm.h"
#include "compute/mmq_q8_imma.h"
#include "core/tensor.h"
#include "exec/weight_dequant.h"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the migrated scratches (A7 step 8)
namespace {

constexpr int kBlk = 32;
constexpr int kBlockBytes = 34;  // half d + 32 s8

void gen_q8_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    ASSERT_EQ(K % kBlk, 0);
    const int bpr = K / kBlk;
    W.resize(static_cast<size_t>(N) * bpr * kBlockBytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    for (size_t b = 0; b < W.size() / kBlockBytes; ++b) {
        uint8_t* bp = W.data() + b * kBlockBytes;
        __half d = __float2half(dd(rng));
        std::memcpy(bp, &d, 2);
        for (int i = 0; i < 32; ++i) bp[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(qd(rng)));
    }
}

// CPU model of the device activation quantizer (quantize_fp16_to_int8_subblock).
void quant_act_cpu(const std::vector<__half>& x, int M, int K, std::vector<int8_t>& xs8,
                   std::vector<float>& xscale) {
    const int subs = K / kBlk;
    xs8.assign(static_cast<size_t>(M) * K, 0);
    xscale.assign(static_cast<size_t>(M) * subs, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int s = 0; s < subs; ++s) {
            float amax = 0.0f;
            for (int i = 0; i < kBlk; ++i)
                amax = std::fmax(amax, std::fabs(__half2float(x[(size_t)m * K + s * kBlk + i])));
            const float inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
            const float scale = (amax > 0.0f) ? amax / 127.0f : 0.0f;
            // device stores the scale as half — model that rounding
            xscale[(size_t)m * subs + s] = __half2float(__float2half(scale));
            for (int i = 0; i < kBlk; ++i) {
                float v = __half2float(x[(size_t)m * K + s * kBlk + i]);
                int q = static_cast<int>(std::nearbyint(v * inv));
                q = std::max(-127, std::min(127, q));
                xs8[(size_t)m * K + s * kBlk + i] = static_cast<int8_t>(q);
            }
        }
    }
}

void run_case(int M, int N, int K, unsigned seed, float tol_exact, float tol_unquant) {
    std::vector<uint8_t> W;
    gen_q8_weight(W, N, K, seed);

    std::vector<__half> x(static_cast<size_t>(M) * K);
    std::mt19937 rng(seed * 7 + 1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half(nd(rng));

    // device
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(M) * N * sizeof(__half)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);

    ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(M) * N);
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(__half), cudaMemcpyDeviceToHost);

    // CPU references
    std::vector<int8_t> xs8;
    std::vector<float> xscale;
    quant_act_cpu(x, M, K, xs8, xscale);
    const int subs = K / kBlk;

    // Exact-model: per-output relative error, denominator floored at output RMS to avoid
    // dividing near-zero outputs. Unquantized: normalized RMS (energy ratio), since
    // activation-quant noise on small outputs has heavy cancellation tails.
    double err2_exact = 0.0, err2_unq = 0.0, ref2 = 0.0, max_rel_exact = 0.0;
    std::vector<double> exact_buf, unq_buf, got_buf;
    std::mt19937 pick(seed * 13 + 5);
    const int rows_to_check = std::min(M, 16);
    for (int rc = 0; rc < rows_to_check; ++rc) {
        const int m = (rows_to_check == M) ? rc : static_cast<int>(pick() % M);
        for (int n = 0; n < N; ++n) {
            double acc_exact = 0.0, acc_unq = 0.0;
            for (int s = 0; s < subs; ++s) {
                const uint8_t* bp =
                    W.data() + (static_cast<size_t>(n) * subs + s) * kBlockBytes;
                __half dh;
                std::memcpy(&dh, bp, 2);
                const double dw = __half2float(dh);
                const int8_t* wq = reinterpret_cast<const int8_t*>(bp + 2);
                long isum = 0;
                double unq = 0.0;
                for (int i = 0; i < kBlk; ++i) {
                    isum += static_cast<long>(xs8[(size_t)m * K + s * kBlk + i]) * wq[i];
                    unq += static_cast<double>(__half2float(x[(size_t)m * K + s * kBlk + i])) *
                           (dw * wq[i]);
                }
                acc_exact += static_cast<double>(xscale[(size_t)m * subs + s]) * dw *
                             static_cast<double>(isum);
                acc_unq += unq;
            }
            const double got = __half2float(out[(size_t)m * N + n]);
            exact_buf.push_back(acc_exact);
            unq_buf.push_back(acc_unq);
            got_buf.push_back(got);
            ref2 += acc_unq * acc_unq;
        }
    }
    const double ref_rms = std::sqrt(ref2 / static_cast<double>(unq_buf.size()));
    for (size_t i = 0; i < got_buf.size(); ++i) {
        const double de = std::fabs(got_buf[i] - exact_buf[i]);
        max_rel_exact = std::max(max_rel_exact, de / std::max(ref_rms, std::fabs(exact_buf[i])));
        err2_exact += de * de;
        const double du = got_buf[i] - unq_buf[i];
        err2_unq += du * du;
    }
    const double nrmse_unq = std::sqrt(err2_unq / static_cast<double>(got_buf.size())) /
                             std::max(ref_rms, 1e-30);
    if (max_rel_exact >= tol_exact) {
        // dump the worst output for diagnosis
        size_t wi = 0;
        double wv = 0.0;
        for (size_t i = 0; i < got_buf.size(); ++i) {
            const double r = std::fabs(got_buf[i] - exact_buf[i]) /
                             std::max(ref_rms, std::fabs(exact_buf[i]));
            if (r > wv) {
                wv = r;
                wi = i;
            }
        }
        fprintf(stderr, "WORST idx=%zu (row-slot %zu, n=%zu): got=%f exact=%f unq=%f ref_rms=%f\n",
                wi, wi / N, wi % N, got_buf[wi], exact_buf[wi], unq_buf[wi], ref_rms);
    }
    EXPECT_LT(max_rel_exact, tol_exact) << "M=" << M << " N=" << N << " K=" << K;
    EXPECT_LT(nrmse_unq, tol_unquant) << "M=" << M << " N=" << N << " K=" << K
                                      << " ref_rms=" << ref_rms;

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    mmq_q8_imma_release_all();
}

TEST(MmqQ8Imma, Square128) { run_case(128, 128, 64, 42, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, PrefillChunkShape) { run_case(512, 128, 256, 43, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, QkvLikeShape) { run_case(512, 1024, 4096, 44, 15e-3f, 1e-2f); }
TEST(MmqQ8Imma, MTail) { run_case(313, 256, 128, 45, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, NTail192) { run_case(128, 192, 128, 51, 15e-3f, 2e-2f); }  // gemma-4 d_ff=704 class
TEST(MmqQ8Imma, MTailNoTail320) { run_case(320, 256, 128, 45, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, MTailSeed47) { run_case(313, 256, 128, 47, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, MTailBigK) { run_case(313, 256, 1024, 45, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, M64Min) { run_case(64, 128, 64, 46, 15e-3f, 2e-2f); }
// Dense small-M (spec-decode verify chunks) routes through the split-K path
// (M <= 32, gridDim.z = K-splits, fp32 partial slices + finalize reduce).
TEST(MmqQ8Imma, SmallMSplitKVerifyShape) { run_case(9, 2560, 2560, 48, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, SmallMSplitKWideK) { run_case(17, 1024, 9728, 49, 15e-3f, 1e-2f); }
TEST(MmqQ8Imma, SmallMMin2) { run_case(2, 256, 512, 50, 15e-3f, 2e-2f); }
TEST(MmqQ8Imma, SmallMNoSplitShortK) { run_case(8, 128, 128, 52, 15e-3f, 2e-2f); }
// ---- Q4_K dense (new stack) — NRMSE vs full-dequant reference ----
namespace q4k_helpers {
constexpr int kSuper = 256;
constexpr int kSuperBytes = 144;
inline void get_scale_min_k4_cpu(int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
    if (j < 4) {
        d_out = q[j] & 63u;
        m_out = q[j + 4] & 63u;
    } else {
        d_out = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
inline void gen_q4k(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    const int bpr = K / kSuper;
    W.resize(static_cast<size_t>(N) * bpr * kSuperBytes);
    std::mt19937 rng(seed);
    for (size_t b = 0; b < W.size() / kSuperBytes; ++b) {
        uint8_t* bp = W.data() + b * kSuperBytes;
        __half dh = __float2half(0.001f + 0.0005f * (rng() % 100));
        __half mh = __float2half(0.0005f + 0.0001f * (rng() % 100));
        std::memcpy(bp, &dh, 2);
        std::memcpy(bp + 2, &mh, 2);
        for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 128; ++i) bp[16 + i] = static_cast<uint8_t>(rng() & 0xFF);
    }
}
inline void dequant_row(const std::vector<uint8_t>& W, int n, int K, std::vector<double>& row) {
    const int bpr = K / kSuper;
    row.assign(K, 0.0);
    for (int sblk = 0; sblk < bpr; ++sblk) {
        const uint8_t* bp = W.data() + (static_cast<size_t>(n) * bpr + sblk) * kSuperBytes;
        __half dh, mh;
        std::memcpy(&dh, bp, 2);
        std::memcpy(&mh, bp + 2, 2);
        const double d = __half2float(dh), dmin = __half2float(mh);
        const uint8_t* scales = bp + 4;
        const uint8_t* qs = bp + 16;
        for (int e = 0; e < kSuper; ++e) {
            const int group = e >> 6, in_grp = e & 63, hi = in_grp >> 5;
            const int byte_q = group * 32 + (in_grp & 31);
            const int sub = group * 2 + hi;
            uint8_t sc, mn;
            get_scale_min_k4_cpu(sub, scales, sc, mn);
            const int nib = hi ? (qs[byte_q] >> 4) : (qs[byte_q] & 0xF);
            row[sblk * kSuper + e] = d * sc * nib - dmin * mn;
        }
    }
}
}  // namespace q4k_helpers

TEST(MmqQ8Imma, Q4KDenseNRMSE) {
    using namespace q4k_helpers;
    const int M = 128, N = 128, K = 512;
    std::vector<uint8_t> W;
    gen_q4k(W, N, K, 77);
    std::vector<__half> x(static_cast<size_t>(M) * K);
    std::mt19937 rng(78);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half(nd(rng));

    uint8_t* d_w; __half *d_x, *d_out;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(M) * N * 2), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q4k_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(M) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    double err2 = 0, ref2 = 0;
    std::vector<double> wrow;
    for (int n = 0; n < N; ++n) {
        dequant_row(W, n, K, wrow);
        for (int m = 0; m < M; m += 7) {
            double ref = 0;
            for (int k = 0; k < K; ++k)
                ref += static_cast<double>(__half2float(x[(size_t)m * K + k])) * wrow[k];
            const double got = __half2float(out[(size_t)m * N + n]);
            err2 += (got - ref) * (got - ref);
            ref2 += ref * ref;
        }
    }
    const double nrmse = std::sqrt(err2 / std::max(ref2, 1e-30));
    EXPECT_LT(nrmse, 2e-2) << "Q4K dense NRMSE";
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_out);
    mmq_q8_imma_release_all();
}

// ---- MoE grouped (Q8_0, ne=4, ragged offsets incl. an empty expert) ----
// ---- Q6_K dense — NRMSE vs full-dequant reference (per-16 scales) ----
namespace q6k_helpers {
constexpr int kSuper = 256;
constexpr int kBytes = 210;
inline void gen_q6k(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    const int bpr = K / kSuper;
    W.resize(static_cast<size_t>(N) * bpr * kBytes);
    std::mt19937 rng(seed);
    for (size_t b = 0; b < W.size() / kBytes; ++b) {
        uint8_t* bp = W.data() + b * kBytes;
        for (int i = 0; i < 192; ++i) bp[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 16; ++i)
            bp[192 + i] = static_cast<uint8_t>(static_cast<int8_t>((rng() % 65) - 32));
        __half dh = __float2half(0.0005f + 0.0003f * (rng() % 100));
        std::memcpy(bp + 208, &dh, 2);
    }
}
inline int q6_element(const uint8_t* bp, int i) {
    const int group = i >> 7, within = i & 127, quad = within >> 5, l = within & 31;
    const int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    const int qh_idx = (group << 5) + l;
    const uint8_t qlb = bp[ql_idx];
    const uint8_t low4 = (quad >= 2) ? ((qlb >> 4) & 0xF) : (qlb & 0xF);
    const uint8_t high2 = (bp[128 + qh_idx] >> (quad * 2)) & 0x3;
    return static_cast<int>((high2 << 4) | low4) - 32;
}
}  // namespace q6k_helpers

// GGUF 210-B blocks read in place; odd K/256 alternates the block 4-B parity row to row.
void run_q6k_dense(int M, int N, int K) {
    using namespace q6k_helpers;
    std::vector<uint8_t> W;
    gen_q6k(W, N, K, 311);
    std::vector<__half> x(static_cast<size_t>(M) * K);
    std::mt19937 rng(312);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half(nd(rng));

    uint8_t* d_w; __half *d_x, *d_out;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(M) * N * 2), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_q6k_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(M) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    const int bpr = K / kSuper;
    double err2 = 0, ref2 = 0;
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; m += 7) {
            double ref = 0;
            for (int sblk = 0; sblk < bpr; ++sblk) {
                const uint8_t* bp = W.data() + (static_cast<size_t>(n) * bpr + sblk) * kBytes;
                __half dh;
                std::memcpy(&dh, bp + 208, 2);
                const double d = __half2float(dh);
                const int8_t* sc = reinterpret_cast<const int8_t*>(bp + 192);
                for (int i = 0; i < kSuper; ++i)
                    ref += static_cast<double>(__half2float(x[(size_t)m * K + sblk * kSuper + i])) *
                           (d * sc[i >> 4] * q6_element(bp, i));
            }
            const double got = __half2float(out[(size_t)m * N + n]);
            err2 += (got - ref) * (got - ref);
            ref2 += ref * ref;
        }
    }
    const double nrmse = std::sqrt(err2 / std::max(ref2, 1e-30));
    EXPECT_LT(nrmse, 2e-2) << "Q6K dense NRMSE K=" << K;
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_out);
    mmq_q8_imma_release_all();
}

TEST(MmqQ8Imma, Q6KDenseNRMSE) { run_q6k_dense(128, 128, 512); }
// K=768: 3 superblocks per row, so the block 4-B parity flips row to row; N=192 leaves a partial tile.
TEST(MmqQ8Imma, Q6KDenseOddSuperblocksNRMSE) { run_q6k_dense(200, 192, 768); }

// ---- Q5_1 MoE grouped — NRMSE vs dequant reference (asymmetric β-form) ----
TEST(MmqQ8Imma, MoeGroupedQ51) {
    const int ne = 2, N = 128, K = 256;
    const int rows_per[ne] = {70, 41};
    int32_t h_off[ne + 1] = {0};
    for (int e = 0; e < ne; ++e) h_off[e + 1] = h_off[e] + rows_per[e];
    const int expanded = h_off[ne];

    // packed q5_1 experts: [ne][N][K/32] 24-B blocks
    const int bpr = K / 32;
    std::vector<uint8_t> W(static_cast<size_t>(ne) * N * bpr * 24);
    std::mt19937 rng(401);
    for (size_t b = 0; b < W.size() / 24; ++b) {
        uint8_t* bp = W.data() + b * 24;
        __half dh = __float2half(0.001f + 0.0005f * (rng() % 100));
        __half mh = __float2half(-0.02f + 0.0004f * (rng() % 100));
        std::memcpy(bp, &dh, 2);
        std::memcpy(bp + 2, &mh, 2);
        for (int i = 0; i < 20; ++i) bp[4 + i] = static_cast<uint8_t>(rng() & 0xFF);
    }
    std::vector<__half> x(static_cast<size_t>(expanded) * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half(nd(rng));

    uint8_t* d_w; __half *d_x, *d_out; int32_t* d_off;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(expanded) * N * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, sizeof(h_off)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_off, sizeof(h_off), cudaMemcpyHostToDevice);
    ASSERT_TRUE(mmq_imma_moe_gemm(d_w, /*qkind=*/3, d_x, d_out, d_off, 70, expanded, ne, N, K,
                                  nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(expanded) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    // dequant reference (unquantized activations; NRMSE bound)
    auto q5 = [&](const uint8_t* bp, int i) -> int {
        uint32_t qh;
        std::memcpy(&qh, bp + 4, 4);
        const uint8_t* qs = bp + 8;
        if (i < 16) return (qs[i] & 0xF) | (((qh >> i) & 1u) << 4);
        return (qs[i - 16] >> 4) | (((qh >> (i)) & 1u) << 4);
    };
    double err2 = 0, ref2 = 0;
    for (int e = 0; e < ne; ++e) {
        for (int r = h_off[e]; r < h_off[e + 1]; ++r) {
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int s2 = 0; s2 < bpr; ++s2) {
                    const uint8_t* bp =
                        W.data() + ((static_cast<size_t>(e) * N + n) * bpr + s2) * 24;
                    __half dh, mh;
                    std::memcpy(&dh, bp, 2);
                    std::memcpy(&mh, bp + 2, 2);
                    const double d = __half2float(dh), mm = __half2float(mh);
                    for (int i = 0; i < 32; ++i)
                        ref += static_cast<double>(
                                   __half2float(x[(size_t)r * K + s2 * 32 + i])) *
                               (d * q5(bp, i) + mm);
                }
                const double got = __half2float(out[(size_t)r * N + n]);
                err2 += (got - ref) * (got - ref);
                ref2 += ref * ref;
            }
        }
    }
    const double nrmse = std::sqrt(err2 / std::max(ref2, 1e-30));
    EXPECT_LT(nrmse, 2e-2) << "Q5_1 MoE NRMSE";
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_out); cudaFree(d_off);
    mmq_q8_imma_release_all();
}

TEST(MmqQ8Imma, MoeGroupedQ8) {
    const int ne = 4, N = 128, K = 256;
    const int rows_per[ne] = {37, 0, 96, 19};  // ragged; expert 1 empty
    int32_t h_off[ne + 1] = {0};
    for (int e = 0; e < ne; ++e) h_off[e + 1] = h_off[e] + rows_per[e];
    const int expanded = h_off[ne];

    // packed experts: contiguous [ne][N][K/32] Q8_0 blocks
    std::vector<uint8_t> W;
    gen_q8_weight(W, ne * N, K, 91);
    std::vector<__half> x(static_cast<size_t>(expanded) * K);
    std::mt19937 rng(92);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half(nd(rng));

    uint8_t* d_w; __half *d_x, *d_out; int32_t* d_off;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(expanded) * N * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, sizeof(h_off)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_off, sizeof(h_off), cudaMemcpyHostToDevice);

    int max_rows = 0;
    for (int e = 0; e < ne; ++e) max_rows = std::max(max_rows, rows_per[e]);
    ASSERT_TRUE(mmq_imma_moe_gemm(d_w, /*qkind=*/0, d_x, d_out, d_off, max_rows, expanded, ne, N, K,
                                  nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(expanded) * N);
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    // reference: per-expert dense GEMM on dequantized Q8 weights (double),
    // with CPU-modeled activation quantization (NRMSE bound covers it)
    std::vector<int8_t> xs8;
    std::vector<float> xscale;
    quant_act_cpu(x, expanded, K, xs8, xscale);
    const int subs = K / kBlk;
    double err2 = 0, ref2 = 0;
    for (int e = 0; e < ne; ++e) {
        for (int r = h_off[e]; r < h_off[e + 1]; ++r) {
            for (int n = 0; n < N; ++n) {
                double accv = 0;
                for (int s = 0; s < subs; ++s) {
                    const uint8_t* bp = W.data() +
                        ((static_cast<size_t>(e) * N + n) * subs + s) * kBlockBytes;
                    __half dh;
                    std::memcpy(&dh, bp, 2);
                    const double dw = __half2float(dh);
                    const int8_t* wq = reinterpret_cast<const int8_t*>(bp + 2);
                    long isum = 0;
                    for (int i = 0; i < kBlk; ++i)
                        isum += static_cast<long>(xs8[(size_t)r * K + s * kBlk + i]) * wq[i];
                    accv += static_cast<double>(xscale[(size_t)r * subs + s]) * dw * isum;
                }
                const double got = __half2float(out[(size_t)r * N + n]);
                err2 += (got - accv) * (got - accv);
                ref2 += accv * accv;
            }
        }
    }
    const double nrmse = std::sqrt(err2 / std::max(ref2, 1e-30));
    EXPECT_LT(nrmse, 5e-3) << "MoE grouped vs exact model";
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_out); cudaFree(d_off);
    mmq_q8_imma_release_all();
}

// The MoE prefill sizes the grid from the bound rows/expert <= n instead of reading the offsets
// back (no D2H + sync per layer): surplus CTAs must exit on their offsets, and the tile hint must
// not change which rows are written.
TEST(MmqQ8Imma, MoeGroupedGridBoundMatchesExactMaxRows) {
    const int ne = 4, N = 128, K = 256;
    const int rows_per[ne] = {37, 0, 96, 19};
    int32_t h_off[ne + 1] = {0};
    for (int e = 0; e < ne; ++e) h_off[e + 1] = h_off[e] + rows_per[e];
    const int expanded = h_off[ne];
    std::vector<uint8_t> W;
    gen_q8_weight(W, ne * N, K, 93);
    std::vector<__half> x(static_cast<size_t>(expanded) * K);
    std::mt19937 rng(94);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half(nd(rng));
    uint8_t* d_w; __half *d_x, *d_out; int32_t* d_off;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(expanded) * N * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, sizeof(h_off)), cudaSuccess);
    cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_off, sizeof(h_off), cudaMemcpyHostToDevice);

    auto run = [&](int max_rows, int hint) {
        cudaMemset(d_out, 0xFF, static_cast<size_t>(expanded) * N * 2);  // NaN: an unwritten row shows
        EXPECT_TRUE(mmq_imma_moe_gemm(d_w, 0, d_x, d_out, d_off, max_rows, expanded, ne, N, K, nullptr, hint));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<__half> out(static_cast<size_t>(expanded) * N);
        cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);
        return out;
    };
    const auto exact = run(96, 0);          // BM 128, grid from the true max
    const auto bound = run(expanded, 0);    // BM 128, grid from the bound: 152 rows
    const auto small = run(expanded, 32);   // BM 32 by hint, grid from the bound
    for (size_t i = 0; i < exact.size(); ++i) {
        ASSERT_FALSE(std::isnan(__half2float(small[i]))) << "row " << i / N << " never written";
        ASSERT_EQ(std::memcmp(&exact[i], &bound[i], 2), 0) << "element " << i;
    }
    double err2 = 0, ref2 = 0;
    for (size_t i = 0; i < exact.size(); ++i) {
        const double a = __half2float(exact[i]), b = __half2float(small[i]);
        err2 += (a - b) * (a - b);
        ref2 += a * a;
    }
    EXPECT_LT(std::sqrt(err2 / std::max(ref2, 1e-30)), 1e-3) << "BM 32 vs BM 128";
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_out); cudaFree(d_off);
    mmq_q8_imma_release_all();
}

TEST(MmqQ8Imma, DeclineShapes) {
    // N odd / K not multiple of 64 / M < 2 → false. (M down to 2 is accepted
    // since the small-M split-K path — spec-decode verify chunks.)
    std::vector<uint8_t> W;
    gen_q8_weight(W, 64, 64, 1);
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_out = nullptr;
    cudaMalloc(&d_w, W.size());
    cudaMalloc(&d_x, 64 * 64 * sizeof(__half));
    cudaMalloc(&d_out, 64 * 64 * sizeof(__half));
    EXPECT_FALSE(mmq_q8_imma_gemm(d_w, d_x, d_out, 1, 128, 64, nullptr));   // M < 2
    EXPECT_FALSE(mmq_q8_imma_gemm(d_w, d_x, d_out, 64, 63, 64, nullptr));   // N odd
    EXPECT_FALSE(mmq_q8_imma_gemm(d_w, d_x, d_out, 64, 128, 32, nullptr));  // K % 64
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    mmq_q8_imma_release_all();
}

// #2468: the per-call split into caller scratch feeds the same kernel as the cached plane, so the
// output must be bit-identical; a scratch below imma_q8_plane_bytes declines.
TEST(MmqQ8Imma, TransientScratchMatchesCachedPlane) {
    for (const int M : {16, 200, 512}) {
        const int N = 384, K = 2048;
        std::vector<uint8_t> W;
        gen_q8_weight(W, N, K, 91 + M);
        std::vector<__half> x((size_t)M * K);
        std::mt19937 rng(M);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : x)
            v = __float2half(nd(rng));
        uint8_t* d_w = nullptr;
        void* d_scratch = nullptr;
        __half *d_x = nullptr, *d_a = nullptr, *d_b = nullptr;
        const size_t need = imma_q8_plane_bytes(N, K);
        ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_scratch, need), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_a, (size_t)M * N * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_b, (size_t)M * N * sizeof(__half)), cudaSuccess);
        cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemset(d_scratch, 0x5A, need);  // stale bytes must not survive the split
        ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_a, M, N, K, nullptr));
        EXPECT_FALSE(mmq_q8_imma_gemm_scratch(d_w, d_scratch, need - 1, d_x, d_b, M, N, K, nullptr));
        ASSERT_TRUE(mmq_q8_imma_gemm_scratch(d_w, d_scratch, need, d_x, d_b, M, N, K, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint16_t> a((size_t)M * N), b((size_t)M * N);
        cudaMemcpy(a.data(), d_a, a.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, b.size() * 2, cudaMemcpyDeviceToHost);
        EXPECT_EQ(a, b) << "M=" << M;
        cudaFree(d_w);
        cudaFree(d_scratch);
        cudaFree(d_x);
        cudaFree(d_a);
        cudaFree(d_b);
        mmq_q8_imma_release_all();
    }
}

// #2468 bound: plane IMMA (weights pre-split, the best an in-place kernel can reach) vs the
// uncached route (dequant_q8_0 to FP16 + cuBLAS) on the Qwen3.6 GDN Q8_0 shapes, weights rotated
// over >= 400 MB so both read DRAM. Prints us per GEMM; no assertion on speed.
TEST(MmqQ8Imma, BoundVsDequantCublas) {
    struct Shape {
        int N, K;
    };
    void* d_deq = nullptr;
    ASSERT_EQ(cudaMalloc(&d_deq, (size_t)8192 * 4096 * sizeof(__half)), cudaSuccess);
    for (const Shape sh : {Shape{8192, 2048}, Shape{4096, 2048}, Shape{2048, 4096}}) {
        std::vector<uint8_t> W;
        gen_q8_weight(W, sh.N, sh.K, 7);
        const int ring = (int)((400ull << 20) / W.size()) + 1;
        std::vector<void*> d_w(ring);
        for (auto& p : d_w) {
            ASSERT_EQ(cudaMalloc(&p, W.size()), cudaSuccess);
            cudaMemcpy(p, W.data(), W.size(), cudaMemcpyHostToDevice);
        }
        for (int M : {512, 1024, 2048, 4096}) {
            std::vector<__half> x((size_t)M * sh.K);
            std::mt19937 rng(M);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto& v : x)
                v = __float2half(nd(rng));
            __half *d_x = nullptr, *d_out = nullptr;
            ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(__half)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_out, (size_t)M * sh.N * sizeof(__half)), cudaSuccess);
            cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
            int64_t xs[2] = {M, sh.K}, os[2] = {M, sh.N}, ws[2] = {sh.N, sh.K};
            Tensor X(d_x, QType::F16, 2, xs, true), O(d_out, QType::F16, 2, os, true);
            Tensor Wd(d_deq, QType::F16, 2, ws, true);
            auto imma = [&](int i) { return mmq_q8_imma_gemm(d_w[i % ring], d_x, d_out, M, sh.N, sh.K, nullptr); };
            auto transient = [&](int i) {
                return mmq_q8_imma_gemm_scratch(d_w[i % ring], d_deq, (size_t)8192 * 4096 * sizeof(__half), d_x,
                                                d_out, M, sh.N, sh.K, nullptr);
            };
            auto deq = [&](int i) {
                Tensor Wq(d_w[i % ring], QType::Q8_0, 2, ws, true);
                dequant_weight_fp16(Wq, d_deq, nullptr);
                gemm(X, Wd, O, 1.0f, 0.0f, nullptr);
                return true;
            };
            auto time_us = [&](auto&& fn) {
                for (int i = 0; i < 2 * ring; i++)
                    EXPECT_TRUE(fn(i));  // warm: planes built, clocks up
                cudaEvent_t t0, t1;
                cudaEventCreate(&t0);
                cudaEventCreate(&t1);
                std::vector<float> ms(5);
                for (auto& m : ms) {
                    cudaEventRecord(t0);
                    for (int i = 0; i < 40; i++)
                        fn(i);
                    cudaEventRecord(t1);
                    cudaEventSynchronize(t1);
                    cudaEventElapsedTime(&m, t0, t1);
                }
                std::sort(ms.begin(), ms.end());
                cudaEventDestroy(t0);
                cudaEventDestroy(t1);
                return 1000.0 * ms[2] / 40;
            };
            const double us_deq = time_us(deq);
            const double us_tr = time_us(transient);
            const double us_imma = time_us(imma);
            printf("Q8Bound N=%d K=%d M=%d ring=%d: dequant+cuBLAS %.1f us, transient split+IMMA %.1f us, "
                   "plane IMMA %.1f us\n",
                   sh.N, sh.K, M, ring, us_deq, us_tr, us_imma);
            cudaFree(d_x);
            cudaFree(d_out);
            mmq_q8_imma_release_all();
        }
        for (void* p : d_w)
            cudaFree(p);
    }
    cudaFree(d_deq);
}

}  // namespace
}  // namespace imp
