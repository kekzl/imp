// Radix top-k sampler (sampling_topk_radix.cu) against the legacy multi-block kernels it replaces:
// every scratch byte (block max/sum, candidate values/indices in order) and the token must match,
// single-row and rows, for ties, -inf masks, signed zeros, denormals and FLT_MAX extremes.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "compute/sampling.h"
#include "compute/sampling_internal.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace imp {
namespace {

enum class Gen { Normal, Ties, TopTies, Masked, Extremes, AllEqual, AllNegInf };
constexpr Gen kGens[] = {Gen::Normal,   Gen::Ties,     Gen::TopTies,  Gen::Masked,
                         Gen::Extremes, Gen::AllEqual, Gen::AllNegInf};

size_t used_bytes(int top_k) {
    return sizeof(int32_t) + (SAMPLE_NBLOCKS * 2 * sizeof(float)) +
           (static_cast<size_t>(SAMPLE_NBLOCKS) * top_k * (sizeof(float) + sizeof(int32_t)));
}

std::vector<float> make_logits(Gen g, int vocab, std::mt19937& rng) {
    std::vector<float> h(vocab);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::uniform_int_distribution<int> small(-6, 6);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const float ninf = -std::numeric_limits<float>::infinity();
    const float extremes[] = {FLT_MAX, -FLT_MAX, 0.0f,   -0.0f, 1e-40f, -1e-40f, 1e-38f, 1.0f,
                              -1.0f,   88.0f,    -88.0f, ninf,  3.5f,   3.5f,    FLT_MAX};
    for (int i = 0; i < vocab; ++i) {
        switch (g) {
            case Gen::Normal:
                h[i] = nd(rng);
                break;
            case Gen::Ties:
                h[i] = static_cast<float>(small(rng)) * 0.5f;
                break;
            case Gen::TopTies:
                h[i] = nd(rng);
                break;
            case Gen::Masked:
                h[i] = (u01(rng) < 0.998f) ? ninf : static_cast<float>(small(rng) / 3);
                break;
            case Gen::Extremes:
                h[i] = (u01(rng) < 0.5f) ? extremes[rng() % (sizeof(extremes) / sizeof(float))] : nd(rng);
                break;
            case Gen::AllEqual:
                h[i] = 1.25f;
                break;
            case Gen::AllNegInf:
                h[i] = ninf;
                break;
        }
    }
    if (g == Gen::TopTies) {
        // copy the 200 largest values onto random other slots: ties inside the candidate set
        std::vector<float> s = h;
        std::nth_element(s.begin(), s.begin() + 200, s.end(), std::greater<float>());
        for (int j = 0; j < 200; ++j)
            for (int c = 0; c < 3; ++c)
                h[rng() % vocab] = s[j];
    }
    return h;
}

struct Slots {
    float* d_logits = nullptr;
    int32_t* d_legacy = nullptr;
    int32_t* d_radix = nullptr;
    int n = 0;
    Slots(int vocab, int rows) : n(rows) {
        EXPECT_EQ(cudaMalloc(&d_logits, sizeof(float) * vocab * rows), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_legacy, SAMPLE_SCRATCH_BYTES * rows), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_radix, SAMPLE_SCRATCH_BYTES * rows), cudaSuccess);
    }
    ~Slots() {
        cudaFree(d_logits);
        cudaFree(d_legacy);
        cudaFree(d_radix);
    }
    void poison() {
        ASSERT_EQ(cudaMemset(d_legacy, 0xA5, SAMPLE_SCRATCH_BYTES * n), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_radix, 0x5A, SAMPLE_SCRATCH_BYTES * n), cudaSuccess);
    }
    std::vector<char> bytes(const int32_t* d, int row, size_t len) const {
        std::vector<char> h(len);
        EXPECT_EQ(cudaMemcpy(h.data(), reinterpret_cast<const char*>(d) + (SAMPLE_SCRATCH_BYTES * row), len,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return h;
    }
};

int32_t token_of(const std::vector<char>& b) {
    int32_t t = 0;
    std::memcpy(&t, b.data(), sizeof(t));
    return t;
}

TEST(TopkRadixSampler, SingleRowMatchesLegacyBytesAndTokens) {
    std::mt19937 rng(20261018u);
    int configs = 0;
    int tokens = 0;
    for (const int vocab : {32000, 151936, 262144}) {
        ASSERT_TRUE(topk_radix_supported(vocab, SAMPLE_MAX_TOP_K));
        Slots s(vocab, 1);
        for (const Gen g : kGens) {
            for (int rep = 0; rep < 2; ++rep) {
                const std::vector<float> h = make_logits(g, vocab, rng);
                ASSERT_EQ(cudaMemcpy(s.d_logits, h.data(), sizeof(float) * vocab, cudaMemcpyHostToDevice),
                          cudaSuccess);
                for (const int k : {1, 2, 5, 40, 64, 97, 128}) {
                    for (const float temp : {1.0f, 0.7f}) {
                        for (int si = 0; si < 12; ++si) {
                            const unsigned int seed = 1000u + (rng() % 100000u);
                            const float top_p = (si % 3 == 0) ? 1.0f : ((si % 3 == 1) ? 0.95f : 0.5f);
                            s.poison();
                            ASSERT_EQ(launch_topk_topp_multiblock_legacy(s.d_logits, vocab, k, top_p,
                                                                         1.0f / temp, seed, s.d_legacy,
                                                                         nullptr, nullptr),
                                      cudaSuccess);
                            ASSERT_EQ(launch_topk_topp_radix(s.d_logits, vocab, k, top_p, 1.0f / temp, seed,
                                                             s.d_radix, nullptr, nullptr),
                                      cudaSuccess);
                            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                            const size_t len = (si == 0) ? used_bytes(k) : sizeof(int32_t);
                            const std::vector<char> a = s.bytes(s.d_legacy, 0, len);
                            const std::vector<char> b = s.bytes(s.d_radix, 0, len);
                            ASSERT_EQ(token_of(a), token_of(b))
                                << "vocab " << vocab << " gen " << static_cast<int>(g) << " k " << k
                                << " seed " << seed;
                            ASSERT_EQ(std::memcmp(a.data(), b.data(), len), 0)
                                << "scratch differs: vocab " << vocab << " gen " << static_cast<int>(g)
                                << " k " << k;
                            ++tokens;
                            configs += (si == 0) ? 1 : 0;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(configs, 3 * 7 * 2 * 7 * 2);
    EXPECT_EQ(tokens, configs * 12);
}

// Rows: per-row top_k under one max_top_k, all three finalize slot widths (8, 16, 32).
TEST(TopkRadixSampler, RowsMatchLegacyBytesAndTokens) {
    std::mt19937 rng(7u);
    const int ks_by_max[3][6] = {{1, 2, 32, 7, 3, 32}, {40, 1, 64, 2, 17, 33}, {128, 1, 2, 40, 97, 64}};
    const int maxes[3] = {32, 64, 128};
    for (const int vocab : {32000, 151936, 262144}) {
        constexpr int kRows = 6;
        Slots s(vocab, kRows);
        TopkRowArgs* d_args = nullptr;
        ASSERT_EQ(cudaMalloc(&d_args, sizeof(TopkRowArgs) * kRows), cudaSuccess);
        for (int mi = 0; mi < 3; ++mi) {
            for (int rep = 0; rep < 4; ++rep) {
                for (int r = 0; r < kRows; ++r) {
                    const std::vector<float> h = make_logits(kGens[(r + rep) % 7], vocab, rng);
                    ASSERT_EQ(cudaMemcpy(s.d_logits + (static_cast<size_t>(r) * vocab), h.data(),
                                         sizeof(float) * vocab, cudaMemcpyHostToDevice),
                              cudaSuccess);
                }
                std::vector<TopkRowArgs> args(kRows);
                s.poison();
                for (const bool radix : {false, true}) {
                    for (int r = 0; r < kRows; ++r) {
                        char* slot = reinterpret_cast<char*>(radix ? s.d_radix : s.d_legacy) +
                                     (SAMPLE_SCRATCH_BYTES * r);
                        args[r] = TopkRowArgs{s.d_logits + (static_cast<size_t>(r) * vocab),
                                              reinterpret_cast<int32_t*>(slot),
                                              (r % 2) ? 1.0f : 1.0f / 0.8f,
                                              (r % 3 == 0) ? 1.0f : 0.9f,
                                              77u + (rep * 13u) + r,
                                              ks_by_max[mi][r]};
                    }
                    ASSERT_EQ(cudaMemcpy(d_args, args.data(), sizeof(TopkRowArgs) * kRows,
                                         cudaMemcpyHostToDevice),
                              cudaSuccess);
                    ASSERT_EQ(radix ? launch_topk_topp_rows_radix(d_args, kRows, maxes[mi], vocab, nullptr)
                                    : launch_topk_topp_rows_legacy(d_args, kRows, maxes[mi], vocab, nullptr),
                              cudaSuccess);
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                }
                for (int r = 0; r < kRows; ++r) {
                    const size_t len = used_bytes(ks_by_max[mi][r]);
                    const std::vector<char> a = s.bytes(s.d_legacy, r, len);
                    const std::vector<char> b = s.bytes(s.d_radix, r, len);
                    ASSERT_EQ(token_of(a), token_of(b))
                        << "vocab " << vocab << " row " << r << " max " << maxes[mi];
                    ASSERT_EQ(std::memcmp(a.data(), b.data(), len), 0)
                        << "scratch differs: vocab " << vocab << " row " << r << " max " << maxes[mi];
                }
            }
        }
        cudaFree(d_args);
    }
}

TEST(TopkRadixSampler, ShapesPastTheRadixBoundStayLegacy) {
    EXPECT_TRUE(topk_radix_supported(262144, 128));
    EXPECT_FALSE(topk_radix_supported(262145, 40));
    EXPECT_FALSE(topk_radix_supported(151936, 129));
    EXPECT_FALSE(topk_radix_supported(151936, 0));
}

}  // namespace
}  // namespace imp
