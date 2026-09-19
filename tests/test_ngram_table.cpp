// model/ngram_table.h: the Qwen4Exp n-gram hash against Qwen4ExpTextNGramEmbedding.forward.
// Cases: EOS-filled context at a sequence start, shifts blocked by an EOS inside the window,
// Python-style modulo on a negative wrapped product, bigram vs trigram head groups.

#include "model/ngram_table.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace imp {
namespace {

constexpr int kEos = 99;
constexpr int64_t kMult[3] = {7, 11, 13};
constexpr int64_t kVocab[4] = {100, 101, 1000, 1001};
constexpr int64_t kOff[4] = {0, 100, 201, 1201};

NGramHashParams params() { return NGramHashParams{kMult, kOff, kVocab, 3, 4, kEos}; }

int64_t py_mod(int64_t a, int64_t m) {
    const int64_t r = a % m;
    return r < 0 ? r + m : r;
}

// Reference: history = ctx ++ tokens, torch semantics spelled out per position.
std::vector<int64_t> reference(const std::vector<int32_t>& ctx, const std::vector<int32_t>& tok) {
    std::vector<int32_t> hist(ctx);
    hist.insert(hist.end(), tok.begin(), tok.end());
    std::vector<int64_t> out;
    for (size_t t = ctx.size(); t < hist.size(); t++) {
        int64_t sh[3];
        for (int s = 0; s < 3; s++) {
            bool eos_between = false;
            for (int k = 1; k <= s; k++)
                if (hist[t - k] == kEos)
                    eos_between = true;
            sh[s] = eos_between ? kEos : hist[t - s];
        }
        const uint64_t m2 = static_cast<uint64_t>(sh[0]) * 7u ^ static_cast<uint64_t>(sh[1]) * 11u;
        const uint64_t m3 = m2 ^ static_cast<uint64_t>(sh[2]) * 13u;
        for (int h = 0; h < 2; h++)
            out.push_back(py_mod(static_cast<int64_t>(m2), kVocab[h]) + kOff[h]);
        for (int h = 2; h < 4; h++)
            out.push_back(py_mod(static_cast<int64_t>(m3), kVocab[h]) + kOff[h]);
    }
    return out;
}

void check(const std::vector<int32_t>& ctx, const std::vector<int32_t>& tok) {
    std::vector<int64_t> got(tok.size() * 4);
    ngram_hash(params(), ctx.data(), tok.data(), static_cast<int>(tok.size()), got.data());
    EXPECT_EQ(got, reference(ctx, tok));
}

TEST(NGramHash, SequenceStartIsEosFilled) {
    check({kEos, kEos}, {5, 6, 7, 8});
    // First token: bigram of (5, EOS), trigram of (5, EOS, EOS).
    std::vector<int64_t> got(4);
    const int32_t ctx[2] = {kEos, kEos};
    const int32_t tok[1] = {5};
    ngram_hash(params(), ctx, tok, 1, got.data());
    const int64_t m2 = static_cast<int64_t>(5u * 7u ^ 99u * 11u);
    const int64_t m3 = static_cast<int64_t>(5u * 7u ^ 99u * 11u ^ 99u * 13u);
    EXPECT_EQ(got[0], py_mod(m2, 100));
    EXPECT_EQ(got[1], py_mod(m2, 101) + 100);
    EXPECT_EQ(got[2], py_mod(m3, 1000) + 201);
    EXPECT_EQ(got[3], py_mod(m3, 1001) + 1201);
}

TEST(NGramHash, EosInsideWindowBlocksTheShift) {
    // Token after an EOS sees EOS for both shifts; two after sees its predecessor but EOS at shift 2.
    check({1, 2}, {3, kEos, 4, 5, 6});
    const int32_t ctx[2] = {1, 2};
    const int32_t tok[3] = {kEos, 4, 5};
    std::vector<int64_t> got(12);
    ngram_hash(params(), ctx, tok, 3, got.data());
    // t=1 (token 4): shifted = {4, EOS, EOS}; t=2 (token 5): shifted = {5, 4, EOS}.
    const int64_t m2_t2 = static_cast<int64_t>(5u * 7u ^ 4u * 11u);
    const int64_t m3_t2 = static_cast<int64_t>(5u * 7u ^ 4u * 11u ^ 99u * 13u);
    EXPECT_EQ(got[8], py_mod(m2_t2, 100));
    EXPECT_EQ(got[10], py_mod(m3_t2, 1000) + 201);
}

TEST(NGramHash, CarriedContextMatchesOneShot) {
    // Hashing tokens in two chunks with the carried 2-token context equals one pass.
    const std::vector<int32_t> all = {10, 11, kEos, 12, 13, 14, 15};
    std::vector<int64_t> one(all.size() * 4);
    const int32_t eos_ctx[2] = {kEos, kEos};
    ngram_hash(params(), eos_ctx, all.data(), static_cast<int>(all.size()), one.data());
    std::vector<int64_t> a(3 * 4), b(4 * 4);
    ngram_hash(params(), eos_ctx, all.data(), 3, a.data());
    const int32_t ctx2[2] = {all[1], all[2]};
    ngram_hash(params(), ctx2, all.data() + 3, 4, b.data());
    a.insert(a.end(), b.begin(), b.end());
    EXPECT_EQ(a, one);
}

TEST(NGramHash, NegativeWrappedProductUsesPythonModulo) {
    const int64_t big_mult[3] = {INT64_C(0x7fffffffffffffff), 3, 5};
    NGramHashParams p{big_mult, kOff, kVocab, 3, 4, kEos};
    const int32_t ctx[2] = {kEos, kEos};
    const int32_t tok[1] = {2};
    std::vector<int64_t> got(4);
    ngram_hash(p, ctx, tok, 1, got.data());
    const uint64_t m2 = 2u * UINT64_C(0x7fffffffffffffff) ^ 99u * 3u;  // wraps negative
    ASSERT_LT(static_cast<int64_t>(m2), 0);
    EXPECT_EQ(got[0], py_mod(static_cast<int64_t>(m2), 100));
    EXPECT_GE(got[0], 0);
}

}  // namespace
}  // namespace imp
