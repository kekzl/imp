// Group-limited top-k (src/compute/moe_group_limit.h, the function topk_gating_kernel runs on
// thread 0) against Ling-3.0's BailingMoeV3Gate in torch (tests/refs/moe_group_topk_golden.h).
// The host replays the kernel's selection: mask non-kept groups, top-k on sigmoid + bias,
// weights from the unbiased sigmoid, normalised, * scale. Weights to 1e-5 abs.

#include "compute/moe_group_limit.h"
#include "kda_reference.h"
#include "refs/moe_group_topk_golden.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace imp {
namespace {

namespace gold = moe_group_topk_golden;

struct Routed {
    std::vector<int> idx;
    std::vector<float> w;
};

Routed route(const float* logits, const std::vector<float>& bias, MoeGroupLimit grp) {
    const int ne = gold::kExperts;
    std::vector<float> score(ne), sel(ne);
    for (int e = 0; e < ne; ++e) {
        score[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        sel[e] = score[e] + bias[e];
    }
    unsigned char keep[kMoeMaxGroups];
    if (moe_group_limit_mask(sel.data(), ne, grp, keep)) {
        const int gs = ne / grp.n_group;
        for (int e = 0; e < ne; ++e)
            if (!keep[e / gs])
                sel[e] = -3.0e38f;
    }
    Routed r;
    for (int k = 0; k < gold::kTopK; ++k) {
        const int best = static_cast<int>(std::max_element(sel.begin(), sel.end()) - sel.begin());
        r.idx.push_back(best);
        sel[best] = -3.4e38f;
    }
    std::sort(r.idx.begin(), r.idx.end());
    float sum = 0.0f;
    for (int e : r.idx)
        sum += score[e];
    for (int e : r.idx)
        r.w.push_back(score[e] / sum * gold::kScale);
    return r;
}

struct Inputs {
    std::vector<float> logits, bias;
};

Inputs inputs() {
    kda_ref::Lcg rng(gold::kSeed);
    Inputs in;
    in.logits = rng.fill(static_cast<size_t>(gold::kTokens) * gold::kExperts, 4.0);
    in.bias = rng.fill(gold::kExperts, 0.2);
    return in;
}

TEST(MoeGroupLimit, MatchesLingGate) {
    const Inputs in = inputs();
    const MoeGroupLimit grp{gold::kNGroup, gold::kTopkGroup};
    for (int t = 0; t < gold::kTokens; ++t) {
        const Routed r = route(in.logits.data() + (static_cast<size_t>(t) * gold::kExperts), in.bias, grp);
        for (int k = 0; k < gold::kTopK; ++k) {
            EXPECT_EQ(r.idx[k], gold::kIdx[(t * gold::kTopK) + k]) << "token " << t << " slot " << k;
            EXPECT_NEAR(r.w[k], gold::kWeight[(t * gold::kTopK) + k], 1e-5) << "token " << t;
        }
    }
}

// FP16 round to nearest even (normal and subnormal range), as the device bias is stored.
float round_to_half(float x) {
    if (std::abs(x) < 0x1p-14f)
        return std::nearbyint(x * 0x1p24f) * 0x1p-24f;
    int e = 0;
    (void)std::frexp(x, &e);
    const float s = std::ldexp(1.0f, 11 - e);
    return std::nearbyint(x * s) / s;
}

// The kernel reads the expert bias as FP16 (tests/test_moe_group_topk.cu): same selection.
TEST(MoeGroupLimit, SelectionStableUnderFp16Bias) {
    Inputs in = inputs();
    for (auto& b : in.bias)
        b = round_to_half(b);
    const MoeGroupLimit grp{gold::kNGroup, gold::kTopkGroup};
    for (int t = 0; t < gold::kTokens; ++t) {
        const Routed r = route(in.logits.data() + (static_cast<size_t>(t) * gold::kExperts), in.bias, grp);
        EXPECT_TRUE(std::equal(r.idx.begin(), r.idx.end(), gold::kIdx + (t * gold::kTopK))) << "token " << t;
    }
}

// Without the group limit the same inputs route differently on the golden's flagged tokens.
TEST(MoeGroupLimit, UngroupedDiffers) {
    const Inputs in = inputs();
    int differs = 0;
    for (int t = 0; t < gold::kTokens; ++t) {
        const Routed r = route(in.logits.data() + (static_cast<size_t>(t) * gold::kExperts), in.bias, {});
        differs += !std::equal(r.idx.begin(), r.idx.end(), gold::kIdx + (t * gold::kTopK));
    }
    EXPECT_EQ(differs, gold::kTokensWhereGroupingMatters);
}

TEST(MoeGroupLimit, RefusesUnservableGeometry) {
    std::vector<float> sel(10, 0.0f);
    unsigned char keep[kMoeMaxGroups];
    EXPECT_FALSE(moe_group_limit_mask(sel.data(), 10, MoeGroupLimit{3, 1}, keep));   // 10 % 3
    EXPECT_FALSE(moe_group_limit_mask(sel.data(), 10, MoeGroupLimit{10, 2}, keep));  // group of 1
    EXPECT_FALSE(moe_group_limit_mask(sel.data(), 10, MoeGroupLimit{2, 2}, keep));   // keeps all
    EXPECT_TRUE(moe_group_limit_mask(sel.data(), 10, MoeGroupLimit{5, 2}, keep));
}

}  // namespace
}  // namespace imp
