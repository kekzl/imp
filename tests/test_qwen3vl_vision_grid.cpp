// Qwen3-VL encoder grid math: token order, RoPE positions, position-embedding
// resample.
//
// Every failure here is silent — the encoder runs either way and returns
// embeddings that are merely wrong. So the resample is tested against a real
// oracle rather than against itself: a bilinear interpolation of a table that is
// an AFFINE function of (row, col) must reproduce that affine function exactly
// at the resampled coordinate. That pins the taps and the weights independently,
// without reimplementing the formula under test.

#include "vision/qwen3vl_vision_grid.h"

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <utility>
#include <string>
#include <vector>

namespace imp {
namespace {

QwenVisionGrid build(int h, int w, int merge = 2, int side = 48) {
    auto g = qwen3vl_build_vision_grid(h, w, merge, side);
    EXPECT_TRUE(g) << g.error();
    return g.value_or(QwenVisionGrid{});
}

// The patchifier emits tokens grouped by 2x2 merge block, so the merger can
// consume four CONSECUTIVE tokens. Raster order would still run and would put
// four unrelated patches in every merged token.
TEST(Qwen3VLVisionGrid, TokenOrderIsMergeBlockGroupedNotRaster) {
    const auto g = build(4, 4);
    ASSERT_EQ(g.tokens, 16);

    // 4x4 grid, merge 2 -> raster indices 0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15
    const int expect_raster[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
    for (int i = 0; i < 16; ++i) {
        const int raster = g.row[i] * 4 + g.col[i];
        EXPECT_EQ(raster, expect_raster[i]) << "token " << i;
    }
}

TEST(Qwen3VLVisionGrid, EveryPatchAppearsExactlyOnce) {
    for (auto [h, w] : {std::pair{4, 6}, std::pair{8, 8}, std::pair{2, 10}, std::pair{6, 2}}) {
        const auto g = build(h, w);
        std::set<int> seen;
        for (int i = 0; i < g.tokens; ++i) {
            ASSERT_GE(g.row[i], 0);
            ASSERT_LT(g.row[i], h);
            ASSERT_GE(g.col[i], 0);
            ASSERT_LT(g.col[i], w);
            EXPECT_TRUE(seen.insert(g.row[i] * w + g.col[i]).second) << h << "x" << w << " token " << i;
        }
        EXPECT_EQ(static_cast<int>(seen.size()), h * w);
    }
}

// A partition of unity. If the weights did not sum to 1 the position embedding
// would be scaled per token, which reads as a strange but plausible encoder.
TEST(Qwen3VLVisionGrid, InterpolationWeightsSumToOne) {
    for (auto [h, w] : {std::pair{4, 4}, std::pair{32, 48}, std::pair{96, 64}, std::pair{2, 2}}) {
        const auto g = build(h, w);
        for (int i = 0; i < g.tokens; ++i) {
            float s = 0.0f;
            for (int t = 0; t < kQwenVisionPosTaps; ++t)
                s += g.pos_weights[i * kQwenVisionPosTaps + t];
            EXPECT_NEAR(s, 1.0f, 1e-5f) << h << "x" << w << " token " << i;
        }
    }
}

// The oracle: resampling an affine table must return the affine function at the
// source coordinate. align_corners maps target index i on an axis of length n to
// source i*(side-1)/(n-1), so the expected value is known in closed form.
TEST(Qwen3VLVisionGrid, ResamplesAnAffineTableExactly) {
    const int side = 48;
    // table(r, c) = 3 - 0.25*r + 0.5*c
    auto table = [](int r, int c) { return 3.0 - 0.25 * r + 0.5 * c; };

    for (auto [h, w] : {std::pair{4, 4}, std::pair{48, 48}, std::pair{96, 32}, std::pair{30, 70}}) {
        const auto g = build(h, w, 2, side);
        for (int i = 0; i < g.tokens; ++i) {
            double got = 0.0;
            for (int t = 0; t < kQwenVisionPosTaps; ++t) {
                const int idx = g.pos_taps[i * kQwenVisionPosTaps + t];
                got += static_cast<double>(g.pos_weights[i * kQwenVisionPosTaps + t]) *
                       table(idx / side, idx % side);
            }
            const double src_r = g.row[i] * static_cast<double>(side - 1) / std::max(h - 1, 1);
            const double src_c = g.col[i] * static_cast<double>(side - 1) / std::max(w - 1, 1);
            EXPECT_NEAR(got, 3.0 - 0.25 * src_r + 0.5 * src_c, 1e-4)
                << h << "x" << w << " token " << i << " (r=" << g.row[i] << ", c=" << g.col[i] << ")";
        }
    }
}

// align_corners: the four corners of the image must land exactly on the four
// corners of the learned table, with no second tap bleeding in.
TEST(Qwen3VLVisionGrid, CornersLandExactlyOnTheTableCorners) {
    const int side = 48, h = 30, w = 70;
    const auto g = build(h, w, 2, side);
    const std::pair<int, int> corners[] = {{0, 0}, {0, w - 1}, {h - 1, 0}, {h - 1, w - 1}};
    for (auto [cr, cc] : corners) {
        int found = -1;
        for (int i = 0; i < g.tokens && found < 0; ++i)
            if (g.row[i] == cr && g.col[i] == cc)
                found = i;
        ASSERT_GE(found, 0) << cr << "," << cc;
        const int want = (cr == 0 ? 0 : side - 1) * side + (cc == 0 ? 0 : side - 1);
        float total = 0.0f;
        for (int t = 0; t < kQwenVisionPosTaps; ++t) {
            const float wt = g.pos_weights[found * kQwenVisionPosTaps + t];
            if (wt > 0.0f) {
                EXPECT_EQ(g.pos_taps[found * kQwenVisionPosTaps + t], want) << cr << "," << cc;
                total += wt;
            }
        }
        EXPECT_NEAR(total, 1.0f, 1e-6f);
    }
}

// A grid the same size as the table must be an identity gather, not a blur.
TEST(Qwen3VLVisionGrid, MatchingGridIsAnIdentityGather) {
    const int side = 48;
    const auto g = build(side, side, 2, side);
    for (int i = 0; i < g.tokens; ++i) {
        const int want = g.row[i] * side + g.col[i];
        float mass_on_target = 0.0f, mass_elsewhere = 0.0f;
        for (int t = 0; t < kQwenVisionPosTaps; ++t) {
            const float wt = g.pos_weights[i * kQwenVisionPosTaps + t];
            (g.pos_taps[i * kQwenVisionPosTaps + t] == want ? mass_on_target : mass_elsewhere) += wt;
        }
        EXPECT_NEAR(mass_on_target, 1.0f, 1e-6f) << "token " << i;
        EXPECT_NEAR(mass_elsewhere, 0.0f, 1e-6f) << "token " << i;
    }
}

TEST(Qwen3VLVisionGrid, TapsStayInsideTheTable) {
    const int side = 48;
    for (auto [h, w] : {std::pair{2, 2}, std::pair{4, 4}, std::pair{200, 4}, std::pair{48, 96}}) {
        const auto g = build(h, w, 2, side);
        for (size_t k = 0; k < g.pos_taps.size(); ++k) {
            EXPECT_GE(g.pos_taps[k], 0);
            EXPECT_LT(g.pos_taps[k], side * side);
        }
    }
}

// A grid that is not a multiple of the merge size would drop the tail of the
// last block, silently, since the token count still comes out of the loop.
TEST(Qwen3VLVisionGrid, RefusesAGridThatIsNotAMultipleOfTheMergeSize) {
    for (auto [h, w] : {std::pair{5, 4}, std::pair{4, 5}}) {
        const auto g = qwen3vl_build_vision_grid(h, w, 2, 48);
        ASSERT_FALSE(g.has_value()) << h << "x" << w;
        EXPECT_NE(g.error().find("multiple"), std::string::npos) << g.error();
    }
}

TEST(Qwen3VLVisionGrid, RefusesNonPositiveDimensions) {
    EXPECT_FALSE(qwen3vl_build_vision_grid(0, 4, 2, 48).has_value());
    EXPECT_FALSE(qwen3vl_build_vision_grid(4, 0, 2, 48).has_value());
    EXPECT_FALSE(qwen3vl_build_vision_grid(4, 4, 0, 48).has_value());
    EXPECT_FALSE(qwen3vl_build_vision_grid(4, 4, 2, 0).has_value());
}

// merge == 1 is the degenerate case the same code has to keep handling: token
// order collapses to raster.
TEST(Qwen3VLVisionGrid, MergeOneIsRasterOrder) {
    const auto g = build(3, 5, 1);
    ASSERT_EQ(g.tokens, 15);
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(g.row[i], i / 5);
        EXPECT_EQ(g.col[i], i % 5);
    }
}

}  // namespace
}  // namespace imp
