// (t, h, w) positions for a prompt containing images.
//
// The failure mode is not a crash: wrong positions mean the model reads the
// image as if its tokens sat somewhere else on the grid, and describes a
// different picture. The oracle is `Qwen3VLModel.get_rope_index` /
// `get_vision_position_ids`, reimplemented here as explicit expectations rather
// than as a second copy of the loop.

#include "model/mrope_positions.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace imp {
namespace {

struct Built {
    std::vector<int32_t> pos;
    int next = 0;
    size_t n = 0;

    int t(size_t i) const { return pos[i]; }
    int h(size_t i) const { return pos[n + i]; }
    int w(size_t i) const { return pos[2 * n + i]; }
};

Built build(const std::vector<uint8_t>& mask, const std::vector<MRopeImageGrid>& grids, int start = 0) {
    Built b;
    b.n = mask.size();
    const auto r = qwen_build_mrope_positions(mask, grids, start);
    EXPECT_TRUE(r) << r.error();
    if (r) {
        b.pos = r->pos;
        b.next = r->next_pos;
    }
    return b;
}

// The invariant that keeps M-RoPE a no-op for every text-only model.
TEST(MRopePositions, TextOnlyIsPlainAscendingOnAllThreeAxes) {
    const Built b = build(std::vector<uint8_t>(7, 0), {});
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_EQ(b.t(i), static_cast<int>(i));
        EXPECT_EQ(b.h(i), static_cast<int>(i));
        EXPECT_EQ(b.w(i), static_cast<int>(i));
    }
    EXPECT_EQ(b.next, 7);
}

TEST(MRopePositions, ContinuationStartsWhereTheCallerSays) {
    const Built b = build(std::vector<uint8_t>(3, 0), {}, 100);
    EXPECT_EQ(b.t(0), 100);
    EXPECT_EQ(b.t(2), 102);
    EXPECT_EQ(b.next, 103);
}

// An image: one shared temporal position, row/column on the other two axes, in
// raster order over the MERGED grid.
TEST(MRopePositions, ImageTokensCarryTheirRowAndColumn) {
    // 2 text, then a 2x3 image (6 tokens), then 2 text.
    std::vector<uint8_t> mask = {0, 0, 1, 1, 1, 1, 1, 1, 0, 0};
    const Built b = build(mask, {{2, 3}});

    EXPECT_EQ(b.t(0), 0);
    EXPECT_EQ(b.t(1), 1);
    // Image starts at position 2.
    for (int k = 0; k < 6; ++k) {
        const size_t i = 2 + static_cast<size_t>(k);
        EXPECT_EQ(b.t(i), 2) << "token " << k << ": all image tokens share one temporal position";
        EXPECT_EQ(b.h(i), 2 + k / 3) << "token " << k;
        EXPECT_EQ(b.w(i), 2 + k % 3) << "token " << k;
    }
}

// The whole point of the layout: six tokens cost three positions, not six.
TEST(MRopePositions, AnImageCostsMaxOfRowsAndColsNotItsTokenCount) {
    std::vector<uint8_t> mask = {0, 1, 1, 1, 1, 1, 1, 0};
    const Built b = build(mask, {{2, 3}});
    EXPECT_EQ(b.t(0), 0);
    // Image occupies positions 1..3 on the width axis (1 + max(2,3) - 1 = 3).
    EXPECT_EQ(b.t(7), 4) << "the token after the image resumes at 1 + max(rows, cols)";
    EXPECT_EQ(b.h(7), 4);
    EXPECT_EQ(b.w(7), 4);
    EXPECT_EQ(b.next, 5);

    // A tall image costs its row count instead.
    const Built tall = build({1, 1, 1, 1, 1, 1, 0}, {{3, 2}});
    EXPECT_EQ(tall.t(6), 3);
    EXPECT_EQ(tall.next, 4);
}

TEST(MRopePositions, TwoImagesEachAdvanceIndependently) {
    // img(1x2) text img(2x2)
    std::vector<uint8_t> mask = {1, 1, 0, 1, 1, 1, 1};
    const Built b = build(mask, {{1, 2}, {2, 2}});

    EXPECT_EQ(b.h(0), 0);
    EXPECT_EQ(b.w(0), 0);
    EXPECT_EQ(b.w(1), 1);
    // First image costs max(1,2) = 2, so the text token sits at 2.
    EXPECT_EQ(b.t(2), 2);
    // Second image starts at 3.
    EXPECT_EQ(b.t(3), 3);
    EXPECT_EQ(b.h(3), 3);
    EXPECT_EQ(b.w(3), 3);
    EXPECT_EQ(b.h(6), 4);
    EXPECT_EQ(b.w(6), 4);
    EXPECT_EQ(b.next, 5);  // 3 + max(2,2)
}

TEST(MRopePositions, ASingleImageWithNoTextAround) {
    const Built b = build({1, 1, 1, 1}, {{2, 2}});
    EXPECT_EQ(b.h(0), 0);
    EXPECT_EQ(b.w(3), 1);
    EXPECT_EQ(b.next, 2);
}

// A run length that does not match the grid means the placeholder expansion and
// the preprocessor disagree. Every position after it would be shifted.
TEST(MRopePositions, RefusesARunThatDoesNotMatchItsGrid) {
    const auto r = qwen_build_mrope_positions({0, 1, 1, 1, 0}, {{2, 2}}, 0);
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("4 tokens"), std::string::npos) << r.error();
    EXPECT_NE(r.error().find("reserves 3"), std::string::npos) << r.error();
}

TEST(MRopePositions, RefusesAGridCountMismatch) {
    // Two runs, one grid.
    {
        const auto r = qwen_build_mrope_positions({1, 0, 1}, {{1, 1}}, 0);
        ASSERT_FALSE(r.has_value());
        EXPECT_NE(r.error().find("more image runs"), std::string::npos) << r.error();
    }
    // One run, two grids.
    {
        const auto r = qwen_build_mrope_positions({1, 0}, {{1, 1}, {1, 1}}, 0);
        ASSERT_FALSE(r.has_value());
        EXPECT_NE(r.error().find("grids were supplied"), std::string::npos) << r.error();
    }
}

TEST(MRopePositions, RefusesAnEmptyGrid) {
    {
        const auto r = qwen_build_mrope_positions({1}, {{0, 3}}, 0);
        ASSERT_FALSE(r.has_value());
        EXPECT_NE(r.error().find("empty grid"), std::string::npos) << r.error();
    }
}

TEST(MRopePositions, EmptySequenceIsFine) {
    const auto r = qwen_build_mrope_positions({}, {}, 5);
    ASSERT_TRUE(r) << r.error();
    EXPECT_TRUE(r->pos.empty());
    EXPECT_EQ(r->next_pos, 5);
}

// Two adjacent images with no text between them are one contiguous run of image
// tokens, and must NOT be read as a single image.
TEST(MRopePositions, AdjacentImagesWithoutSeparatorAreRefused) {
    // 4 contiguous image tokens, declared as two 1x2 images.
    const auto r = qwen_build_mrope_positions({1, 1, 1, 1}, {{1, 2}, {1, 2}}, 0);
    ASSERT_FALSE(r.has_value());
    EXPECT_FALSE(r.error().empty()) << "a merged run must not silently become one image";
}

}  // namespace
}  // namespace imp
