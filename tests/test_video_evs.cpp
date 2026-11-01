// EVS video token pruning, host half (vLLM video_prune/evs.py): retained count, stable top-k mask,
// pruned prompt layout and the M-RoPE positions of the kept tokens. Goldens derived by hand from
// evs.py semantics; the positions are those of the unpruned prompt (HF get_rope_index).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "model/image_placeholders.h"
#include "model/mrope_positions.h"
#include "runtime/config.h"
#include "vision/video_evs.h"

namespace imp {
namespace {

// Token ids of the hand-built prompts: text, vision_start, vision_end, video pad, stamps.
constexpr int32_t kT = 1, kS = 10, kE = 11, kVid = 12, kImg = 13, kStamp0 = 100;

TEST(VideoEvs, RetainedCountTruncatesAndFloorsAtOnePair) {
    EXPECT_EQ(evs_retained_count(4, 10, 0.0), 40);
    EXPECT_EQ(evs_retained_count(4, 10, 0.5), 20);
    EXPECT_EQ(evs_retained_count(4, 10, 0.25), 30);
    EXPECT_EQ(evs_retained_count(4, 10, 0.75), 10);
    // 128 * 0.05 = 6.4 -> 6, below one pair: the floor keeps 16.
    EXPECT_EQ(evs_retained_count(8, 16, 0.95), 16);
    // 7 * 9 * 0.5 = 31.5 -> 31 (Python int() truncates).
    EXPECT_EQ(evs_retained_count(7, 9, 0.5), 31);
    EXPECT_EQ(evs_retained_count(0, 16, 0.5), 0);
    EXPECT_EQ(evs_retained_count(4, 0, 0.5), 0);
}

TEST(VideoEvs, ConfigRateIsReadAsItsDecimal) {
    // float 0.1f as a double keeps 899 of 1000; Python's 0.1 keeps 900.
    EXPECT_EQ(evs_retained_count(10, 100, static_cast<double>(0.1f)), 899);
    EXPECT_EQ(evs_rate_from_config(0.1f), 0.1);
    EXPECT_EQ(evs_retained_count(10, 100, evs_rate_from_config(0.1f)), 900);
}

TEST(VideoEvs, FirstPairIsAlwaysKept) {
    // Pair 0 scores lowest (and one NaN) on input; evs.py overwrites it with 255.
    const std::vector<float> d = {
        0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.9f, 0.8f, 0.7f, 0.95f, 0.85f, 0.75f};
    const auto mask = evs_retention_mask(d, 3, 3, 3);
    EXPECT_EQ(mask, (std::vector<uint8_t>{1, 1, 1, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(evs_kept_indices(evs_retention_mask(d, 3, 3, 5)), (std::vector<int32_t>{0, 1, 2, 3, 6}));
}

TEST(VideoEvs, TiesKeepTheLowerIndex) {
    // Pair 1: 0.3, 0.5, 0.3, 0.3. keep 6 = pair 0 + index 5 + the first of the three 0.3 ties.
    const std::vector<float> d = {0, 0, 0, 0, 0.3f, 0.5f, 0.3f, 0.3f};
    EXPECT_EQ(evs_kept_indices(evs_retention_mask(d, 2, 4, 6)), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(evs_kept_indices(evs_retention_mask(d, 2, 4, 7)), (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}));
    // keep past the total marks everything; a size mismatch marks nothing.
    EXPECT_EQ(evs_kept_indices(evs_retention_mask(d, 2, 4, 99)).size(), 8u);
    EXPECT_EQ(evs_kept_indices(evs_retention_mask(d, 3, 4, 6)).size(), 0u);
}

// Template block [kS, kVid, kE] around text, expanded with one stamp token per pair.
std::vector<int32_t> pruned_prompt(int groups, int tpg, int keep) {
    VideoPlaceholderLayout v;
    v.tokens_per_group = tpg;
    for (int g = 0; g < groups; ++g)
        v.stamp_ids.push_back({kStamp0 + g});
    evs_prune_layout(v, keep);
    std::vector<int32_t> tokens = {kT, kS, kVid, kE, kT};
    EXPECT_TRUE(expand_video_placeholders(tokens, kVid, kS, kE, {v}).has_value());
    return tokens;
}

TEST(VideoEvs, PrunedLayoutPutsEveryKeptPadInPairZero) {
    const auto tokens = pruned_prompt(2, 4, 6);
    EXPECT_EQ(tokens, (std::vector<int32_t>{kT, kS, kStamp0, kS, kVid, kVid, kVid, kVid, kVid, kVid, kE,
                                            kStamp0 + 1, kS, kE, kE, kT}));
    // Unpruned (no group_tokens) stays the HF layout.
    VideoPlaceholderLayout v;
    v.tokens_per_group = 2;
    v.stamp_ids = {{kStamp0}, {kStamp0 + 1}};
    std::vector<int32_t> plain = {kS, kVid, kE};
    ASSERT_TRUE(expand_video_placeholders(plain, kVid, kS, kE, {v}).has_value());
    EXPECT_EQ(plain,
              (std::vector<int32_t>{kS, kStamp0, kS, kVid, kVid, kE, kStamp0 + 1, kS, kVid, kVid, kE, kE}));
    // A per-pair count list of the wrong length is refused, tokens untouched.
    v.group_tokens = {2};
    std::vector<int32_t> bad = {kS, kVid, kE};
    EXPECT_FALSE(expand_video_placeholders(bad, kVid, kS, kE, {v}).has_value());
    EXPECT_EQ(bad, (std::vector<int32_t>{kS, kVid, kE}));
}

std::vector<int32_t> axis(const MRopePositions& p, int a) {
    const size_t n = p.pos.size() / 3;
    return {p.pos.begin() + static_cast<std::ptrdiff_t>(a * n),
            p.pos.begin() + static_cast<std::ptrdiff_t>((a + 1) * n)};
}

TEST(VideoEvs, KeptTokensKeepTheirUnprunedPositions) {
    // 2 pairs of 2x2, keep 6: pair 1 scores 0.1, 0.5, 0.2, 0.4 -> kept flat {0, 1, 2, 3, 5, 7}.
    auto tokens = pruned_prompt(2, 4, 6);
    const auto kept = evs_kept_indices(
        evs_retention_mask(std::vector<float>{0, 0, 0, 0, 0.1f, 0.5f, 0.2f, 0.4f}, 2, 4, 6));
    ASSERT_EQ(kept, (std::vector<int32_t>{0, 1, 2, 3, 5, 7}));
    const auto types = fold_video_pads(tokens, kImg, kVid);
    const std::vector<VideoEvsPlan> plans = {{0, 2, 4, 6, 0.25}};
    const std::vector<std::vector<int32_t>> kept_all = {kept};
    const auto pos = qwen_evs_mrope_positions(types, tokens, kE, {{2, 2}, {2, 2}}, plans, kept_all, 0);
    ASSERT_TRUE(pos.has_value()) << pos.error();
    // Unpruned: T0 S1 st2 s3 [pair 0 at t=4] e6 st7 s8 [pair 1 at t=9] e11 E12 T13, next 14.
    // Pruned slots 8, 9 hold flat 5 (pair 1, row 0, col 1) and flat 7 (pair 1, row 1, col 1).
    EXPECT_EQ(axis(*pos, 0), (std::vector<int32_t>{0, 1, 2, 3, 4, 4, 4, 4, 9, 9, 6, 7, 8, 11, 12, 13}));
    EXPECT_EQ(axis(*pos, 1), (std::vector<int32_t>{0, 1, 2, 3, 4, 4, 5, 5, 9, 10, 6, 7, 8, 11, 12, 13}));
    EXPECT_EQ(axis(*pos, 2), (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 4, 5, 10, 10, 6, 7, 8, 11, 12, 13}));
    EXPECT_EQ(pos->next_pos, 14);  // unpruned next position: decode delta 14 - 16 = -2
}

TEST(VideoEvs, EmptyPairsAfterTheFirstStillAdvancePositions) {
    // 3 pairs of 2x2, keep one pair: pairs 1 and 2 cost their positions, no tokens.
    auto tokens = pruned_prompt(3, 4, 4);
    ASSERT_EQ(tokens.size(), 17u);
    const auto types = fold_video_pads(tokens, kImg, kVid);
    const std::vector<VideoEvsPlan> plans = {{0, 3, 4, 4, 0.75}};
    const std::vector<std::vector<int32_t>> kept = {{0, 1, 2, 3}};
    const auto pos = qwen_evs_mrope_positions(types, tokens, kE, {{2, 2}, {2, 2}, {2, 2}}, plans, kept, 0);
    ASSERT_TRUE(pos.has_value()) << pos.error();
    EXPECT_EQ(axis(*pos, 0), (std::vector<int32_t>{0, 1, 2, 3, 4, 4, 4, 4, 6, 7, 8, 11, 12, 13, 16, 17, 18}));
    EXPECT_EQ(pos->next_pos, 19);
}

TEST(VideoEvs, NoPlanIsBitIdenticalToTheUnprunedPath) {
    // An image then an unpruned video: rate 0 creates no plan, and no plan changes nothing.
    VideoPlaceholderLayout v;
    v.tokens_per_group = 6;
    v.stamp_ids = {{kStamp0}, {kStamp0 + 1}, {kStamp0 + 2}};
    std::vector<int32_t> tokens = {kT, kS, kImg, kE, kT, kS, kVid, kE, kT};
    ASSERT_TRUE(expand_image_placeholders(tokens, kImg, {12}).has_value());
    ASSERT_TRUE(expand_video_placeholders(tokens, kVid, kS, kE, {v}).has_value());
    const auto types = fold_video_pads(tokens, kImg, kVid);
    const std::vector<MRopeImageGrid> items = {{3, 4}, {2, 3}, {2, 3}, {2, 3}};
    const auto want = qwen_build_mrope_positions_items(types, items, 0);
    const auto got = qwen_evs_mrope_positions(types, tokens, kE, items, {}, {}, 0);
    ASSERT_TRUE(want.has_value() && got.has_value());
    EXPECT_EQ(got->pos, want->pos);
    EXPECT_EQ(got->next_pos, want->next_pos);
    EXPECT_EQ(evs_retained_count(3, 6, 0.0), 18);  // q = 0: keep == total, the server adds no plan
}

TEST(VideoEvs, RefusesAPlanThePromptDoesNotHold) {
    auto tokens = pruned_prompt(2, 4, 6);
    const auto types = fold_video_pads(tokens, kImg, kVid);
    const std::vector<std::vector<int32_t>> kept = {{0, 1, 2, 3, 4, 5}};
    // keep disagrees with the reserved run.
    EXPECT_FALSE(qwen_evs_mrope_positions(types, tokens, kE, {{2, 2}, {2, 2}},
                                          std::vector<VideoEvsPlan>{{0, 2, 4, 5, 0.25}}, kept, 0)
                     .has_value());
    // A pair grid that is not tokens_per_group.
    EXPECT_FALSE(qwen_evs_mrope_positions(types, tokens, kE, {{2, 2}, {1, 2}},
                                          std::vector<VideoEvsPlan>{{0, 2, 4, 6, 0.25}}, kept, 0)
                     .has_value());
    // Kept indices out of order.
    const std::vector<std::vector<int32_t>> unordered = {{0, 1, 2, 3, 5, 4}};
    EXPECT_FALSE(qwen_evs_mrope_positions(types, tokens, kE, {{2, 2}, {2, 2}},
                                          std::vector<VideoEvsPlan>{{0, 2, 4, 6, 0.25}}, unordered, 0)
                     .has_value());
    // The prompt lost pair 1's vision_end.
    std::vector<int32_t> cut(tokens.begin(), tokens.begin() + 12);
    std::vector<uint8_t> cut_types(types.begin(), types.begin() + 12);
    EXPECT_FALSE(qwen_evs_mrope_positions(cut_types, cut, kE, {{2, 2}, {2, 2}},
                                          std::vector<VideoEvsPlan>{{0, 2, 4, 6, 0.25}}, kept, 0)
                     .has_value());
}

TEST(VideoEvs, GatherRowsKeepImagesWholeAndVideosPruned) {
    // Items: image (3 rows), video of 2 pairs x 2 (rows 3..6), image (2 rows, rows 7..8).
    const std::vector<int> items = {3, 2, 2, 2};
    const std::vector<VideoEvsPlan> plans = {{1, 2, 2, 3, 0.25}};
    const std::vector<std::vector<int32_t>> kept = {{0, 1, 3}};
    ASSERT_TRUE(evs_plans_match(plans, items));
    EXPECT_EQ(evs_gather_rows(plans, items, kept), (std::vector<int32_t>{0, 1, 2, 3, 4, 6, 7, 8}));
    EXPECT_EQ(evs_gather_rows({}, items, {}), (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(VideoEvs, PlansMustMatchTheEncodedItems) {
    const std::vector<int> items = {3, 2, 2, 2};
    using P = std::vector<VideoEvsPlan>;
    EXPECT_FALSE(evs_plans_match(P{{0, 2, 2, 3, 0.25}}, items));  // pair 0 is a 3-token image
    EXPECT_FALSE(evs_plans_match(P{{1, 2, 2, 4, 0.25}}, items));  // keep == total: nothing pruned
    EXPECT_FALSE(evs_plans_match(P{{2, 3, 2, 3, 0.25}}, items));  // runs past the last item
    EXPECT_FALSE(evs_plans_match(P{{1, 2, 2, 3, 0.25}, {2, 2, 2, 3, 0.25}}, items));  // overlap
    EXPECT_TRUE(evs_plans_match(P{{1, 3, 2, 3, 0.5}}, items));
}

TEST(VideoEvs, ConfigRateOutsideZeroToOneIsRejected) {
    RuntimeConfig cfg;
    EXPECT_TRUE(cfg.apply_overrides({"runtime.video_pruning_rate=0.5"}).empty());
    EXPECT_EQ(cfg.runtime.video_pruning_rate, 0.5f);
    for (const char* bad : {"runtime.video_pruning_rate=1", "runtime.video_pruning_rate=-0.1",
                            "runtime.video_pruning_rate=nan"}) {
        EXPECT_EQ(cfg.apply_overrides({bad}).size(), 1u) << bad;
        EXPECT_EQ(cfg.runtime.video_pruning_rate, 0.5f) << bad << ": the previous value stays";
    }
}

}  // namespace
}  // namespace imp
