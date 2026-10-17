// Banned phrases (roadmap row 118): the pure text rules the engine's rewind relies on.

#include "runtime/phrase_ban.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace imp {
namespace {

TEST(PhraseBan, OnlyMatchesThatEndInTheNewText) {
    const std::vector<std::string> ph = {"delve", "tapestry"};
    // "I will delve" with prev_len 7 ("I will "): "delve" ends in the new text.
    EXPECT_EQ(phrase_match_start("I will delve", 7, ph), 7u);
    // Same text, everything already seen: no new match.
    EXPECT_FALSE(phrase_match_start("I will delve", 12, ph).has_value());
    // A phrase split over the old/new boundary still matches ("del" old, "ve" new).
    EXPECT_EQ(phrase_match_start("I will delve", 10, ph), 7u);
    // Earliest start wins across phrases.
    EXPECT_EQ(phrase_match_start("a tapestry to delve", 9, ph), 2u);
    EXPECT_FALSE(phrase_match_start("deliver", 0, ph).has_value());
}

TEST(PhraseBan, TokenAtByteAndSafePrefix) {
    // tokens "The", " capital", " of": ends 3, 11, 14
    const std::vector<uint32_t> ends = {3, 11, 14};
    EXPECT_EQ(phrase_token_at(ends, 0), 0u);
    EXPECT_EQ(phrase_token_at(ends, 2), 0u);
    EXPECT_EQ(phrase_token_at(ends, 3), 1u);  // byte 3 is the space opening " capital"
    EXPECT_EQ(phrase_token_at(ends, 13), 2u);
    // max phrase 5 bytes: a later match starts at byte >= 14 - 4 = 10, inside token 1.
    EXPECT_EQ(phrase_safe_tokens(ends, 14, 5), 1u);
    // max phrase 1 byte: every finished token is safe.
    EXPECT_EQ(phrase_safe_tokens(ends, 14, 1), 3u);
    // text shorter than the window: nothing is safe yet.
    EXPECT_EQ(phrase_safe_tokens(ends, 14, 40), 0u);
    // A match start is never inside a safe token: the rewind cannot reach delivered text.
    const std::vector<std::string> ph = {"of"};
    const size_t safe = phrase_safe_tokens({3, 11}, 11, 2);  // before " of" arrived
    const auto hit = phrase_match_start("The capital of", 11, ph);
    ASSERT_TRUE(hit.has_value());
    EXPECT_GE(phrase_token_at(ends, *hit), safe);
}

// Hybrids: a rewind to pos restores the slab with the most tokens that still lets the next
// forward read output[consumed] (consumed <= pos - 1); periodic saves never overwrite slot 0.
TEST(PhraseBan, RecurrentSnapshotPickAndSlot) {
    EXPECT_EQ(PhraseBanState{}.snap_consumed, (std::array<int, kPhraseSnapSlots>{-1, -1, -1}));
    const std::array<int, kPhraseSnapSlots> c = {0, 32, 64};
    EXPECT_EQ(phrase_pick_snapshot(c, 70), 2);
    EXPECT_EQ(phrase_pick_snapshot(c, 65), 2);
    EXPECT_EQ(phrase_pick_snapshot(c, 64), 1);
    EXPECT_EQ(phrase_pick_snapshot(c, 33), 1);
    EXPECT_EQ(phrase_pick_snapshot(c, 32), 0);
    EXPECT_EQ(phrase_pick_snapshot(c, 1), 0);
    EXPECT_EQ(phrase_pick_snapshot({-1, -1, -1}, 5), -1);
    EXPECT_EQ(phrase_pick_snapshot({0, -1, 32}, 40), 2);
    EXPECT_EQ(phrase_snapshot_slot({0, -1, -1}), 1);
    EXPECT_EQ(phrase_snapshot_slot({0, 32, -1}), 2);
    EXPECT_EQ(phrase_snapshot_slot({0, 32, 64}), 1);
    EXPECT_EQ(phrase_snapshot_slot({0, 96, 64}), 2);
}

}  // namespace
}  // namespace imp
