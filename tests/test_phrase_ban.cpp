// Banned phrases (roadmap row 118): the pure text rules the engine's rewind relies on.

#include "runtime/phrase_ban.h"
#include "runtime/phrase_regex.h"

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

TEST(PhraseBan, RegexCompileRejectsBadAndEmptyMatching) {
    std::string why;
    EXPECT_EQ(PhraseRegex::compile("(unclosed", why), nullptr);
    EXPECT_NE(why.find("does not compile"), std::string::npos) << why;
    why.clear();
    EXPECT_EQ(PhraseRegex::compile("x*", why), nullptr);
    EXPECT_EQ(why, "pattern matches the empty string");
    EXPECT_NE(PhraseRegex::compile(R"(\d{3,})", why), nullptr);
}

// A match counts once it ends in the new text; zero-length and old matches do not.
TEST(PhraseBan, RegexMatchEndsInNewTextInsideTheWindow) {
    std::string why;
    const auto re = PhraseRegex::compile(R"(\b[0-9]{3}\b)", why);
    ASSERT_NE(re, nullptr) << why;
    EXPECT_EQ(re->match_start("call 911 now", 0, 0), 5u);
    EXPECT_EQ(re->match_start("call 911 now", 0, 8), std::nullopt) << "ended at byte 8, nothing new";
    EXPECT_EQ(re->match_start("call 911 now 112", 0, 13), 13u) << "the old match is skipped";

    PhraseBanState s;
    s.phrases = {"now"};
    s.regex = {re};
    EXPECT_EQ(phrase_hit(s, "call 911 now", 0), 5u) << "the earlier start wins";
    EXPECT_EQ(phrase_hit(s, "call 911 now", 9), 9u);
    // Only the last kMaxBannedPhraseBytes bytes are searched: a match starting before them passes.
    PhraseBanState wide;
    wide.regex = {PhraseRegex::compile("1 +x", why)};
    ASSERT_NE(wide.regex[0], nullptr) << why;
    const std::string far = "911" + std::string(kMaxBannedPhraseBytes, ' ') + "x";
    EXPECT_EQ(phrase_hit(wide, far, far.size() - 1), std::nullopt);
    EXPECT_EQ(phrase_hit(wide, "911   x", 6), 2u);

    // A token boundary splits a UTF-8 character: no error, the match after it still counts.
    const auto word = PhraseRegex::compile("Stra(ss|ß)e", why);
    ASSERT_NE(word, nullptr) << why;
    const std::string split = std::string("gro") + "\xC3";  // first byte of U+00DF only
    EXPECT_EQ(word->match_start(split, 0, 0), std::nullopt);
    EXPECT_EQ(word->match_start(split + " Strasse", 0, 4), 5u);
}

}  // namespace
}  // namespace imp
