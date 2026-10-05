// Jump-ahead drafts from the request's own tokenization (roadmap row 46).
#include "runtime/jump_draft.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

// Toy vocabulary: id -> piece.
const std::vector<std::string> kVocab = {"{\"", "desc",  "ription", "_of",  "_",
                                         "the", "\":\"", "x",       "\"},", "_the"};

std::string piece(int32_t id) { return kVocab[static_cast<size_t>(id)]; }

}  // namespace

TEST(JumpDraft, ReusesTheModelsOwnSplitOfARepeatedSpan) {
    // Earlier output: {"description_of_the":"x"},  spelled with "_" + "the", not "_the".
    const std::vector<int32_t> out = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const auto d = imp::jump_draft_from_history(out, piece, "{\"description_of_the\":\"", 1024, 3);
    EXPECT_EQ(d, (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}));
}

TEST(JumpDraft, PartialCoverIsAPrefixAndShortCoversAreRefused) {
    const std::vector<int32_t> out = {0, 1, 2, 7};
    // Covers "{\"description" of a longer span: 3 tokens, a prefix.
    EXPECT_EQ(imp::jump_draft_from_history(out, piece, "{\"description_of", 1024, 3),
              (std::vector<int32_t>{0, 1, 2}));
    EXPECT_TRUE(imp::jump_draft_from_history(out, piece, "{\"description_of", 1024, 4).empty());
    EXPECT_TRUE(imp::jump_draft_from_history(out, piece, "zzz", 1024, 1).empty());
}

TEST(JumpDraft, LongestCoverWinsOverTheMostRecentShortOne) {
    // Recent "{\"desc" + "x" covers 2 tokens; the earlier run covers the whole span.
    const std::vector<int32_t> out = {0, 1, 2, 3, 7, 0, 1, 7};
    EXPECT_EQ(imp::jump_draft_from_history(out, piece, "{\"description_of", 1024, 2),
              (std::vector<int32_t>{0, 1, 2, 3}));
    // Outside the window the full cover is not seen.
    EXPECT_EQ(imp::jump_draft_from_history(out, piece, "{\"description_of", 3, 2),
              (std::vector<int32_t>{0, 1}));
}
