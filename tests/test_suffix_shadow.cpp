// SuffixShadow (diagnostics.spec_trace): a suffix draft scored against the tokens the request emitted
// afterwards, one line per request step, and the match length SuffixDraftIndex::draft reports.

#include "runtime/suffix_draft.h"
#include "runtime/suffix_shadow.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using imp::SuffixDraftIndex;
using imp::SuffixShadow;

TEST(SuffixShadow, LineWaitsForTheOutcome) {
    SuffixShadow s;
    std::vector<int32_t> out = {1, 2};
    s.drafted(7, 2, {10, 11, 12, 13}, 6);
    s.stepped(7, /*n=*/4, /*rows=*/4, /*nd=*/3, /*matched=*/2, /*emitted=*/3);
    out.insert(out.end(), {10, 11});  // 2 of 4 draft tokens emitted, both agree: undecided
    EXPECT_TRUE(s.settle(7, out, /*last=*/false).empty());
    out.push_back(99);  // the third differs
    const auto lines = s.settle(7, out, /*last=*/false);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], "spec-sfx-shadow id=7 pos=2 n=4 T=4 mtp=3:2:3 sfx=4:6:2");
    EXPECT_TRUE(s.settle(7, out, /*last=*/true).empty());  // a settled record leaves
}

TEST(SuffixShadow, FullAgreementAndNoDraft) {
    SuffixShadow s;
    std::vector<int32_t> out = {5};
    s.drafted(1, 1, {6, 7}, 12);
    s.stepped(1, 1, 5, 4, 4, 5);
    out.insert(out.end(), {6, 7, 8, 9, 3});
    s.drafted(1, 6, {}, 9);  // no suffix match at this step: the match length reads 0
    s.stepped(1, 1, 5, 4, 0, 1);
    const auto lines = s.settle(1, out, /*last=*/false);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0], "spec-sfx-shadow id=1 pos=1 n=1 T=5 mtp=4:4:5 sfx=2:12:2");
    EXPECT_EQ(lines[1], "spec-sfx-shadow id=1 pos=6 n=1 T=5 mtp=4:0:1 sfx=0:0:0");
}

TEST(SuffixShadow, LastStepSettlesAnOpenDraft) {
    SuffixShadow s;
    std::vector<int32_t> out = {1, 2, 3};
    s.drafted(3, 2, {3, 4, 5}, 8);  // the request ends 1 token into the draft
    s.stepped(3, 2, 4, 3, 1, 1);
    EXPECT_TRUE(s.settle(3, out, /*last=*/false).empty());
    const auto lines = s.settle(3, out, /*last=*/true);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], "spec-sfx-shadow id=3 pos=2 n=2 T=4 mtp=3:1:1 sfx=3:8:1");
}

TEST(SuffixShadow, DeclinedStepIsReplaced) {
    SuffixShadow s;
    std::vector<int32_t> out = {1};
    s.drafted(2, 1, {50, 51}, 6);                           // the step was declined: no stepped()
    out.push_back(8);                                       // a plain step emitted this token
    EXPECT_TRUE(s.settle(2, out, /*last=*/false).empty());  // decided, but no step ran: no line
    s.drafted(2, 2, {9}, 7);
    s.stepped(2, 1, 5, 4, 1, 2);
    out.insert(out.end(), {9, 4});
    const auto lines = s.settle(2, out, /*last=*/true);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], "spec-sfx-shadow id=2 pos=2 n=1 T=5 mtp=4:1:2 sfx=1:7:1");
}

TEST(SuffixShadow, IndexReportsTheMatchLength) {
    SuffixDraftIndex idx(/*min_match=*/3, /*max_match=*/8);
    const std::vector<int32_t> h = {1, 2, 3, 4, 5, 6, 77, 9, 2, 3, 4, 5};  // 4 tokens recur, then 1 != 9
    idx.append(h);
    int match = -1;
    const auto d = idx.draft(2, 2, &match);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d.tokens[0], 6);
    EXPECT_EQ(match, 4);
    SuffixDraftIndex none(3, 8);
    none.append(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7});
    match = -1;
    EXPECT_TRUE(none.draft(2, 2, &match).empty());
    EXPECT_EQ(match, 0);
}
