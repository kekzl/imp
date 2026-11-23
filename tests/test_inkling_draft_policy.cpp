// Inkling2 DSpark draft: the row choice of exec/inkling_draft_policy.h.

#include "exec/inkling_draft_policy.h"

#include <gtest/gtest.h>

namespace imp {

TEST(InklingDraftPolicy, RowsFollowTheMeasuredChanceNotTheStatedOne) {
    const int32_t tok[7] = {5, 6, 7, 8, 9, 10, 11};
    // A sure head keeps the block: 0.97 * 0.96 * 0.97 * 0.96^4 = 0.77.
    const float sure[7] = {0.9995f, 0.9995f, 0.9995f, 0.9995f, 0.9995f, 0.9995f, 0.9995f};
    EXPECT_EQ(inkling_draft_rows(tok, sure, 7, 100), 7);
    // A stated 0.6 holds 37 % at row 0 and 28 % at row 1: 0.37 stays, 0.37 * 0.28 = 0.10 goes. The
    // stated product (0.36) would keep two rows.
    const float half_sure[7] = {0.6f, 0.6f, 0.9995f, 0.9995f, 0.9995f, 0.9995f, 0.9995f};
    EXPECT_EQ(inkling_draft_rows(tok, half_sure, 7, 100), 1);
    // A stated 0.4 holds 21 %: below kDraftKeep, no row.
    const float weak[7] = {0.4f, 0.9995f, 0.9995f, 0.9995f, 0.9995f, 0.9995f, 0.9995f};
    EXPECT_EQ(inkling_draft_rows(tok, weak, 7, 100), 0);
    // 0.97 * 0.86 * 0.86 = 0.72, then row 3 at a stated 0.75 holds 28 %: 0.20, three rows.
    const float fading[7] = {0.9995f, 0.997f, 0.997f, 0.75f, 0.9995f, 0.9995f, 0.9995f};
    EXPECT_EQ(inkling_draft_rows(tok, fading, 7, 100), 3);
}

TEST(InklingDraftPolicy, ATokenOutsideTheVocabularyEndsTheDraft) {
    const float sure[4] = {0.9995f, 0.9995f, 0.9995f, 0.9995f};
    const int32_t past[4] = {5, 6, 100, 7};
    EXPECT_EQ(inkling_draft_rows(past, sure, 4, 100), 2);
    const int32_t none[4] = {-1, 6, 7, 8};
    EXPECT_EQ(inkling_draft_rows(none, sure, 4, 100), 0);
}

TEST(InklingDraftPolicy, TheChanceRisesWithTheStatedConfidence) {
    for (int row = 0; row < 4; ++row)
        for (int b = 1; b < kDraftConfBuckets; ++b)
            EXPECT_GE(kDraftHit[row][b], kDraftHit[row][b - 1]) << row << " " << b;
    EXPECT_FLOAT_EQ(inkling_draft_hit(0, 0.0f), 0.09f);
    EXPECT_FLOAT_EQ(inkling_draft_hit(0, 0.3f), 0.21f);
    EXPECT_FLOAT_EQ(inkling_draft_hit(9, 1.0f), 0.96f);
}

}  // namespace imp
