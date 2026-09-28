// Minimum prompt chunk (#2152): runtime/prompt_tail.h is header-only, so the chunk-split and
// prefix-reuse caps run in the unit lane. A prompt row below 33 rows takes decode kernels.
#include "runtime/prompt_tail.h"

#include <gtest/gtest.h>

using namespace imp;

TEST(PromptTail, IssueSplitKeeps33Rows) {
    // 350-token prompt, 336-token chunk: 14-row tail becomes 33.
    EXPECT_EQ(keep_prompt_tail(336, 350), 317);
    EXPECT_EQ(350 - keep_prompt_tail(336, 350), kMinPromptChunkRows);
}

TEST(PromptTail, LongTailUnchanged) {
    EXPECT_EQ(keep_prompt_tail(176, 350), 176);
    EXPECT_EQ(keep_prompt_tail(2048, 2048 + 33), 2048);
}

TEST(PromptTail, NoTailUnchanged) { EXPECT_EQ(keep_prompt_tail(350, 350), 350); }

TEST(PromptTail, ShorteningBelowMinimumDeclined) {
    // 40 rows left, chunk 20: shortening to 7 would break the minimum itself.
    EXPECT_EQ(keep_prompt_tail(20, 40), 20);
}

TEST(PromptTail, ReuseCapLeavesMinimumRows) {
    // 289-token resend, block 16: reuse <= 16 blocks (256 tokens), 33 rows re-prefilled.
    EXPECT_EQ(prompt_reuse_cap_blocks(289, 16), 16);
    EXPECT_GE(289 - prompt_reuse_cap_blocks(289, 16) * 16, kMinPromptChunkRows);
    EXPECT_EQ(prompt_reuse_cap_blocks(282, 16), 15);
}

TEST(PromptTail, ShortPromptNoReuse) {
    EXPECT_EQ(prompt_reuse_cap_blocks(33, 16), 0);
    EXPECT_EQ(prompt_reuse_cap_blocks(48, 16), 0);
    EXPECT_EQ(prompt_reuse_cap_blocks(49, 16), 1);
}
