// Unit tests for reconcile_thinking_with_prompt_tail (reasoning_split.h):
// the rendered chat-prompt tail is ground truth for whether generation begins
// inside an open <think> block. Regression cover for #934 — Qwen3.5-4B's
// template mentions enable_thinking but defaults to a *closed* empty block, so
// the heuristic's ON default must be overridden to OFF or the whole answer is
// trapped in reasoning_content.

#include <gtest/gtest.h>

#include "reasoning_split.h"

using imp::server::reconcile_thinking_with_prompt_tail;

// current, explicit_set, tail_has_think, tail_has_close
namespace {
constexpr bool kNotExplicit = false;
constexpr bool kExplicit = true;
}  // namespace

TEST(ThinkingReconcile, ClosedBlockDefaultsOff) {
    // #934: heuristic defaulted ON, template rendered a pre-closed block.
    EXPECT_FALSE(reconcile_thinking_with_prompt_tail(/*current=*/true, kNotExplicit,
                                                     /*think=*/true, /*close=*/true));
}

TEST(ThinkingReconcile, OpenPrefixTurnsOn) {
    // Qwen3.6-style open <think>\n prefix: model is mid-reasoning.
    EXPECT_TRUE(reconcile_thinking_with_prompt_tail(/*current=*/false, kNotExplicit,
                                                    /*think=*/true, /*close=*/false));
}

TEST(ThinkingReconcile, OpenPrefixWinsEvenWhenExplicit) {
    // An open block is unambiguous ground truth regardless of the request flag.
    EXPECT_TRUE(reconcile_thinking_with_prompt_tail(/*current=*/false, kExplicit,
                                                    /*think=*/true, /*close=*/false));
}

TEST(ThinkingReconcile, NoThinkKeepsCurrent) {
    // Templateless / force-append path: no <think> in the tail — leave the
    // upstream decision (heuristic or explicit) untouched, both directions.
    EXPECT_TRUE(reconcile_thinking_with_prompt_tail(/*current=*/true, kNotExplicit,
                                                    /*think=*/false, /*close=*/false));
    EXPECT_FALSE(reconcile_thinking_with_prompt_tail(/*current=*/false, kNotExplicit,
                                                     /*think=*/false, /*close=*/false));
}

TEST(ThinkingReconcile, ExplicitRequestNotDowngradedOnClosedBlock) {
    // A caller that explicitly asked for thinking keeps their choice; a
    // closed-block template cannot honor it, but we do not silently flip it.
    EXPECT_TRUE(reconcile_thinking_with_prompt_tail(/*current=*/true, kExplicit,
                                                    /*think=*/true, /*close=*/true));
}

TEST(ThinkingReconcile, StrayCloseWithoutOpenKeepsCurrent) {
    // </think> with no <think> in-window is not a closed block — keep current.
    EXPECT_TRUE(reconcile_thinking_with_prompt_tail(/*current=*/true, kNotExplicit,
                                                    /*think=*/false, /*close=*/true));
}
