// Unit tests for reconcile_thinking_with_prompt_tail (reasoning_split.h):
// the rendered chat-prompt tail is ground truth for whether generation begins
// inside an open <think> block. Regression cover for #934 — Qwen3.5-4B's
// template mentions enable_thinking but defaults to a *closed* empty block, so
// the heuristic's ON default must be overridden to OFF or the whole answer is
// trapped in reasoning_content.

#include <gtest/gtest.h>

#include "reasoning_split.h"
#include "utils.h"  // extract_reasoning, strip_think_block

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

// ---------------------------------------------------------------------------
// strip_think_block edge cases exercised by the shared split_last_think helper.
// extract_reasoning + the strip_think_block happy path are already covered in
// test_sse_stream_utils.cpp (issue #557, which also uses extract_reasoning as
// the oracle for the streaming splitter). These pin the strip-only edge paths
// that the shared-helper refactor restructured — an unclosed trailing/leading
// <think> is discarded (the model never finished the block).
// ---------------------------------------------------------------------------

TEST(StripThinkBlock, DiscardsUnclosedTrailingThink) {
    std::string text = "<think>reasoning</think><think>unclosed";
    strip_think_block(text);
    EXPECT_EQ(text, "");
}

TEST(StripThinkBlock, EmptyContentAfterCloseClears) {
    std::string text = "<think>a</think>";
    strip_think_block(text);
    EXPECT_EQ(text, "");
}

TEST(StripThinkBlock, UnclosedLeadingThinkClears) {
    std::string text = "<think>never finished thinking";
    strip_think_block(text);
    EXPECT_EQ(text, "");
}

// ---------------------------------------------------------------------------
// Structured output vs thinking (#1431)
// ---------------------------------------------------------------------------

using imp::server::should_stamp_thinking_off;
using imp::server::structured_output_excludes_thinking;

// json_schema was the entry MISSING from this list, and its absence returned
// empty content on any model whose </think> is a multi-token BPE sequence: the
// gate held the mask open for a reasoning block that never closed in the text
// the splitter reads. Each flag is pinned separately so a future edit cannot
// drop one silently.
TEST(StructuredOutputThinking, EveryWholeReplyConstraintExcludesThinking) {
    EXPECT_TRUE(structured_output_excludes_thinking(true, false, false, false, false));   // json_mode
    EXPECT_TRUE(structured_output_excludes_thinking(false, true, false, false, false));   // tools
    EXPECT_TRUE(structured_output_excludes_thinking(false, false, true, false, false));   // json_schema
    EXPECT_TRUE(structured_output_excludes_thinking(false, false, false, true, false));   // regex
    EXPECT_TRUE(structured_output_excludes_thinking(false, false, false, false, true));   // grammar
}

TEST(StructuredOutputThinking, AnUnconstrainedRequestMayStillThink) {
    EXPECT_FALSE(structured_output_excludes_thinking(false, false, false, false, false));
}

// Not stamping is not the same as stamping false: unstamped, a template uses
// its OWN default, and Qwen3.8's is an open <think>. So a request that does not
// want thinking must stamp it off even when the budget is non-zero, which is
// the second half of #1431.
TEST(StructuredOutputThinking, StampsOffWhenThinkingIsUnwantedNotOnlyOnZeroBudget) {
    EXPECT_TRUE(should_stamp_thinking_off(/*is_think_model=*/true, /*enable_thinking=*/false,
                                          /*budget_disabled=*/false, /*want_thinking=*/false));
    // The pre-existing reason still stamps.
    EXPECT_TRUE(should_stamp_thinking_off(true, false, /*budget_disabled=*/true, /*want=*/true));
}

TEST(StructuredOutputThinking, NeverStampsWhenThinkingIsActuallyOn) {
    // enable_thinking true means the prompt carries an open block on purpose;
    // stamping it off here would contradict the render.
    EXPECT_FALSE(should_stamp_thinking_off(true, /*enable_thinking=*/true, false, true));
    EXPECT_FALSE(should_stamp_thinking_off(true, /*enable_thinking=*/true, true, false));
    // Not a reasoning model at all: nothing to say either way.
    EXPECT_FALSE(should_stamp_thinking_off(/*is_think_model=*/false, false, true, false));
}
