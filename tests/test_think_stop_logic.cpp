// CPU unit tests for the think/stop state machine (Test-Audit Phase 2, Risk #4).
//
// These exercise the *pure* host logic extracted into runtime/think_stop_logic.h
// from engine_sampling_stop.cpp + engine_workspace_warmup.cpp. The week these
// were written, three stacked bugs lived in this exact code with zero coverage:
//   (a) budget recount ignored prompt-injected <think> prefixes (no opener in
//       output) -> seeded by started_in_think,
//   (b) think-id cache required CONTROL token type, but Qwen3-GGUF tags
//       <think> USER_DEFINED -> accept any special (non-NORMAL) type,
//   (c) </think> split across BPE tokens was never detected for SafeTensors
//       quants that ship the marker as special=False text.
//
// Ground truth here is derived by hand from the documented semantics and stated
// as a comment per case — no snapshot of the code's own output.

#include "runtime/think_stop_logic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace imp::think_logic;

// ---------------------------------------------------------------------------
// Warmup: think-token-type acceptance (bug (b))
// ---------------------------------------------------------------------------
// Token-type codes (tokenizer.h): NORMAL=1, CONTROL=3, USER_DEFINED=4.
// is_special_token() returns (type != NORMAL). The engine passes
// `has_token_types() && is_special_token(ts)` as `is_special`.

TEST(ThinkTokenAccept, ControlTypeAccepted) {
    // GGUF metadata path: <think> tagged CONTROL -> is_special=true -> accept.
    EXPECT_TRUE(accept_think_token(/*start_id=*/151648, /*has_token_types=*/true,
                                   /*is_special=*/true, /*is_added=*/false, /*vocab=*/152064));
}

TEST(ThinkTokenAccept, UserDefinedTypeAccepted) {
    // THE Qwen3-GGUF case: <think> tagged USER_DEFINED (type 4). is_special is
    // still true (4 != NORMAL). The old code required CONTROL and rejected this,
    // leaving think_end_id_ == -1 so the budget never fired. Must accept now.
    EXPECT_TRUE(accept_think_token(/*start_id=*/151648, /*has_token_types=*/true,
                                   /*is_special=*/true, /*is_added=*/false, /*vocab=*/152064));
}

TEST(ThinkTokenAccept, AddedButNotSpecialAccepted) {
    // THE Qwen3/Qwen3.x NVFP4 SafeTensors case: </think> is added_tokens id
    // 151668 with special=false -> is_special=false but is_added=true. The old
    // is_special-only gate rejected it (think_end_id_ == -1), forcing every
    // think chat onto the eager decode path. An explicit added marker must be
    // accepted so the conditional-graph loop runs.
    EXPECT_TRUE(accept_think_token(/*start_id=*/151667, /*has_token_types=*/true,
                                   /*is_special=*/false, /*is_added=*/true, /*vocab=*/151936));
}

TEST(ThinkTokenAccept, NormalTypeRejected) {
    // Nemotron: "<think>" is plain text at ID 12, type NORMAL, NOT in
    // added_tokens -> is_special=false, is_added=false. Must NOT be treated as a
    // think marker (else every literal "<think>" in ordinary text would toggle
    // reasoning mode).
    EXPECT_FALSE(accept_think_token(/*start_id=*/12, /*has_token_types=*/true,
                                    /*is_special=*/false, /*is_added=*/false, /*vocab=*/256000));
}

TEST(ThinkTokenAccept, AbsentTokenRejected) {
    // find_token("<think>") == -1: the model has no <think> token at all.
    EXPECT_FALSE(accept_think_token(/*start_id=*/-1, /*has_token_types=*/true,
                                    /*is_special=*/false, /*is_added=*/false, /*vocab=*/152064));
}

TEST(ThinkTokenAccept, NoTypeTableHeuristicTopOfVocab) {
    // Legacy GGUF without a token-type table: accept iff the id sits in the top
    // 1% of the vocab range (added/special tokens cluster there).
    // vocab=1000 -> threshold = 1000*99/100 = 990; accept strictly above 990.
    EXPECT_TRUE(accept_think_token(/*start_id=*/995, /*has_token_types=*/false,
                                   /*is_special=*/false, /*is_added=*/false, /*vocab=*/1000));
    EXPECT_FALSE(accept_think_token(/*start_id=*/990, /*has_token_types=*/false,
                                    /*is_special=*/false, /*is_added=*/false, /*vocab=*/1000));
    EXPECT_FALSE(accept_think_token(/*start_id=*/5, /*has_token_types=*/false,
                                    /*is_special=*/false, /*is_added=*/false, /*vocab=*/1000));
}

// ---------------------------------------------------------------------------
// Budget recount + force-token (bug (a))
// ---------------------------------------------------------------------------
// think_start_id=100, think_end_id=200 in these fixtures. Reasoning tokens are
// the ones emitted while "currently_thinking" is true; the opener/closer ids
// themselves are NOT counted.

// ---------------------------------------------------------------------------
// Prompt-tail seed (the flag pair Engine::add_request derives from the prompt)
// ---------------------------------------------------------------------------
// The defect this covers: add_request set only `in_think_block` and never
// `started_in_think`, so on every model whose template ends the generation
// prompt with "<think>\n" (Qwen3/3.5/3.6/3.8, DeepSeek-R1) the recount below
// started outside think, counted 0, should_force_think_end never fired, and the
// request spent all of max_tokens reasoning. Both flags come from one tail, so
// they are derived in one place and cannot drift apart again.

TEST(ThinkSeedFromPromptTail, TrailingOpenerSeedsBothFlags) {
    // The Qwen3.x generation prompt: "...<|im_start|>assistant\n<think>\n".
    ThinkSeed s = seed_from_prompt_tail("<|im_start|>assistant\n<think>\n");
    EXPECT_TRUE(s.in_think);
    EXPECT_TRUE(s.started_in_think) << "the budget recount never starts in-think";
}

TEST(ThinkSeedFromPromptTail, ClosedBlockSeedsNeither) {
    // A template that renders a pre-closed think block (the agent-scan family):
    // generation starts OUTSIDE think, so neither flag may be set.
    ThinkSeed s = seed_from_prompt_tail("<think>\n\n</think>\n\n");
    EXPECT_FALSE(s.in_think);
    EXPECT_FALSE(s.started_in_think);
}

TEST(ThinkSeedFromPromptTail, ReopenedAfterCloseSeedsBothFlags) {
    // A prior turn's block closed, this turn's opener follows: the LAST marker
    // wins. "</think>" contains "think>", so the precedence test must skip the
    // "</" the closer contributes (open_pos > close_pos + 1).
    ThinkSeed s = seed_from_prompt_tail("</think>\nanswer\n<|im_start|>assistant\n<think>\n");
    EXPECT_TRUE(s.in_think);
    EXPECT_TRUE(s.started_in_think);
}

TEST(ThinkSeedFromPromptTail, NoMarkerSeedsNeither) {
    ThinkSeed s = seed_from_prompt_tail("<|im_start|>assistant\n");
    EXPECT_FALSE(s.in_think);
    EXPECT_FALSE(s.started_in_think);
}

TEST(BudgetRecount, OpenerInOutputCountsBetweenMarkers) {
    // Output: [open, r, r, r, close, c]. With started_in_think=false the count
    // is the 3 tokens strictly between open and close. Ends not thinking.
    std::vector<int32_t> out = {100, 1, 2, 3, 200, 4};
    bool thinking = true;  // sentinel, must be overwritten
    int n = count_reasoning_tokens(out, 100, 200, /*started_in_think=*/false, thinking);
    EXPECT_EQ(n, 3);
    EXPECT_FALSE(thinking);
}

TEST(BudgetRecount, OpenerOnlyInPromptSeedsInThink) {
    // Prompt-injected "<think>\n" prefix: the opener is in the PROMPT, so the
    // output has NO opener. Every output token before any close is reasoning.
    // started_in_think=true makes the recount start in-think; here 4 reasoning
    // tokens and the model is still thinking at the end.
    std::vector<int32_t> out = {7, 8, 9, 10};
    bool thinking = false;
    int n = count_reasoning_tokens(out, 100, 200, /*started_in_think=*/true, thinking);
    EXPECT_EQ(n, 4);
    EXPECT_TRUE(thinking);
}

TEST(BudgetRecount, OpenerOnlyInPromptWithoutSeedCountsZero) {
    // Same output, but started_in_think=false (the pre-fix behaviour): with no
    // opener in the output the recount never enters the thinking state, so it
    // counts 0 and the budget can never fire. This is exactly bug (a).
    std::vector<int32_t> out = {7, 8, 9, 10};
    bool thinking = true;
    int n = count_reasoning_tokens(out, 100, 200, /*started_in_think=*/false, thinking);
    EXPECT_EQ(n, 0);
    EXPECT_FALSE(thinking);
}

TEST(BudgetRecount, NoThinkAtAllCountsZero) {
    std::vector<int32_t> out = {4, 5, 6};
    bool thinking = true;
    int n = count_reasoning_tokens(out, 100, 200, /*started_in_think=*/false, thinking);
    EXPECT_EQ(n, 0);
    EXPECT_FALSE(thinking);
}

TEST(ForceThinkEnd, FiresWhenBudgetReachedWhileThinking) {
    // max_tokens=10, budget=0.5 -> limit = int(10*0.5) = 5. Prompt-injected
    // prefix (started_in_think=true), 5 reasoning tokens emitted, still thinking
    // -> force </think>.
    std::vector<int32_t> out = {1, 2, 3, 4, 5};
    EXPECT_TRUE(should_force_think_end(/*budget=*/0.5f, /*think_end_id=*/200,
                                       /*max_tokens=*/10, out, /*think_start_id=*/100,
                                       /*started_in_think=*/true));
}

TEST(ForceThinkEnd, DoesNotFireBelowBudget) {
    // 4 reasoning tokens < limit 5 -> no force.
    std::vector<int32_t> out = {1, 2, 3, 4};
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 10, out, 100, /*started_in_think=*/true));
}

TEST(ForceThinkEnd, DoesNotFireAfterThinkClosed) {
    // Even with many tokens, once </think> (id 200) has been emitted the model
    // is no longer thinking, so the budget must not force another close.
    // out: open + 8 reasoning + close -> ends not thinking. limit=int(10*.5)=5,
    // but currently_thinking=false suppresses the force.
    std::vector<int32_t> out = {100, 1, 2, 3, 4, 5, 6, 7, 8, 200};
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 10, out, 100, /*started_in_think=*/false));
}

TEST(ForceThinkEnd, DisabledWhenBudgetZeroOrNoCloseId) {
    std::vector<int32_t> out = {1, 2, 3, 4, 5, 6, 7, 8};
    // budget 0 disables the whole mechanism.
    EXPECT_FALSE(should_force_think_end(0.0f, 200, 10, out, 100, true));
    // No </think> id known (-1) -> cannot force a token that doesn't exist.
    EXPECT_FALSE(should_force_think_end(0.5f, -1, 10, out, 100, true));
    // Empty output -> nothing to count.
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 10, {}, 100, true));
}

// --- Answer-reserve cap (kMaxAnswerReserve) -------------------------------
// For generous max_tokens the fractional budget over-reserves; the cap lets the
// model think up to max_tokens - kMaxAnswerReserve so it is not cut off
// mid-thought (the reasoning-into-content leak). The cap only ever GROWS the
// think limit, so small-max_tokens behaviour is unchanged.

TEST(ForceThinkEnd, ReserveCapGrantsMoreThinkingForLargeMaxTokens) {
    // max_tokens=1024, budget=0.5 -> frac_limit=512, reserve_limit=1024-256=768.
    // think_limit = max(512, 768) = 768. The model has thought 600 tokens (above
    // the old 512 limit, below 768) and is still thinking -> must NOT be forced.
    std::vector<int32_t> out(600, /*non-marker token*/ 1);
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 1024, out, 100, /*started_in_think=*/true));

    // At 800 reasoning tokens it crosses 768 -> force.
    std::vector<int32_t> out2(800, 1);
    EXPECT_TRUE(should_force_think_end(0.5f, 200, 1024, out2, 100, /*started_in_think=*/true));
}

TEST(ForceThinkEnd, ReserveCapDoesNotChangeSmallMaxTokens) {
    // max_tokens=200, budget=0.5 -> frac_limit=100, reserve_limit=200-256=-56.
    // think_limit = max(100, -56) = 100 (unchanged from the pure-fraction rule).
    std::vector<int32_t> at_limit(100, 1);
    EXPECT_TRUE(should_force_think_end(0.5f, 200, 200, at_limit, 100, /*started_in_think=*/true));
    std::vector<int32_t> below(99, 1);
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 200, below, 100, /*started_in_think=*/true));
}

// --- The reserve scales above the flat floor (#1248) -----------------------
//
// A FLAT reserve makes the answer length independent of max_tokens: think_limit
// takes the LATER of the two limits, so past 2*kMaxAnswerReserve the reserve
// always wins and the answer is pinned at 256 tokens whatever the caller asks
// for. Measured on Qwen3.6-35B-A3B-NVFP4, same request, only max_tokens varied:
// 600/1500/3000/4096 returned 935/1084/968/934 characters — five times the
// budget for fifty more characters, and never finish_reason "stop".

TEST(ForceThinkEnd, AnswerRoomGrowsWithMaxTokens) {
    // The property the flat cap broke: a bigger budget must buy answer room.
    auto answer_tokens = [](int max_tokens) {
        int frac = static_cast<int>(max_tokens * 0.5f);
        int reserve = max_tokens - answer_reserve_for(max_tokens);
        return max_tokens - (frac > reserve ? frac : reserve);
    };
    EXPECT_GT(answer_tokens(3000), answer_tokens(1500)) << "5x the budget bought no answer room";
    EXPECT_GT(answer_tokens(4096), answer_tokens(3000));
    EXPECT_EQ(answer_tokens(4096), 1024);
    EXPECT_EQ(answer_tokens(3000), 750);
}

TEST(ForceThinkEnd, ScaledReserveLeavesSmallBudgetsAlone) {
    // Everything at or below 4*kMaxAnswerReserve keeps the flat floor — that is
    // the range the reasoning-into-content leak was reported in, and the range
    // the two tests above pin.
    EXPECT_EQ(answer_reserve_for(200), kMaxAnswerReserve);
    EXPECT_EQ(answer_reserve_for(600), kMaxAnswerReserve);
    EXPECT_EQ(answer_reserve_for(1024), kMaxAnswerReserve);
    // 1024 is the boundary: one token more and the scaled reserve takes over.
    EXPECT_EQ(answer_reserve_for(1028), 257);
}

TEST(ForceThinkEnd, ScaledReserveStillForcesTheCloseInTime) {
    // The forced close must still fire while the answer room is intact:
    // max_tokens=4096 -> reserve 1024, think_limit = max(2048, 3072) = 3072.
    std::vector<int32_t> below(3000, 1);
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 4096, below, 100, /*started_in_think=*/true));
    std::vector<int32_t> at(3072, 1);
    EXPECT_TRUE(should_force_think_end(0.5f, 200, 4096, at, 100, /*started_in_think=*/true));
}

// ---------------------------------------------------------------------------
// The reserve is a config key (runtime.think_answer_reserve), not a constant
// ---------------------------------------------------------------------------

TEST(AnswerReserveKey, DefaultMatchesTheOldConstant) {
    // The key's default is the compile-time value every call site used before.
    EXPECT_EQ(answer_reserve_for(600), answer_reserve_for(600, kMaxAnswerReserve));
    EXPECT_EQ(answer_reserve_for(600, 256), 256);
}

TEST(AnswerReserveKey, RaisedFloorForcesTheCloseEarlier) {
    // reserve 1024 at max_tokens 2000: think_limit = max(1000, 976) = 1000.
    // reserve 1400: think_limit = max(1000, 600) = 1000 (the fraction wins).
    // The reserve only ever moves the limit DOWN to the fraction, never below.
    EXPECT_EQ(answer_reserve_for(2000, 1024), 1024);
    std::vector<int32_t> at(1000, 1);
    EXPECT_TRUE(should_force_think_end(0.5f, 200, 2000, at, 100, true, 1024));
    std::vector<int32_t> below(999, 1);
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 2000, below, 100, true, 1024));
}

TEST(AnswerReserveKey, ZeroReserveLeavesTheFractionInCharge) {
    // reserve 0 at max_tokens 260, budget 0.5: the flat floor is gone, only
    // max_tokens/4 = 65 is left, so limit = max(130, 260 - 65 = 195) = 195.
    EXPECT_EQ(answer_reserve_for(260, 0), 65);  // max_tokens/4 still applies
    std::vector<int32_t> at_130(130, 1);
    EXPECT_FALSE(should_force_think_end(0.5f, 200, 260, at_130, 100, true, 0));
    // ...and with the 256 default it does: max(130, 260 - 256 = 4) = 130.
    EXPECT_TRUE(should_force_think_end(0.5f, 200, 260, at_130, 100, true, 256));
}

TEST(AnswerReserveKey, NegativeReserveClampsToZero) {
    // A corrupted config must not hand the model more than max_tokens of room.
    EXPECT_EQ(answer_reserve_for(600, -1), answer_reserve_for(600, 0));
    EXPECT_EQ(answer_reserve_for(600, -100000), 150);
}

// ---------------------------------------------------------------------------
// One limit for every enforcement path
// ---------------------------------------------------------------------------
// Three paths enforce the think budget: the eager sampler
// (engine_sampling_stop.cpp), the CUDA-graph loop's device counter
// (engine_graph_decode.cpp, twice) and the n-gram/scheduler gates. The graph
// loop used to compute int(max_tokens * budget) ALONE - no answer reserve - so
// it cut thinking earlier than the documented rule and ignored
// runtime.think_answer_reserve entirely. think_limit() is now the single
// formula; these pin that the two readings agree.

TEST(ThinkLimit, GraphPathAndHostPathAgree) {
    // The graph path's old formula, kept here as the thing that must NOT come
    // back: it is the fraction with no reserve term.
    auto fraction_only = [](int max_tokens, float budget) { return static_cast<int>(max_tokens * budget); };
    for (int max_tokens : {64, 150, 260, 4096}) {
        const int limit = think_limit(max_tokens, 0.5f, kMaxAnswerReserve);
        // The host rule, recomputed by hand from the documented formula.
        const int expect = std::max(static_cast<int>(max_tokens * 0.5f),
                                    max_tokens - std::max(256, max_tokens / 4));
        EXPECT_EQ(limit, expect) << "max_tokens=" << max_tokens;
        // ...and it is never SMALLER than the fraction, i.e. the reserve only
        // ever grants thinking room, never takes it away.
        EXPECT_GE(limit, fraction_only(max_tokens, 0.5f)) << "max_tokens=" << max_tokens;
    }
    // 4096 is where the two readings actually differed: fraction 2048 against
    // the reserve rule's 3072, so the graph loop cut a third of the thinking.
    EXPECT_EQ(think_limit(4096, 0.5f), 3072);
    EXPECT_EQ(fraction_only(4096, 0.5f), 2048);
}

TEST(ThinkLimit, IsTheCountShouldForceThinkEndFiresAt) {
    // The limit is not a separate number: should_force_think_end fires exactly
    // when the reasoning count reaches it, so a device-side counter armed with
    // think_limit() forces </think> on the same token as the host.
    for (int max_tokens : {64, 150, 260, 4096}) {
        const int limit = think_limit(max_tokens, 0.5f);
        std::vector<int32_t> below(limit - 1, 1);
        std::vector<int32_t> at(limit, 1);
        EXPECT_FALSE(should_force_think_end(0.5f, 200, max_tokens, below, 100, true))
            << "max_tokens=" << max_tokens;
        EXPECT_TRUE(should_force_think_end(0.5f, 200, max_tokens, at, 100, true))
            << "max_tokens=" << max_tokens;
    }
}

TEST(ThinkLimit, FollowsTheConfiguredReserve) {
    // The knob reaches the shared formula, so it reaches every path that uses it.
    EXPECT_EQ(think_limit(2000, 0.5f, /*reserve=*/1024), 1000);  // fraction wins
    EXPECT_EQ(think_limit(2000, 0.1f, /*reserve=*/1024), 976);   // reserve wins
    EXPECT_EQ(think_limit(2000, 0.1f, /*reserve=*/0), 1500);     // no flat floor
}

// ---------------------------------------------------------------------------
// Text-tail </think> detection across token boundaries (bug (c))
// ---------------------------------------------------------------------------

TEST(TextThink, EntersThenExitsAcrossSplitTokens) {
    // SafeTensors quants emit </think> as ['</','think','>']. The single-id
    // compare never sees it; the sliding-window text match must.
    TextThinkState s;
    // Start inside a think block (chat-template primed <think>\n).
    s.in_think_block = true;
    EXPECT_FALSE(s.feed_piece("</"));     // partial, no match yet
    EXPECT_FALSE(s.feed_piece("think"));  // still partial ("</think")
    EXPECT_TRUE(s.feed_piece(">"));       // completes "</think>" -> exit fires
    EXPECT_FALSE(s.in_think_block);
}

TEST(TextThink, ExitMidDeltaSingledPiece) {
    // The whole "</think>" arrives in one decoded piece (e.g. surrounded by
    // other chars). Exit fires on that feed.
    TextThinkState s;
    s.in_think_block = true;
    EXPECT_TRUE(s.feed_piece("reasoning done</think>"));
    EXPECT_FALSE(s.in_think_block);
}

TEST(TextThink, EntersOnLiteralOpenWhenNotThinking) {
    // Model emits a literal "<think>" while not yet thinking (and no closer in
    // the same window) -> enter the block.
    TextThinkState s;
    ASSERT_FALSE(s.in_think_block);
    EXPECT_TRUE(s.feed_piece("hmm <think> let me"));
    EXPECT_TRUE(s.in_think_block);
}

TEST(TextThink, NoMarkerNoTransition) {
    TextThinkState s;
    s.in_think_block = true;
    EXPECT_FALSE(s.feed_piece("just some normal reasoning text"));
    EXPECT_TRUE(s.in_think_block);  // still inside; no </think> seen
}

TEST(TextThink, MarkerSurvivesWindowEviction) {
    // The 32-char sliding window must not drop a marker that is being assembled.
    // Feed >32 chars of filler then the split marker: the marker bytes are the
    // most recent, so the window still contains the full "</think>".
    TextThinkState s;
    s.in_think_block = true;
    s.feed_piece(std::string(40, 'x'));  // window now full of filler
    EXPECT_FALSE(s.feed_piece("</th"));
    EXPECT_TRUE(s.feed_piece("ink>"));
    EXPECT_FALSE(s.in_think_block);
}

// ---------------------------------------------------------------------------
// Post-</think> grace period (should_stop)
// ---------------------------------------------------------------------------
// kMinAnswerAfterThink == 16: after the block closes, suppress stop until at
// least 16 content tokens have been produced.

TEST(GracePeriod, BlocksStopImmediatelyAfterExitWithNoContent) {
    // Empty-think case: </think> at 10, EOS at 11, no content emitted yet -> 1
    // token since exit < 16 -> blocked (force the model to produce something).
    EXPECT_TRUE(grace_blocks_stop(/*think_exit_idx=*/10, /*output_size=*/11,
                                  /*content_after_think=*/false));
}

TEST(GracePeriod, HonorsStopOnceContentSeen) {
    // THE FIX: the model produced a real answer (content_after_think=true) and
    // then emitted its stop token only 2 tokens after </think>. Previously the
    // raw-distance grace blocked this and padded/repeated the answer; now the
    // stop is honored the instant content exists. (A complete "Paris"/"4".)
    EXPECT_FALSE(grace_blocks_stop(/*think_exit_idx=*/10, /*output_size=*/12,
                                   /*content_after_think=*/true));
}

TEST(GracePeriod, HardCapReleasesEvenWithoutContent) {
    // Even with NO content, the grace lifts after kMinAnswerAfterThink so a
    // model that only emits stops still finishes (bounded). 10 -> 26 (==16).
    EXPECT_FALSE(grace_blocks_stop(/*think_exit_idx=*/10, /*output_size=*/26,
                                   /*content_after_think=*/false));
    // 15 since exit, still no content -> blocked (just under the cap).
    EXPECT_TRUE(grace_blocks_stop(/*think_exit_idx=*/10, /*output_size=*/25,
                                  /*content_after_think=*/false));
}

TEST(GracePeriod, NoBlockWhenNeverThought) {
    // think_exit_idx < 0 means the request never entered/exited a think block.
    EXPECT_FALSE(grace_blocks_stop(/*think_exit_idx=*/-1, /*output_size=*/2,
                                   /*content_after_think=*/false));
}

TEST(GracePeriod, WhitespacePieceIsNotContent) {
    // THE post-#798 FIX: a "\n" / "\n\n" / "  " token after </think> must NOT
    // count as answer content (else a stop right after it yields empty content).
    EXPECT_TRUE(piece_is_whitespace("\n"));
    EXPECT_TRUE(piece_is_whitespace("\n\n"));
    EXPECT_TRUE(piece_is_whitespace("   "));
    EXPECT_TRUE(piece_is_whitespace(" \t\r\n"));
    EXPECT_TRUE(piece_is_whitespace(""));  // empty decode (e.g. some specials)
}

TEST(GracePeriod, RealTextPieceIsContent) {
    EXPECT_FALSE(piece_is_whitespace("7"));
    EXPECT_FALSE(piece_is_whitespace("Paris"));
    EXPECT_FALSE(piece_is_whitespace(" red"));     // leading space, real word
    EXPECT_FALSE(piece_is_whitespace("\n4"));      // newline + digit
}
