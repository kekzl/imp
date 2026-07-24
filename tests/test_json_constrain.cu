#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/json_constrain.h"
#include "compute/constrain_common.h"
#include "compute/preamble_gate.h"
#include "model/tokenizer.h"

#include <string>
#include <cfloat>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// ===========================================================================
// Test 1: classify_token("0") includes NUMBER category
// ===========================================================================
TEST(JsonConstrainTest, ClassifyDigit) {
    uint16_t cat = classify_token("0");
    EXPECT_TRUE(cat & CAT_NUMBER_START) << "Digit '0' should be classified as NUMBER_START";
    EXPECT_TRUE(cat & CAT_NUMBER_CONT) << "Digit '0' should be classified as NUMBER_CONT";
}

// ===========================================================================
// Test 2: classify_token("{") returns OPEN_BRACE
// ===========================================================================
TEST(JsonConstrainTest, ClassifyBrace) {
    uint16_t cat = classify_token("{");
    EXPECT_TRUE(cat & CAT_OPEN_BRACE) << "'{' should be classified as OPEN_BRACE";
    EXPECT_FALSE(cat & CAT_CLOSE_BRACE) << "'{' should NOT be CLOSE_BRACE";

    uint16_t cat2 = classify_token("}");
    EXPECT_TRUE(cat2 & CAT_CLOSE_BRACE) << "'}' should be classified as CLOSE_BRACE";
}

// ===========================================================================
// Test 3: classify_token('"') returns QUOTE
// ===========================================================================
TEST(JsonConstrainTest, ClassifyQuote) {
    uint16_t cat = classify_token("\"");
    EXPECT_TRUE(cat & CAT_QUOTE) << "'\"' should be classified as QUOTE";
}

// ===========================================================================
// Test 4: classify_token for multi-char string content
// ===========================================================================
TEST(JsonConstrainTest, ClassifyStringContent) {
    // "hello" is all lowercase printable — should be STRING_CHAR
    uint16_t cat = classify_token("hello");
    EXPECT_TRUE(cat & CAT_STRING_CHAR) << "'hello' should be classified as STRING_CHAR";
    // Also a literal continuation (all lowercase)
    EXPECT_TRUE(cat & CAT_LITERAL_CONT) << "'hello' (all lowercase) should also be LITERAL_CONT";
}

// ===========================================================================
// Test 5: classify_token for structural tokens
// ===========================================================================
TEST(JsonConstrainTest, ClassifyStructural) {
    EXPECT_TRUE(classify_token("[") & CAT_OPEN_BRACKET);
    EXPECT_TRUE(classify_token("]") & CAT_CLOSE_BRACKET);
    EXPECT_TRUE(classify_token(":") & CAT_COLON);
    EXPECT_TRUE(classify_token(",") & CAT_COMMA);
}

// ===========================================================================
// Test 6: classify_token for literal starts
// ===========================================================================
TEST(JsonConstrainTest, ClassifyLiteralStarts) {
    EXPECT_TRUE(classify_token("t") & CAT_TRUE_START);
    EXPECT_TRUE(classify_token("f") & CAT_FALSE_START);
    EXPECT_TRUE(classify_token("n") & CAT_NULL_START);
    EXPECT_TRUE(classify_token("true") & CAT_TRUE_START);
    EXPECT_TRUE(classify_token("false") & CAT_FALSE_START);
    EXPECT_TRUE(classify_token("null") & CAT_NULL_START);
}

// ===========================================================================
// Test 7: classify_token for whitespace
// ===========================================================================
TEST(JsonConstrainTest, ClassifyWhitespace) {
    EXPECT_TRUE(classify_token(" ") & CAT_WHITESPACE);
    EXPECT_TRUE(classify_token("\n") & CAT_WHITESPACE);
    EXPECT_TRUE(classify_token("") & CAT_WHITESPACE);  // empty = whitespace
}

// ===========================================================================
// Test 8: classify_token for number patterns
// ===========================================================================
TEST(JsonConstrainTest, ClassifyNumbers) {
    EXPECT_TRUE(classify_token("-") & CAT_NUMBER_START);
    EXPECT_TRUE(classify_token("123") & CAT_NUMBER_START);
    EXPECT_TRUE(classify_token("123") & CAT_NUMBER_CONT);
    EXPECT_TRUE(classify_token(".") & CAT_NUMBER_CONT);
}

// ===========================================================================
// FSM grammar tests (#1067) — a container close must restore the continuation
// pushed at its matching opener. The old code popped it and *peeked the
// grandparent's* continuation instead, so after `[..., {"k": [..]` the `]`
// left the FSM in array context and it accepted `,"bare-string"` + `]]`.
// ===========================================================================
TEST(JsonConstrainFsmTest, Issue1067NestedArrayCloseKeepsObjectContext) {
    JsonConstrainer c;
    // Exact shape from the degen_suite repro: object inside array, whose last
    // value is an array — then a dangling key and ']]' closing the '{'.
    EXPECT_FALSE(c.sim_token_valid(
        "[\"_classifications\",{\"sentiment\":\"positive\","
        "\"topics\":[\"phone\",\"camera\"],\"score\"  ]]"));
}

TEST(JsonConstrainFsmTest, ValidNestedDocumentsAccepted) {
    JsonConstrainer c;
    EXPECT_TRUE(c.sim_token_valid("[\"a\",{\"k\":\"v\",\"t\":[\"x\",\"y\"],\"n\":1}]"));
    EXPECT_TRUE(c.sim_token_valid("{\"a\":{\"b\":1},\"c\":2}"));
    EXPECT_TRUE(c.sim_token_valid("{\"a\":[true,false,null],\"b\":{\"c\":[1,2.5e3]}}"));
    EXPECT_TRUE(c.sim_token_valid("[[1,2],[3,4]]"));
    EXPECT_TRUE(c.sim_token_valid("[{\"a\":1},{\"b\":2}]"));
}

TEST(JsonConstrainFsmTest, MismatchedClosersRejected) {
    JsonConstrainer c;
    EXPECT_FALSE(c.sim_token_valid("{\"a\":1]"));
    EXPECT_FALSE(c.sim_token_valid("[1}"));
    EXPECT_FALSE(c.sim_token_valid("[[1,2]]]"));
    EXPECT_FALSE(c.sim_token_valid("{\"a\":{\"b\":1}}}"));
}

TEST(JsonConstrainFsmTest, TrailingContentAfterRootRejected) {
    JsonConstrainer c;
    EXPECT_FALSE(c.sim_token_valid("{\"a\":1}{"));
    EXPECT_FALSE(c.sim_token_valid("[1] 2"));
}

TEST(JsonConstrainFsmTest, MidDocumentStateAfterNestedArrayClose) {
    JsonConstrainer c;
    c.advance_text("{\"t\":[1,2]");
    // Back in the object after the nested array closed: ']' is illegal, a
    // comma must be followed by a key (not a bare value).
    EXPECT_FALSE(c.sim_token_valid("]"));
    EXPECT_TRUE(c.sim_token_valid(",\"k\":3}"));
    EXPECT_FALSE(c.sim_token_valid(",5"));
}

// ===========================================================================
// Test 9: GPU mask kernel — constrain_mask_kernel masks invalid tokens
// ===========================================================================
TEST(JsonConstrainTest, MaskAllowsValidTokens) {
    SKIP_IF_NO_CUDA();

    // Simulate 4 tokens: "{", "hello", "0", "}"
    // with allowed_mask = CAT_OPEN_BRACE | CAT_OPEN_BRACKET (START state)
    constexpr int vocab = 4;
    uint16_t h_cats[vocab] = {
        CAT_OPEN_BRACE,                                             // "{"
        CAT_STRING_CHAR,                                            // "hello"
        static_cast<uint16_t>(CAT_NUMBER_START | CAT_NUMBER_CONT),  // "0"
        CAT_CLOSE_BRACE                                             // "}"
    };
    uint16_t h_mask = CAT_OPEN_BRACE | CAT_OPEN_BRACKET | CAT_WHITESPACE;

    float h_logits[vocab] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Upload to device
    uint16_t *d_cats, *d_mask;
    float* d_logits;
    cudaMalloc(&d_cats, vocab * sizeof(uint16_t));
    cudaMalloc(&d_mask, sizeof(uint16_t));
    cudaMalloc(&d_logits, vocab * sizeof(float));

    cudaMemcpy(d_cats, h_cats, vocab * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, &h_mask, sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_logits, h_logits, vocab * sizeof(float), cudaMemcpyHostToDevice);

    constrain_mask_kernel<<<1, vocab>>>(d_logits, d_cats, d_mask, vocab);
    cudaDeviceSynchronize();

    float h_out[vocab];
    cudaMemcpy(h_out, d_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);

    // Token 0 ("{") matches CAT_OPEN_BRACE -> should be untouched
    EXPECT_FLOAT_EQ(h_out[0], 1.0f);
    // Token 1 ("hello"), 2 ("0"), 3 ("}") don't match -> should be -FLT_MAX
    EXPECT_FLOAT_EQ(h_out[1], -FLT_MAX);
    EXPECT_FLOAT_EQ(h_out[2], -FLT_MAX);
    EXPECT_FLOAT_EQ(h_out[3], -FLT_MAX);

    cudaFree(d_cats);
    cudaFree(d_mask);
    cudaFree(d_logits);
}

// ===========================================================================
// PreambleGate tests — reasoning-model thinking pass-through
// ===========================================================================
//
// Token IDs used in these tests are fictitious — the gate only cares about
// matching the configured close_token id and inspecting token text for a JSON
// start char.

constexpr int32_t TOK_THINK_OPEN = 100;
constexpr int32_t TOK_THINK_CLOSE = 101;
constexpr int32_t TOK_TEXT = 200;
constexpr int32_t TOK_OPEN_BRACE = 300;

TEST(PreambleGateTest, DisabledByDefault) {
    PreambleGate g;
    EXPECT_FALSE(g.active());
    EXPECT_FALSE(g.absorb(TOK_TEXT, "hi"));  // gate inactive → don't absorb
}

TEST(PreambleGateTest, ConfigureNoCloseTokenAndNoBudgetStaysDisabled) {
    PreambleGate g;
    g.configure(-1, 0);
    EXPECT_FALSE(g.active());
    EXPECT_FALSE(g.absorb(TOK_TEXT, "hi"));
}

TEST(PreambleGateTest, BudgetOnlyModeActivates) {
    // Non-reasoning model: no </think> token, but we still want a small
    // slack window for markdown fences (```json) or short verbal preambles.
    PreambleGate g;
    g.configure(-1, 8);
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, BudgetOnlyModeAbsorbsMarkdownFenceTokens) {
    PreambleGate g;
    g.configure(-1, 8);

    // Tokens like "```", "json", "\n" all absorbed.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "```"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "json"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "\n"));
    EXPECT_TRUE(g.active());

    // Then `{` triggers transition (forwarded to FSM).
    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, BudgetOnlyModeBudgetExhaustionForcesTransition) {
    PreambleGate g;
    g.configure(-1, 3);
    for (int i = 0; i < 2; i++) {
        EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
        EXPECT_TRUE(g.active());
    }
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, BudgetOnlyModeIgnoresNegativeTokenIds) {
    // close_token_=-1 should not match any real token. Use TOK_THINK_CLOSE
    // as a stand-in for "any non-{ token" and verify it's absorbed (not
    // treated as a close-match).
    PreambleGate g;
    g.configure(-1, 8);
    EXPECT_TRUE(g.absorb(TOK_THINK_CLOSE, "</think>"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ActivatesAfterConfigure) {
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, AbsorbsThinkingTokensThenTransitionsOnClose) {
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);

    // Free-form thinking tokens are all absorbed.
    EXPECT_TRUE(g.absorb(TOK_THINK_OPEN, "<think>"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "let me reason"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, " about this"));
    EXPECT_TRUE(g.active());

    // </think> is consumed by the gate (NOT forwarded to JSON FSM).
    EXPECT_TRUE(g.absorb(TOK_THINK_CLOSE, "</think>"));
    EXPECT_FALSE(g.active());

    // After transition, tokens are no longer absorbed.
    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "{"));
}

TEST(PreambleGateTest, JsonStartCharForcesEarlyTransition) {
    // Model that doesn't think — emits `{` directly. Gate must transition
    // and forward the token so the FSM sees the open brace.
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);
    EXPECT_TRUE(g.active());

    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "{"));  // forwarded
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, BracketAlsoTriggersTransition) {
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);
    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "["));
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, BudgetExhaustionForcesTransition) {
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 3);  // tiny budget
    for (int i = 0; i < 2; i++) {
        EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
        EXPECT_TRUE(g.active());
    }
    // Third token hits the budget — absorbed, then gate goes inactive.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, ResetReactivatesGate) {
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);
    g.absorb(TOK_THINK_CLOSE, "</think>");
    EXPECT_FALSE(g.active());

    g.reset();
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "thinking again"));
}

TEST(PreambleGateTest, ResetWhenDisabledStaysDisabled) {
    PreambleGate g;
    EXPECT_FALSE(g.active());
    g.reset();
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, MidStringJsonCharStillTriggers) {
    // Some tokenizers merge punctuation: a token like "Sure!{" should
    // also trigger the transition because { appears in the text.
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);
    EXPECT_FALSE(g.absorb(TOK_TEXT, "Sure!{"));
    EXPECT_FALSE(g.active());
}

// ===========================================================================
// Tool-aware tri-state tests
// ===========================================================================

constexpr int32_t TOK_TOOL_OPEN = 400;   // synthetic <tool_call>
constexpr int32_t TOK_TOOL_CLOSE = 401;  // synthetic </tool_call>

TEST(PreambleGateTest, ToolOpenerTokenTransitionsToToolBody) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, /*budget=*/64,
                           /*open_tokens=*/{TOK_TOOL_OPEN},
                           /*close_tokens=*/{TOK_TOOL_CLOSE},
                           /*open_prefix=*/"",
                           /*close_suffix=*/"");
    EXPECT_TRUE(g.active());

    // Free-form preamble before tool: still absorbed, still active.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "Sure! "));
    EXPECT_TRUE(g.active());

    // Opener token: absorbed, gate stays "not masking" but is now in TOOL_BODY.
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.active());  // active() still means "no mask"

    // Tool body content (including `{`!) does NOT trigger preamble exit
    // anymore — we are inside a tool body.
    EXPECT_TRUE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "\"name\": \"x\"}"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolCloseTokenTransitionsToTerminalOff) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    g.absorb(TOK_TOOL_OPEN, "<tool_call>");
    g.absorb(TOK_TEXT, "{...}");

    // Close token: absorbed, terminal OFF.
    EXPECT_TRUE(g.absorb(TOK_TOOL_CLOSE, "</tool_call>"));
    EXPECT_TRUE(g.active());  // TERMINAL_OFF still reads as "no mask"

    // Subsequent tokens — including `{` — are absorbed, FSM never re-engages.
    EXPECT_TRUE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "free text after"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolModeStillExitsOnJsonStartIfNoTool) {
    // Model emits free-text JSON instead of a tool call: gate exits to FSM
    // exactly like non-tool mode.
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    EXPECT_TRUE(g.active());
    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_FALSE(g.active());  // OFF (preamble exit), FSM enforces
}

TEST(PreambleGateTest, ToolModeBudgetExhaustExitsToFsm) {
    // Long preamble without a tool opener: budget exhausts, FSM kicks in.
    PreambleGate g;
    g.configure_with_tools(/*close_token=*/-1, /*budget=*/3,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "blah"));
    EXPECT_FALSE(g.active());  // budget exhausted → FSM kicks in
}

TEST(PreambleGateTest, ToolModeParallelCallsStayTerminalOff) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    g.absorb(TOK_TOOL_OPEN, "<tool_call>");
    g.absorb(TOK_TEXT, "{a}");
    g.absorb(TOK_TOOL_CLOSE, "</tool_call>");
    EXPECT_TRUE(g.active());

    // Second tool call — opener and body both absorbed in TERMINAL_OFF.
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "{b}"));
    EXPECT_TRUE(g.absorb(TOK_TOOL_CLOSE, "</tool_call>"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolModeResetReturnsToActive) {
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, 64,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");
    g.absorb(TOK_TOOL_OPEN, "<tool_call>");
    g.absorb(TOK_TOOL_CLOSE, "</tool_call>");
    EXPECT_TRUE(g.active());  // TERMINAL_OFF → active()=true

    g.reset();
    EXPECT_TRUE(g.active());
    // After reset, an opener token works fresh.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "hi"));
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolModeCharPrefixFallback) {
    // Llama3 dialect: <function= is multi-token. Use char-prefix only;
    // open_tokens is empty.
    PreambleGate g;
    g.configure_with_tools(/*close_token=*/-1, /*budget=*/64,
                           /*open_tokens=*/{},
                           /*close_tokens=*/{},
                           /*open_prefix=*/"<function=",
                           /*close_suffix=*/"</function>");

    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "<"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "function"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "="));  // prefix complete here
    EXPECT_TRUE(g.active());

    // Body content with `{` is absorbed (TOOL_BODY).
    EXPECT_TRUE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_TRUE(g.active());

    // Close suffix split across tokens.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "</"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "function"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, ">"));
    EXPECT_TRUE(g.active());  // TERMINAL_OFF
    EXPECT_TRUE(g.absorb(TOK_TEXT, "anything"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, LegacyConfigureKeepsBinaryBehavior) {
    // The two-arg configure() must not enable tool detection — protects
    // existing JsonConstrainer/SchemaConstrainer callers that don't know
    // about tools.
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192);
    EXPECT_TRUE(g.active());

    // A token id that *would* be a tool-opener if registered must be
    // treated as ordinary text here.
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.active());  // still in ACTIVE (preamble), not TOOL_BODY

    // `{` still triggers preamble exit (legacy behaviour).
    EXPECT_FALSE(g.absorb(TOK_OPEN_BRACE, "{"));
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, ToolModeReasoningCloseStaysActiveForToolDetection) {
    // Reasoning models (Qwen3.6, Gemma-4 thinking) emit <think>...</think>
    // before any structured output. With tools+schema both set, the gate
    // must stay ACTIVE after </think> so a subsequent <tool_call> opener
    // is recognised. In legacy (non-tool) mode, </think> still exits to
    // OFF — see ResetReactivatesGate / AbsorbsThinkingTokensThenTransitionsOnClose.
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, /*budget=*/64,
                           /*open_tokens=*/{TOK_TOOL_OPEN},
                           /*close_tokens=*/{TOK_TOOL_CLOSE},
                           /*open_prefix=*/"",
                           /*close_suffix=*/"");

    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "let me think"));
    EXPECT_TRUE(g.absorb(TOK_THINK_CLOSE, "</think>"));
    EXPECT_TRUE(g.active());  // still ACTIVE in tool-aware mode

    // Tool opener now lands in TOOL_BODY just like the no-think path.
    EXPECT_TRUE(g.absorb(TOK_TOOL_OPEN, "<tool_call>"));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_OPEN_BRACE, "{"));  // body content, not preamble exit
    EXPECT_TRUE(g.active());

    // Close-token works as before, → TERMINAL_OFF.
    EXPECT_TRUE(g.absorb(TOK_TOOL_CLOSE, "</tool_call>"));
    EXPECT_TRUE(g.active());
}

TEST(PreambleGateTest, ToolModeReasoningBudgetResetsAfterThinkClose) {
    // After </think>, the post-think budget should be fresh — i.e. the
    // budget counter resets so a long thinking block doesn't exhaust the
    // slack window before the tool opener can fire.
    PreambleGate g;
    g.configure_with_tools(TOK_THINK_CLOSE, /*budget=*/4,
                           {TOK_TOOL_OPEN}, {TOK_TOOL_CLOSE}, "", "");

    // Burn 3 tokens of "thinking" (under budget).
    EXPECT_TRUE(g.absorb(TOK_TEXT, "a"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "b"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "c"));
    EXPECT_TRUE(g.active());

    // </think> resets the budget — we should now be able to absorb 3+
    // more text tokens before exhaust.
    EXPECT_TRUE(g.absorb(TOK_THINK_CLOSE, "</think>"));
    EXPECT_TRUE(g.active());

    // Three more tokens still fit under fresh budget.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "d"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "e"));
    EXPECT_TRUE(g.absorb(TOK_TEXT, "f"));
    EXPECT_TRUE(g.active());

    // Fourth post-think token exhausts the fresh budget.
    EXPECT_TRUE(g.absorb(TOK_TEXT, "g"));
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, ThinkingAlreadyClosedStartsEnforcing) {
    // Reasoning model, but the prompt already closed <think> (e.g. /no_think):
    // there is no </think> to wait for, so the gate must enforce immediately.
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192, /*thinking_open=*/false);
    EXPECT_FALSE(g.active()) << "thinking already closed -> mask enforced from token 0";
    EXPECT_FALSE(g.absorb(TOK_TEXT, "Okay")) << "must not absorb a free-form preamble";
}

TEST(PreambleGateTest, ThinkingOpenStillAbsorbsUntilClose) {
    // Normal reasoning request (thinking open) is unchanged: absorb until </think>.
    PreambleGate g;
    g.configure(TOK_THINK_CLOSE, 8192, /*thinking_open=*/true);
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_TEXT, "reasoning..."));
    EXPECT_TRUE(g.active());
    EXPECT_TRUE(g.absorb(TOK_THINK_CLOSE, "</think>"));
    EXPECT_FALSE(g.active());
}

TEST(PreambleGateTest, BudgetOnlyModeUnaffectedByThinkingFlag) {
    // Non-reasoning model (close_token < 0) keeps the small budget window even
    // when thinking_open=false (the flag only matters in close-token mode).
    PreambleGate g;
    g.configure(/*close_token=*/-1, /*max_tokens=*/8, /*thinking_open=*/false);
    EXPECT_TRUE(g.active());
}


TEST(JsonConstrainTest, RawControlCharInStringMasked) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5      6
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "\"\n", "\"ok"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    JsonConstrainer jc;
    ASSERT_TRUE(jc.init(tok));
    jc.update(3);  // '{' → OBJECT_START, next token may open a key string

    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, vocab * sizeof(float));
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    jc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    EXPECT_GT(h[6], -1e30f) << "'\"ok' (quote + printable key chars) must be allowed";
    EXPECT_FLOAT_EQ(h[5], -FLT_MAX) << "'\"<newline>' must be masked — raw control char in string";
}

TEST(JsonConstrainTest, ModelVocabLargerThanTokenizerMasksPadding) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5    6
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "}", "0"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    JsonConstrainer jc;
    ASSERT_TRUE(jc.init(tok));

    const int tok_vocab = static_cast<int>(toks.size());
    const int model_vocab = tok_vocab + 9;  // simulated lm_head padding rows

    std::vector<float> h(model_vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, model_vocab * sizeof(float));
    cudaMemcpy(d, h.data(), model_vocab * sizeof(float), cudaMemcpyHostToDevice);
    jc.apply_mask(d, model_vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, model_vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    // '{' starts a JSON document — must stay alive.
    EXPECT_GT(h[3], -1e30f) << "'{' must be allowed at document start";
    // Every padding id (no tokenizer entry) must be masked.
    for (int i = tok_vocab; i < model_vocab; i++) {
        EXPECT_FLOAT_EQ(h[i], -FLT_MAX) << "padding id " << i << " leaked through the json mask";
    }
}

}  // namespace
}  // namespace imp
