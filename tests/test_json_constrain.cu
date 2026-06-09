#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/json_constrain.h"
#include "compute/constrain_common.h"
#include "compute/preamble_gate.h"
#include "compute/json_schema.h"
#include "compute/schema_constrain.h"
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

// ---------------------------------------------------------------------------
// RegexNfa — direct unit tests for JSON-schema `pattern` enforcement.
// Isolated from the model / preamble gate to pinpoint over-masking.
// ---------------------------------------------------------------------------

// Feed a whole string; return final state set ({} if it died).
static std::vector<int> nfa_run(const RegexNfa& n, const std::string& s) {
    std::vector<int> st = n.start_set();
    for (char c : s) {
        st = n.step(st, static_cast<unsigned char>(c));
        if (st.empty())
            return st;
    }
    return st;
}

TEST(RegexNfaTest, Literal) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("abc"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_TRUE(n.accepts(nfa_run(n, "abc")));
    EXPECT_TRUE(nfa_run(n, "abx").empty());        // diverges at 3rd char
    EXPECT_FALSE(n.accepts(nfa_run(n, "ab")));     // prefix alive but not accepting
    EXPECT_TRUE(nfa_run(n, "x").empty());          // wrong first char dies
}

TEST(RegexNfaTest, CharClassRange) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("[A-Z]"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_TRUE(n.accepts(nfa_run(n, "D")));
    EXPECT_TRUE(nfa_run(n, "d").empty());          // lowercase not in class
}

TEST(RegexNfaTest, CountedRepeat) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("[A-Z]{3}"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_FALSE(n.step(n.start_set(), 'D').empty());  // first char must survive
    EXPECT_TRUE(n.accepts(nfa_run(n, "DEU")));
    EXPECT_FALSE(n.accepts(nfa_run(n, "DE")));     // only 2 — not yet accepting
    EXPECT_TRUE(nfa_run(n, "DEUX").empty());       // 4th char dies
    EXPECT_TRUE(nfa_run(n, "Dx").empty());         // 2nd char wrong class
}

TEST(RegexNfaTest, Anchored) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("^[A-Z]{3}$"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_FALSE(n.step(n.start_set(), 'D').empty());
    EXPECT_TRUE(n.accepts(nfa_run(n, "DEU")));
    EXPECT_FALSE(n.accepts(nfa_run(n, "DE")));
}

// End-to-end SchemaConstrainer: a `pattern` value must allow pattern-valid tokens
// and mask the rest — and crucially must NOT mask everything (the over-masking
// regression that produced "!!!!"). No model / preamble gate involved.
TEST(SchemaConstrainTest, PatternEnforcementMasksCorrectly) {
    SKIP_IF_NO_CUDA();

    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "code",
                                     ":",     "}",   "D",    "DEU", "abc"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string","pattern":"^[A-Z]{3}$"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);

    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    // Walk the FSM to the string value: {  "code"  :  "
    for (int t : {3, 4, 5, 4, 6, 4})
        sc.update(t);

    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, vocab * sizeof(float));
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    auto allowed = [&](int i) { return h[i] > -1e30f; };
    int n_allowed = 0;
    for (int i = 0; i < vocab; i++)
        if (allowed(i)) n_allowed++;

    EXPECT_GT(n_allowed, 0) << "over-masking regression: every token forbidden in STRING_PATTERN";
    EXPECT_TRUE(allowed(8)) << "'D' (uppercase, pattern-alive) must be allowed";
    EXPECT_TRUE(allowed(9)) << "'DEU' (full ^[A-Z]{3}$ match) must be allowed";
    EXPECT_FALSE(allowed(10)) << "'abc' (lowercase) must be masked by the pattern";
}

// At OBJECT_OPEN, a multi-char token that begins with the opening quote
// (`"code`, `"Why`) opens the key string AND fills key chars in one step.
// Such tokens must be narrowed to valid key prefixes — otherwise a non-key
// token (`"Why`) slips through on its CAT_QUOTE bit and the FSM gets stuck
// mid-key, degenerating into "!!!!". No model / preamble gate involved.
TEST(SchemaConstrainTest, ObjectOpenQuotePrefixedKeyMasked) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4    5     6        7       8     9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "}", "\"", "\"code", "\"Why", ":", "code"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);

    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    sc.update(3);  // "{"  → OBJECT_OPEN

    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, vocab * sizeof(float));
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    auto allowed = [&](int i) { return h[i] > -1e30f; };

    EXPECT_TRUE(allowed(5)) << "bare opening quote must be allowed";
    EXPECT_TRUE(allowed(6)) << "'\"code' (quote + valid complete key) must be allowed";
    EXPECT_FALSE(allowed(7)) << "'\"Why' (quote + invalid key) must be masked — the OBJECT_OPEN hole";
    EXPECT_FALSE(allowed(9)) << "'code' without opening quote must be masked at OBJECT_OPEN (category)";
    EXPECT_FALSE(allowed(4)) << "'}' must be masked: required key 'code' not yet emitted";
}

// Key order in a JSON object is not significant: {"type":"string","enum":[...]}
// and the alphabetically-reordered {"enum":[...],"type":"string"} (what a
// request round-trip through a JSON library produces) must both parse to ENUM.
// A later "type":"string" must not demote the node back to a free string.
TEST(SchemaConstrainTest, EnumPrecedenceIsOrderIndependent) {
    auto a = parse_json_schema(R"({"type":"string","enum":["en","de","fr"]})");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->type, SchemaType::ENUM) << "type-then-enum must be ENUM";

    auto b = parse_json_schema(R"({"enum":["en","de","fr"],"type":"string"})");
    ASSERT_TRUE(b != nullptr);
    EXPECT_EQ(b->type, SchemaType::ENUM) << "enum-then-type must still be ENUM";
    EXPECT_EQ(b->enum_values.size(), 3u);
}

// Run apply_mask over a vocab of `n` tokens and return which token ids survive.
static std::vector<bool> schema_allowed(SchemaConstrainer& sc, int n) {
    std::vector<float> h(n, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
    std::vector<bool> out(n);
    for (int i = 0; i < n; i++)
        out[i] = h[i] > -1e30f;
    return out;
}

// A combined token like `{}` opens AND closes an object in one step; it must be
// rejected while required keys are unmet (the first-char category mask only
// sees CAT_OPEN_BRACE and would let it through → empty object, schema violated).
TEST(SchemaConstrainTest, PrematureObjectCloseRejected) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5     6     7
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "}", "{}", "\"", "code"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    // Fresh root: phase VALUE_START expecting the object to open.
    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(a[3]) << "'{' must open the object";
    EXPECT_FALSE(a[4]) << "bare '}' masked by category (not a value start)";
    EXPECT_FALSE(a[5]) << "'{}' combined token must be rejected — required 'code' unmet";
}

// After the last property's value, a comma would dangle (no key can follow) —
// it must be masked, leaving only the closing brace. Prevents `{"a":"x",}`.
TEST(SchemaConstrainTest, TrailingCommaRejected) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5    6    7    8    9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "a", "x", ":", "}", ","};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    for (int t : {3, 4, 5, 4, 7, 4, 6, 4})  // { "a" : "x"  -> OBJECT_AFTER_VALUE
        sc.update(t);
    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(a[8]) << "'}' must close the object after the only property";
    EXPECT_FALSE(a[9]) << "',' must be masked — no further property can follow";
}

// JSON forbids leading zeros: after a single '0' the integer part is done, so
// another digit is illegal (also bounds `0999...` degeneration).
TEST(SchemaConstrainTest, IntegerLeadingZeroRejected) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5    6    7    8    9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "n", ":", "0", "5", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    for (int t : {3, 4, 5, 4, 6, 7})  // { "n" : 0  -> NUMBER_VALUE (leading zero)
        sc.update(t);
    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_FALSE(a[8]) << "digit after a leading '0' must be masked";
    EXPECT_TRUE(a[9]) << "'}' must be allowed to close the number/object";
}

// A combined value token must be validated against an enum/integer constraint,
// not just its opening quote/digit category.
TEST(SchemaConstrainTest, EnumAndIntegerComboTokensValidated) {
    SKIP_IF_NO_CUDA();
    //          0       1     2      3   4    5            6    7      8       9      10     11    12     13
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "sentiment", ":", "\"en\"", "\":\"", "\"x", "5", "5.", "5.0", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    // --- enum property ---
    {
        auto schema = parse_json_schema(
            R"({"type":"object","properties":{"sentiment":{"type":"string","enum":["en","de","fr"]}},"required":["sentiment"]})");
        ASSERT_TRUE(schema != nullptr);
        SchemaConstrainer sc;
        ASSERT_TRUE(sc.init(tok, std::move(schema)));
        for (int t : {3, 4, 5, 4, 6})  // { " sentiment " :  -> value frame (enum)
            sc.update(t);
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[7]) << "'\"en\"' (exact enum value in one token) must be allowed";
        EXPECT_TRUE(a[4]) << "bare opening '\"' must be allowed";
        EXPECT_FALSE(a[8]) << "'\":\"' (quote+colon+quote, not an enum value) must be masked";
        EXPECT_FALSE(a[9]) << "'\"x' (invalid enum prefix) must be masked";
    }

    // --- integer property: reject float-shaped combined tokens ---
    {
        auto schema = parse_json_schema(
            R"({"type":"object","properties":{"sentiment":{"type":"integer"}},"required":["sentiment"]})");
        ASSERT_TRUE(schema != nullptr);
        SchemaConstrainer sc;
        ASSERT_TRUE(sc.init(tok, std::move(schema)));
        for (int t : {3, 4, 5, 4, 6})  // { " sentiment " :  -> value frame (integer)
            sc.update(t);
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[10]) << "'5' must be allowed for an integer";
        EXPECT_FALSE(a[11]) << "'5.' must be masked for an integer";
        EXPECT_FALSE(a[12]) << "'5.0' must be masked for an integer";
    }
}

// ---------------------------------------------------------------------------
// $ref / $defs (issue #555) — pydantic/zod emit $defs+$ref for EVERY nested
// model, so this is the agent-framework path, not an exotic corner.
// ---------------------------------------------------------------------------

// Parse-level: $defs collected, $ref resolves, unsupported/unresolvable refs
// fail the parse (decline constrained decoding instead of enforcing a wrong
// grammar). No CUDA needed.
TEST(SchemaConstrainTest, RefDefsParseAndResolve) {
    // pydantic-style nested model
    auto a = parse_json_schema(
        R"({"type":"object","properties":{"inner":{"$ref":"#/$defs/Inner"}},)"
        R"("required":["inner"],"$defs":{"Inner":{"type":"object",)"
        R"("properties":{"x":{"type":"integer"}},"required":["x"]}}})");
    ASSERT_TRUE(a != nullptr);
    ASSERT_EQ(a->properties.size(), 1u);
    const SchemaNode* inner_ref = a->properties[0].second.get();
    EXPECT_EQ(inner_ref->type, SchemaType::REF);
    const SchemaNode* inner = resolve_schema_ref(a.get(), inner_ref);
    ASSERT_TRUE(inner != nullptr);
    EXPECT_EQ(inner->type, SchemaType::OBJECT);
    ASSERT_EQ(inner->properties.size(), 1u);
    EXPECT_EQ(inner->properties[0].first, "x");

    // "definitions" spelling + clone preserves refs/defs
    auto b = parse_json_schema(
        R"({"definitions":{"S":{"type":"string"}},"type":"object",)"
        R"("properties":{"s":{"$ref":"#/definitions/S"}}})");
    ASSERT_TRUE(b != nullptr);
    auto b2 = b->clone();
    const SchemaNode* s_res = resolve_schema_ref(b2.get(), b2->properties[0].second.get());
    ASSERT_TRUE(s_res != nullptr);
    EXPECT_EQ(s_res->type, SchemaType::STRING);

    // root self-ref "#"
    auto c = parse_json_schema(
        R"({"type":"object","properties":{"next":{"$ref":"#"}},"required":[]})");
    ASSERT_TRUE(c != nullptr);
    EXPECT_EQ(resolve_schema_ref(c.get(), c->properties[0].second.get()), c.get());

    // unresolvable name → parse fails
    EXPECT_EQ(parse_json_schema(
                  R"({"type":"object","properties":{"a":{"$ref":"#/$defs/Missing"}}})"),
              nullptr);
    // external / deep-pointer refs → parse fails
    EXPECT_EQ(parse_json_schema(
                  R"({"properties":{"a":{"$ref":"https://example.com/s.json"}}})"),
              nullptr);
    EXPECT_EQ(parse_json_schema(
                  R"({"properties":{"a":{"$ref":"#/properties/b"}}})"),
              nullptr);
    // pure ref->ref cycle → parse fails (no structure to terminate resolution)
    EXPECT_EQ(parse_json_schema(
                  R"({"$ref":"#/$defs/A","$defs":{"A":{"$ref":"#/$defs/B"},)"
                  R"("B":{"$ref":"#/$defs/A"}}})"),
              nullptr);
}

// Enforcement through a $ref: the referenced object's keys/required are
// enforced exactly as an inline schema would be.
TEST(SchemaConstrainTest, RefNestedModelEnforced) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5        6    7    8    9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "inner", ":", "x", "5", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"inner":{"$ref":"#/$defs/Inner"}},)"
        R"("required":["inner"],"$defs":{"Inner":{"type":"object",)"
        R"("properties":{"x":{"type":"integer"}},"required":["x"]}}})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    for (int t : {3, 4, 5, 4, 6})  // { "inner" :  → value frame is the REF target
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[3]) << "'{' must open the $ref'd inner object";
        EXPECT_FALSE(a[8]) << "bare digit masked — inner value is an object, not a number";
    }
    for (int t : {3, 4, 7, 4, 6})  // { "x" :  → inner integer value
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[8]) << "'5' must be allowed for Inner.x (integer)";
        EXPECT_FALSE(a[9]) << "'}' masked — required Inner.x not yet emitted";
    }
}

// True recursion: a tree node whose children are arrays of itself. The frame
// stack must follow $ref depth arbitrarily (here 2 levels) — the gap the old
// NFA backend could not represent.
TEST(SchemaConstrainTest, RecursiveSchemaEnforced) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5    6    7    8    9    10   11
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "v", ":", "1", ",", "kids", "[", "]"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"$ref":"#/$defs/Node","$defs":{"Node":{"type":"object",)"
        R"("properties":{"v":{"type":"integer"},"kids":{"type":"array",)"
        R"("items":{"$ref":"#/$defs/Node"}}},"required":["v"]}}})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    // { "v" : 1 , "kids" : [
    for (int t : {3, 4, 5, 4, 6, 7, 8, 4, 9, 4, 6, 10})
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[3]) << "'{' must start a recursive child Node inside kids[]";
        EXPECT_TRUE(a[11]) << "']' must be allowed (empty kids)";
        EXPECT_FALSE(a[7]) << "bare digit masked — items are Node objects";
    }
    // open child: { "v" : 1 — the recursive frame enforces Node's grammar at
    // depth 2: a second "kids" key (','+key) must be offered, digits must not.
    for (int t : {3, 4, 5, 4, 6, 7})
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[8]) << "',' after child's v — 'kids' is still emittable at depth 2";
        EXPECT_FALSE(a[4]) << "bare quote masked mid-number at depth 2";
    }
}

// ===========================================================================
// Vocab-mismatch regression (SIGBUS 2026-06-09): SafeTensors models have
// MORE logits than tokenizer entries (Qwen3-8B-NVFP4: lm_head vocab 151936
// vs tokenizer.json 151669). apply_mask receives the LOGITS vocab size; the
// constrainer classified only the TOKENIZER vocab. The host validation loop
// then read token_texts_[i] out of bounds (SIGBUS — killed imp-server on the
// first json_mode request), and the mask kernels read the category/allow
// device buffers out of bounds. Contract under test: apply_mask with a
// larger vocab_size must (a) not crash and (b) mask every logit in the
// padding range [tokenizer_vocab, model_vocab) — padding ids are unknown to
// the grammar and untrained in the model, so they must never be sampleable.
// ===========================================================================

// ===========================================================================
// Raw-control-char regression (2026-06-10): JSON forbids unescaped U+0000–
// U+001F inside strings, but both FSMs accepted "any content char" there.
// Multi-char tokens whose FIRST char passes the category mask (`"<newline>`
// opens a string and smuggles a raw newline in one step) produced output that
// json.loads() rejects ("Invalid control character"). Observed live on
// Qwen3-8B-NVFP4 json_schema generation.
// ===========================================================================

TEST(SchemaConstrainTest, RawControlCharInStringMasked) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5       6    7     8        9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "code", ":", "}", "\"\n", "\"ok"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    // Walk to OBJECT_COLON: {  "code"  :   — next token opens the string value.
    for (int t : {3, 4, 5, 4, 6})
        sc.update(t);

    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(a[9]) << "'\"ok' (quote + printable content) must be allowed";
    EXPECT_FALSE(a[8]) << "'\"<newline>' must be masked — raw control char inside a string";
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

TEST(SchemaConstrainTest, ModelVocabLargerThanTokenizerMasksPadding) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5       6    7
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "code", ":", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    const int tok_vocab = static_cast<int>(toks.size());
    const int model_vocab = tok_vocab + 9;

    std::vector<float> h(model_vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, model_vocab * sizeof(float));
    cudaMemcpy(d, h.data(), model_vocab * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, model_vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, model_vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    EXPECT_GT(h[3], -1e30f) << "'{' must be allowed at schema root";
    for (int i = tok_vocab; i < model_vocab; i++) {
        EXPECT_FLOAT_EQ(h[i], -FLT_MAX) << "padding id " << i << " leaked through the schema mask";
    }
}

}  // namespace
}  // namespace imp
