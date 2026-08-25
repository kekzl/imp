// Qwen3.8-27B chat template: the `reasoning_effort` control field.
//
// Qwen3.8's chat_template.jinja (tests/fixtures/qwen38_chat_template.jinja,
// embedded verbatim in tests/refs/qwen38_template_jinja.h) branches on
// `reasoning_effort` and injects a different system-prompt preamble for each
// of its three supported values:
//
//   xhigh (its own default)  long "think carefully / validate assumptions" text
//   low                      short "keep your thinking brief" text
//   medium                   NO preamble at all
//   anything else            raise_exception
//
// Before this test existed, imp accepted `reasoning_effort` on the wire and
// never stamped it into the Jinja context, so every request rendered the
// template's `default('xhigh')` branch. Measured on Qwen3.8-27B-NVFP4 through
// /v1/chat/completions: `low` and `xhigh` both produced 67 prompt_tokens and
// identical output. The field was inert.
//
// This asserts on the RENDERED PROMPT, which is what the model actually sees,
// via ChatTemplate::render_jinja() — the production context builder, not a
// hand-rebuilt parallel one (the same reason the Harmony golden moved onto it
// in #1572). No vocabulary and no model needed, so it runs in the CPU lane.

#include "model/chat_template.h"
#include "refs/qwen38_template_jinja.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace imp {
namespace {

// The two preambles the template injects, verbatim from the fixture. A test
// that only grepped for "xhigh" would pass on the broken build, because the
// word also appears in the template's own exception message.
constexpr const char* kXhighPreamble =
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.";
constexpr const char* kLowPreamble =
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.";

// Minimal ChatML vocabulary — render_jinja only needs bos/eos text plus the
// control tokens the template emits as literal strings.
Tokenizer make_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    tokens.push_back("<unk>");
    scores.push_back(0.0f);
    tokens.push_back("<s>");
    scores.push_back(0.0f);
    tokens.push_back("</s>");
    scores.push_back(0.0f);
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.push_back(buf);
        scores.push_back(-10.0f);
    }
    tokens.push_back("<|im_start|>");
    scores.push_back(0.0f);
    tokens.push_back("<|im_end|>");
    scores.push_back(0.0f);

    Tokenizer tok;
    tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2);
    tok.set_type("spm");
    tok.set_add_bos(false);
    tok.set_add_space_prefix(false);
    return tok;
}

ChatTemplate make_template(const Tokenizer& tok) {
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok, qwen38_golden::k_chat_template_jinja));
    EXPECT_TRUE(tpl.has_jinja()) << "Qwen3.8 template must drive rendering through Jinja";
    return tpl;
}

std::vector<ChatMessage> one_user_turn() {
    return {{"user", "What is 17*23?"}};
}

std::string render_with(const ChatTemplate& tpl, const Tokenizer& tok, const std::string& effort) {
    return tpl.render_jinja(tok, one_user_turn(), /*add_generation_prompt=*/true,
                            /*suppress_thinking=*/false, /*force_thinking=*/false, effort);
}

// ---- the template's own default, with no caller opinion ----

TEST(Qwen38ReasoningEffort, DefaultsToXhighWhenUnset) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl = make_template(tok);
    const std::string out = render_with(tpl, tok, "");
    EXPECT_NE(out.find(kXhighPreamble), std::string::npos)
        << "unset reasoning_effort must fall through to the template's default('xhigh')";
    EXPECT_EQ(out.find(kLowPreamble), std::string::npos);
}

// ---- each supported value changes the rendered prompt ----

TEST(Qwen38ReasoningEffort, LowSelectsTheBriefPreamble) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl = make_template(tok);
    const std::string out = render_with(tpl, tok, "low");
    EXPECT_NE(out.find(kLowPreamble), std::string::npos)
        << "reasoning_effort=low must reach the Jinja context";
    EXPECT_EQ(out.find(kXhighPreamble), std::string::npos)
        << "low must not also carry the xhigh preamble";
}

TEST(Qwen38ReasoningEffort, MediumSelectsNoPreamble) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl = make_template(tok);
    const std::string out = render_with(tpl, tok, "medium");
    EXPECT_EQ(out.find(kXhighPreamble), std::string::npos);
    EXPECT_EQ(out.find(kLowPreamble), std::string::npos)
        << "medium is the template's no-preamble branch";
    // It must still be a real render, not an empty string that trivially
    // satisfies both assertions above.
    EXPECT_NE(out.find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(out.find("What is 17*23?"), std::string::npos);
}

TEST(Qwen38ReasoningEffort, ThreeValuesGiveThreeDistinctPrompts) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl = make_template(tok);
    const std::string lo = render_with(tpl, tok, "low");
    const std::string me = render_with(tpl, tok, "medium");
    const std::string hi = render_with(tpl, tok, "xhigh");
    EXPECT_NE(lo, me);
    EXPECT_NE(me, hi);
    EXPECT_NE(lo, hi);
    // The measured symptom of the bug was that low and xhigh rendered the same
    // number of prompt tokens. Length is the proxy that has no vocabulary.
    EXPECT_LT(me.size(), lo.size());
    EXPECT_LT(lo.size(), hi.size());
}

// ---- interaction with the thinking switch ----

TEST(Qwen38ReasoningEffort, SuppressedThinkingDropsThePreamble) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl = make_template(tok);
    // The template guards the whole reasoning_effort block behind
    // `enable_thinking is undefined or enable_thinking is true`, so a caller
    // that suppresses thinking must get no preamble regardless of effort.
    const std::string out = tpl.render_jinja(tok, one_user_turn(), /*add_generation_prompt=*/true,
                                             /*suppress_thinking=*/true, /*force_thinking=*/false,
                                             "xhigh");
    EXPECT_EQ(out.find(kXhighPreamble), std::string::npos)
        << "no reasoning preamble when thinking is off";
    EXPECT_EQ(out.find(kLowPreamble), std::string::npos);
}

// ---- an unsupported value must not silently render as the default ----

TEST(Qwen38ReasoningEffort, UnsupportedValueYieldsNoPreamble) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl = make_template(tok);
    // The template calls raise_exception for anything outside its three values.
    // imp's Jinja logs that and keeps rendering, so the observable contract is:
    // the prompt carries NEITHER preamble. What must not happen is silently
    // getting the xhigh text, which would make a typo look like it worked.
    const std::string out = render_with(tpl, tok, "ludicrous");
    EXPECT_EQ(out.find(kXhighPreamble), std::string::npos)
        << "an unsupported effort must not render as the default";
    EXPECT_EQ(out.find(kLowPreamble), std::string::npos);
}

}  // namespace
}  // namespace imp
