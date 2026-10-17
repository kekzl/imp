// Assistant prefill (continue_final_message): rendered prompt vs HF transformers
// apply_chat_template(continue_final_message=True), transformers in hfref:cpu, 2026-10-17.
// Qwen3.8-27B trims content (`content|trim`), the Qwen3 ChatML golden keeps it verbatim.

#include "model/chat_template.h"
#include "refs/chat_template_goldens.h"
#include "refs/qwen38_template_jinja.h"

#include <gtest/gtest.h>
#include <cstdio>
#include <string>
#include <vector>

namespace imp {
namespace {

Tokenizer make_tokenizer() {
    std::vector<std::string> tokens = {"<unk>", "<s>", "</s>"};
    std::vector<float> scores = {0.0f, 0.0f, 0.0f};
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
    EXPECT_TRUE(tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2));
    tok.set_type("spm");
    tok.set_add_bos(false);
    tok.set_add_space_prefix(false);
    return tok;
}

std::vector<ChatMessage> prefilled(const std::string& prefix) {
    return {{"user", "What is 17*23?"}, {"assistant", prefix}};
}

std::string render_continued(const std::string& jinja, const std::string& prefix) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl;
    EXPECT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok, jinja));
    tpl.set_continue_final(true);
    return tpl.render_jinja(tok, prefilled(prefix), /*add_generation_prompt=*/true);
}

constexpr const char* kQwen38Head =
    "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
    "and clarity in the final answer.<|im_end|>\n<|im_start|>user\nWhat is 17*23?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n";
constexpr const char* kQwen3Head =
    "<|im_start|>user\nWhat is 17*23?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

TEST(ChatTemplateContinue, Qwen38TrimmedContentMatchesHf) {
    const std::string head = kQwen38Head;
    EXPECT_EQ(render_continued(qwen38_golden::k_chat_template_jinja, "The answer is"),
              head + "The answer is");
    EXPECT_EQ(render_continued(qwen38_golden::k_chat_template_jinja, "The answer is "),
              head + "The answer is");
    EXPECT_EQ(render_continued(qwen38_golden::k_chat_template_jinja, "  Sure:\n"), head + "Sure:");
}

TEST(ChatTemplateContinue, Qwen3VerbatimContentMatchesHf) {
    const std::string head = kQwen3Head;
    EXPECT_EQ(render_continued(chat_goldens::k_chatml_template, "The answer is"), head + "The answer is");
    EXPECT_EQ(render_continued(chat_goldens::k_chatml_template, "The answer is "), head + "The answer is ");
    EXPECT_EQ(render_continued(chat_goldens::k_chatml_template, "  Sure:\n"), head + "  Sure:\n");
}

// Off (the default) renders the closed turn plus the generation prompt, as before.
TEST(ChatTemplateContinue, OffKeepsTheClosedTurn) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl;
    ASSERT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok, chat_goldens::k_chatml_template));
    const std::string out = tpl.render_jinja(tok, prefilled("The answer is"), true);
    EXPECT_NE(out.find("The answer is<|im_end|>\n<|im_start|>assistant\n"), std::string::npos) << out;
}

// No assistant turn last: continuation is a no-op.
TEST(ChatTemplateContinue, UserLastIsUnchanged) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl;
    ASSERT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok, chat_goldens::k_chatml_template));
    const std::vector<ChatMessage> msgs = {{"user", "What is 17*23?"}};
    const std::string off = tpl.render_jinja(tok, msgs, true);
    tpl.set_continue_final(true);
    EXPECT_EQ(tpl.render_jinja(tok, msgs, true), off);
}

// Hardcoded family (no Jinja): history with the generation prompt, then the prefix text.
TEST(ChatTemplateContinue, HardcodedFamilyAppendsAfterGenerationPrompt) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl;
    ASSERT_TRUE(tpl.init(ChatTemplateFamily::CHATML, tok));
    ASSERT_FALSE(tpl.has_jinja());
    std::vector<int32_t> expect = tpl.apply(tok, {{"user", "What is 17*23?"}});
    const std::vector<int32_t> text = tok.encode("The answer is");
    expect.insert(expect.end(), text.begin(), text.end());
    tpl.set_continue_final(true);
    EXPECT_EQ(tpl.apply(tok, prefilled("The answer is")), expect);
}

TEST(ChatTemplateContinue, CutRule) {
    EXPECT_EQ(ChatTemplate::cut_after_final_content("a X b X<e>", "X"), "a X b X");
    EXPECT_EQ(ChatTemplate::cut_after_final_content("a X <e>", " X "), "a X ");
    EXPECT_EQ(ChatTemplate::cut_after_final_content("a X<e>", "X\n"), "a X");
    EXPECT_EQ(ChatTemplate::cut_after_final_content("a b<e>", "X"), "");
    EXPECT_EQ(ChatTemplate::cut_after_final_content("a b<e>", " \n"), "");
}

}  // namespace
}  // namespace imp
