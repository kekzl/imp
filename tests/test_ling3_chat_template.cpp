// Ling-3.0 (bailing_hybrid) chat template through imp's Jinja engine vs jinja2 (HF
// apply_chat_template semantics), goldens from tests/refs/gen_ling3_template_golden.py.
// Whole rendered prompt compared (tests/CLAUDE.md: goldens, not markers).

#include "model/chat_template.h"
#include "refs/ling3_template_golden.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>

namespace imp {
namespace {

Tokenizer make_tokenizer() {
    std::vector<std::string> tokens = {"<unk>", "<|startoftext|>", "<|role_end|>"};
    std::vector<float> scores(tokens.size(), 0.0f);
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.emplace_back(buf);
        scores.push_back(-10.0f);
    }
    Tokenizer tok;
    EXPECT_TRUE(tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2));
    tok.set_type("spm");
    tok.set_add_bos(false);
    tok.set_add_space_prefix(false);
    return tok;
}

class Ling3ChatTemplate : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(tpl_.init(ChatTemplateFamily::RAW, tok_, ling3_golden::k_chat_template_jinja));
        ASSERT_TRUE(tpl_.has_jinja()) << "Ling-3.0 template must parse in imp's Jinja engine";
    }
    Tokenizer tok_ = make_tokenizer();
    ChatTemplate tpl_;
};

TEST_F(Ling3ChatTemplate, OneTurnThinkingDefault) {
    EXPECT_EQ(tpl_.render_jinja(tok_, {{"user", "Hello"}}), ling3_golden::kOneTurnDefault);
}

TEST_F(Ling3ChatTemplate, MultiTurnThinkingOff) {
    const std::vector<ChatMessage> msgs = {{"system", "You are a helpful assistant."},
                                           {"user", "Hi"},
                                           {"assistant", "Hello! How can I help?"},
                                           {"user", "What is 2+2?"}};
    EXPECT_EQ(tpl_.render_jinja(tok_, msgs, true, /*suppress_thinking=*/true),
              ling3_golden::kMultiTurnNoThink);
}

TEST_F(Ling3ChatTemplate, OneTurnThinkingForced) {
    EXPECT_EQ(tpl_.render_jinja(tok_, {{"user", "Hi"}}, true, false, /*force_thinking=*/true),
              ling3_golden::kOneTurnThink);
}

}  // namespace
}  // namespace imp
