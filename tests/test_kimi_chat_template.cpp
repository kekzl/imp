// Kimi-Linear chat template through imp's Jinja engine vs HF transformers render_jinja_template
// (what apply_chat_template calls), goldens from tests/refs/gen_kimi_template_golden.py.
// Whole rendered prompt compared (tests/CLAUDE.md: goldens, not markers).

#include "model/chat_template.h"
#include "refs/kimi_template_golden.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>

namespace imp {
namespace {

Tokenizer make_tokenizer() {
    std::vector<std::string> tokens = {"<unk>", "[BOS]", "[EOS]"};
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

class KimiChatTemplate : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(tpl_.init(ChatTemplateFamily::RAW, tok_, kimi_golden::k_chat_template_jinja));
        ASSERT_TRUE(tpl_.has_jinja()) << "Kimi-Linear template must parse in imp's Jinja engine";
    }
    Tokenizer tok_ = make_tokenizer();
    ChatTemplate tpl_;
};

TEST_F(KimiChatTemplate, OneTurn) {
    EXPECT_EQ(tpl_.render_jinja(tok_, {{"user", "Hello"}}), kimi_golden::kOneTurn);
}

TEST_F(KimiChatTemplate, MultiTurnWithSystem) {
    const std::vector<ChatMessage> msgs = {{"system", "You are a helpful assistant."},
                                           {"user", "Hi"},
                                           {"assistant", "Hello! How can I help?"},
                                           {"user", "What is 2+2?"}};
    EXPECT_EQ(tpl_.render_jinja(tok_, msgs), kimi_golden::kMultiTurnSystem);
}

// Structured tool_calls (arguments as an object, tojson) and a tool result message.
TEST_F(KimiChatTemplate, ToolCallHistory) {
    std::vector<ChatMessage> msgs(3);
    msgs[0].role = "user";
    msgs[0].content = "Weather in Paris?";
    msgs[1].role = "assistant";
    msgs[1].tool_calls_json =
        R"([{"id":"functions.get_weather:0","type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}])";
    msgs[2].role = "tool";
    msgs[2].tool_call_id = "functions.get_weather:0";
    msgs[2].content = "18 C, sunny";
    EXPECT_EQ(tpl_.render_jinja(tok_, msgs), kimi_golden::kToolHistory);
}

}  // namespace
}  // namespace imp
