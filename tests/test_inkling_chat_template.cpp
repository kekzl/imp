// Inkling2 chat template through imp's Jinja engine vs HF transformers render_jinja_template, goldens
// from tests/refs/gen_inkling_template_golden.py: the thinking-effort line for every named level and
// for numbers, and what the template advertises as /v1/models meta.reasoning_effort.

#include "model/chat_template.h"
#include "refs/inkling_template_golden.h"

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

class InklingChatTemplate : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(tpl_.init(ChatTemplateFamily::RAW, tok_, inkling_golden::k_chat_template_jinja));
        ASSERT_TRUE(tpl_.has_jinja()) << "Inkling2 template must parse in imp's Jinja engine";
    }
    std::string render(const std::string& effort,
                       std::vector<ChatMessage> msgs = {{"user", "What is 17*23?"}}) {
        return tpl_.render_jinja(tok_, msgs, /*add_generation_prompt=*/true, /*suppress_thinking=*/false,
                                 /*force_thinking=*/false, effort);
    }
    Tokenizer tok_ = make_tokenizer();
    ChatTemplate tpl_;
};

TEST_F(InklingChatTemplate, AdvertisesItsLevelsAndDefault) {
    EXPECT_EQ(tpl_.reasoning_effort_values(),
              (std::vector<std::string>{"none", "minimal", "low", "medium", "high", "xhigh", "max"}));
    EXPECT_EQ(tpl_.reasoning_effort_default(), "high");
}

TEST_F(InklingChatTemplate, NamedLevelsRenderTheirNumber) {
    EXPECT_EQ(render(""), inkling_golden::kDefault);
    EXPECT_EQ(render("none"), inkling_golden::kNone);
    EXPECT_EQ(render("minimal"), inkling_golden::kMinimal);
    EXPECT_EQ(render("low"), inkling_golden::kLow);
    EXPECT_EQ(render("medium"), inkling_golden::kMedium);
    EXPECT_EQ(render("high"), inkling_golden::kHigh);
    EXPECT_EQ(render("xhigh"), inkling_golden::kXhigh);
    EXPECT_EQ(render("max"), inkling_golden::kMax);
    EXPECT_EQ(render("low", {{"system", "Be brief."}, {"user", "What is 17*23?"}}),
              inkling_golden::kSystemFirst);
}

// A number reaches the template as a number: as a string it is no key of the map and the line is empty.
TEST_F(InklingChatTemplate, NumbersRenderAsNumbers) {
    EXPECT_EQ(render("0.5"), inkling_golden::kNumberHalf);
    EXPECT_EQ(render("0.25"), inkling_golden::kNumberQuarter);
    EXPECT_EQ(render("0"), inkling_golden::kNumberZero);
}

TEST_F(InklingChatTemplate, RefusesWhatTheTemplateRaisesOn) {
    for (const char* ok : {"", "none", "max", "0", "0.0", "0.5", "0.99"})
        EXPECT_EQ(tpl_.reasoning_effort_error(ok), "") << ok;
    for (const char* bad : {"weird", "1", "1.5", "-0.1", "0.5x", "High"})
        EXPECT_NE(tpl_.reasoning_effort_error(bad), "") << bad;
    EXPECT_NE(tpl_.reasoning_effort_error("weird").find("none, minimal, low, medium, high, xhigh, max"),
              std::string::npos);
}

// A template with a plain list or none keeps taking any value: its own branch decides.
TEST(InklingEffortSpec, OtherTemplatesTakeAnyValue) {
    Tokenizer tok = make_tokenizer();
    ChatTemplate tpl;
    ASSERT_TRUE(
        tpl.init(ChatTemplateFamily::RAW, tok,
                 "{% if reasoning_effort in ('high', 'low') %}x{% endif %}{{ messages[0].content }}"));
    EXPECT_EQ(tpl.reasoning_effort_values(), (std::vector<std::string>{"high", "low"}));
    EXPECT_EQ(tpl.reasoning_effort_error("weird"), "");
    EXPECT_EQ(tpl.reasoning_effort_error("7"), "");
}

}  // namespace
}  // namespace imp
