// test(P2.7): gpt-oss Harmony chat-template golden parity vs HF.
//
// gpt-oss-20b ships a Harmony chat_template.jinja. imp renders it through its
// own jinja engine (src/model/jinja.cpp) in production (init() sets
// use_jinja_ for any parseable template). Until now only a token-level
// template SMOKE existed (test_chat_template.cpp) — no string-level parity
// against the REAL HF tokenizer output.
//
// This test renders the COMMITTED chat_template.jinja (embedded verbatim in
// tests/refs/harmony_template_jinja.h, copied from
// tests/fixtures/gpt_oss_chat_template.jinja) through imp's jinja engine,
// building the context exactly as ChatTemplate::apply_jinja does, and
// compares the rendered string against goldens produced by the HF reference
// (transformers AutoTokenizer.apply_chat_template; see
// tests/refs/gen_harmony_golden.py + tests/refs/harmony_golden.h).
//
// Channel markers (<|start|>, <|channel|>, <|message|>, <|end|>), the system
// preamble, the developer-role mapping for user system messages, and the
// assistant-history `final` channel are all asserted by EXACT string compare.
//
// DOCUMENTED, INTENTIONAL normalization: the Harmony template injects
// `Current date: <today>` via strftime_now (line 202). That is inherently
// non-deterministic, so BOTH the golden and the imp render have that one line
// normalized to a placeholder before comparison. Everything else — every
// channel marker and structural byte — is compared exactly. If imp's render
// diverges anywhere else, the test fails loudly (and any such divergence is a
// real jinja-engine / parity bug to report, not to fudge).

#include "model/jinja.h"
#include "refs/harmony_golden.h"
#include "refs/harmony_template_jinja.h"

#include <gtest/gtest.h>
#include <regex>
#include <string>
#include <vector>

namespace imp {
namespace {

// Normalize the single non-deterministic line so the rest is an exact compare.
std::string normalize_date(const std::string& s) {
    return std::regex_replace(s, std::regex("Current date: [0-9]{4}-[0-9]{2}-[0-9]{2}"),
                              "Current date: <DATE>");
}

// Build the jinja context exactly as ChatTemplate::apply_jinja does for a
// messages-only conversation (add_generation_prompt=true, enable_thinking
// left undefined). bos/eos are empty: the Harmony template does not consume
// them, and the HF golden has no BOS prefix.
jinja::Value::Array make_messages(const std::vector<std::pair<std::string, std::string>>& msgs) {
    jinja::Value::Array arr;
    for (const auto& m : msgs)
        arr.push_back(jinja::Value::object({{"role", jinja::Value(m.first)},
                                            {"content", jinja::Value(m.second)}}));
    return arr;
}

std::string render(const std::string& tpl_src,
                   const std::vector<std::pair<std::string, std::string>>& msgs) {
    jinja::Template tpl;
    EXPECT_TRUE(tpl.parse(tpl_src)) << "jinja parse failed: " << tpl.error();
    jinja::Context ctx;
    ctx["messages"] = jinja::Value(make_messages(msgs));
    ctx["add_generation_prompt"] = jinja::Value(true);
    ctx["bos_token"] = jinja::Value(std::string(""));
    ctx["eos_token"] = jinja::Value(std::string(""));
    std::string out = tpl.render(ctx);
    EXPECT_FALSE(out.empty()) << "jinja render empty: " << tpl.error();
    return out;
}

class HarmonyGoldenTest : public ::testing::Test {
protected:
    void SetUp() override {
        tpl_src_ = harmony_golden::k_chat_template_jinja;
        if (tpl_src_.empty())
            GTEST_SKIP() << "embedded Harmony template empty";
    }
    std::string tpl_src_;
};

TEST_F(HarmonyGoldenTest, UserOnly) {
    std::string got = render(tpl_src_, {{"user", "What is the capital of France?"}});
    EXPECT_EQ(normalize_date(got), normalize_date(harmony_golden::k_user_only));
}

TEST_F(HarmonyGoldenTest, SystemUser) {
    std::string got = render(tpl_src_, {{"system", "You are a terse assistant."},
                                        {"user", "Hello there."}});
    EXPECT_EQ(normalize_date(got), normalize_date(harmony_golden::k_system_user));
}

TEST_F(HarmonyGoldenTest, MultiTurnAssistantFinalChannel) {
    std::string got = render(tpl_src_, {{"user", "Hi"},
                                        {"assistant", "Hello! How can I help?"},
                                        {"user", "What is 2+2?"}});
    EXPECT_EQ(normalize_date(got), normalize_date(harmony_golden::k_multi_turn));
}

// Structural guard independent of date: the channel markers must all be
// present in imp's render of a multi-turn conversation.
TEST_F(HarmonyGoldenTest, ChannelMarkersPresent) {
    std::string got = render(tpl_src_, {{"user", "Hi"},
                                        {"assistant", "Hello!"},
                                        {"user", "Bye"}});
    for (const char* marker : {"<|start|>system<|message|>", "<|end|>", "<|start|>user<|message|>",
                               "<|start|>assistant<|channel|>final<|message|>", "<|start|>assistant"}) {
        EXPECT_NE(got.find(marker), std::string::npos) << "missing Harmony marker: " << marker;
    }
}

}  // namespace
}  // namespace imp
