// Registry fields the golden (test_tool_call_dialect_golden.cpp) cannot see: the engine-side gate
// pair, the handler-side JSON envelope literals, XML body, native token and tool-response placement.
// Expected values = the per-family switches they replaced (main cdf37707, #2464).

#include <gtest/gtest.h>
#include "tool_call_dialect.h"
#include "model/chat_template.h"

#include <string>

using imp::ChatTemplateFamily;

namespace {

struct Expect {
    ChatTemplateFamily family;
    const char* dialect;
    const char* gate_open;
    const char* gate_close;
    bool gate_tokens;
    bool json_envelope;
    bool xml_body;
    const char* native_token;
    bool joins_assistant;
};

// constraint_manager.cpp resolve_tool_dialect switch, handlers_chat_core.cpp envelope literals and
// CHATML xml check, gemma_native_tool_call_ token, handlers_chat_params.cpp GEMMA tool glue.
constexpr Expect kExpect[] = {
    {ChatTemplateFamily::RAW, "chatml-hint", "<tool_call>", "</tool_call>", true, false, false, nullptr,
     false},
    {ChatTemplateFamily::CHATML, "chatml", "<tool_call>", "</tool_call>", true, true, true, nullptr, false},
    {ChatTemplateFamily::LLAMA2, "chatml-hint", "<tool_call>", "</tool_call>", true, false, false, nullptr,
     false},
    {ChatTemplateFamily::MISTRAL_V3, "chatml-hint", "<tool_call>", "</tool_call>", true, false, false,
     nullptr, false},
    {ChatTemplateFamily::LLAMA3, "llama3", "<function=", "</function>", false, false, false, nullptr, false},
    {ChatTemplateFamily::NEMOTRON, "chatml-hint", "<tool_call>", "</tool_call>", true, false, false, nullptr,
     false},
    {ChatTemplateFamily::GEMMA, "gemma", "<|tool_call>", "<tool_call|>", true, false, false, "<|tool_call>",
     true},
    {ChatTemplateFamily::DEEPSEEK_R1, "chatml-hint", "<tool_call>", "</tool_call>", true, false, false,
     nullptr, false},
    {ChatTemplateFamily::PHI, "chatml-hint", "<tool_call>", "</tool_call>", true, false, false, nullptr,
     false},
    {ChatTemplateFamily::HARMONY, "harmony", "<tool_call>", "</tool_call>", true, false, false, nullptr,
     false},
};

}  // namespace

TEST(ToolCallDialectRegistry, EveryFamilyMatchesThePreRegistrySwitches) {
    for (const Expect& e : kExpect) {
        SCOPED_TRACE(imp::chat_template_family_name(e.family));
        const ToolCallDialect& d = tool_call_dialect(e.family);
        EXPECT_STREQ(d.name, e.dialect);
        const imp::ToolCallGate g = tool_call_gate(e.family);
        EXPECT_EQ(g.open, e.gate_open);
        EXPECT_EQ(g.close, e.gate_close);
        EXPECT_EQ(g.tokens, e.gate_tokens);
        EXPECT_EQ(d.json_envelope_open != nullptr, e.json_envelope);
        if (e.json_envelope) {
            EXPECT_STREQ(d.json_envelope_open, "<tool_call>\n");
            EXPECT_STREQ(d.json_envelope_close, "\n</tool_call>");
        }
        EXPECT_EQ(d.xml_body, e.xml_body);
        EXPECT_EQ(std::string(d.native_token ? d.native_token : ""),
                  std::string(e.native_token ? e.native_token : ""));
        EXPECT_EQ(d.tool_response_joins_assistant, e.joins_assistant);
        EXPECT_NE(d.prompt, nullptr);
        EXPECT_NE(d.parse, nullptr);
        EXPECT_NE(d.scan, nullptr);
        EXPECT_NE(d.render_call, nullptr);
    }
}

// Request default (no server, e.g. C API) keeps the pre-registry CHATML gate.
TEST(ToolCallDialectRegistry, DefaultGateIsChatml) {
    const imp::ToolCallGate g;
    EXPECT_EQ(g.open, "<tool_call>");
    EXPECT_EQ(g.close, "</tool_call>");
    EXPECT_TRUE(g.tokens);
}
