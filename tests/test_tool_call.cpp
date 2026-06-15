// =============================================================================
// Unit tests for tools/imp-server/tool_call.cpp — TEST_AUDIT.md §7 Tier-2.
//
// WHY THIS EXISTS: tool_call.cpp (parse_tool_calls_{chatml,llama3,gemma},
// validate_tool_call) had ZERO unit tests — tool_call.h was included by no test.
// The only coverage was the mark-excluded real-server test_tools.py, which does
// NOT validate arguments against the schema. A wrong arg-parse or a validation
// bypass is an agent-correctness/security surface (the model's tool calls drive
// side-effecting actions), and this is CPU-only so it runs in CI where the real
// handlers.cpp does not. These assert the CURRENT parser/validator contract.
//
// ORACLE: hand-constructed inputs in each family's documented wire format, with
// the expected (name, arguments-JSON, id, valid) spelled out — no imp-vs-imp.
// =============================================================================

#include <gtest/gtest.h>
#include "tool_call.h"
#include "model/chat_template.h"

#include <atomic>
#include <string>

using imp::ChatTemplateFamily;

namespace {

// OpenAI-shaped tools array with one function + a JSON-schema for its params.
json weather_tools() {
    return json::array({{{"type", "function"},
                         {"function",
                          {{"name", "get_weather"},
                           {"description", "Get weather"},
                           {"parameters",
                            {{"type", "object"},
                             {"properties",
                              {{"city", {{"type", "string"}}}, {"days", {{"type", "integer"}}}}},
                             {"required", json::array({"city"})}}}}}}});
}

}  // namespace

// ---------------------------------------------------------------------------
// ChatML (Qwen3, Hermes) — <tool_call>{json}</tool_call>
// ---------------------------------------------------------------------------
TEST(ToolCallChatML, SingleCallNameAndArgs) {
    std::atomic<int> id(0);
    std::string text =
        "Sure, let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": "
        "\"Paris\"}}\n</tool_call>";
    auto [content, calls] = parse_tool_calls_chatml(text, id);
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].id, "call_imp_0");
    // arguments is the serialized arguments object.
    json args = json::parse(calls[0].arguments);
    EXPECT_EQ(args["city"], "Paris");
    // Content is the prose before the first tag, trailing whitespace trimmed.
    EXPECT_EQ(content, "Sure, let me check.");
}

TEST(ToolCallChatML, MultipleCallsGetSequentialIds) {
    std::atomic<int> id(0);
    std::string text =
        "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>"
        "<tool_call>{\"name\": \"b\", \"arguments\": {\"x\": 1}}</tool_call>";
    auto [content, calls] = parse_tool_calls_chatml(text, id);
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].name, "a");
    EXPECT_EQ(calls[0].id, "call_imp_0");
    EXPECT_EQ(calls[1].name, "b");
    EXPECT_EQ(calls[1].id, "call_imp_1");
}

TEST(ToolCallChatML, NoToolCallReturnsFullContent) {
    std::atomic<int> id(0);
    std::string text = "Just a normal answer, no tools.";
    auto [content, calls] = parse_tool_calls_chatml(text, id);
    EXPECT_EQ(calls.size(), 0u);
    EXPECT_EQ(content, text);
}

TEST(ToolCallChatML, ArgumentsKeyAbsentUsesRestOfObject) {
    // No "arguments" key → everything except "name" becomes the arguments.
    std::atomic<int> id(0);
    std::string text = "<tool_call>{\"name\": \"get_weather\", \"city\": \"Rome\"}</tool_call>";
    auto [content, calls] = parse_tool_calls_chatml(text, id);
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    json args = json::parse(calls[0].arguments);
    EXPECT_EQ(args["city"], "Rome");
    EXPECT_FALSE(args.contains("name"));
}

TEST(ToolCallChatML, MalformedJsonBodyProducesNoCall) {
    // A body that is neither valid JSON nor the <function=...> fallback yields
    // no parsed call (and must not throw).
    std::atomic<int> id(0);
    std::string text = "<tool_call>{not valid json,,,}</tool_call>";
    auto [content, calls] = parse_tool_calls_chatml(text, id);
    EXPECT_EQ(calls.size(), 0u);
}

TEST(ToolCallChatML, DriftMissingCloseTagStillParses) {
    // Models sometimes emit a second opening <tool_call> instead of the close;
    // the parser treats either as the delimiter.
    std::atomic<int> id(0);
    std::string text =
        "<tool_call>{\"name\": \"a\", \"arguments\": {}}<tool_call>{\"name\": \"b\", "
        "\"arguments\": {}}</tool_call>";
    auto [content, calls] = parse_tool_calls_chatml(text, id);
    EXPECT_GE(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "a");
}

// ---------------------------------------------------------------------------
// Llama3 — <function=NAME>{json}</function>
// ---------------------------------------------------------------------------
TEST(ToolCallLlama3, SingleCall) {
    std::atomic<int> id(0);
    std::string text = "<function=get_weather>{\"city\": \"Berlin\", \"days\": 3}</function>";
    auto [content, calls] = parse_tool_calls_llama3(text, id);
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    json args = json::parse(calls[0].arguments);
    EXPECT_EQ(args["city"], "Berlin");
    EXPECT_EQ(args["days"], 3);
}

TEST(ToolCallLlama3, NoCallReturnsContent) {
    std::atomic<int> id(0);
    std::string text = "plain text";
    auto [content, calls] = parse_tool_calls_llama3(text, id);
    EXPECT_EQ(calls.size(), 0u);
    EXPECT_EQ(content, "plain text");
}

// ---------------------------------------------------------------------------
// Gemma — <|tool_call>call:NAME{key:value}<tool_call|>
// ---------------------------------------------------------------------------
TEST(ToolCallGemma, SingleCall) {
    // Gemma string values are wrapped in the kGemmaQuote sequence <|"|>...<|"|>,
    // not ASCII quotes; an integer value sidesteps that and still exercises
    // name + args parsing. (String quoting is covered indirectly via build/parse
    // round-trips elsewhere.)
    std::atomic<int> id(0);
    std::string text = "<|tool_call>call:get_weather{days:3}<tool_call|>";
    auto [content, calls] = parse_tool_calls_gemma(text, id);
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    json args = json::parse(calls[0].arguments);
    EXPECT_EQ(args["days"], 3);
}

TEST(ToolCallGemma, NoCallReturnsContent) {
    std::atomic<int> id(0);
    std::string text = "no tool here";
    auto [content, calls] = parse_tool_calls_gemma(text, id);
    EXPECT_EQ(calls.size(), 0u);
    EXPECT_EQ(content, "no tool here");
}

// ---------------------------------------------------------------------------
// Dispatch: parse_tool_calls(family, ...) routes to the right family parser.
// ---------------------------------------------------------------------------
TEST(ToolCallDispatch, RoutesByFamily) {
    {
        std::atomic<int> id(0);
        auto [c, calls] = parse_tool_calls(ChatTemplateFamily::LLAMA3,
                                           "<function=f>{\"a\":1}</function>", id);
        ASSERT_EQ(calls.size(), 1u);
        EXPECT_EQ(calls[0].name, "f");
    }
    {
        std::atomic<int> id(0);
        auto [c, calls] = parse_tool_calls(ChatTemplateFamily::GEMMA,
                                           "<|tool_call>call:g{x:1}<tool_call|>", id);
        ASSERT_EQ(calls.size(), 1u);
        EXPECT_EQ(calls[0].name, "g");
    }
    {
        std::atomic<int> id(0);
        auto [c, calls] = parse_tool_calls(ChatTemplateFamily::CHATML,
                                           "<tool_call>{\"name\":\"h\",\"arguments\":{}}</tool_call>", id);
        ASSERT_EQ(calls.size(), 1u);
        EXPECT_EQ(calls[0].name, "h");
    }
}

// ---------------------------------------------------------------------------
// validate_tool_call — the schema-conformance gate.
// ---------------------------------------------------------------------------
TEST(ToolCallValidate, ValidArgsStayValid) {
    ParsedToolCall tc;
    tc.name = "get_weather";
    tc.arguments = "{\"city\": \"Paris\", \"days\": 2}";
    validate_tool_call(tc, weather_tools());
    EXPECT_TRUE(tc.valid) << tc.error;
}

TEST(ToolCallValidate, MissingRequiredIsRejected) {
    ParsedToolCall tc;
    tc.name = "get_weather";
    tc.arguments = "{\"days\": 2}";  // missing required "city"
    validate_tool_call(tc, weather_tools());
    EXPECT_FALSE(tc.valid);
    EXPECT_NE(tc.error.find("city"), std::string::npos) << tc.error;
}

TEST(ToolCallValidate, WrongTypeIsRejected) {
    ParsedToolCall tc;
    tc.name = "get_weather";
    tc.arguments = "{\"city\": 123}";  // city should be string
    validate_tool_call(tc, weather_tools());
    EXPECT_FALSE(tc.valid);
    EXPECT_NE(tc.error.find("city"), std::string::npos) << tc.error;
}

TEST(ToolCallValidate, NonObjectArgumentsRejected) {
    ParsedToolCall tc;
    tc.name = "get_weather";
    tc.arguments = "[1,2,3]";  // not an object
    validate_tool_call(tc, weather_tools());
    EXPECT_FALSE(tc.valid);
}

TEST(ToolCallValidate, UnknownToolIsNoOp) {
    // No schema located → validation is a no-op, call stays valid.
    ParsedToolCall tc;
    tc.name = "nonexistent_tool";
    tc.arguments = "{\"anything\": true}";
    validate_tool_call(tc, weather_tools());
    EXPECT_TRUE(tc.valid);
}

TEST(ToolCallValidate, AdditionalPropertiesAllowed) {
    // A property not in the schema is not rejected (only declared types checked).
    ParsedToolCall tc;
    tc.name = "get_weather";
    tc.arguments = "{\"city\": \"Oslo\", \"extra\": 9}";
    validate_tool_call(tc, weather_tools());
    EXPECT_TRUE(tc.valid) << tc.error;
}
