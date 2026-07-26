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
    return json::array(
        {{{"type", "function"},
          {"function",
           {{"name", "get_weather"},
            {"description", "Get weather"},
            {"parameters",
             {{"type", "object"},
              {"properties", {{"city", {{"type", "string"}}}, {"days", {{"type", "integer"}}}}},
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

// Llama 3.2 drops the <function=> envelope and emits a bare JSON object. Found
// by the cross-engine agentic comparison: the model and the grammar were
// correct, but the call came back as `content`, so an agent saw no tool call.
TEST(ToolCallLlama3, BareJsonObjectIsACall) {
    std::atomic<int> id(0);
    std::string text = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Berlin\"}}";
    auto [content, calls] = parse_tool_calls_llama3(text, id, {"get_weather"});
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(json::parse(calls[0].arguments)["city"], "Berlin");
    EXPECT_TRUE(content.empty());
    EXPECT_FALSE(calls[0].id.empty());
}

TEST(ToolCallLlama3, BareJsonAcceptsArgumentsKeyToo) {
    std::atomic<int> id(0);
    std::string text = "  {\"name\": \"f\", \"arguments\": {\"a\": 1}}  ";
    auto [content, calls] = parse_tool_calls_llama3(text, id, {"f"});
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "f");
    EXPECT_EQ(json::parse(calls[0].arguments)["a"], 1);
}

// The strictness matters: a plain JSON answer must NOT become a tool call.
TEST(ToolCallLlama3, PlainJsonAnswerIsNotACall) {
    std::atomic<int> id(0);
    for (const char* text : {
             "{\"name\": \"Alice\", \"age\": 30}",                  // name, but no params object
             "{\"parameters\": {\"city\": \"Berlin\"}}",            // params, but no name
             "{\"name\": \"\", \"parameters\": {}}",                // empty name
             "{\"name\": \"f\", \"parameters\": \"str\"}",          // parameters not an object
             "[{\"name\": \"f\", \"parameters\": {}}]",             // not an object
             "here you go: {\"name\": \"f\", \"parameters\": {}}",  // not bare
         }) {
        auto [content, calls] = parse_tool_calls_llama3(text, id, {"f", "get_weather"});
        EXPECT_EQ(calls.size(), 0u) << "should not be a tool call: " << text;
        EXPECT_EQ(content, text) << "content must pass through: " << text;
    }
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
        auto [c, calls] = parse_tool_calls(ChatTemplateFamily::LLAMA3, "<function=f>{\"a\":1}</function>",
                                           id);
        ASSERT_EQ(calls.size(), 1u);
        EXPECT_EQ(calls[0].name, "f");
    }
    {
        std::atomic<int> id(0);
        auto [c, calls] = parse_tool_calls(ChatTemplateFamily::GEMMA, "<|tool_call>call:g{x:1}<tool_call|>",
                                           id);
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

// ---------------------------------------------------------------------------
// Qwen3.6 XML fallback — <function=NAME><parameter=KEY>VAL</parameter></function>
// The distinct parse_qwen36_xml_call path had no direct coverage.
// ---------------------------------------------------------------------------
TEST(ToolCallQwen36Xml, NoFunctionTagReturnsFalse) {
    ParsedToolCall tc;
    EXPECT_FALSE(parse_qwen36_xml_call("just some text", tc));
}

TEST(ToolCallQwen36Xml, SingleStringParam) {
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=get_weather><parameter=city>Paris</parameter></function>", tc));
    EXPECT_EQ(tc.name, "get_weather");
    json args = json::parse(tc.arguments);
    EXPECT_EQ(args["city"], "Paris");
}

TEST(ToolCallQwen36Xml, CoercesScalarTypes) {
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=f>"
                              "<parameter=s>hello</parameter>"
                              "<parameter=i>42</parameter>"
                              "<parameter=d>3.5</parameter>"
                              "<parameter=b>true</parameter>"
                              "<parameter=n>null</parameter>"
                              "</function>",
                              tc));
    json args = json::parse(tc.arguments);
    EXPECT_TRUE(args["s"].is_string());
    EXPECT_EQ(args["s"], "hello");
    EXPECT_TRUE(args["i"].is_number_integer());
    EXPECT_EQ(args["i"], 42);
    EXPECT_TRUE(args["d"].is_number_float());
    EXPECT_DOUBLE_EQ(args["d"].get<double>(), 3.5);
    EXPECT_EQ(args["b"], true);
    EXPECT_TRUE(args["n"].is_null());
}

TEST(ToolCallQwen36Xml, NameIsTrimmed) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call("<function= spaced_name >\n</function>", tc));
    EXPECT_EQ(tc.name, "spaced_name");
}

TEST(ToolCallQwen36Xml, EmptyNameReturnsFalse) {
    ParsedToolCall tc;
    EXPECT_FALSE(parse_qwen36_xml_call("<function=   ></function>", tc));
}

TEST(ToolCallQwen36Xml, MissingCloseFunctionTagStillParses) {
    // No </function> — the parser scans to end of body.
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call("<function=f><parameter=k>v</parameter>", tc));
    EXPECT_EQ(tc.name, "f");
    EXPECT_EQ(json::parse(tc.arguments)["k"], "v");
}

TEST(ToolCallQwen36Xml, NoParamsGivesEmptyArgs) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call("<function=ping></function>", tc));
    EXPECT_EQ(tc.name, "ping");
    EXPECT_EQ(json::parse(tc.arguments), json::object());
}

// ---------------------------------------------------------------------------
// format_tool_response — render a role=tool message into the family wire format
// ---------------------------------------------------------------------------
TEST(ToolResponseFormat, Llama3AndChatMLPassContentThrough) {
    json msg = {{"content", "sunny"}};
    EXPECT_EQ(format_tool_response(ChatTemplateFamily::LLAMA3, msg), "sunny");
    EXPECT_EQ(format_tool_response(ChatTemplateFamily::CHATML, msg), "sunny");
}

TEST(ToolResponseFormat, NonStringContentSerialized) {
    // A structured payload must be serialized, not dropped to "".
    json msg = {{"content", json{{"temp", 20}}}};
    std::string out = format_tool_response(ChatTemplateFamily::LLAMA3, msg);
    EXPECT_EQ(json::parse(out), (json{{"temp", 20}}));
}

TEST(ToolResponseFormat, NullOrAbsentContentIsEmpty) {
    EXPECT_EQ(format_tool_response(ChatTemplateFamily::CHATML, json{{"content", nullptr}}), "");
    EXPECT_EQ(format_tool_response(ChatTemplateFamily::CHATML, json::object()), "");
}

TEST(ToolResponseFormat, GemmaWrapsWithNameAndQuotes) {
    json msg = {{"content", "sunny"}, {"name", "get_weather"}};
    EXPECT_EQ(format_tool_response(ChatTemplateFamily::GEMMA, msg),
              "<|tool_response>response:get_weather{value:<|\"|>sunny<|\"|>}<tool_response|>");
}

TEST(ToolResponseFormat, GemmaDefaultsNameToTool) {
    std::string s = format_tool_response(ChatTemplateFamily::GEMMA, json{{"content", "x"}});
    EXPECT_NE(s.find("response:tool{"), std::string::npos);
}

// ---------------------------------------------------------------------------
// reconstruct_tool_call_output — rebuild the assistant tool-call wire text
// ---------------------------------------------------------------------------
TEST(ToolCallReconstruct, LeadingContentPreserved) {
    json calls = json::array();
    EXPECT_EQ(reconstruct_tool_call_output(ChatTemplateFamily::CHATML, calls, "hello"), "hello");
    // "null" content is treated as empty.
    EXPECT_EQ(reconstruct_tool_call_output(ChatTemplateFamily::CHATML, calls, "null"), "");
}

TEST(ToolCallReconstruct, ChatMLWrapsParsedCall) {
    json calls = json::array({{{"function", {{"name", "foo"}, {"arguments", "{\"a\":1}"}}}}});
    std::string out = reconstruct_tool_call_output(ChatTemplateFamily::CHATML, calls, "");
    // Extract and re-parse the inner JSON to avoid key-order/spacing fragility.
    auto open = out.find("<tool_call>\n");
    auto close = out.find("\n</tool_call>");
    ASSERT_NE(open, std::string::npos);
    ASSERT_NE(close, std::string::npos);
    std::string inner = out.substr(open + 12, close - (open + 12));
    json parsed = json::parse(inner);
    EXPECT_EQ(parsed["name"], "foo");
    EXPECT_EQ(parsed["arguments"], (json{{"a", 1}}));
}

TEST(ToolCallReconstruct, Llama3WrapsCall) {
    json calls = json::array({{{"function", {{"name", "foo"}, {"arguments", "{\"a\":1}"}}}}});
    std::string out = reconstruct_tool_call_output(ChatTemplateFamily::LLAMA3, calls, "");
    EXPECT_NE(out.find("<function=foo>"), std::string::npos);
    EXPECT_NE(out.find("</function>"), std::string::npos);
}

TEST(ToolCallReconstruct, GemmaWrapsCall) {
    json calls = json::array({{{"function", {{"name", "foo"}, {"arguments", "{\"a\":1}"}}}}});
    std::string out = reconstruct_tool_call_output(ChatTemplateFamily::GEMMA, calls, "");
    EXPECT_NE(out.find("<|tool_call>call:foo{"), std::string::npos);
    EXPECT_NE(out.find("}<tool_call|>"), std::string::npos);
}

TEST(ToolCallReconstruct, SkipsEntryWithoutFunction) {
    json calls = json::array({{{"id", "x"}}});  // no "function" key
    EXPECT_EQ(reconstruct_tool_call_output(ChatTemplateFamily::CHATML, calls, "keep"), "keep");
}

// Qwen-Coder XML dialect (xml=true): prior calls replay in the shape the
// template's own tool_calls branch renders — raw multi-line string values,
// non-strings stringified — never the ChatML JSON body (a JSON replay teaches
// the model the wrong dialect for its next call).
TEST(ToolCallReconstruct, XmlDialectWrapsCall) {
    json calls = json::array(
        {{{"function",
           {{"name", "write_file"},
            {"arguments", "{\"path\":\"/tmp/x.py\",\"content\":\"line1\\nline2\",\"limit\":3}"}}}}});
    std::string out = reconstruct_tool_call_output(ChatTemplateFamily::CHATML, calls, "", /*xml=*/true);
    EXPECT_NE(out.find("<tool_call>\n<function=write_file>\n"), std::string::npos);
    EXPECT_NE(out.find("<parameter=path>\n/tmp/x.py\n</parameter>\n"), std::string::npos);
    // Raw value — the newline stays a newline, not an escape.
    EXPECT_NE(out.find("<parameter=content>\nline1\nline2\n</parameter>\n"), std::string::npos);
    EXPECT_NE(out.find("<parameter=limit>\n3\n</parameter>\n"), std::string::npos);
    EXPECT_NE(out.find("</function>\n</tool_call>"), std::string::npos);
    EXPECT_EQ(out.find("{\"name\""), std::string::npos) << "no JSON body on the XML dialect";
}

// The enforced grammar only ends a value at "\n</parameter>" — a raw value may
// legally CONTAIN a bare close tag (code writing about tool calls). The parser
// must anchor on the newline form first and keep such text inside the value.
TEST(ToolCallQwen36Xml, BareCloseTagInsideValueStaysValueText) {
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=f>\n"
                              "<parameter=doc>\n"
                              "call format: </parameter> and </function> inline\n"
                              "</parameter>\n"
                              "<parameter=n>\n42\n</parameter>\n"
                              "</function>",
                              tc));
    EXPECT_EQ(tc.name, "f");
    json args = json::parse(tc.arguments);
    EXPECT_EQ(args["doc"], "call format: </parameter> and </function> inline");
    EXPECT_EQ(args["n"], 42);
}

// A hallucinated function name must never become a call. Llama-3.2-3B answers a
// plain chat turn with {"name":"print",...} when tools are in context; without
// this gate that fabricates a tool call the caller never offered.
TEST(ToolCallLlama3, UnknownFunctionNameIsNotACall) {
    std::atomic<int> id(0);
    std::string text = "{\"name\": \"print\", \"parameters\": {\"text\": \"hello\"}}";
    auto [content, calls] = parse_tool_calls_llama3(text, id, {"get_weather"});
    EXPECT_EQ(calls.size(), 0u);
    EXPECT_EQ(content, text);
    // And with no names known at all, the bare-JSON form stays off entirely.
    auto [c2, calls2] = parse_tool_calls_llama3(text, id);
    EXPECT_EQ(calls2.size(), 0u);
    EXPECT_EQ(c2, text);
}

TEST(ToolNamesFromRequest, ExtractsOpenAIAndBareShapes) {
    EXPECT_EQ(tool_names_from_request(weather_tools()), (std::vector<std::string>{"get_weather"}));
    EXPECT_TRUE(tool_names_from_request(json::array()).empty());
    EXPECT_TRUE(tool_names_from_request(json("not an array")).empty());
}

// A small model asked for one call can emit several, "; "-separated. Parsing the
// whole string fails; the first balanced object is the call. Brace counting is
// string-aware so a '}' inside a value does not terminate it early.
TEST(ToolCallLlama3, TakesFirstOfSeveralConcatenatedObjects) {
    std::atomic<int> id(0);
    std::string text =
        "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Berlin\", \"unit\": \"c\"}}; "
        "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Berlin\", \"unit\": \"f\"}}";
    auto [content, calls] = parse_tool_calls_llama3(text, id, {"get_weather"});
    ASSERT_EQ(calls.size(), 1u);
    json args = json::parse(calls[0].arguments);
    EXPECT_EQ(args["city"], "Berlin");
    EXPECT_EQ(args["unit"], "c");
}

TEST(ToolCallLlama3, BraceInsideStringDoesNotEndTheObject) {
    std::atomic<int> id(0);
    std::string text = "{\"name\": \"f\", \"parameters\": {\"q\": \"a } b\", \"n\": 1}}";
    auto [content, calls] = parse_tool_calls_llama3(text, id, {"f"});
    ASSERT_EQ(calls.size(), 1u);
    json args = json::parse(calls[0].arguments);
    EXPECT_EQ(args["q"], "a } b");
    EXPECT_EQ(args["n"], 1);
}
