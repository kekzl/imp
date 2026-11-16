// GLM-4.x <arg_key>/<arg_value> tool calls: parsing, schema typing as vLLM 0.33 glm47 (tool_arg_coerce)
// and the raw body under diagnostics.tool_fault_raw. Split from test_tool_call.cpp (file size gate).
#include <gtest/gtest.h>
#include "tool_arg_coerce.h"
#include "tool_call.h"

#include <atomic>
#include <fstream>
#include <string>

namespace {

// GLM-4.7-Flash body, as vLLM 0.33 glm47: without a tool schema every value stays the emitted text.
TEST(ToolCallGlm, ArgKeyBodyParsesThroughTheChatmlParser) {
    std::atomic<int> next_id{0};
    const std::string text =
        "Let me check.\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>units</arg_key>"
        "<arg_value>{\"temp\": \"C\"}</arg_value></tool_call>";
    auto [content, calls] = parse_tool_calls_chatml(text, next_id);
    EXPECT_EQ(content, "Let me check.");
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].name, "get_weather");
    const json args = json::parse(calls[0].arguments);
    EXPECT_EQ(args["city"], "Paris");
    EXPECT_EQ(args["days"], "3");
    EXPECT_EQ(args["units"], "{\"temp\": \"C\"}");
}

// diagnostics.tool_fault_raw: a GLM call that fails validation carries its emitted body into the fault line
// (familiar on Flint: edit_file without old_string logged no raw text, raw_body was XML-only).
TEST(ToolCallGlm, FaultLineCarriesTheRawBody) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"edit_file","parameters":{
        "type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"}},
        "required":["path","old_string"]}}}])");
    ParsedToolCall tc;
    ASSERT_TRUE(parse_glm_arg_key_call("edit_file<arg_key>path</arg_key><arg_value>a.js</arg_value>", tc));
    validate_tool_call(tc, tools);
    EXPECT_FALSE(tc.valid);
    const std::string with_raw = tool_call_fault(tc, tools, true);
    EXPECT_NE(with_raw.find("missing required argument \"old_string\""), std::string::npos) << with_raw;
    EXPECT_NE(with_raw.find("raw=\"edit_file<arg_key>path</arg_key><arg_value>a.js</arg_value>\""),
              std::string::npos)
        << with_raw;
    EXPECT_EQ(tool_call_fault(tc, tools, false).find("raw="), std::string::npos);
}

TEST(ToolCallGlm, NoArgumentsAndMalformedPairs) {
    ParsedToolCall tc;
    EXPECT_FALSE(parse_glm_arg_key_call("<arg_key>x</arg_key><arg_value>1</arg_value>", tc));  // no name
    ParsedToolCall ok;
    ASSERT_TRUE(parse_glm_arg_key_call("list_files<arg_key>dir</arg_key><arg_value>/tmp</arg_value>", ok));
    EXPECT_EQ(ok.name, "list_files");
    EXPECT_EQ(json::parse(ok.arguments)["dir"], "/tmp");
}

// Pairs as vLLM 0.33 glm47 extracts them (re.finditer, non-greedy, \s* between the tags), each row checked
// against vLLM's _glm47_arg_converter: complete pairs only, a broken tail is ignored.
TEST(ToolCallGlm, PairsMatchVllmFinditer) {
    const auto args = [](const std::string& body) {
        ParsedToolCall tc;
        EXPECT_TRUE(parse_glm_arg_key_call(body, tc)) << body;
        return json::parse(tc.arguments);
    };
    // familiar run 4 on Flint: a complete pair, then another format's closing tags.
    EXPECT_EQ(args("bash<arg_key>command</arg_key><arg_value>echo ok</arg_value><arg_key>parameter>\n"
                   "</parameter>\n</function>\n"),
              json({{"command", "echo ok"}}));
    EXPECT_EQ(args("f<arg_key>x</arg_key>"), json::object());  // key without value: no pair, still a call
    EXPECT_EQ(args("f<arg_key> k </arg_key>\n <arg_value>v</arg_value>"), json({{"k", "v"}}));
    EXPECT_EQ(
        args("f<arg_key>a</arg_key>junk<arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value>"),
        json({{"a</arg_key>junk<arg_value>1</arg_value><arg_key>b", "2"}}));
    EXPECT_EQ(args("f<arg_key>a</arg_key><arg_value>1</arg_value>tail<arg_key>b</arg_key><arg_value>2"),
              json({{"a", "1"}}));
}

// Schema typing as vLLM's glm47 parser (coerce_to_schema_type): a string param keeps the emitted
// text ("42", "true", a quoted JSON literal), an integer param parses, string|null maps "null" to null,
// keys are stripped; a second pass (stream driver, then validate) changes nothing.
TEST(ToolCallGlm, StringSchemaKeepsTheEmittedText) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"run","parameters":{
        "type":"object","properties":{"cmd":{"type":"string"},"flag":{"type":"string"},
        "n":{"type":"integer"},"tag":{"type":["string","null"]},"path":{"type":"string"},
        "opts":{"type":"object"}}}}}])");
    ParsedToolCall tc;
    ASSERT_TRUE(parse_glm_arg_key_call(
        "run<arg_key>cmd</arg_key><arg_value>42</arg_value><arg_key>flag</arg_key>"
        "<arg_value>true</arg_value><arg_key> n\n</arg_key><arg_value>7</arg_value>"
        "<arg_key>tag</arg_key><arg_value>null</arg_value><arg_key>path</arg_key>"
        "<arg_value>\"/home/x\"</arg_value><arg_key>opts</arg_key><arg_value>{\"a\": 1}</arg_value>",
        tc));
    restore_raw_string_params(tc, tools);
    validate_tool_call(tc, tools);
    EXPECT_TRUE(tc.valid) << tc.error;
    const json args = json::parse(tc.arguments);
    EXPECT_EQ(args["cmd"], "42");
    EXPECT_EQ(args["flag"], "true");
    EXPECT_EQ(args["n"], 7);
    EXPECT_TRUE(args["tag"].is_null());
    EXPECT_EQ(args["path"], "\"/home/x\"");
    EXPECT_EQ(args["opts"]["a"], 1);
}

// tool_arg_coerce against vLLM 0.33 itself: tests/fixtures/vllm_tool_arg_coerce.json holds
// coerce_to_schema_type(text, extract_types_from_schema(schema)) per row
// (tools/analysis/vllm_tool_arg_coerce_cases.py). Python big ints are the one known gap.
TEST(ToolArgCoerce, MatchesVllmCoerceToSchemaType) {
    std::ifstream f(std::string(IMP_TEST_FIXTURES_DIR) + "/vllm_tool_arg_coerce.json");
    ASSERT_TRUE(f.good());
    const json fx = json::parse(f);
    int checked = 0;
    for (const auto& row : fx["rows"]) {
        const std::string text = row[1].get<std::string>();
        if (text == "99999999999999999999" && row[2].is_number())
            continue;  // vLLM: Python int; imp: beyond uint64
        const json got = imp::coerce_arg_text(text, imp::schema_arg_types(
                                                        fx["schemas"][row[0].get<std::string>()]));
        EXPECT_EQ(got.dump(), row[2].dump()) << "schema " << row[0] << " text \"" << text << "\"";
        ++checked;
    }
    EXPECT_GT(checked, 400);
}

}  // namespace
