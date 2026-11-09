// Qwen XML tool calls: a <parameter> value against the schema, and how a value ends.
#include <gtest/gtest.h>
#include "tool_call.h"

#include <string>

namespace {
json read_file_tools() {
    return json::parse(R"([{"type":"function","function":{"name":"read_file","parameters":{
        "type":"object","required":["path"],"properties":{"path":{"type":"string"}}}}}])");
}
}  // namespace

TEST(ToolCallFault, CleanCallHasNoLine) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call("<function=read_file><parameter=path>a.py</parameter></function>", tc));
    validate_tool_call(tc, read_file_tools());
    EXPECT_EQ(tool_call_fault(tc, read_file_tools(), true), "");
}

TEST(ToolCallFault, RawTextOnlyOnRequest) {
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=read_file>\n<parameter=limit>40</parameter>\n</function>", tc));
    validate_tool_call(tc, read_file_tools());
    EXPECT_EQ(tool_call_fault(tc, read_file_tools(), false), "read_file: missing required argument \"path\"");
    const std::string line = tool_call_fault(tc, read_file_tools(), true);
    EXPECT_NE(line.find(" raw=\"<function=read_file>\\n<parameter=limit>40"), std::string::npos) << line;
}

TEST(ToolCallFault, NameTheRequestDidNotOfferIsAFault) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call("<function=run_tests.py\n</parameter>\n</function>", tc));
    validate_tool_call(tc, read_file_tools());
    EXPECT_TRUE(tc.valid);
    EXPECT_EQ(tool_call_fault(tc, read_file_tools(), false),
              "\"run_tests.py\\n</parameter\": tool not offered");
    EXPECT_EQ(tool_call_fault(tc, json::array(), false), "") << "no tools in the request: nothing to compare";
}

TEST(ToolCallFault, RawTextIsCappedAt2000Bytes) {
    ParsedToolCall tc;
    tc.name = "read_file";
    tc.valid = false;
    tc.error = "x";
    tc.raw_body = std::string(5000, 'a');
    const std::string line = tool_call_fault(tc, read_file_tools(), true);
    EXPECT_EQ(line, "read_file: x raw=\"" + std::string(2000, 'a') + "\"");
}

TEST(ToolCallQwen36Xml, ArrayAndObjectSchemasParseTheEmittedText) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"multi_edit","parameters":{
        "type":"object","required":["path","edits"],"properties":{"path":{"type":"string"},
        "edits":{"type":"array"},"opts":{"type":"object"},"note":{"type":["array","string"]}}}}}])");
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=multi_edit><parameter=path>[a].txt</parameter>"
                              "<parameter=edits>\n[{\"old\": \"foo\", \"new\": \"bar\"}]\n</parameter>"
                              "<parameter=opts>{\"dry\": true}</parameter>"
                              "<parameter=note>[1]</parameter></function>",
                              tc));
    validate_tool_call(tc, tools);
    EXPECT_TRUE(tc.valid) << tc.error;
    const json args = json::parse(tc.arguments);
    ASSERT_TRUE(args["edits"].is_array());
    EXPECT_EQ(args["edits"][0]["new"], "bar");
    EXPECT_EQ(args["opts"]["dry"], true);
    EXPECT_EQ(args["path"], "[a].txt") << "a string schema keeps the text";
    EXPECT_EQ(args["note"], "[1]") << "a schema that allows a string keeps the text";
}

TEST(ToolCallQwen36Xml, BareCloseTagDoesNotSwallowTheNextParameter) {
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=read_file>\n<parameter=offset>10</parameter>\n"
                              "<parameter=path>\na.py\n</parameter>\n</function>",
                              tc));
    const json args = json::parse(tc.arguments);
    EXPECT_EQ(args["offset"], 10);
    EXPECT_EQ(args["path"], "a.py");
}

TEST(ToolCallQwen36Xml, BareCloseTagInsideAValueStaysText) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call(
        "<function=write>\n<parameter=code>\nx = \"</parameter>\"\ny = 1\n</parameter>\n"
        "<parameter=path>\na.py\n</parameter>\n</function>",
        tc));
    const json args = json::parse(tc.arguments);
    EXPECT_EQ(args["code"], "x = \"</parameter>\"\ny = 1");
    EXPECT_EQ(args["path"], "a.py");
}

// Only the template's own newlines go: a re-render of the parsed call is the model's text again.
TEST(ToolCallQwen36Xml, ValueKeepsIndentationAndEdgeSpaces) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call(
        "<function=edit>\n<parameter=old>\n    def foo():\n        pass\n\n</parameter>\n"
        "<parameter=pattern>\ndef \n</parameter>\n<parameter=n>\n120\n</parameter>\n</function>",
        tc));
    const json args = json::parse(tc.arguments);
    EXPECT_EQ(args["old"], "    def foo():\n        pass\n");
    EXPECT_EQ(args["pattern"], "def ");
    EXPECT_EQ(args["n"], 120);
}

TEST(ToolCallQwen36Xml, ArraySchemaWithUnparsableTextStaysInvalid) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"multi_edit","parameters":{
        "type":"object","properties":{"edits":{"type":"array"}}}}}])");
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=multi_edit><parameter=edits>[{'old': 1}]</parameter></function>",
                              tc));
    validate_tool_call(tc, tools);
    EXPECT_FALSE(tc.valid);
    EXPECT_EQ(json::parse(tc.arguments)["edits"], "[{'old': 1}]");
}

TEST(ToolCallQwen36Xml, ArraySchemaRejectsAnObject) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"multi_edit","parameters":{
        "type":"object","properties":{"edits":{"type":"array"}}}}}])");
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=multi_edit><parameter=edits>{\"a\": 1}</parameter></function>", tc));
    validate_tool_call(tc, tools);
    EXPECT_FALSE(tc.valid);
    EXPECT_EQ(json::parse(tc.arguments)["edits"], "{\"a\": 1}") << "a value of the wrong kind stays the text";
}
