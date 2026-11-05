// Qwen XML tool calls: a <parameter> value against the schema, and how a value ends.
#include <gtest/gtest.h>
#include "tool_call.h"

#include <string>

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
