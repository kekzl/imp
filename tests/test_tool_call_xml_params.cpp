// Qwen XML tool calls (parse_qwen36_xml_call, <function=NAME><parameter=KEY>VAL</parameter></function>):
// scalar coercion, a <parameter> value against the schema, how a value ends, tool_call_fault lines.
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
              "\"run_tests.py\\n</parameter\": tool not offered (1 offered: read_file)");
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

// An emitted call replays as the model wrote it (parameter order), under either id form, while its
// arguments still parse equal; an edited or unknown call renders canonically (keys sorted).
TEST(ToolCallQwen36Xml, ReplayedCallKeepsTheEmittedParameterOrder) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"read_file","parameters":{
        "type":"object","required":["path"],"properties":{"path":{"type":"string"},
        "offset":{"type":"integer"},"limit":{"type":"integer"}}}}}])");
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=read_file>\n<parameter=path>\na.py\n</parameter>\n"
                              "<parameter=offset>\n200\n</parameter>\n<parameter=limit>\n120\n"
                              "</parameter>\n</function>",
                              tc));
    tc.id = "call_imp_987654";
    validate_tool_call(tc, tools);
    const std::string emitted =
        "<parameter=path>\na.py\n</parameter>\n<parameter=offset>\n200\n</parameter>\n"
        "<parameter=limit>\n120\n</parameter>\n";
    const std::string sorted =
        "<parameter=limit>\n120\n</parameter>\n<parameter=offset>\n200\n</parameter>\n"
        "<parameter=path>\na.py\n</parameter>\n";
    auto replay = [](const char* id, const char* args) {
        const json calls = json::array(
            {{{"id", id}, {"type", "function"}, {"function", {{"name", "read_file"}, {"arguments", args}}}}});
        return reconstruct_tool_call_output(imp::ChatTemplateFamily::CHATML, calls, "", true);
    };
    const char* same = R"({"limit":120,"offset":200,"path":"a.py"})";
    EXPECT_NE(replay("toolu_987654", same).find(emitted), std::string::npos);
    EXPECT_NE(replay("call_imp_987654", same).find(emitted), std::string::npos);
    EXPECT_NE(replay("toolu_987654", R"({"limit":10,"offset":200,"path":"a.py"})")
                  .find("<parameter=limit>\n10\n</parameter>\n<parameter=offset>"),
              std::string::npos)
        << "edited arguments render canonically";
    EXPECT_NE(replay("toolu_123", same).find(sorted), std::string::npos) << "unknown id renders canonically";
}

// Qwen3.8 wrote this in 4 of 951 turns (familiar session 20261114-142819): the call came back as text.
TEST(ToolCallQwen36Xml, ParameterTagNamingTheFunctionIsACall) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"bash","parameters":{
        "type":"object","required":["command"],"properties":{"command":{"type":"string"},
        "timeout":{"type":"integer"}}}}}])");
    const std::string body =
        "<parameter=bash>\n<parameter=command>\ncd /x && sed -n '48,72p' main.js\n"
        "</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>";
    ASSERT_TRUE(qwen_xml_parameter_named_call(body));
    ParsedToolCall tc;
    ASSERT_TRUE(parse_stream_tool_body(body, false, "", tc));
    EXPECT_EQ(tc.name, "bash");
    validate_tool_call(tc, tools);
    EXPECT_TRUE(tc.valid) << tc.error;
    const json args = json::parse(tc.arguments);
    EXPECT_EQ(args.size(), 2u) << "the name tag is not an argument";
    EXPECT_EQ(args["command"], "cd /x && sed -n '48,72p' main.js");
    EXPECT_EQ(args["timeout"], 30);
    EXPECT_EQ(tool_call_fault(tc, tools, false), "");
}

TEST(ToolCallQwen36Xml, ParameterBodiesThatAreNotANamedCallStayText) {
    ParsedToolCall tc;
    // The first parameter has a value: nothing names a function.
    const std::string valued = "<parameter=command>\nls\n</parameter>\n<parameter=timeout>\n3\n</parameter>";
    EXPECT_FALSE(qwen_xml_parameter_named_call(valued));
    EXPECT_FALSE(parse_stream_tool_body(valued, false, "", tc));
    // A lone name tag, a cut tag and an empty name.
    EXPECT_FALSE(parse_stream_tool_body("<parameter=bash>\n", false, "", tc));
    EXPECT_FALSE(parse_stream_tool_body("<parameter=bash", false, "", tc));
    EXPECT_FALSE(
        parse_stream_tool_body("<parameter=>\n<parameter=command>\nls\n</parameter>", false, "", tc));
    // A regular call whose first value is empty keeps its <function= name.
    ASSERT_TRUE(parse_stream_tool_body(
        "<function=bash>\n<parameter=note>\n<parameter=command>\nls\n</parameter>\n</function>", false, "",
        tc));
    EXPECT_EQ(tc.name, "bash");
}

// Qwen3.6 XML fallback (<function=NAME><parameter=KEY>VAL</parameter></function>):
// parse_qwen36_xml_call had no direct coverage before this.
TEST(ToolCallQwen36Xml, NoFunctionTagReturnsFalse) {
    ParsedToolCall tc;
    EXPECT_FALSE(parse_qwen36_xml_call("just some text", tc));
}

// Qwen3.8-Flash-Next under Codex writes the freeform apply_patch body straight into the function tag,
// sometimes followed by stray close tags of other dialects.
TEST(ToolCallQwen36Xml, BodyWithoutParameterIsTheFreeformInput) {
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=apply_patch>\n*** Begin Patch\n*** Add File: a.txt\n+</div>\n"
                              "*** End Patch\n</parameter>\n</invoke>\n</parameter>\n</apply_patch>",
                              tc));
    EXPECT_EQ(tc.name, "apply_patch");
    EXPECT_EQ(json::parse(tc.arguments)["input"],
              "*** Begin Patch\n*** Add File: a.txt\n+</div>\n*** End Patch");
}

TEST(ToolCallQwen36Xml, NoArgumentCallStaysEmpty) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_qwen36_xml_call("<function=get_goal>\n</function>", tc));
    EXPECT_EQ(tc.arguments, "{}");
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

// #2103: stod/stoll read a numeric prefix, so a hex id became 600 / 84.0 / 1e10.
TEST(ToolCallQwen36Xml, DigitLeadingNonNumberStaysAString) {
    for (const char* id :
         {"600acab9a967586f", "84ed1861359f79ce", "1e10aa00bb11cc22", "007", "1.2.3", "-x"}) {
        ParsedToolCall tc;
        ASSERT_TRUE(
            parse_qwen36_xml_call(std::string("<function=f><parameter=id>") + id + "</parameter></function>",
                                  tc));
        EXPECT_EQ(json::parse(tc.arguments)["id"], id) << id;
    }
}

TEST(ToolCallQwen36Xml, StringSchemaKeepsTheEmittedText) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"read_source","parameters":{
        "type":"object","properties":{"source_id":{"type":"string"},"offset":{"type":"integer"},
        "tag":{"type":["string","null"]}}}}}])");
    ParsedToolCall tc;
    ASSERT_TRUE(
        parse_qwen36_xml_call("<function=read_source><parameter=source_id>0600</parameter>"
                              "<parameter=offset>0</parameter><parameter=tag>1.50</parameter>"
                              "</function>",
                              tc));
    validate_tool_call(tc, tools);
    EXPECT_TRUE(tc.valid) << tc.error;
    json args = json::parse(tc.arguments);
    EXPECT_EQ(args["source_id"], "0600");
    EXPECT_EQ(args["offset"], 0);
    EXPECT_EQ(args["tag"], "1.50");
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
