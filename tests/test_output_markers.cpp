// Inkling2 chat-path marker decode: <|message_model|> NAME? <|content_*|> body <|end_message|> becomes
// the <think> / <tool_call> text the shared reasoning and tool parsers read.
#include <gtest/gtest.h>
#include "output_markers.h"

#include <string>
#include <vector>

namespace {

enum : int32_t { kModel = 1, kThinking, kText, kTool, kEnd };

// "@N" = marker N, anything else = plain text.
std::string run(const std::vector<std::string>& pieces) {
    const OutputMarkerAliases table = inkling_alias_table({kModel, kThinking, kText, kTool, kEnd});
    OutputMarkerState st;
    std::string out;
    for (const std::string& p : pieces) {
        const OutputMarkerAlias* a = nullptr;
        if (p.size() == 2 && p[0] == '@')
            for (const auto& e : table)
                if (e.id == p[1] - '0')
                    a = &e;
        out += output_marker_step(st, a, a ? std::string() : p);
    }
    return out;
}

}  // namespace

TEST(OutputMarkersInkling, ThinkingThenTextKeepsOnlyTheThinkTags) {
    EXPECT_EQ(run({"@1", "@2", "The user", " asks.", "@5", "@1", "@3", "Paris", "@5"}),
              "<think>The user asks.</think>Paris");
}

TEST(OutputMarkersInkling, TextWithoutThinkingHasNoTags) {
    EXPECT_EQ(run({"@1", "@3", "Hello", "@5"}), "Hello");
}

TEST(OutputMarkersInkling, ToolCallDropsTheNameHeaderAndRenamesArgs) {
    EXPECT_EQ(
        run({"@1", "@2", "call it", "@5", "@1", "get_", "weather", "@4", "{\"name\":\"get_weather\",\"ar",
             "gs\":{\"city\":\"Berlin\"}}", "@5"}),
        "<think>call it</think><tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}"
        "</tool_call>");
}

TEST(OutputMarkersInkling, OnlyTheFirstArgsKeyIsRenamed) {
    EXPECT_EQ(run({"@1", "f", "@4", "{\"name\":\"f\",\"args\":{\"args\":1}}", "@5"}),
              "<tool_call>{\"name\":\"f\",\"arguments\":{\"args\":1}}</tool_call>");
}

TEST(OutputMarkersInkling, CutCallStillFlushesItsHeldBytes) {
    // max_tokens inside the body: the closer delivers what the rewrite still held.
    EXPECT_EQ(run({"@1", "f", "@4", "{\"name\":\"f\",\"ar", "@5"}),
              "<tool_call>{\"name\":\"f\",\"ar</tool_call>");
}

TEST(OutputMarkersInkling, HeaderTextBeforeATextMarkerIsKept) {
    EXPECT_EQ(run({"@1", "x", "@3", "y", "@5"}), "xy");
}

TEST(OutputMarkersInkling, ExplicitThinkingOffDropsTheThinkingBlock) {
    OutputMarkerAliases table = inkling_alias_table({kModel, kThinking, kText, kTool, kEnd});
    OutputMarkerAliases untouched = table;
    EXPECT_FALSE(silence_marker_thinking(untouched, false));
    ASSERT_TRUE(silence_marker_thinking(table, true));
    OutputMarkerState st;
    std::string out;
    const std::vector<std::string> pieces = {"@1",          "@2", "The user",
                                             " asks.",      "@5", "@1",
                                             "get_weather", "@4", "{\"name\":\"get_weather\",\"args\":{}}",
                                             "@5",          "@1", "@3",
                                             "Paris",       "@5"};
    for (const std::string& p : pieces) {
        const OutputMarkerAlias* a = nullptr;
        if (p.size() == 2 && p[0] == '@')
            for (const auto& e : table)
                if (e.id == p[1] - '0')
                    a = &e;
        out += output_marker_step(st, a, a ? std::string() : p);
    }
    EXPECT_EQ(out, "<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>Paris");
    // Cohere2 shape: every marker has its own closer, the template switches thinking.
    OutputMarkerAliases plain(2);
    plain[0].id = 7, plain[0].text = "<think>";
    plain[1].id = 8, plain[1].text = "</think>";
    EXPECT_FALSE(silence_marker_thinking(plain, true));
    EXPECT_EQ(plain[0].text, "<think>");
}
