// =============================================================================
// Unit tests for tools/imp-server/tool_stream_filter.h + the streaming tag
// scanner/body parser in tool_call.cpp (scan_tool_tag, parse_stream_tool_body).
//
// WHY THIS EXISTS: the streaming tool-call state machines previously lived
// inline in run_chat_stream_ / run_anthropic_stream_ and only recognized
// ChatML <tool_call> and Llama3 <function=. Real agents stream (stream:true),
// so on Gemma-4 the raw <|tool_call> markers leaked as visible text and
// Qwen3.6 XML bodies were silently swallowed by a catch(...). The machinery
// is now the shared StreamToolCallFilter; these tests pin its contract across
// arbitrary token boundaries (char-by-char and random chunk sizes) — token
// pieces split markers at any byte.
//
// ORACLE: hand-constructed inputs per documented wire format with the expected
// (content, calls) spelled out — no imp-vs-imp.
// =============================================================================

#include <gtest/gtest.h>
#include "tool_call.h"
#include "tool_stream_filter.h"
#include "model/chat_template.h"

#include <random>
#include <string>
#include <vector>

using imp::ChatTemplateFamily;
using imp::server::StreamToolCallFilter;

namespace {

struct Collected {
    std::string content;
    std::vector<ParsedToolCall> calls;
    std::string leftover;  // filter.finish() at stream end
};

Collected feed_chunks(ChatTemplateFamily fam, const std::string& input,
                      const std::vector<size_t>& chunk_sizes) {
    StreamToolCallFilter filter(fam);
    Collected out;
    size_t pos = 0, ci = 0;
    while (pos < input.size()) {
        size_t n = chunk_sizes.empty() ? 1 : chunk_sizes[ci++ % chunk_sizes.size()];
        n = std::min(n, input.size() - pos);
        auto segs = filter.feed(input.substr(pos, n));
        for (auto& seg : segs) {
            if (seg.is_call)
                out.calls.push_back(std::move(seg.call));
            else
                out.content += seg.text;
        }
        pos += n;
    }
    out.leftover = filter.finish();
    return out;
}

Collected feed_char_by_char(ChatTemplateFamily fam, const std::string& input) {
    return feed_chunks(fam, input, {1});
}

Collected feed_whole(ChatTemplateFamily fam, const std::string& input) {
    return feed_chunks(fam, input, {input.size()});
}

}  // namespace

// ---------------------------------------------------------------------------
// Gemma-4 native dialect: <|tool_call>call:NAME{...}<tool_call|>
// ---------------------------------------------------------------------------

static const char* kGemmaInput =
    "I'll check the weather.\n"
    "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>,days:3}<tool_call|>\n"
    "Done.";

static void check_gemma_result(const Collected& r) {
    ASSERT_EQ(r.calls.size(), 1u);
    EXPECT_EQ(r.calls[0].name, "get_weather");
    json args = json::parse(r.calls[0].arguments);
    EXPECT_EQ(args["location"], "Paris");
    EXPECT_EQ(args["days"], 3);
    // Content = prose before the marker + after-text (leading ws trimmed).
    EXPECT_EQ(r.content, "I'll check the weather.\nDone.");
    // No marker bytes may leak into the visible stream.
    EXPECT_EQ(r.content.find("<|tool_call>"), std::string::npos);
    EXPECT_EQ(r.content.find("<tool_call|>"), std::string::npos);
    EXPECT_TRUE(r.leftover.empty());
}

TEST(ToolStreamFilterGemma, CharByChar) {
    check_gemma_result(feed_char_by_char(ChatTemplateFamily::GEMMA, kGemmaInput));
}

TEST(ToolStreamFilterGemma, WholeString) {
    check_gemma_result(feed_whole(ChatTemplateFamily::GEMMA, kGemmaInput));
}

TEST(ToolStreamFilterGemma, RandomChunkSizes) {
    std::mt19937 rng(42);
    for (int trial = 0; trial < 32; ++trial) {
        std::vector<size_t> sizes;
        for (int i = 0; i < 16; ++i)
            sizes.push_back(1 + rng() % 7);
        auto r = feed_chunks(ChatTemplateFamily::GEMMA, kGemmaInput, sizes);
        check_gemma_result(r);
    }
}

TEST(ToolStreamFilterGemma, ChatmlFallbackStillRecognized) {
    // The GEMMA family keeps recognizing the ChatML marker (text-based tool
    // prompts instruct the <tool_call> JSON format).
    auto r = feed_char_by_char(ChatTemplateFamily::GEMMA,
                               "<tool_call>{\"name\":\"g\",\"arguments\":{\"x\":1}}</tool_call>");
    ASSERT_EQ(r.calls.size(), 1u);
    EXPECT_EQ(r.calls[0].name, "g");
    EXPECT_EQ(r.content, "");
}

TEST(ToolStreamFilterGemma, TwoCallsInOneFeed) {
    std::string input =
        "<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:2}<tool_call|>";
    auto r = feed_whole(ChatTemplateFamily::GEMMA, input);
    ASSERT_EQ(r.calls.size(), 2u);
    EXPECT_EQ(r.calls[0].name, "a");
    EXPECT_EQ(r.calls[1].name, "b");
    EXPECT_EQ(r.content, "");
}

// ---------------------------------------------------------------------------
// ChatML dialect (existing behavior must be preserved) + Qwen3.6 XML fallback
// ---------------------------------------------------------------------------

TEST(ToolStreamFilterChatML, JsonBodyChunked) {
    std::string input =
        "Sure.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>";
    for (size_t chunk : {size_t(1), size_t(3), size_t(5), input.size()}) {
        auto r = feed_chunks(ChatTemplateFamily::CHATML, input, {chunk});
        ASSERT_EQ(r.calls.size(), 1u) << "chunk=" << chunk;
        EXPECT_EQ(r.calls[0].name, "f");
        json args = json::parse(r.calls[0].arguments);
        EXPECT_EQ(args["x"], 1);
        EXPECT_EQ(r.content, "Sure.\n");
    }
}

TEST(ToolStreamFilterChatML, Qwen36XmlFallback) {
    // json::parse fails on the XML-styled body; the filter must fall back to
    // parse_qwen36_xml_call instead of swallowing the call.
    std::string input =
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    auto r = feed_char_by_char(ChatTemplateFamily::CHATML, input);
    ASSERT_EQ(r.calls.size(), 1u);
    EXPECT_EQ(r.calls[0].name, "get_weather");
    json args = json::parse(r.calls[0].arguments);
    EXPECT_EQ(args["city"], "Paris");
    EXPECT_EQ(args["days"], 3);
    EXPECT_EQ(r.content, "");
}

TEST(ToolStreamFilterChatML, UnparseableBodyRestoredVerbatim) {
    // Neither JSON nor XML: the raw text (markers included) must be restored
    // to the content stream, not silently dropped.
    std::string input = "before <tool_call>garbage{{{</tool_call> after";
    auto r = feed_chunks(ChatTemplateFamily::CHATML, input, {4});
    EXPECT_EQ(r.calls.size(), 0u);
    EXPECT_NE(r.content.find("before "), std::string::npos);
    EXPECT_NE(r.content.find("<tool_call>garbage{{{</tool_call>"), std::string::npos);
    EXPECT_NE(r.content.find("after"), std::string::npos);
}

TEST(ToolStreamFilterChatML, BareLessThanIsNotHeldForever) {
    // Prose containing '<' that is provably not a tool tag must be released
    // (the previous inline machines could hold such content to stream end).
    std::string input = "a < b and 3 < 5 hold";
    auto r = feed_chunks(ChatTemplateFamily::CHATML, input, {2});
    EXPECT_EQ(r.calls.size(), 0u);
    EXPECT_EQ(r.content + r.leftover, input);
    // Everything except (at most) a trailing potential-tag fragment is out.
    EXPECT_EQ(r.leftover, "");
}

TEST(ToolStreamFilterChatML, PartialTagAtEofReturnedByFinish) {
    std::string input = "text <tool_call>{\"name\":";
    auto r = feed_char_by_char(ChatTemplateFamily::CHATML, input);
    EXPECT_EQ(r.calls.size(), 0u);
    EXPECT_EQ(r.content, "text ");
    EXPECT_EQ(r.leftover, "<tool_call>{\"name\":");
}

// ---------------------------------------------------------------------------
// Llama3 dialect: <function=NAME>{json}</function>
// ---------------------------------------------------------------------------

TEST(ToolStreamFilterLlama3, Chunked) {
    std::string input = "Looking it up. <function=lookup>{\"q\": \"imp\"}</function>";
    for (size_t chunk : {size_t(1), size_t(4), input.size()}) {
        auto r = feed_chunks(ChatTemplateFamily::LLAMA3, input, {chunk});
        ASSERT_EQ(r.calls.size(), 1u) << "chunk=" << chunk;
        EXPECT_EQ(r.calls[0].name, "lookup");
        json args = json::parse(r.calls[0].arguments);
        EXPECT_EQ(args["q"], "imp");
        EXPECT_EQ(r.content, "Looking it up. ");
    }
}

// ---------------------------------------------------------------------------
// scan_tool_tag: the '<'/'|' holdback the Gemma marker depends on
// ---------------------------------------------------------------------------

TEST(ToolTagScan, GemmaMarkerPrefixesAreHeldBack) {
    using Kind = ToolTagScan::Kind;
    EXPECT_EQ(scan_tool_tag("<", ChatTemplateFamily::GEMMA).kind, Kind::PARTIAL);
    EXPECT_EQ(scan_tool_tag("<|", ChatTemplateFamily::GEMMA).kind, Kind::PARTIAL);
    EXPECT_EQ(scan_tool_tag("<|tool_ca", ChatTemplateFamily::GEMMA).kind, Kind::PARTIAL);
    EXPECT_EQ(scan_tool_tag("<|x", ChatTemplateFamily::GEMMA).kind, Kind::NONE);
    auto open = scan_tool_tag("pre<|tool_call>call:", ChatTemplateFamily::GEMMA);
    EXPECT_EQ(open.kind, Kind::OPEN);
    EXPECT_EQ(open.content_len, 3u);
    EXPECT_TRUE(open.gemma_body);
    EXPECT_STREQ(open.close_tag, "<tool_call|>");
}

TEST(ToolTagScan, ChatmlAndLlama3Unchanged) {
    using Kind = ToolTagScan::Kind;
    // ChatML family does NOT treat "<|" as a potential tag.
    EXPECT_EQ(scan_tool_tag("<|", ChatTemplateFamily::CHATML).kind, Kind::NONE);
    EXPECT_EQ(scan_tool_tag("<tool_c", ChatTemplateFamily::CHATML).kind, Kind::PARTIAL);
    auto open = scan_tool_tag("<tool_call>", ChatTemplateFamily::CHATML);
    EXPECT_EQ(open.kind, Kind::OPEN);
    EXPECT_STREQ(open.close_tag, "</tool_call>");
    // Llama3: full open tag needs the trailing '>' (name terminator).
    EXPECT_EQ(scan_tool_tag("<function=foo", ChatTemplateFamily::LLAMA3).kind, Kind::PARTIAL);
    auto l3 = scan_tool_tag("<function=foo>", ChatTemplateFamily::LLAMA3);
    EXPECT_EQ(l3.kind, Kind::OPEN);
    EXPECT_EQ(l3.fn_name, "foo");
}

// ---------------------------------------------------------------------------
// parse_stream_tool_body + exported single-call parsers
// ---------------------------------------------------------------------------

TEST(ParseStreamToolBody, GemmaBody) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_stream_tool_body("call:go{speed:5}", /*gemma_body=*/true, "", tc));
    EXPECT_EQ(tc.name, "go");
    EXPECT_EQ(json::parse(tc.arguments)["speed"], 5);
}

TEST(ParseStreamToolBody, JsonThenXmlFallbackThenFail) {
    ParsedToolCall tc;
    ASSERT_TRUE(parse_stream_tool_body("{\"name\":\"j\",\"arguments\":{}}", false, "", tc));
    EXPECT_EQ(tc.name, "j");
    ParsedToolCall tc2;
    ASSERT_TRUE(parse_stream_tool_body(
        "<function=x>\n<parameter=k>\nv\n</parameter>\n</function>", false, "", tc2));
    EXPECT_EQ(tc2.name, "x");
    EXPECT_EQ(json::parse(tc2.arguments)["k"], "v");
    ParsedToolCall tc3;
    EXPECT_FALSE(parse_stream_tool_body("not a call at all", false, "", tc3));
}

TEST(ParseGemmaToolCallBody, RejectsMalformed) {
    ParsedToolCall tc;
    EXPECT_FALSE(parse_gemma_tool_call_body("", tc));
    EXPECT_FALSE(parse_gemma_tool_call_body("call:", tc));
    EXPECT_FALSE(parse_gemma_tool_call_body("nope{}", tc));
    // Unparseable args degrade to "{}" but the call is kept (matches the
    // non-streaming parser).
    ASSERT_TRUE(parse_gemma_tool_call_body("call:f{broken", tc));
    EXPECT_EQ(tc.name, "f");
    EXPECT_EQ(tc.arguments, "{}");
}
