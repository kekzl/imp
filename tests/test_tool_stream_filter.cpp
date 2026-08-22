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
#include "stream_pipeline.h"
#include "model/chat_template.h"

#include <random>
#include <string>
#include <tuple>
#include <vector>

using imp::ChatTemplateFamily;
using imp::server::StreamToolCallFilter;

namespace {

struct Collected {
    std::string content;
    std::vector<ParsedToolCall> calls;  // buffered CALLs + streamed CALL_ENDs, in order
    std::string leftover;               // filter.finish() at stream end

    // Incremental-streaming bookkeeping (JSON layouts):
    int n_streamed_calls = 0;           // CALL_BEGIN count
    int n_arg_deltas = 0;               // CALL_ARGS_DELTA count
    std::string open_deltas;            // deltas of the currently-open call
    std::vector<std::string> delta_concats;  // per streamed call: concat(deltas)
    std::vector<std::string> all_deltas;     // every CALL_ARGS_DELTA, in order
};

Collected feed_chunks(ChatTemplateFamily fam, const std::string& input,
                      const std::vector<size_t>& chunk_sizes) {
    StreamToolCallFilter filter(fam);
    Collected out;
    using Kind = StreamToolCallFilter::Segment::Kind;
    size_t pos = 0, ci = 0;
    while (pos < input.size()) {
        size_t n = chunk_sizes.empty() ? 1 : chunk_sizes[ci++ % chunk_sizes.size()];
        n = std::min(n, input.size() - pos);
        auto segs = filter.feed(input.substr(pos, n));
        for (auto& seg : segs) {
            switch (seg.kind) {
                case Kind::TEXT:
                    out.content += seg.text;
                    break;
                case Kind::CALL:
                    out.calls.push_back(std::move(seg.call));
                    break;
                case Kind::CALL_BEGIN:
                    out.n_streamed_calls++;
                    out.open_deltas.clear();
                    break;
                case Kind::CALL_ARGS_DELTA:
                    out.n_arg_deltas++;
                    out.open_deltas += seg.text;
                    out.all_deltas.push_back(seg.text);
                    break;
                case Kind::CALL_END:
                    out.delta_concats.push_back(out.open_deltas);
                    out.calls.push_back(std::move(seg.call));
                    break;
            }
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
        // JSON layout streams incrementally: BEGIN + deltas whose
        // concatenation IS the final arguments string, chunking-invariant.
        EXPECT_EQ(r.n_streamed_calls, 1) << "chunk=" << chunk;
        ASSERT_EQ(r.delta_concats.size(), 1u);
        EXPECT_EQ(r.delta_concats[0], r.calls[0].arguments);
    }
}

// ---------------------------------------------------------------------------
// Incremental argument streaming (the 20-60s zero-bytes fix): the arguments
// of a JSON-layout call must flow out WHILE the body arrives, not after the
// close tag.
// ---------------------------------------------------------------------------

TEST(ToolStreamFilterChatML, ArgsStreamBeforeCloseTag) {
    // Feed everything except the close tag: the name and (most of) the
    // arguments must already have been emitted.
    std::string args_json =
        "{\"path\": \"/src/main.cpp\", \"content\": \"int main() { return 0; } // "
        "a reasonably long payload so multiple deltas confirm before the tail\"}";
    std::string body = "<tool_call>{\"name\": \"edit_file\", \"arguments\": " + args_json + "}";
    StreamToolCallFilter filter(ChatTemplateFamily::CHATML);
    using Kind = StreamToolCallFilter::Segment::Kind;
    bool begun = false;
    std::string streamed;
    for (char c : body) {
        for (auto& seg : filter.feed(std::string(1, c))) {
            if (seg.kind == Kind::CALL_BEGIN) {
                begun = true;
                EXPECT_EQ(seg.call.name, "edit_file");
            } else if (seg.kind == Kind::CALL_ARGS_DELTA) {
                streamed += seg.text;
            }
        }
    }
    EXPECT_TRUE(begun) << "CALL_BEGIN must fire before the close tag";
    EXPECT_TRUE(filter.call_open());
    // Everything except the close-tag-straddle holdback (≤ close-tag length)
    // must already be on the wire before the close marker arrives.
    EXPECT_GE(streamed.size() + 12, args_json.size());
    EXPECT_EQ(args_json.compare(0, streamed.size(), streamed), 0)
        << "streamed bytes must be a prefix of the arguments";
    // The close tag disambiguates the held tail and completes the call.
    auto segs = filter.feed("</tool_call>");
    ASSERT_FALSE(segs.empty());
    for (size_t i = 0; i + 1 < segs.size(); ++i) {
        EXPECT_EQ(segs[i].kind, Kind::CALL_ARGS_DELTA);
        streamed += segs[i].text;
    }
    EXPECT_EQ(segs.back().kind, Kind::CALL_END);
    EXPECT_EQ(streamed, args_json);
    EXPECT_EQ(segs.back().call.arguments, args_json);
    EXPECT_FALSE(filter.call_open());
}

TEST(ToolStreamFilterChatML, StreamedArgsHandleStringsEscapesAndAngles) {
    // Strings containing '<', escaped quotes, and nested structures must not
    // confuse the nesting tracker or the close-tag holdback.
    std::string args_json =
        "{\"html\": \"<div class=\\\"x\\\">a < b</div>\", "
        "\"nested\": {\"arr\": [1, {\"k\": \"}]\"}]}}";
    std::string input =
        "<tool_call>{\"name\": \"render\", \"arguments\": " + args_json + "}</tool_call>";
    for (size_t chunk : {size_t(1), size_t(7), input.size()}) {
        auto r = feed_chunks(ChatTemplateFamily::CHATML, input, {chunk});
        ASSERT_EQ(r.calls.size(), 1u) << "chunk=" << chunk;
        EXPECT_EQ(r.calls[0].name, "render");
        EXPECT_EQ(r.calls[0].arguments, args_json);
        ASSERT_EQ(r.delta_concats.size(), 1u);
        EXPECT_EQ(r.delta_concats[0], args_json);
        EXPECT_NO_THROW(std::ignore = json::parse(r.calls[0].arguments)) << "chunk=" << chunk;
        EXPECT_EQ(r.content, "");
    }
}

TEST(ToolStreamFilterChatML, StringEncodedArgumentsFallBackToBuffered) {
    // "arguments" as a JSON-encoded STRING is not streamable as raw bytes
    // (the client expects the decoded object text) — must take the buffered
    // CALL path, whose parser normalizes it.
    std::string input =
        "<tool_call>{\"name\": \"s\", \"arguments\": \"{\\\"x\\\": 2}\"}</tool_call>";
    auto r = feed_char_by_char(ChatTemplateFamily::CHATML, input);
    EXPECT_EQ(r.n_streamed_calls, 0);
    ASSERT_EQ(r.calls.size(), 1u);
    EXPECT_EQ(r.calls[0].name, "s");
    // The buffered parser's arguments normalization is pre-existing behavior
    // (string-encoded args may stay in string form) — only the routing to the
    // buffered path is under test here. Whatever form: it must decode to x=2.
    json args = json::parse(r.calls[0].arguments);
    if (args.is_string())
        args = json::parse(args.get<std::string>());
    EXPECT_EQ(args["x"], 2);
}

TEST(ToolStreamFilterChatML, StreamedCutoffKeepsEmittedDeltas) {
    // Stream ends mid-arguments after CALL_BEGIN: nothing restorable —
    // finish() must be empty, call_open() true, streamed_arguments() has the
    // emitted prefix.
    std::string input =
        "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": "
        "\"a-long-enough-value-to-clear-the-close-tag-straddle-holdback";
    StreamToolCallFilter filter(ChatTemplateFamily::CHATML);
    using Kind = StreamToolCallFilter::Segment::Kind;
    std::string streamed;
    bool begun = false;
    for (char c : input) {
        for (auto& seg : filter.feed(std::string(1, c))) {
            if (seg.kind == Kind::CALL_BEGIN)
                begun = true;
            if (seg.kind == Kind::CALL_ARGS_DELTA)
                streamed += seg.text;
        }
    }
    EXPECT_TRUE(begun);
    EXPECT_TRUE(filter.call_open());
    EXPECT_EQ(filter.finish(), "");
    EXPECT_EQ(filter.streamed_arguments(), streamed);
    EXPECT_FALSE(streamed.empty());
}

TEST(ToolStreamFilterLlama3, ArgsStreamIncrementally) {
    std::string args_json = "{\"query\": \"a longer search string for deltas\"}";
    std::string input = "go <function=search>" + args_json + "</function>";
    auto r = feed_char_by_char(ChatTemplateFamily::LLAMA3, input);
    ASSERT_EQ(r.calls.size(), 1u);
    EXPECT_EQ(r.calls[0].name, "search");
    EXPECT_EQ(r.calls[0].arguments, args_json);
    EXPECT_EQ(r.n_streamed_calls, 1);
    ASSERT_EQ(r.delta_concats.size(), 1u);
    EXPECT_EQ(r.delta_concats[0], args_json);
    EXPECT_EQ(r.content, "go ");
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

// ---- #1554: an argument delta never ends mid-codepoint ----
//
// The emit loop pulls `limit` back by close_tag_.size() - 1 BYTES so a
// partially arrived close tag cannot leak into the arguments. That cut lands
// inside a multi-byte character whenever one sits at the boundary, and each
// half is JSON-encoded into its own SSE delta, where dump_safe turns it into
// U+FFFD. Measured on Qwen3-8B-Q8_0 with a forced tool_choice: ten replacement
// characters in one argument string, and the non-streaming control clean.
//
// The first attempt at this issue hardened the BUFFERED 48-byte chunker
// instead. That path is real but is not the one a shipped model takes here, so
// the defect survived the fix. These tests drive the streaming path.

// Every emitted delta must be valid UTF-8 on its own, because each one is
// JSON-encoded separately.
static bool all_deltas_are_whole_utf8(const Collected& c) {
    for (const auto& d : c.all_deltas) {
        if (imp::stream::utf8_complete_len(d) != d.size())
            return false;
    }
    return true;
}

TEST(ToolStreamFilterUtf8, ArgumentDeltasNeverEndMidCodepoint) {
    // Umlauts spread through the value so at least one lands on the hold
    // boundary for some chunk size.
    const std::string body =
        "<tool_call>{\"name\": \"note\", \"arguments\": {\"text\": "
        "\"ÄÖÜäöüß ÄÖÜäöüß ÄÖÜäöüß ÄÖÜäöüß\"}}</tool_call>";
    const std::string want_args = "{\"text\": \"ÄÖÜäöüß ÄÖÜäöüß ÄÖÜäöüß ÄÖÜäöüß\"}";

    // Byte-at-a-time is the worst case: every multi-byte character arrives
    // split across feeds.
    {
        auto c = feed_chunks(ChatTemplateFamily::CHATML, body, {1});
        ASSERT_EQ(c.delta_concats.size(), 1u);
        EXPECT_EQ(c.delta_concats[0], want_args);
        EXPECT_TRUE(all_deltas_are_whole_utf8(c)) << "a delta ended mid-character";
    }
    // And a spread of chunk sizes, including ones that straddle the 2-byte
    // characters at every offset.
    for (size_t n : {2u, 3u, 4u, 5u, 7u, 11u, 16u, 48u}) {
        auto c = feed_chunks(ChatTemplateFamily::CHATML, body, {n});
        ASSERT_EQ(c.delta_concats.size(), 1u) << "chunk=" << n;
        EXPECT_EQ(c.delta_concats[0], want_args) << "chunk=" << n;
        EXPECT_TRUE(all_deltas_are_whole_utf8(c)) << "chunk=" << n;
    }
}

TEST(ToolStreamFilterUtf8, HoldingBackTheTailDoesNotLoseIt) {
    // The tail held back for the next feed has to come out eventually. A
    // three-byte character as the very last thing before the close tag is the
    // case where "hold it back" and "there is no next feed" meet.
    const std::string body = "<tool_call>{\"name\": \"note\", \"arguments\": {\"t\": \"中\"}}</tool_call>";
    for (size_t n : {1u, 2u, 3u, 5u, 9u}) {
        auto c = feed_chunks(ChatTemplateFamily::CHATML, body, {n});
        ASSERT_EQ(c.delta_concats.size(), 1u) << "chunk=" << n;
        EXPECT_EQ(c.delta_concats[0], "{\"t\": \"中\"}") << "chunk=" << n;
        EXPECT_TRUE(all_deltas_are_whole_utf8(c)) << "chunk=" << n;
    }
}

TEST(ToolStreamFilterUtf8, FourByteCharactersSurviveToo) {
    const std::string body =
        "<tool_call>{\"name\": \"note\", \"arguments\": {\"t\": \"a😀b😀c\"}}</tool_call>";
    for (size_t n : {1u, 2u, 3u, 4u, 6u, 13u}) {
        auto c = feed_chunks(ChatTemplateFamily::CHATML, body, {n});
        ASSERT_EQ(c.delta_concats.size(), 1u) << "chunk=" << n;
        EXPECT_EQ(c.delta_concats[0], "{\"t\": \"a😀b😀c\"}") << "chunk=" << n;
        EXPECT_TRUE(all_deltas_are_whole_utf8(c)) << "chunk=" << n;
    }
}

TEST(ToolStreamFilterUtf8, AsciiArgumentsAreUnchanged) {
    // Negative control: holding a partial codepoint back must not change the
    // delta sequence for input that has none.
    const std::string body =
        "<tool_call>{\"name\": \"note\", \"arguments\": {\"t\": \"plain ascii\"}}</tool_call>";
    auto c = feed_chunks(ChatTemplateFamily::CHATML, body, {1});
    ASSERT_EQ(c.delta_concats.size(), 1u);
    EXPECT_EQ(c.delta_concats[0], "{\"t\": \"plain ascii\"}");
    EXPECT_GT(c.n_arg_deltas, 1) << "the value should still stream incrementally";
}
