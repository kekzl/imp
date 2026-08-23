// =============================================================================
// Server streaming-pipeline unit tests (issue #557 item 2).
//
// The SSE chunk envelopes, the reasoning/think split and the Gemma-4 channel
// split were only exercised indirectly via degen_suite against a live server —
// the NUL-leak (#510) and the think-leak classes lived exactly here. These
// tests pin the pure text-level contracts on the CPU:
//
//   * sse_chunk / sse_completion_chunk emit `data: <valid JSON>\n\n`
//   * SSEChunkWriter's hand-built envelope is byte-compatible with the
//     json-built sse_chunk (the writer exists as a hot-path optimization —
//     if the two drift, streaming silently changes shape)
//   * json_escape_into handles quotes/backslash/control chars/UTF-8
//   * no NUL byte ever reaches a delta (the #510 regression class)
//   * extract_reasoning: closed, unclosed, missing and multiple think blocks
//   * strip_think_block / strip_channel_headers / split_channel_segments
// =============================================================================

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "utils.h"            // tools/imp-server/utils.h
#include "reasoning_split.h"  // StreamReasoningSplitter (shared streaming demux)

namespace {

// Collect everything written to the sink. DataSink is neither copyable nor
// movable — configure the caller's instance in place.
void bind_sink(httplib::DataSink& sink, std::string& out) {
    sink.write = [&out](const char* d, size_t n) {
        out.append(d, n);
        return true;
    };
    sink.done = [] {};
    sink.is_writable = [] { return true; };
}

// Parse one `data: {...}\n\n` SSE frame into json.
json parse_sse(const std::string& frame) {
    EXPECT_TRUE(frame.rfind("data: ", 0) == 0) << frame;
    EXPECT_TRUE(frame.size() >= 8 && frame.substr(frame.size() - 2) == "\n\n") << frame;
    return json::parse(frame.substr(6, frame.size() - 8));
}

// ---------------------------------------------------------------------------
// SSE chunk shape
// ---------------------------------------------------------------------------

TEST(SseChunk, ChatChunkIsValidJsonWithEnvelope) {
    json delta = {{"content", "Hello"}};
    std::string s = sse_chunk("chatcmpl-1", 1234, "test-model", delta, nullptr);
    json j = parse_sse(s);
    EXPECT_EQ(j["object"], "chat.completion.chunk");
    EXPECT_EQ(j["id"], "chatcmpl-1");
    EXPECT_EQ(j["created"], 1234);
    EXPECT_EQ(j["model"], "test-model");
    ASSERT_EQ(j["choices"].size(), 1u);
    EXPECT_EQ(j["choices"][0]["index"], 0);
    EXPECT_EQ(j["choices"][0]["delta"]["content"], "Hello");
    EXPECT_TRUE(j["choices"][0]["finish_reason"].is_null());
}

TEST(SseChunk, FinishReasonAndLogprobs) {
    json delta = json::object();
    json lp = {{"content", json::array()}};
    std::string s = sse_chunk("id", 1, "m", delta, "stop", lp);
    json j = parse_sse(s);
    EXPECT_EQ(j["choices"][0]["finish_reason"], "stop");
    EXPECT_TRUE(j["choices"][0].contains("logprobs"));
}

TEST(SseChunk, CompletionChunkShape) {
    std::string s = sse_completion_chunk("cmpl-9", 7, "m", "txt", nullptr);
    json j = parse_sse(s);
    EXPECT_EQ(j["object"], "text_completion");
    EXPECT_EQ(j["choices"][0]["text"], "txt");
    EXPECT_TRUE(j["choices"][0]["finish_reason"].is_null());
}

// ---------------------------------------------------------------------------
// SSEChunkWriter — must stay byte-compatible with the json-built sse_chunk
// ---------------------------------------------------------------------------

TEST(SseWriter, ContentFrameMatchesJsonBuiltChunk) {
    SSEChunkWriter w("chatcmpl-42", 1700000000, "imp-model");
    std::string got;
    httplib::DataSink sink;
    bind_sink(sink, got);
    ASSERT_TRUE(w.write_content(std::string("Hello \"world\"\n"), sink));

    json expect_delta = {{"content", "Hello \"world\"\n"}};
    json want = parse_sse(sse_chunk("chatcmpl-42", 1700000000, "imp-model", expect_delta, nullptr));
    json got_j = parse_sse(got);
    EXPECT_EQ(got_j, want);
}

TEST(SseWriter, ReasoningFrameUsesReasoningContentKey) {
    SSEChunkWriter w("id", 1, "m");
    std::string got;
    httplib::DataSink sink;
    bind_sink(sink, got);
    ASSERT_TRUE(w.write_reasoning(std::string("step 1"), sink));
    json j = parse_sse(got);
    EXPECT_EQ(j["choices"][0]["delta"]["reasoning_content"], "step 1");
    EXPECT_FALSE(j["choices"][0]["delta"].contains("content"));
}

TEST(SseWriter, EscapesIdAndModel) {
    SSEChunkWriter w("we\"ird", 1, "mo\\del");
    std::string got;
    httplib::DataSink sink;
    bind_sink(sink, got);
    ASSERT_TRUE(w.write_content(std::string("x"), sink));
    json j = parse_sse(got);  // would throw on broken escaping
    EXPECT_EQ(j["id"], "we\"ird");
    EXPECT_EQ(j["model"], "mo\\del");
}

TEST(SseWriter, NoNulByteEverReachesTheFrame) {
    // #510 regression class: a NUL leaked into every delta via an off-by-one
    // flush past pending_text's end. The writer path must never emit one even
    // when the input piece itself contains an embedded NUL.
    SSEChunkWriter w("id", 1, "m");
    std::string got;
    httplib::DataSink sink;
    bind_sink(sink, got);
    std::string piece("a\0b", 3);
    ASSERT_TRUE(w.write_content(piece, sink));
    EXPECT_EQ(got.find('\0'), std::string::npos) << "raw NUL byte in SSE frame";
    json j = parse_sse(got);  // \0 escape is fine — the frame stays valid JSON
    EXPECT_EQ(j["choices"][0]["delta"]["content"].get<std::string>(), piece);
}

TEST(JsonEscape, ControlCharsQuotesAndUtf8) {
    std::string out;
    std::string in = "q\"b\\s\nn\tt\rr\x01z – ü";  // quote, backslash, \n, \t, \r, 0x01, UTF-8
    json_escape_into(out, in.data(), in.size());
    // Wrap as a JSON string and parse back — the round trip is the contract.
    json j = json::parse("\"" + out + "\"");
    EXPECT_EQ(j.get<std::string>(), in);
}

// ---------------------------------------------------------------------------
// Reasoning split (<think>)
// ---------------------------------------------------------------------------

TEST(ExtractReasoning, ClosedThinkBlockSplits) {
    auto [reasoning, content] = extract_reasoning("<think>\nstep A\nstep B\n</think>\n\nParis.");
    EXPECT_EQ(reasoning, "step A\nstep B");
    EXPECT_EQ(content, "Paris.");
}

TEST(ExtractReasoning, UnclosedThinkIsAllReasoning) {
    // Truncated think (max_tokens hit): NOTHING may spill into content —
    // the production bug class where the non-stream path dumped the whole
    // buffer into content.
    auto [reasoning, content] = extract_reasoning("<think>\nThe user wants X, I should");
    EXPECT_EQ(reasoning, "The user wants X, I should");
    EXPECT_EQ(content, "");
}

TEST(ExtractReasoning, NoThinkTagsIsAllContent) {
    auto [reasoning, content] = extract_reasoning("Just a plain answer.");
    EXPECT_EQ(reasoning, "");
    EXPECT_EQ(content, "Just a plain answer.");
}

TEST(ExtractReasoning, MissingOpenTagTreatsPrefixAsReasoning) {
    // Template-injected <think> means the model's output may START inside the
    // block: only </think> appears in the generated text.
    auto [reasoning, content] = extract_reasoning("hmm, 2+2=4\n</think>\n4");
    EXPECT_EQ(reasoning, "hmm, 2+2=4");
    EXPECT_EQ(content, "4");
}

TEST(ExtractReasoning, LastCloseTagWins) {
    auto [reasoning, content] =
        extract_reasoning("<think>a</think>mid<think>b</think>final answer");
    EXPECT_EQ(content, "final answer");
    // Everything before the LAST </think> is reasoning (minus the first <think>).
    EXPECT_EQ(reasoning, "a</think>mid<think>b");
}

TEST(StripThinkBlock, RemovesBlockAndKeepsAnswer) {
    std::string t = "<think>\nreasoning here\n</think>\n\nThe answer is 4.";
    strip_think_block(t);
    EXPECT_EQ(t.find("<think>"), std::string::npos);
    EXPECT_EQ(t.find("</think>"), std::string::npos);
    EXPECT_NE(t.find("The answer is 4."), std::string::npos);
}

// ---------------------------------------------------------------------------
// Gemma-4 channel split
// ---------------------------------------------------------------------------

TEST(ChannelSplit, ThoughtAndFinalSeparate) {
    auto seg = split_channel_segments("<|channel>thought\nlet me think...\n<|channel>final\nParis.");
    EXPECT_NE(seg.reasoning.find("let me think..."), std::string::npos);
    EXPECT_NE(seg.content.find("Paris."), std::string::npos);
    EXPECT_EQ(seg.content.find("let me think"), std::string::npos);
    EXPECT_TRUE(seg.other.empty());
}

TEST(ChannelSplit, PreChannelTextIsContent) {
    auto seg = split_channel_segments("hello there");
    EXPECT_EQ(seg.reasoning, "");
    EXPECT_NE(seg.content.find("hello there"), std::string::npos);
}

TEST(ChannelSplit, UnknownChannelGoesToOther) {
    auto seg = split_channel_segments("<|channel>commentary\nmeta text\n<|channel>final\nanswer");
    EXPECT_NE(seg.other.find("meta text"), std::string::npos);
    EXPECT_NE(seg.content.find("answer"), std::string::npos);
    EXPECT_EQ(seg.content.find("meta text"), std::string::npos);
}

TEST(ChannelSplit, AnalysisCountsAsReasoning) {
    auto seg = split_channel_segments("<|channel>analysis\ndeep dive\n<|channel>final\nok");
    EXPECT_NE(seg.reasoning.find("deep dive"), std::string::npos);
    EXPECT_NE(seg.content.find("ok"), std::string::npos);
}

TEST(StripChannelHeaders, HeadersGoneBodyStays) {
    std::string t = "<|channel>thought\nsome thought\n<|channel>final\nThe answer";
    strip_channel_headers(t);
    EXPECT_EQ(t.find("<|channel>"), std::string::npos);
    EXPECT_NE(t.find("The answer"), std::string::npos);
}

// gpt-oss Harmony output: <|channel|>NAME<|message|>BODY<|end|> blocks.
TEST(HarmonySplit, AnalysisToReasoningFinalToContent) {
    auto seg = split_harmony_channels(
        "<|channel|>analysis<|message|>Let me think. The answer is Paris.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Paris");
    EXPECT_NE(seg.reasoning.find("Let me think"), std::string::npos);
    EXPECT_EQ(seg.content, "Paris");
    // No control markup or role names leak into either field.
    EXPECT_EQ(seg.reasoning.find("<|"), std::string::npos);
    EXPECT_EQ(seg.content.find("<|"), std::string::npos);
    EXPECT_EQ(seg.content.find("assistant"), std::string::npos);
}

TEST(HarmonySplit, FinalEndsAtReturnMarker) {
    auto seg = split_harmony_channels("<|channel|>final<|message|>The answer.<|return|>");
    EXPECT_EQ(seg.content, "The answer.");
    EXPECT_TRUE(seg.reasoning.empty());
}

TEST(HarmonySplit, CommentaryCountsAsReasoning) {
    auto seg = split_harmony_channels(
        "<|channel|>commentary<|message|>meta<|end|><|channel|>final<|message|>ans");
    EXPECT_NE(seg.reasoning.find("meta"), std::string::npos);
    EXPECT_EQ(seg.content, "ans");
}

// ---------------------------------------------------------------------------
// Streaming reasoning-split (BUGREPORT-qwen36-reasoning-leaks-into-content)
//
// StreamReasoningSplitter (reasoning_split.h) is the shared, pure demux used by
// both streaming handlers. These tests drive the REAL splitter (no copy) and
// check it against the non-streaming oracle extract_reasoning (LastCloseTagWins
// above). The bug: a model that re-deliberates after closing its first <think>
// block leaked that second reasoning pass into `content` on the streaming path,
// because the CONTENT-phase re-entry was a token-id compare that never fires for
// Qwen3.6's multi-BPE markers (request.h:84-90). The fix adds a text-scan
// re-entry + overlap holdback so the second pass is re-routed to reasoning.

using imp::server::StreamReasoningSplitter;
using imp::server::ThinkPhase;

// Drive the splitter over a piece sequence (start phase REASONING, mirroring an
// enable_thinking request whose <think> opener lives in the prompt). token id
// is -1 throughout — the realistic Qwen3.6 case where markers are decoded text,
// not single special tokens, so only the text-scan paths fire.
struct SplitDrive {
    std::string reasoning, content;
    void run(const std::vector<std::string>& pieces, ThinkPhase start = ThinkPhase::REASONING) {
        StreamReasoningSplitter s(start, /*think_start_id=*/-1, /*think_end_id=*/-1);
        for (const auto& p : pieces) {
            auto r = s.feed(p, /*token=*/-1);
            reasoning += r.reasoning;
            content += r.content;
        }
        auto r = s.finish();
        reasoning += r.reasoning;
        content += r.content;
    }
};

// The report scenario: a clean first think block, then the model re-deliberates
// ("The user wants…/I should…") inside a SECOND block before the real answer.
// Delivered whole (one burst). Non-streaming keeps the deliberation in
// reasoning_content; the splitter must too.
TEST(StreamReasoningSplit, SecondThinkPassDoesNotReachContent) {
    const std::string out =
        "<think>weather is 18C cloudy</think>"
        "<think>The user wants the forecast. I should formulate in German.</think>"
        "Morgen wird es 18 Grad.";

    auto [oracle_reasoning, oracle_content] = extract_reasoning(out);
    ASSERT_EQ(oracle_content, "Morgen wird es 18 Grad.");  // sanity: oracle is right

    SplitDrive d;
    d.run({out});

    EXPECT_EQ(d.content, oracle_content)
        << "streaming content leaked reasoning: " << d.content;
    EXPECT_EQ(d.content.find("The user wants"), std::string::npos)
        << "reasoning opener leaked into content: " << d.content;
}

// The same re-deliberation, but the second <think> opener is split across SSE
// piece boundaries ("<th" + "ink>…") — the multi-token reality for Qwen3.6. The
// overlap holdback must still catch it.
TEST(StreamReasoningSplit, ReentryAcrossPieceBoundaries) {
    SplitDrive d;
    d.run({"<think>real reasoning</think>", "<th", "ink>second pass</think>", "the answer"});

    EXPECT_EQ(d.content, "the answer") << "leaked: " << d.content;
    EXPECT_EQ(d.content.find("second pass"), std::string::npos)
        << "second reasoning pass leaked into content: " << d.content;
    EXPECT_NE(d.reasoning.find("second pass"), std::string::npos)
        << "second pass should be reasoning_content";
}

// A bare stray </think> (no opener) seen within the same burst is reclassified:
// the text before it was reasoning, not content.
TEST(StreamReasoningSplit, StrayCloseReclassifiedWithinBurst) {
    SplitDrive d;
    d.run({"<think>first</think>still thinking</think>done"});
    EXPECT_EQ(d.content, "done") << "leaked: " << d.content;
    EXPECT_EQ(d.content.find("still thinking"), std::string::npos);
}

// Guard: a single think block streams correctly (no over-eager reclassification)
// — proves the splitter is faithful for the common case.
TEST(StreamReasoningSplit, SingleBlockStreamsCleanly) {
    SplitDrive d;
    d.run({"<think>some reasoning</think>", "The answer is 4."});
    EXPECT_EQ(d.content, "The answer is 4.");
    EXPECT_EQ(d.reasoning, "some reasoning");
}

// Guard: content tokens streamed one piece at a time (post-think) must arrive
// intact despite the 7-byte overlap holdback (the last bytes flush at finish()).
TEST(StreamReasoningSplit, ContentTailNotLostByOverlap) {
    SplitDrive d;
    d.run({"<think>r</think>", "Paris", " is the", " capital."});
    EXPECT_EQ(d.content, "Paris is the capital.");
}

// Guard: an unclosed think block (budget exhausted mid-think) keeps everything
// in reasoning — nothing spills to content. Mirrors extract_reasoning's
// UnclosedThinkIsAllReasoning.
TEST(StreamReasoningSplit, UnclosedThinkIsAllReasoning) {
    SplitDrive d;
    d.run({"<think>The user wants X, I should"});
    EXPECT_EQ(d.content, "");
    EXPECT_EQ(d.reasoning, "The user wants X, I should");
}

// Guard: pass-through when reasoning extraction is off (start CONTENT) — the
// splitter must not hold back or mangle plain content.
TEST(StreamReasoningSplit, PassThroughInContentPhase) {
    SplitDrive d;
    d.run({"Hello ", "world."}, ThinkPhase::CONTENT);
    EXPECT_EQ(d.content, "Hello world.");
    EXPECT_EQ(d.reasoning, "");
}


// ---------------------------------------------------------------------------
// OpenAI compliance helpers (utils.cpp): max_completion_tokens precedence and
// the 16-entry stop-sequence cap used by parse_chat_request_params.
// ---------------------------------------------------------------------------

TEST(ParseMaxTokensField, DefaultWhenAbsent) {
    EXPECT_EQ(parse_max_tokens_field(json::object(), 8192), 8192);
}

TEST(ParseMaxTokensField, MaxTokensHonored) {
    json body = {{"max_tokens", 100}};
    EXPECT_EQ(parse_max_tokens_field(body, 8192), 100);
}

TEST(ParseMaxTokensField, MaxCompletionTokensHonored) {
    // Current OpenAI SDKs send only max_completion_tokens.
    json body = {{"max_completion_tokens", 55}};
    EXPECT_EQ(parse_max_tokens_field(body, 8192), 55);
}

TEST(ParseMaxTokensField, MaxCompletionTokensTakesPrecedence) {
    json body = {{"max_tokens", 100}, {"max_completion_tokens", 55}};
    EXPECT_EQ(parse_max_tokens_field(body, 8192), 55);
}

TEST(ParseMaxTokensField, NullAndNonNumberIgnored) {
    json body = {{"max_tokens", nullptr}, {"max_completion_tokens", "nope"}};
    EXPECT_EQ(parse_max_tokens_field(body, 8192), 8192);
}

TEST(ParseStopField, AbsentAndNull) {
    std::vector<std::string> out;
    EXPECT_FALSE(parse_stop_field(json::object(), 16, out));
    EXPECT_TRUE(out.empty());
    json body = {{"stop", nullptr}};
    EXPECT_FALSE(parse_stop_field(body, 16, out));
    EXPECT_TRUE(out.empty());
}

TEST(ParseStopField, SingleString) {
    std::vector<std::string> out;
    json body = {{"stop", "END"}};
    EXPECT_FALSE(parse_stop_field(body, 16, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], "END");
}

TEST(ParseStopField, ArrayUpTo16Kept) {
    // 16 sequences (Anthropic clients routinely send >4) all survive.
    json arr = json::array();
    for (int i = 0; i < 16; ++i)
        arr.push_back("s" + std::to_string(i));
    json body = {{"stop", arr}};
    std::vector<std::string> out;
    EXPECT_FALSE(parse_stop_field(body, 16, out));
    ASSERT_EQ(out.size(), 16u);
    EXPECT_EQ(out[15], "s15");
}

TEST(ParseStopField, TruncatesBeyondCapAndReportsIt) {
    json arr = json::array();
    for (int i = 0; i < 20; ++i)
        arr.push_back("s" + std::to_string(i));
    json body = {{"stop", arr}};
    std::vector<std::string> out;
    EXPECT_TRUE(parse_stop_field(body, 16, out));
    ASSERT_EQ(out.size(), 16u);
    EXPECT_EQ(out[0], "s0");
    EXPECT_EQ(out[15], "s15");
}

TEST(ParseStopField, NonStringEntriesSkipped) {
    json body = {{"stop", json::array({"a", 5, nullptr, "b"})}};
    std::vector<std::string> out;
    EXPECT_FALSE(parse_stop_field(body, 16, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], "a");
    EXPECT_EQ(out[1], "b");
}

// -----------------------------------------------------------------------------
// Split multi-byte characters. A BPE vocabulary cuts "größer" into a piece
// ending in 0xC3 and one starting with 0xB6; each delta is serialized alone, so
// without stitching dump_safe turns the halves into U+FFFD and the client reads
// "gr??ßer" — while the same generation is correct over the non-streaming path.
// -----------------------------------------------------------------------------

TEST(Utf8Stitch, RejoinsCharacterSplitAcrossTwoPieces) {
    Utf8Stitch st;
    EXPECT_EQ(st.feed("gr\xC3"), "gr");        // lead byte held back
    EXPECT_EQ(st.feed("\xB6\xC3\x9F"), "ö\xC3\x9F");
}

TEST(Utf8Stitch, PassesCompleteInputThrough) {
    Utf8Stitch st;
    EXPECT_EQ(st.feed("Straße"), "Straße");
    EXPECT_EQ(st.feed(""), "");
    EXPECT_EQ(st.feed("weiß"), "weiß");
}

TEST(Utf8Stitch, ReassemblesFourByteCharacterOneByteAtATime) {
    Utf8Stitch st;  // U+1F600, the worst case: 4 bytes over 4 tokens
    EXPECT_EQ(st.feed("\xF0"), "");
    EXPECT_EQ(st.feed("\x9F"), "");
    EXPECT_EQ(st.feed("\x98"), "");
    EXPECT_EQ(st.feed("\x80"), "😀");
}

TEST(Utf8Stitch, InvalidLeadFollowedByContinuationsIsNotHeldBackForever) {
    // feed()'s `<= 3` bound exists because utf8_complete_len parks on an invalid
    // lead byte, and an invalid lead followed by continuation bytes parks
    // arbitrarily far back - without the bound those bytes are carried forever
    // and the stream stalls.
    //
    // DoesNotStallOrLoseBytesOnInvalidInput below cannot reach that: its input
    // "\xFF\xFE\xFD\xFC\xFB" contains no continuation byte (0x80-0xBF), so the
    // walk-back stops at the last byte, the tail is 1, and the bound never
    // binds. Widening the bound leaves that test green.
    Utf8Stitch st;
    const std::string in1 = "\xFF\x80\x80\x80";  // invalid lead + 3 continuations
    const std::string in2 = "ok";
    const std::string first = st.feed(in1);
    EXPECT_FALSE(first.empty())
        << "a 4-byte tail that can never complete must be passed through, not parked";
    const std::string out = first + st.feed(in2);
    EXPECT_EQ(out.size(), in1.size() + in2.size()) << "bytes were dropped or stalled";
    EXPECT_EQ(out.substr(out.size() - 2), "ok");
}

TEST(Utf8Stitch, DoesNotStallOrLoseBytesOnInvalidInput) {
    // Bytes that can never complete a character must keep moving: a stitch that
    // waits for a completion that never comes would hang the stream. Nothing is
    // dropped either — dump_safe deals with what is genuinely ill-formed.
    Utf8Stitch st;
    const std::string in1 = "\xFF\xFE\xFD\xFC\xFB", in2 = "ok";
    // Sequenced deliberately: feed() is stateful and the evaluation order of
    // `f(a) + f(b)` is unspecified.
    const std::string first = st.feed(in1);
    const std::string out = first + st.feed(in2);
    EXPECT_EQ(out.size(), in1.size() + in2.size());
    EXPECT_EQ(out.substr(out.size() - 2), "ok");
}

TEST(HoldbackDecision, FlushCutLandsOnCharacterBoundary) {
    // Even with well-formed input the byte-offset cut can fall inside a
    // character; the flushed prefix must still be decodable on its own.
    // "größer" is 8 bytes (ö and ß are 2 each); max_stop_len 4 puts the raw cut
    // at byte 5 — the first half of ß.
    const std::string pending = "größer";
    auto d = imp::stream::holdback_decision(pending, 4, {"</tool>"});
    EXPECT_FALSE(d.complete_match);
    const std::string flushed = pending.substr(0, d.flush_len);
    EXPECT_EQ(imp::stream::utf8_complete_len(flushed), flushed.size());
    EXPECT_EQ(flushed, "grö");
}

TEST(HoldbackDecision, StopMatchStillCutsExactlyAtTheMatch) {
    const std::string pending = "grüß dich</tool>";
    auto d = imp::stream::holdback_decision(pending, 8, {"</tool>"});
    EXPECT_TRUE(d.complete_match);
    EXPECT_EQ(pending.substr(0, d.flush_len), "grüß dich");
}

}  // namespace

// ---- the empty-answer diagnostic -----------------------------------------
//
// An empty `content` beside a full `reasoning_content` is not a defect: the
// reply shares the token budget with the thinking, and on a long conversation
// the thinking can consume it before the answer starts. The server logs a line
// saying so, because the alternative is a caller bisecting an engine that did
// what it was asked (measured on Qwen3.8-27B: empty replies at max_tokens 260,
// 74/74 clean at 600).
//
// Covered here rather than by a live run because the state depends on how long
// the model chooses to think: it showed up repeatedly across 74-turn sessions
// and could not be produced on demand with a short prompt, a tiny budget or a
// stop sequence. A rule that fires rarely is the one that needs a test.

TEST(AnswerLostToReasoning, FiresOnlyWhenThinkingAteTheWholeReply) {
    EXPECT_TRUE(answer_lost_to_reasoning(false, "", "the model was still thinking"));
}

TEST(AnswerLostToReasoning, StaysQuietWhenThereIsAnAnswer) {
    EXPECT_FALSE(answer_lost_to_reasoning(false, "Paris", "some thinking"));
    // Short answers are answers: "name only" prompts are satisfied by one word.
    EXPECT_FALSE(answer_lost_to_reasoning(false, "8347", "some thinking"));
}

TEST(AnswerLostToReasoning, StaysQuietWithoutReasoning) {
    // Empty content and no thinking either is a different situation (a stop
    // sequence that matched immediately, for one) and not this diagnostic.
    EXPECT_FALSE(answer_lost_to_reasoning(false, "", ""));
}

TEST(AnswerLostToReasoning, StaysQuietOnAToolCall) {
    // A tool call legitimately carries a null/empty content; saying the answer
    // was lost would be wrong on every forced-tool request.
    EXPECT_FALSE(answer_lost_to_reasoning(true, "", "thinking about which tool"));
    EXPECT_FALSE(answer_lost_to_reasoning(true, "", ""));
}

// A floored KV pool is not "the last request failed", it is "this process
// cannot serve". Reported from production: `docker compose restart` while the
// previous process still held the card came up with 16 blocks against a planned
// 3066, and /health said ok throughout.
TEST(HealthUnservable, FlooredPoolIsUnservableAndNamesThePool) {
    const std::string why = health_unservable_reason(false, true, 16, 32);
    EXPECT_FALSE(why.empty());
    EXPECT_NE(why.find("512"), std::string::npos) << "16 blocks of 32 is the capacity to state";
    EXPECT_NE(why.find("restart"), std::string::npos) << "the only fix belongs in the message";
    EXPECT_STREQ(health_unservable_code(false, true), "kv_pool_floored");
}

TEST(HealthUnservable, HealthyPoolSaysNothing) {
    EXPECT_TRUE(health_unservable_reason(false, false, 3066, 32).empty());
    EXPECT_STREQ(health_unservable_code(false, false), "");
}

// The engine being wedged already answered 503 before this existed (#874); it
// keeps that answer and only gains the identifier.
TEST(HealthUnservable, FaultedEngineKeepsItsOwnCode) {
    EXPECT_FALSE(health_unservable_reason(true, false, 3066, 32).empty());
    EXPECT_STREQ(health_unservable_code(true, false), "engine_faulted");
    // Both at once: the wedged engine is the louder fault and wins the code,
    // but a caller must still not be told the server is fine.
    EXPECT_STREQ(health_unservable_code(true, true), "engine_faulted");
    EXPECT_FALSE(health_unservable_reason(true, true, 16, 32).empty());
}

// Before a model is loaded there is no pool to judge, and an unknown capacity
// must not read as a floored one: the server is starting, not broken.
TEST(HealthUnservable, UnknownCapacityIsNotAVerdict) {
    EXPECT_TRUE(health_unservable_reason(false, false, -1, -1).empty());
    EXPECT_STREQ(health_unservable_code(false, false), "");
}

// ---- #1554: tool-argument chunks end on codepoint boundaries ----
//
// Buffered tool calls were sliced every 48 BYTES and each slice JSON-encoded on
// its own, so a multi-byte character straddling a boundary was cut in half and
// dump_safe replaced each half with U+FFFD. The client concatenates the pieces
// and gets a corrupt argument. Two of the three dialects did this; /v1/responses
// does not chunk and was immune.

TEST(Utf8ChunkLen, NeverSplitsAMultiByteCharacter) {
    // "ü" is 2 bytes. With max=5 over "aaaaü", a byte slice takes 5 bytes and
    // cuts the ü in half.
    const std::string s =
        "aaaa\xc3\xbc"
        "bbbb";
    const size_t n = utf8_chunk_len(s, 0, 5);
    EXPECT_EQ(n, 4u) << "must stop before the split character, not inside it";
    // Continuing from there takes the whole character.
    EXPECT_EQ(utf8_chunk_len(s, 4, 5), 5u);
}

TEST(Utf8ChunkLen, HandlesThreeAndFourByteSequences) {
    const std::string cjk = "ab\xe4\xb8\xad";  // 'ab' + U+4E2D (3 bytes)
    EXPECT_EQ(utf8_chunk_len(cjk, 0, 3), 2u);  // stop before the CJK char
    EXPECT_EQ(utf8_chunk_len(cjk, 0, 4), 2u);
    EXPECT_EQ(utf8_chunk_len(cjk, 0, 5), 5u);       // whole string fits
    const std::string emoji = "x\xf0\x9f\x98\x80";  // 'x' + U+1F600 (4 bytes)
    EXPECT_EQ(utf8_chunk_len(emoji, 0, 3), 1u);
    EXPECT_EQ(utf8_chunk_len(emoji, 0, 5), 5u);
}

TEST(Utf8ChunkLen, ReassemblingTheChunksReproducesTheInput) {
    // The property that actually matters: chunking and concatenating is the
    // identity, for any max, on real tool-argument content.
    const std::string args = R"({"city":"München","note":"Grüße aus Köln 😀","path":"/tmp/übung"})";
    for (size_t max : {1u, 2u, 3u, 4u, 5u, 7u, 16u, 48u}) {
        std::string rebuilt;
        for (size_t off = 0; off < args.size();) {
            const size_t n = utf8_chunk_len(args, off, max);
            ASSERT_GT(n, 0u) << "max=" << max << " off=" << off;
            const std::string piece = args.substr(off, n);
            // Every piece must be valid UTF-8 on its own - that is the whole
            // point, since each one is JSON-encoded separately.
            EXPECT_EQ(imp::stream::utf8_complete_len(piece), piece.size())
                << "max=" << max << " produced a piece ending mid-character";
            rebuilt += piece;
            off += n;
        }
        EXPECT_EQ(rebuilt, args) << "max=" << max;
    }
}

TEST(Utf8ChunkLen, AlwaysMakesProgress) {
    const std::string s = "\xc3\xbc\xc3\xbc";  // two 2-byte chars
    // max=1 cannot fit a character; it must still advance rather than loop.
    EXPECT_GT(utf8_chunk_len(s, 0, 1), 0u);
    EXPECT_EQ(utf8_chunk_len(s, 4, 8), 0u);  // nothing left
}

// ---- #1607: request-body nesting depth, counted without parsing ----

TEST(JsonNestingDepth, CountsPlainNesting) {
    EXPECT_EQ(json_nesting_depth("{}", 100), 1);
    EXPECT_EQ(json_nesting_depth("[[[]]]", 100), 3);
    EXPECT_EQ(json_nesting_depth(R"({"a":{"b":{"c":1}}})", 100), 3);
    EXPECT_EQ(json_nesting_depth("", 100), 0);
    EXPECT_EQ(json_nesting_depth("123", 100), 0);
}

TEST(JsonNestingDepth, IgnoresBracesInsideStrings) {
    // A brace in a string is text. Counting it would reject legitimate bodies -
    // a prompt containing JSON, which is most agent traffic.
    EXPECT_EQ(json_nesting_depth(R"({"a":"{{{{{{{{{{"})", 100), 1);
    EXPECT_EQ(json_nesting_depth(R"({"a":"[[[["})", 100), 1);
    // ...including an escaped quote, which must not end the string early.
    EXPECT_EQ(json_nesting_depth(R"({"a":"he said \"{{{\" to me"})", 100), 1);
    EXPECT_EQ(json_nesting_depth(R"({"a":"trailing backslash \\"})", 100), 1);
}

TEST(JsonNestingDepth, TracksTheMaximumNotTheFinalDepth) {
    // Two siblings, each one level inside the root: depth returns to 1 between
    // them and the maximum is 2, not the sum and not the last value.
    EXPECT_EQ(json_nesting_depth(R"({"a":{"b":1},"c":{"d":1}})", 100), 2);
    // Deepen only the second sibling: the maximum has to follow it.
    EXPECT_EQ(json_nesting_depth(R"({"a":{"b":1},"c":{"d":{"e":{"f":1}}}})", 100), 4);
}

TEST(JsonNestingDepth, StopsEarlyOnceTheLimitIsExceeded) {
    // 200 levels against a limit of 100: the answer only has to prove "over".
    std::string deep(200, '[');
    deep.append(200, ']');
    const int d = json_nesting_depth(deep, 100);
    EXPECT_GT(d, 100);
    EXPECT_LE(d, 101) << "the scan must stop at the limit, not count to 200";
}

TEST(JsonNestingDepth, TheHostileBodyIsOverTheLimitAndARealOneIsNot) {
    // The payload from the issue: ~100 KB of '[' segfaults nlohmann (measured:
    // 50k levels parse fine, 100k crash).
    std::string bomb(100000, '[');
    EXPECT_GT(json_nesting_depth(bomb, 100), 100);

    // A realistic body with tools, nested schema and multi-part content is
    // nowhere near it. This is the arm that says the limit is not too tight.
    const std::string real =
        R"({"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],)"
        R"("tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object",)"
        R"("properties":{"a":{"type":"array","items":{"type":"object","properties":)"
        R"({"b":{"type":"string"}}}}}}}}]})";
    EXPECT_LE(json_nesting_depth(real, 100), 12);
}
