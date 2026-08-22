// The two logprobs shapes and the token attribution behind them
// (#1588, #1589, #1601).
//
// Three separate defects met here:
//
//   * The streaming chat path attached logprobs only when the request carried
//     NO stop sequence. With any stop present every chunk went out through the
//     logprob-free writer, so the field was absent from the whole stream.
//   * /v1/completions returned the CHAT logprobs object on a `text_completion`
//     response. An OpenAI SDK reading `.logprobs.tokens` sees nothing there.
//   * safe_token_json and token_bytes_json, which every one of those paths
//     calls, had no test in any lane.

#include "stream_pipeline.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using imp::stream::TokenSpans;

namespace {

imp::TokenLogprobInfo lp(const std::string& text, float logprob,
                         std::vector<std::pair<std::string, float>> top = {}) {
    imp::TokenLogprobInfo info;
    info.text = text;
    info.logprob = logprob;
    for (auto& [t, l] : top)
        info.top.push_back({0, l, t});
    return info;
}

// ---------------------------------------------------------------------------
// #1601 - the two token helpers every logprobs path calls
// ---------------------------------------------------------------------------

TEST(LogprobTokens, ValidUtf8SurvivesUnchanged) {
    EXPECT_EQ(safe_token_json("hello").get<std::string>(), "hello");
    EXPECT_EQ(safe_token_json("グ").get<std::string>(), "グ");
    EXPECT_EQ(safe_token_json("").get<std::string>(), "");
}

// A token is a byte string, not a string: a multi-byte codepoint is split
// across two tokens by most BPE vocabularies, so half of one reaches here.
// It must not throw and must not corrupt the surrounding JSON.
TEST(LogprobTokens, AHalfCodepointIsRepresentableAndSerialises) {
    const std::string half = "\xe3\x82";  // first two bytes of グ
    const json j = safe_token_json(half);
    EXPECT_NO_THROW((void)j.dump());
    json wrapper = {{"token", j}};
    EXPECT_NO_THROW((void)wrapper.dump());
}

// `bytes` exists precisely so a client can reassemble what `token` cannot
// represent. It is the raw bytes, unsigned, in order.
TEST(LogprobTokens, BytesAreTheRawUnsignedBytes) {
    const json b = token_bytes_json("AB");
    ASSERT_TRUE(b.is_array());
    ASSERT_EQ(b.size(), 2u);
    EXPECT_EQ(b[0].get<int>(), 65);
    EXPECT_EQ(b[1].get<int>(), 66);

    const json g = token_bytes_json("グ");
    ASSERT_EQ(g.size(), 3u);
    EXPECT_EQ(g[0].get<int>(), 0xE3);
    EXPECT_EQ(g[1].get<int>(), 0x82);
    EXPECT_EQ(g[2].get<int>(), 0xB0);

    // The half codepoint above: two bytes, no replacement character.
    const json h = token_bytes_json("\xe3\x82");
    ASSERT_EQ(h.size(), 2u);
    EXPECT_EQ(h[0].get<int>(), 0xE3);
}

// ---------------------------------------------------------------------------
// #1589 - two shapes, and they are not interchangeable
// ---------------------------------------------------------------------------

TEST(LogprobShapes, ChatShapeIsAnArrayUnderContent) {
    std::vector<imp::TokenLogprobInfo> lps = {lp("Hel", -0.5f, {{"Hel", -0.5f}, {"Hi", -1.5f}}),
                                              lp("lo", -0.25f)};
    const json j = chat_logprobs_json(lps, lps.size());

    ASSERT_TRUE(j.contains("content"));
    ASSERT_EQ(j["content"].size(), 2u);
    EXPECT_EQ(j["content"][0]["token"].get<std::string>(), "Hel");
    EXPECT_FLOAT_EQ(j["content"][0]["logprob"].get<float>(), -0.5f);
    EXPECT_TRUE(j["content"][0]["bytes"].is_array());
    // top_logprobs is an ARRAY of objects here.
    ASSERT_TRUE(j["content"][0]["top_logprobs"].is_array());
    EXPECT_EQ(j["content"][0]["top_logprobs"][1]["token"].get<std::string>(), "Hi");

    // What the Completions reader would look for, and not find.
    EXPECT_FALSE(j.contains("tokens"));
    EXPECT_FALSE(j.contains("text_offset"));
}

TEST(LogprobShapes, CompletionsShapeIsFourParallelArrays) {
    std::vector<imp::TokenLogprobInfo> lps = {lp("Hel", -0.5f, {{"Hel", -0.5f}, {"Hi", -1.5f}}),
                                              lp("lo", -0.25f)};
    const json j = completions_logprobs_json(lps, lps.size(), "Hello");

    ASSERT_TRUE(j.contains("tokens"));
    EXPECT_EQ(j["tokens"][0].get<std::string>(), "Hel");
    EXPECT_EQ(j["tokens"][1].get<std::string>(), "lo");
    EXPECT_FLOAT_EQ(j["token_logprobs"][1].get<float>(), -0.25f);
    // top_logprobs is an OBJECT per position here, token -> logprob.
    ASSERT_TRUE(j["top_logprobs"][0].is_object());
    EXPECT_FLOAT_EQ(j["top_logprobs"][0]["Hi"].get<float>(), -1.5f);
    // Offsets walk the completion string.
    EXPECT_EQ(j["text_offset"][0].get<int>(), 0);
    EXPECT_EQ(j["text_offset"][1].get<int>(), 3);

    // And what the Chat reader would look for, and not find.
    EXPECT_FALSE(j.contains("content"));
}

// The offsets are derived by walking the completion, not by summing token
// lengths, because the two disagree whenever the assembled text was trimmed.
TEST(LogprobShapes, OffsetsStopAdvancingWhenTheTextDisagrees) {
    std::vector<imp::TokenLogprobInfo> lps = {lp("ab", -1.0f), lp("cd", -1.0f), lp("ef", -1.0f)};
    // A stop sequence trimmed everything after "abcd".
    const json j = completions_logprobs_json(lps, lps.size(), "abcd");
    EXPECT_EQ(j["text_offset"][0].get<int>(), 0);
    EXPECT_EQ(j["text_offset"][1].get<int>(), 2);
    // "ef" is not in the returned text: the offset does not run past its end.
    EXPECT_EQ(j["text_offset"][2].get<int>(), 4);
    EXPECT_LE(j["text_offset"][2].get<int>(), 4);
}

TEST(LogprobShapes, TheStreamingVariantCarriesOneTokenAndItsOffset) {
    const json j = completions_logprobs_json_one(lp("lo", -0.25f, {{"lo", -0.25f}}), 3);
    ASSERT_EQ(j["tokens"].size(), 1u);
    EXPECT_EQ(j["tokens"][0].get<std::string>(), "lo");
    EXPECT_EQ(j["text_offset"][0].get<int>(), 3);
    EXPECT_FLOAT_EQ(j["top_logprobs"][0]["lo"].get<float>(), -0.25f);
}

TEST(LogprobShapes, LimitTruncatesBothShapes) {
    std::vector<imp::TokenLogprobInfo> lps = {lp("a", -1.0f), lp("b", -1.0f), lp("c", -1.0f)};
    EXPECT_EQ(chat_logprobs_json(lps, 2)["content"].size(), 2u);
    EXPECT_EQ(completions_logprobs_json(lps, 2, "abc")["tokens"].size(), 2u);
}

// ---------------------------------------------------------------------------
// #1588 - which token produced which held-back bytes
// ---------------------------------------------------------------------------

TEST(TokenSpansTest, AFlushOnATokenBoundaryAttributesEachPiece) {
    TokenSpans t;
    t.append(3, 0);  // "Hel"
    t.append(2, 1);  // "lo"

    const auto out = t.flush(5);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].offset, 0u);
    EXPECT_EQ(out[0].length, 3u);
    EXPECT_EQ(out[0].token_index, 0);
    EXPECT_EQ(out[1].offset, 3u);
    EXPECT_EQ(out[1].length, 2u);
    EXPECT_EQ(out[1].token_index, 1);
    EXPECT_TRUE(t.empty());
}

// The case the live counter gets wrong: the stop matcher cuts mid-token.
TEST(TokenSpansTest, AFlushInsideATokenIsNotAttributed) {
    TokenSpans t;
    t.append(3, 0);
    t.append(4, 1);

    const auto out = t.flush(5);  // all of token 0, two bytes of token 1
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].token_index, 0);
    EXPECT_EQ(out[0].length, 3u);
    // Two bytes belonging to a token that is not finished: -1, not "the
    // nearest index", because guessing here is what produces a wrong logprob.
    EXPECT_EQ(out[1].token_index, -1);
    EXPECT_EQ(out[1].offset, 3u);
    EXPECT_EQ(out[1].length, 2u);
}

// Whatever is left keeps its identity, rebased to the shortened buffer.
TEST(TokenSpansTest, TheRemainderIsRebasedNotDropped) {
    TokenSpans t;
    t.append(2, 7);
    t.append(2, 8);
    t.append(2, 9);

    (void)t.flush(2);  // drop token 7's bytes
    const auto out = t.flush(4);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].offset, 0u);
    EXPECT_EQ(out[0].token_index, 8);
    EXPECT_EQ(out[1].offset, 2u);
    EXPECT_EQ(out[1].token_index, 9);
    EXPECT_TRUE(t.empty());
}

TEST(TokenSpansTest, EmptyAppendsAndZeroFlushesAreNoOps) {
    TokenSpans t;
    t.append(0, 3);
    EXPECT_TRUE(t.empty());
    EXPECT_TRUE(t.flush(0).empty());

    t.append(2, 4);
    EXPECT_TRUE(t.flush(0).empty());
    EXPECT_FALSE(t.empty());
}

// Nothing recorded but bytes to flush: the buffer was filled by a path that
// does not track tokens. It must still emit, just unattributed.
TEST(TokenSpansTest, UntrackedBytesEmitAsUnattributed) {
    TokenSpans t;
    const auto out = t.flush(4);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].token_index, -1);
    EXPECT_EQ(out[0].length, 4u);
}

}  // namespace
