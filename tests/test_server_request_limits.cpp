// What one HTTP request is allowed to cost (#1614, #1615, #1616, #1617, #1618).
//
// Every case here is a place where the server counted a request as one unit
// while the request decided how much work that unit was, or where a limit
// keyed on something the client writes. The pieces that can be reached without
// a running server are tested here; the rest is in the API battery.

#include "handlers.h"
#include "rate_limit.h"
#include "utils.h"

#include <gtest/gtest.h>

#include <chrono>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// #1614 - the rate-limit key
// ---------------------------------------------------------------------------

TEST(RateLimitKey, IgnoresForwardedForFromAnUntrustedPeer) {
    RateLimiter state;
    // No --trusted-proxy: the header is a string the client writes, and
    // believing it means one client is unlimited buckets.
    EXPECT_EQ(state.key("203.0.113.7", "1.2.3.4"), "203.0.113.7");
    EXPECT_EQ(state.key("203.0.113.7", ""), "203.0.113.7");
}

TEST(RateLimitKey, BelievesForwardedForFromANamedProxy) {
    RateLimiter state;
    state.trusted_proxies.insert("10.0.0.5");
    EXPECT_EQ(state.key("10.0.0.5", "1.2.3.4"), "1.2.3.4");
    // A proxy appends, so the first element is the original client.
    EXPECT_EQ(state.key("10.0.0.5", "1.2.3.4, 10.0.0.5"), "1.2.3.4");
    EXPECT_EQ(state.key("10.0.0.5", "  1.2.3.4 , 10.0.0.5"), "1.2.3.4");
    // Still the peer for anyone else.
    EXPECT_EQ(state.key("198.51.100.9", "1.2.3.4"), "198.51.100.9");
}

TEST(RateLimitKey, BoundsTheKeyLength) {
    RateLimiter state;
    state.trusted_proxies.insert("10.0.0.5");
    const std::string huge(4096, 'a');
    EXPECT_LE(state.key("10.0.0.5", huge).size(), 64u);
    // An all-whitespace header is not an identity.
    EXPECT_EQ(state.key("10.0.0.5", "   "), "10.0.0.5");
}

// ---------------------------------------------------------------------------
// #1614 - the tracker is bounded
// ---------------------------------------------------------------------------

TEST(RateLimiter, EvictsBucketsThatHaveGoneQuiet) {
    RateLimiter state;
    state.limit = 1000;  // high enough that nothing is refused here
    const auto t0 = std::chrono::steady_clock::now();

    // 4000 distinct keys, which is what a client gets for free when the key is
    // a header it writes. All still inside the window.
    for (int i = 0; i < 4000; i++)
        state.allow("client-" + std::to_string(i), t0);
    EXPECT_EQ(state.tracked(), 4000u);

    // Two minutes later every one of those buckets is stale. The sweep runs
    // once per 256 admissions, so drive it past that with one live key.
    const auto t1 = t0 + std::chrono::seconds(120);
    for (int i = 0; i < 300; i++)
        state.allow("still-here", t1);

    // Before the sweep existed this stayed at 4001 for the life of the
    // process: only the bucket being asked about was ever pruned.
    EXPECT_EQ(state.tracked(), 1u);
}

TEST(RateLimiter, TheWindowSlides) {
    RateLimiter state;
    state.limit = 2;
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_TRUE(state.allow("a", t0));
    EXPECT_TRUE(state.allow("a", t0));
    EXPECT_FALSE(state.allow("a", t0));
    // 59 s later the first two are still inside the 60 s window.
    EXPECT_FALSE(state.allow("a", t0 + std::chrono::seconds(59)));
    // 61 s later they are not.
    EXPECT_TRUE(state.allow("a", t0 + std::chrono::seconds(61)));
}

TEST(RateLimiter, RefusesPastTheLimitAndAdmitsBelowIt) {
    RateLimiter state;
    state.limit = 3;
    EXPECT_TRUE(state.allow("a"));
    EXPECT_TRUE(state.allow("a"));
    EXPECT_TRUE(state.allow("a"));
    EXPECT_FALSE(state.allow("a"));
    // A different key is a different bucket.
    EXPECT_TRUE(state.allow("b"));
}

// ---------------------------------------------------------------------------
// #1618 - what goes back into an error body
// ---------------------------------------------------------------------------

TEST(SanitizeForEcho, ReplacesEverythingOutsidePrintableAscii) {
    EXPECT_EQ(sanitize_for_echo("/v1/chat/completions", 128), "/v1/chat/completions");
    // The byte that made json::dump() throw.
    EXPECT_EQ(sanitize_for_echo("/v1/\x80\xff", 128), "/v1/..");
    EXPECT_EQ(sanitize_for_echo("/a\nb\tc", 128), "/a.b.c");
}

TEST(SanitizeForEcho, TruncatesWithAMarker) {
    const std::string huge(4096, 'x');
    const std::string out = sanitize_for_echo(huge, 128);
    EXPECT_EQ(out.size(), 131u);  // 128 + "..."
    EXPECT_EQ(out.substr(128), "...");
}

// The whole point: the sanitised string survives serialisation. Before the
// fix the 404 handler called .dump() on the raw path, which throws
// json::type_error.316 on ill-formed UTF-8 - so a 404 became a 500 with an
// empty body, which is the shape the envelope exists to prevent.
TEST(SanitizeForEcho, TheResultSerialisesWhereTheRawPathThrows) {
    const std::string bad_path = "/v1/\x80\xff";

    nlohmann::json raw = {{"error", {{"message", "Unknown endpoint: GET " + bad_path}}}};
    EXPECT_THROW((void)raw.dump(), nlohmann::json::type_error);

    nlohmann::json clean = {
        {"error", {{"message", "Unknown endpoint: GET " + sanitize_for_echo(bad_path, 128)}}}};
    EXPECT_NO_THROW((void)clean.dump());
    EXPECT_NO_THROW((void)dump_safe(raw));  // the other half of the fix
}

// ---------------------------------------------------------------------------
// #1590 / #1595 / #1602 - what the response says about itself
// ---------------------------------------------------------------------------

// The engine has two finish reasons OpenAI does not. Shipping them verbatim on
// a 200 sends a client through its default branch, where a failed generation
// looks like a normal one.
TEST(FinishReason, EngineOnlyValuesMapIntoTheOpenAiEnum) {
    EXPECT_STREQ(openai_finish_reason("cancelled"), "length");
    EXPECT_STREQ(openai_finish_reason("capacity"), "length");
}

TEST(FinishReason, TheEnumMembersPassThroughUnchanged) {
    for (const char* v : {"stop", "length", "tool_calls", "content_filter", "function_call"})
        EXPECT_STREQ(openai_finish_reason(v), v);
    // nullptr is "the generation ended without a recorded reason", which is a
    // normal stop, not a crash.
    EXPECT_STREQ(openai_finish_reason(nullptr), "stop");
}

TEST(ErrorEnvelope, ParamAndCodeAppearOnlyWhenSupplied) {
    httplib::Response res;
    send_json_error(res, 400, "invalid_request_error", "plain");
    json j = json::parse(res.body);
    EXPECT_EQ(j["error"]["type"], "invalid_request_error");
    // A client that checks `"code" in err` must not see a key saying nothing.
    EXPECT_FALSE(j["error"].contains("param"));
    EXPECT_FALSE(j["error"].contains("code"));

    httplib::Response res2;
    send_json_error(res2, 400, "invalid_request_error", "too long", "messages", "context_length_exceeded");
    json j2 = json::parse(res2.body);
    EXPECT_EQ(j2["error"]["param"], "messages");
    EXPECT_EQ(j2["error"]["code"], "context_length_exceeded");
    EXPECT_EQ(res2.status, 400);
}

// The fingerprint exists so a client can tell "same backend" from "different
// backend" in one comparison. Both halves matter: stable, and not constant.
TEST(SystemFingerprint, IsStablePerModelAndDiffersAcrossModels) {
    const std::string a = system_fingerprint("Qwen3-8B-Q8_0.gguf");
    EXPECT_EQ(a, system_fingerprint("Qwen3-8B-Q8_0.gguf"));
    EXPECT_NE(a, system_fingerprint("Qwen3-4B-Q8_0.gguf"));
    EXPECT_EQ(a.rfind("fp_", 0), 0u);
    EXPECT_EQ(a.size(), 19u);  // "fp_" + 16 hex
}

TEST(SystemFingerprint, AnEmptyModelNameStillProducesOne) {
    const std::string f = system_fingerprint("");
    EXPECT_EQ(f.rfind("fp_", 0), 0u);
    EXPECT_NE(f, system_fingerprint("x"));
}

// ---------------------------------------------------------------------------
// The Anthropic error envelope (#1551, #1556, #1561)
// ---------------------------------------------------------------------------

TEST(AnthropicEnvelope, PathDecidesTheShape) {
    EXPECT_TRUE(is_anthropic_path("/v1/messages"));
    EXPECT_TRUE(is_anthropic_path("/v1/messages/count_tokens"));
    EXPECT_FALSE(is_anthropic_path("/v1/chat/completions"));
    EXPECT_FALSE(is_anthropic_path("/v1/completions"));
}

// A 429 came back in the OpenAI shape on /v1/messages, so an Anthropic SDK
// could not classify it: no top-level "type":"error" (#1551).
TEST(AnthropicEnvelope, DialectErrorPicksTheRightWrapper) {
    httplib::Response anth;
    send_dialect_error(anth, "/v1/messages", 429, "rate_limit_error", "overloaded_error", "busy");
    json a = json::parse(anth.body);
    EXPECT_EQ(a["type"], "error");
    EXPECT_EQ(a["error"]["type"], "overloaded_error");
    EXPECT_EQ(anth.status, 429);

    httplib::Response oai;
    send_dialect_error(oai, "/v1/chat/completions", 429, "rate_limit_error", "overloaded_error", "busy");
    json o = json::parse(oai.body);
    EXPECT_FALSE(o.contains("type"));
    EXPECT_EQ(o["error"]["type"], "rate_limit_error");
}

TEST(AnthropicEnvelope, RequestIdRidesBodyAndHeader) {
    httplib::Response res;
    send_anthropic_error(res, 500, "api_error", "boom", "req_imp_0000000000000001");
    json j = json::parse(res.body);
    EXPECT_EQ(j["request_id"], "req_imp_0000000000000001");
    EXPECT_EQ(res.get_header_value("request-id"), "req_imp_0000000000000001");
}

TEST(AnthropicEnvelope, NoRequestIdMeansNoKeyAndNoHeader) {
    httplib::Response res;
    send_anthropic_error(res, 400, "invalid_request_error", "bad");
    json j = json::parse(res.body);
    EXPECT_FALSE(j.contains("request_id"));
    EXPECT_FALSE(res.has_header("request-id"));
}

// server_error and capacity_error are this server's inventions; neither is an
// Anthropic error type, and both reached SDK clients verbatim (#1556).
TEST(AnthropicEnvelope, TypeTranslationCoversTheInventedOnes) {
    EXPECT_STREQ(anthropic_error_type_for("server_error", 500), "api_error");
    EXPECT_STREQ(anthropic_error_type_for("capacity_error", 503), "overloaded_error");
    // The name decides, not the status: at 500 the fallback would answer
    // api_error anyway, so only a 4xx proves the mapping is doing the work.
    EXPECT_STREQ(anthropic_error_type_for("server_error", 400), "api_error");
    EXPECT_STREQ(anthropic_error_type_for("capacity_error", 429), "overloaded_error");
    // Already-valid types pass through.
    EXPECT_STREQ(anthropic_error_type_for("invalid_request_error", 400), "invalid_request_error");
    EXPECT_STREQ(anthropic_error_type_for("rate_limit_error", 429), "rate_limit_error");
    EXPECT_STREQ(anthropic_error_type_for("not_found_error", 404), "not_found_error");
    // Unknown: the status decides, so nothing outside the set can escape.
    EXPECT_STREQ(anthropic_error_type_for("something_new", 500), "api_error");
    EXPECT_STREQ(anthropic_error_type_for("something_new", 422), "invalid_request_error");
    EXPECT_STREQ(anthropic_error_type_for("", 500), "api_error");
}

// ---------------------------------------------------------------------------
// What /v1/models may advertise (#1542)
// ---------------------------------------------------------------------------

TEST(ServableContext, ThePoolWinsWhenItIsSmaller) {
    // The reported case: resolver planned 97204, the pool was clamped to 52256.
    EXPECT_EQ(servable_context_tokens(97204, 52256), 52256);
}

TEST(ServableContext, ThePlanStandsWhenThePoolIsLarger) {
    EXPECT_EQ(servable_context_tokens(131072, 209408), 131072);
    EXPECT_EQ(servable_context_tokens(4096, 4096), 4096);
}

// A pool whose size could not be read must not silently advertise zero.
TEST(ServableContext, UnknownCapacityLeavesThePlanAlone) {
    EXPECT_EQ(servable_context_tokens(8192, -1), 8192);
    EXPECT_EQ(servable_context_tokens(8192, 0), 8192);
}

// ---------------------------------------------------------------------------
// Latency histogram ladders (#1577)
// ---------------------------------------------------------------------------

// The defect: inter-token latency was observed on the request-duration ladder,
// whose first bucket is 5 ms. imp decodes at 300-450 tok/s, i.e. 2.2-3.3 ms per
// token, so every observation fell in bucket 0 and histogram_quantile returned
// a function of the bounds rather than of the data.
TEST(LatencyLadder, TheSecondsLadderCannotResolveInterTokenLatency) {
    LatencyHistogram seconds;  // the shared ladder, as ITL used to use it
    for (double tok_per_s : {300.0, 350.0, 400.0, 450.0})
        seconds.observe(1.0 / tok_per_s);
    // All four in the first bucket: indistinguishable.
    EXPECT_EQ(seconds.buckets[0].load(), 4);
}

TEST(LatencyLadder, TheItlLadderSeparatesThoseSameValues) {
    LatencyHistogram itl{LatencyHistogram::kItlBounds};
    for (double tok_per_s : {300.0, 350.0, 400.0, 450.0})
        itl.observe(1.0 / tok_per_s);
    // 2.22 / 2.50 / 2.86 / 3.33 ms against bounds 2 ms and 3 ms: the values
    // land in different buckets, which is what makes a quantile mean anything.
    EXPECT_EQ(itl.buckets[2].load(), 0);  // le = 2 ms
    EXPECT_EQ(itl.buckets[3].load(), 3);  // le = 3 ms
    EXPECT_EQ(itl.buckets[4].load(), 4);  // le = 5 ms
    EXPECT_EQ(itl.count.load(), 4);
}

TEST(LatencyLadder, DefaultConstructionKeepsTheSecondsLadder) {
    LatencyHistogram h;
    EXPECT_EQ(h.bounds, LatencyHistogram::kSecondsBounds);
    LatencyHistogram i{LatencyHistogram::kItlBounds};
    EXPECT_EQ(i.bounds, LatencyHistogram::kItlBounds);
}

// Cumulative, so a bucket count is "at most this", and the sum is in seconds.
TEST(LatencyLadder, BucketsAreCumulativeAndTheSumIsSeconds) {
    LatencyHistogram h;
    h.observe(0.02);
    h.observe(2.0);
    EXPECT_EQ(h.buckets[0].load(), 0);  // le = 0.005
    EXPECT_EQ(h.buckets[2].load(), 1);  // le = 0.025
    EXPECT_EQ(h.buckets[8].load(), 2);  // le = 2.5
    EXPECT_EQ(h.count.load(), 2);
    EXPECT_NEAR(h.sum_us.load() / 1e6, 2.02, 1e-6);
}

// A negative reading is a clock going backwards, not a fast request.
TEST(LatencyLadder, NegativeObservationsClampToZero) {
    LatencyHistogram h;
    h.observe(-1.0);
    EXPECT_EQ(h.buckets[0].load(), 1);
    EXPECT_EQ(h.sum_us.load(), 0);
}

}  // namespace
