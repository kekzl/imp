// What one HTTP request is allowed to cost (#1614, #1615, #1616, #1617, #1618).
//
// Every case here is a place where the server counted a request as one unit
// while the request decided how much work that unit was, or where a limit
// keyed on something the client writes. The pieces that can be reached without
// a running server are tested here; the rest is in the API battery.

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

}  // namespace
