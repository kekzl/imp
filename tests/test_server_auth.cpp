// =============================================================================
// Unit tests for the server API-key (Bearer) auth check — TEST_AUDIT.md §7.
//
// WHY: the Bearer auth in main.cpp's pre-routing handler had NO test (mock_server
// has no auth). An auth-bypass regression or a re-introduced early-out timing
// leak would ship silently. The constant-time compare was extracted into
// bearer_token_matches() (utils.cpp) precisely so it could be tested here, on the
// CPU, in CI (where the real handler never runs). These assert the accept/reject
// contract; the timing property is documented and structurally enforced by the
// no-early-out implementation (a perf "optimization" back to operator== would
// still pass these, so the comment in utils.cpp is the guard there).
// =============================================================================

#include <gtest/gtest.h>
#include "utils.h"

#include <string>

TEST(BearerAuth, AcceptsCorrectToken) {
    EXPECT_TRUE(bearer_token_matches("Bearer secret123", "secret123"));
}

TEST(BearerAuth, RejectsWrongToken) {
    EXPECT_FALSE(bearer_token_matches("Bearer wrong", "secret123"));
}

TEST(BearerAuth, RejectsMissingBearerPrefix) {
    EXPECT_FALSE(bearer_token_matches("secret123", "secret123"));        // no "Bearer "
    EXPECT_FALSE(bearer_token_matches("Token secret123", "secret123"));  // wrong scheme
    EXPECT_FALSE(bearer_token_matches("bearer secret123", "secret123")); // case-sensitive
}

TEST(BearerAuth, RejectsEmptyHeader) {
    EXPECT_FALSE(bearer_token_matches("", "secret123"));
}

TEST(BearerAuth, RejectsCorrectPrefixWrongSuffix) {
    // A prefix match must not pass — the whole token has to match.
    EXPECT_FALSE(bearer_token_matches("Bearer secret12", "secret123"));   // truncated
    EXPECT_FALSE(bearer_token_matches("Bearer secret1234", "secret123")); // extra char
}

TEST(BearerAuth, RejectsLeadingMatchOnly) {
    // Differ only in the first byte after "Bearer " — exercises the no-early-out
    // path (the whole length is still compared).
    EXPECT_FALSE(bearer_token_matches("Bearer Xecret123", "secret123"));
}

TEST(BearerAuth, EmptyApiKeyMatchesBareBearer) {
    // Documents the edge: with an empty configured key the expected string is
    // exactly "Bearer ". (In main.cpp the empty-key case is guarded earlier so
    // auth is not enforced at all; this just pins the function's own contract.)
    EXPECT_TRUE(bearer_token_matches("Bearer ", ""));
    EXPECT_FALSE(bearer_token_matches("Bearer x", ""));
}

TEST(BearerAuth, HandlesLongTokens) {
    std::string key(4096, 'k');
    EXPECT_TRUE(bearer_token_matches("Bearer " + key, key));
    std::string almost = key;
    almost.back() = 'x';
    EXPECT_FALSE(bearer_token_matches("Bearer " + almost, key));
}
