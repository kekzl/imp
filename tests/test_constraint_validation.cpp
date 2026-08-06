// =============================================================================
// Admission-time validation of constrained-decoding requests (#1256).
//
// WHY: a constraint imp cannot compile used to be dropped and the request
// answered anyway — HTTP 200, free-form text, nothing in the reply telling the
// caller its guarantee was not applied. These assert the 400 contract on the
// CPU, in CI, where the real handler never runs.
//
// The alias coverage is the part most likely to rot: `regex`/`pattern`,
// `grammar`/`gbnf`, `guided_regex`, `guided_grammar` and llama.cpp's bare
// `grammar` all reach the same engine, so validation that misses one is a
// bypass, not a gap. A new spelling added to the parameter parser without a
// matching line here is exactly the regression these catch.
// =============================================================================

#include <gtest/gtest.h>

#include "handlers_internal.h"

#include <nlohmann/json.hpp>
#include <httplib.h>

using json = nlohmann::json;

namespace {

// Returns true when the body is accepted (no 400 written).
bool accepts(const json& body) {
    httplib::Response res;
    return validate_constraints(body, res);
}

// Returns the error message a rejected body produces (empty when accepted).
std::string rejection_message(const json& body) {
    httplib::Response res;
    if (validate_constraints(body, res))
        return {};
    EXPECT_EQ(res.status, 400);
    auto parsed = json::parse(res.body, nullptr, false);
    if (parsed.is_discarded() || !parsed.contains("error"))
        return "<unparseable>";
    EXPECT_EQ(parsed["error"].value("type", ""), "invalid_request_error");
    return parsed["error"].value("message", "");
}

json regex_body(const std::string& pattern) {
    return json{{"response_format", {{"type", "regex"}, {"regex", pattern}}}};
}

json grammar_body(const std::string& gbnf) {
    return json{{"response_format", {{"type", "grammar"}, {"grammar", gbnf}}}};
}

}  // namespace

TEST(ConstraintValidation, UnconstrainedRequestIsUntouched) {
    EXPECT_TRUE(accepts(json::object()));
    EXPECT_TRUE(accepts(json{{"max_tokens", 16}}));
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "text"}}}}));
    // json_object carries no pattern to compile.
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "json_object"}}}}));
}

TEST(ConstraintValidation, EnforceablePatternsAreAccepted) {
    EXPECT_TRUE(accepts(regex_body("[0-9]{3}")));
    // Edge anchors are redundant, not unsupported (#1255) — they must NOT 400.
    EXPECT_TRUE(accepts(regex_body("^[0-9]{3}$")));
    EXPECT_TRUE(accepts(regex_body("^(yes|no)$")));
    // Non-capturing groups compile since #1257.
    EXPECT_TRUE(accepts(regex_body("(?:ab)+")));
    EXPECT_TRUE(accepts(grammar_body("root ::= \"yes\" | \"no\"")));
}

TEST(ConstraintValidation, MalformedRegexIsRejectedWithItsPattern) {
    const std::string msg = rejection_message(regex_body("^[0-9{3}$"));
    EXPECT_NE(msg.find("^[0-9{3}$"), std::string::npos)
        << "the message must quote the pattern the caller sent";
    EXPECT_NE(msg.find("cannot be enforced"), std::string::npos);
}

TEST(ConstraintValidation, UnsupportedConstructsAreRejected) {
    for (const char* bad : {"(?=lookahead)x", "a\\b", "(a)\\1", "a^b"}) {
        EXPECT_FALSE(accepts(regex_body(bad))) << "should have been rejected: " << bad;
    }
}

// A pattern that matches nothing would decode to an empty string with no
// explanation, which is the failure this whole change exists to remove.
TEST(ConstraintValidation, EmptyLanguageIsRejected) {
    EXPECT_FALSE(accepts(regex_body("[z-a]")));
}

TEST(ConstraintValidation, MalformedGrammarIsRejectedWithTheParserError) {
    const std::string msg = rejection_message(grammar_body("root ::= <<<"));
    EXPECT_NE(msg.find("GBNF"), std::string::npos);
    // The parser's own diagnostic is forwarded rather than replaced by a generic one.
    EXPECT_NE(msg.find("unexpected character"), std::string::npos) << "got: " << msg;
}

// Every spelling the parameter parser accepts must be validated, or the check
// is bypassable by using an alias.
TEST(ConstraintValidation, AllRegexSpellingsAreValidated) {
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "regex"}, {"regex", "(?=x)y"}}}}));
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "regex"}, {"pattern", "(?=x)y"}}}}));
    EXPECT_FALSE(accepts(json{{"guided_regex", "(?=x)y"}})) << "vLLM spelling";
}

TEST(ConstraintValidation, AllGrammarSpellingsAreValidated) {
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "grammar"}, {"grammar", "root ::= <<<"}}}}));
    EXPECT_FALSE(accepts(json{{"response_format", {{"type", "grammar"}, {"gbnf", "root ::= <<<"}}}}));
    EXPECT_FALSE(accepts(json{{"grammar", "root ::= <<<"}})) << "llama.cpp spelling";
    EXPECT_FALSE(accepts(json{{"guided_grammar", "root ::= <<<"}})) << "vLLM spelling";
}

// Wrong JSON types must not be read as a constraint — and must not crash.
TEST(ConstraintValidation, NonStringConstraintFieldsAreIgnored) {
    EXPECT_TRUE(accepts(json{{"guided_regex", 42}}));
    EXPECT_TRUE(accepts(json{{"grammar", json::array({1, 2})}}));
    EXPECT_TRUE(accepts(json{{"response_format", {{"type", "regex"}, {"regex", nullptr}}}}));
    EXPECT_TRUE(accepts(json{{"response_format", "not-an-object"}}));
}
