// =============================================================================
// RegexConstrainer FSM tests (CPU lane — no GPU, no tokenizer).
//
// WHY THIS EXISTS: regex-constrained decoding is what an agent uses to pin a
// format the JSON FSMs cannot express (a diff header, an ID, a small DSL). The
// failure that matters is not "rejects a bad token" — it is accepting one, or
// declaring a half-written format finished. Both are asserted here.
//
// The regex engine itself (RegexNfa) is shared with JSON-Schema `pattern`; this
// covers the decode-time wrapper: prefix acceptance, is_done() semantics, and
// that unsupported patterns are refused instead of silently mis-enforced.
// =============================================================================

#include <gtest/gtest.h>

#include "compute/regex_constrain.h"

#include <memory>
#include <string>

using imp::RegexConstrainer;

namespace {

// The constrainer owns a device buffer, so it is non-copyable by design —
// hand out a unique_ptr rather than relaxing that for the tests' convenience.
std::unique_ptr<RegexConstrainer> make(const std::string& pattern) {
    auto rc = std::make_unique<RegexConstrainer>();
    EXPECT_TRUE(rc->init_pattern_only(pattern)) << "pattern should compile: " << pattern;
    return rc;
}

}  // namespace

TEST(RegexConstrain, AcceptsOnlyTheMatchingContinuation) {
    auto rc = make("[0-9]{3}-[0-9]{2}");
    EXPECT_TRUE(rc->would_accept("1"));
    EXPECT_FALSE(rc->would_accept("a"));
    EXPECT_FALSE(rc->would_accept("-"));  // digits come first

    ASSERT_TRUE(rc->update_text("123"));
    EXPECT_TRUE(rc->would_accept("-"));
    EXPECT_FALSE(rc->would_accept("4"));  // the group is exactly three digits
}

// The contract that keeps a half-written format from looking finished.
TEST(RegexConstrain, IsDoneOnlyWhenThePatternIsComplete) {
    auto rc = make("[0-9]{3}-[0-9]{2}");
    EXPECT_FALSE(rc->is_done());
    ASSERT_TRUE(rc->update_text("123"));
    EXPECT_FALSE(rc->is_done());
    ASSERT_TRUE(rc->update_text("-4"));
    EXPECT_FALSE(rc->is_done()) << "one digit short must not count as done";
    ASSERT_TRUE(rc->update_text("5"));
    EXPECT_TRUE(rc->is_done());
}

TEST(RegexConstrain, RejectsTextThatLeavesTheLanguage) {
    auto rc = make("v\\d+\\.\\d+");
    ASSERT_TRUE(rc->update_text("v1."));
    EXPECT_FALSE(rc->update_text("x"));
    // A rejected update must not corrupt the state — the FSM stays where it was.
    EXPECT_TRUE(rc->would_accept("2"));
    ASSERT_TRUE(rc->update_text("2"));
    EXPECT_TRUE(rc->is_done());
}

TEST(RegexConstrain, MultiCharTokensAreValidatedWhole) {
    // A token is several bytes at once; accepting on its first byte alone would
    // let " 12" through where only digits are legal.
    auto rc = make("[0-9]+");
    EXPECT_TRUE(rc->would_accept("123"));
    EXPECT_FALSE(rc->would_accept("12a"));
    EXPECT_FALSE(rc->would_accept(" 12"));
}

TEST(RegexConstrain, AlternationAndOptionalParts) {
    auto rc = make("(cat|dog)s?");
    EXPECT_TRUE(rc->would_accept("dog"));
    EXPECT_FALSE(rc->would_accept("cow"));
    ASSERT_TRUE(rc->update_text("dog"));
    EXPECT_TRUE(rc->is_done()) << "the optional 's' means 'dog' already matches";
    ASSERT_TRUE(rc->update_text("s"));
    EXPECT_TRUE(rc->is_done());
    EXPECT_FALSE(rc->would_accept("s")) << "only one optional 's'";
}

TEST(RegexConstrain, CharacterClassesAndNegation) {
    auto rc = make("[a-z_][a-z0-9_]*");
    EXPECT_FALSE(rc->would_accept("1"));
    ASSERT_TRUE(rc->update_text("my_var1"));
    EXPECT_TRUE(rc->is_done());

    auto neg = make("[^x]+");
    EXPECT_TRUE(neg->would_accept("abc"));
    EXPECT_FALSE(neg->would_accept("axc"));
}

// An unsupported pattern must be refused, not quietly enforced as something
// else — a wrong grammar is worse than no grammar.
TEST(RegexConstrain, RefusesPatternsItCannotEnforce) {
    for (const char* bad : {"(unclosed", "[z-a]", "(?=lookahead)x", "a\\b", "(a)\\1"}) {
        RegexConstrainer rc;
        EXPECT_FALSE(rc.init_pattern_only(bad)) << "should have been refused: " << bad;
        EXPECT_FALSE(rc.is_initialized());
    }
}

// `^...$` used to sit in the list above. It is the most natural way to write a
// pattern, and refusing it meant a client asking for `^[0-9]{3}$` got free-form
// text back with HTTP 200 — the constraint it asked for silently absent. Under
// whole-output matching the anchors are redundant, so they are stripped and the
// pattern is enforced.
TEST(RegexConstrain, EdgeAnchorsAreRedundantAndAccepted) {
    auto rc = make("^[0-9]{3}$");
    EXPECT_TRUE(rc->is_initialized());
    EXPECT_TRUE(rc->would_accept("123"));
    EXPECT_FALSE(rc->would_accept("12a"));
    ASSERT_TRUE(rc->update_text("123"));
    EXPECT_TRUE(rc->is_done()) << "three digits satisfies the pattern";

    // Diagnostics must quote what the caller sent, not the rewritten form.
    EXPECT_EQ(rc->pattern(), "^[0-9]{3}$");

    // One-sided anchors too.
    EXPECT_TRUE(make("^abc")->would_accept("abc"));
    EXPECT_TRUE(make("abc$")->would_accept("abc"));
}

// The remainder is wrapped, so a top-level alternation still binds to the whole
// output: `^a|b$` means "a or b", not "starts with a" or "ends with b".
TEST(RegexConstrain, EdgeAnchorsAroundAlternationBindToWholeOutput) {
    auto rc = make("^yes|no$");
    EXPECT_TRUE(rc->is_initialized());
    EXPECT_TRUE(rc->would_accept("yes"));
    EXPECT_TRUE(rc->would_accept("no"));
    EXPECT_FALSE(rc->would_accept("yesno"));
}

// A `$` the caller escaped is a literal dollar and must survive.
TEST(RegexConstrain, EscapedDollarIsNotAnAnchor) {
    auto rc = make("[0-9]+\\$");
    EXPECT_TRUE(rc->is_initialized());
    ASSERT_TRUE(rc->update_text("42$"));
    EXPECT_TRUE(rc->is_done());
}

// Anchors that are NOT at the edges keep being refused — there the literal
// reading really would enforce something the caller never asked for.
TEST(RegexConstrain, InteriorAnchorsStillRefused) {
    for (const char* bad : {"a^b", "a$b", "^a^b$"}) {
        RegexConstrainer rc;
        EXPECT_FALSE(rc.init_pattern_only(bad)) << "should have been refused: " << bad;
    }
}

// `(?:…)` passed the support check ("a plain non-capturing group and is fine")
// but the engine had no `?:` form: `?` was read as a quantifier with no atom and
// `:` as a literal, so `(?:a|b)c` compiled to `(:a|b)c` — it accepted "bc",
// rejected "ac", and reported a successful compile throughout. Accepted AND
// enforced as something else is the one outcome this layer exists to prevent.
TEST(RegexConstrain, NonCapturingGroupIsHonoured) {
    auto plain = make("(?:abc)");
    EXPECT_TRUE(plain->would_accept("abc"));

    auto alt = make("(?:a|b)c");
    EXPECT_TRUE(alt->would_accept("ac")) << "the first branch was lost to the ':' literal";
    EXPECT_TRUE(alt->would_accept("bc"));
    EXPECT_FALSE(alt->would_accept(":ac")) << "':' must not be matchable";

    EXPECT_TRUE(make("x(?:yz)")->would_accept("xyz"));

    // Quantified — the reason non-capturing groups get written at all.
    // would_accept asks "is this still inside the language", so a partial
    // repetition passes it; completeness is is_done()'s question.
    auto rep = make("(?:ab)+");
    ASSERT_TRUE(rep->update_text("abab"));
    EXPECT_TRUE(rep->is_done());

    auto partial = make("(?:ab)+");
    ASSERT_TRUE(partial->update_text("aba"));
    EXPECT_FALSE(partial->is_done()) << "half a repetition is not a complete match";
}

// Documented tolerance rather than a silent surprise: the shared engine reads a
// reversed bound as the lower one instead of rejecting it, so `a{2,1}` enforces
// exactly two. Pinned so a future engine change that starts rejecting it is a
// visible decision, not a mystery.
TEST(RegexConstrain, ReversedRepetitionBoundIsTreatedAsExact) {
    auto rc = make("a{2,1}");
    EXPECT_FALSE(rc->is_done());
    ASSERT_TRUE(rc->update_text("aa"));
    EXPECT_TRUE(rc->is_done());
    EXPECT_FALSE(rc->would_accept("a")) << "a third 'a' must not be accepted";
}

// Uninitialised must be permissive, never a silent all-reject: that would mask
// every token and degenerate generation.
TEST(RegexConstrain, UninitialisedConstrainsNothing) {
    RegexConstrainer rc;
    EXPECT_FALSE(rc.is_initialized());
    EXPECT_TRUE(rc.would_accept("anything at all"));
    EXPECT_TRUE(rc.update_text("anything at all"));
}

TEST(RegexConstrain, ResetReturnsToTheStart) {
    auto rc = make("ab");
    ASSERT_TRUE(rc->update_text("ab"));
    EXPECT_TRUE(rc->is_done());
    rc->reset();
    EXPECT_FALSE(rc->is_done());
    EXPECT_TRUE(rc->would_accept("a"));
    EXPECT_FALSE(rc->would_accept("b"));
}
