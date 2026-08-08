// =============================================================================
// GBNF grammar engine tests (CPU lane — no GPU, no tokenizer).
//
// WHY THIS EXISTS: a grammar constraint that is merely *usually* right is worse
// than none, because the caller stops checking. The failures that matter are
// (a) accepting a token that leaves the language, (b) declaring a half-written
// derivation finished, and (c) accepting a grammar the engine cannot actually
// enforce. All three are asserted here.
//
// This covers GbnfMatcher — the parser, the pushdown simulator and the UTF-8
// assembly, i.e. everything except the device mask. It runs CPU-only on
// purpose: every past constrained-decoding bug in this tree escaped CI because
// its test needed a GPU.
// =============================================================================

#include <gtest/gtest.h>

#include "compute/gbnf_grammar.h"

#include <string>

using imp::GbnfMatcher;

namespace {

GbnfMatcher make(const std::string& src) {
    GbnfMatcher m;
    std::string err;
    EXPECT_TRUE(m.compile(src, &err)) << "grammar should compile: " << err;
    return m;
}

// Compiles and returns the error message, for the refusal cases.
std::string compile_error(const std::string& src) {
    GbnfMatcher m;
    std::string err;
    if (m.compile(src, &err))
        return "";
    return err.empty() ? "(no message)" : err;
}

}  // namespace

// -----------------------------------------------------------------------------
// Pathological nesting
// -----------------------------------------------------------------------------

// expand() drops a continuation once it is kMaxStackDepth (128) rule references
// deep, because a self-referential grammar otherwise grows the work list without
// bound. No test in this file nested anything, so removing that cap left the
// whole suite green while turning a recursive grammar into a hang.
//
// A grammar that recurses on every open bracket, driven past the cap: the
// matcher must stay responsive and simply stop accepting deeper opens, never
// spin. `would_accept` returning either answer is fine — the assertion is that
// it returns at all, with a bounded amount of work, and that shallow nesting
// still behaves.
TEST(GbnfGrammarTest, DeepSelfRecursionStaysBounded) {
    GbnfMatcher m = make("root ::= \"[\" root \"]\" | \"x\"\n");
    ASSERT_TRUE(m.compiled());

    // Well inside the cap: must accept.
    EXPECT_TRUE(m.would_accept(std::string(8, '[') + "x" + std::string(8, ']')));

    // Well past it: the cap makes this unreachable, and the call must still
    // return rather than expanding for ever.
    const std::string deep_open(512, '[');
    EXPECT_FALSE(m.would_accept(deep_open + "x" + std::string(512, ']')));

    // The matcher is not wedged by the pathological input.
    EXPECT_TRUE(m.would_accept("x"));
}

// -----------------------------------------------------------------------------
// Refusals: a grammar we cannot enforce must be rejected, never approximated.
// -----------------------------------------------------------------------------

TEST(GbnfGrammar, RefusesGrammarWithoutRoot) { EXPECT_FALSE(compile_error("start ::= \"a\"").empty()); }

TEST(GbnfGrammar, RefusesUndefinedRuleReference) {
    const std::string err = compile_error("root ::= value\n");
    ASSERT_FALSE(err.empty());
    EXPECT_NE(err.find("value"), std::string::npos) << err;
}

// Left recursion has no finite expansion in a stack simulator: `a ::= a "x"`
// would push frames forever. Refusing beats hanging the request.
TEST(GbnfGrammar, RefusesLeftRecursion) {
    EXPECT_FALSE(compile_error("root ::= root \"x\" | \"y\"").empty());
    // Indirect, through a nullable prefix — the case a naive check misses.
    EXPECT_FALSE(compile_error("root ::= opt root \"x\" | \"y\"\nopt ::= \"a\"?").empty());
    // A star over a nullable rule is the same trap wearing a different hat.
    EXPECT_FALSE(compile_error("root ::= a*\na ::= \"\"").empty());
}

TEST(GbnfGrammar, RefusesDuplicateAndMalformedRules) {
    EXPECT_FALSE(compile_error("root ::= \"a\"\nroot ::= \"b\"").empty());
    EXPECT_FALSE(compile_error("root ::= \"unterminated").empty());
    EXPECT_FALSE(compile_error("root ::= [a-z").empty());
    EXPECT_FALSE(compile_error("root ::= *").empty());
    EXPECT_FALSE(compile_error("root \"a\"").empty());
    EXPECT_FALSE(compile_error("root ::= [z-a]").empty());
}

// {0,100000} is a grammar bomb, not an intent: it would materialise 100k
// synthetic rules before the first token is decoded.
TEST(GbnfGrammar, RefusesAbsurdRepetitionBounds) {
    EXPECT_FALSE(compile_error("root ::= \"a\"{0,100000}").empty());
    EXPECT_TRUE(compile_error("root ::= \"a\"{0,8}").empty());
}

// -----------------------------------------------------------------------------
// Core language semantics
// -----------------------------------------------------------------------------

TEST(GbnfGrammar, AlternationAcceptsOnlyLiveContinuations) {
    GbnfMatcher m = make("root ::= \"yes\" | \"no\"");
    EXPECT_TRUE(m.would_accept("y"));
    EXPECT_TRUE(m.would_accept("no"));
    EXPECT_FALSE(m.would_accept("x"));
    EXPECT_FALSE(m.would_accept("yo"));

    ASSERT_TRUE(m.update_text("y"));
    EXPECT_FALSE(m.would_accept("n")) << "the 'no' branch is dead after 'y'";
    EXPECT_TRUE(m.would_accept("es"));
}

TEST(GbnfGrammar, IsDoneOnlyWhenTheDerivationIsComplete) {
    GbnfMatcher m = make("root ::= \"yes\" | \"no\"");
    EXPECT_FALSE(m.is_done());
    ASSERT_TRUE(m.update_text("ye"));
    EXPECT_FALSE(m.is_done()) << "one character short must not count as done";
    ASSERT_TRUE(m.update_text("s"));
    EXPECT_TRUE(m.is_done());
    // Anchored: the whole output must be a derivation, so nothing may follow.
    EXPECT_FALSE(m.would_accept("!"));
}

// The reason this engine exists at all: a regex cannot count brackets.
TEST(GbnfGrammar, RecursionBalancesNesting) {
    GbnfMatcher m = make("root ::= \"(\" root \")\" | \"x\"");
    ASSERT_TRUE(m.update_text("(("));
    EXPECT_FALSE(m.is_done());
    EXPECT_FALSE(m.would_accept(")")) << "no inner value yet";
    ASSERT_TRUE(m.update_text("x"));
    EXPECT_FALSE(m.is_done()) << "two parens still open";
    ASSERT_TRUE(m.update_text(")"));
    EXPECT_FALSE(m.is_done()) << "one paren still open";
    ASSERT_TRUE(m.update_text(")"));
    EXPECT_TRUE(m.is_done());
    EXPECT_FALSE(m.would_accept(")")) << "closing more than was opened must fail";
}

TEST(GbnfGrammar, RepetitionOperators) {
    GbnfMatcher star = make("root ::= \"a\"*");
    EXPECT_TRUE(star.is_done()) << "zero repetitions is a complete derivation";
    ASSERT_TRUE(star.update_text("aaaa"));
    EXPECT_TRUE(star.is_done());
    EXPECT_FALSE(star.would_accept("b"));

    GbnfMatcher plus = make("root ::= \"a\"+");
    EXPECT_FALSE(plus.is_done()) << "+ requires at least one";
    ASSERT_TRUE(plus.update_text("a"));
    EXPECT_TRUE(plus.is_done());

    GbnfMatcher opt = make("root ::= \"a\"? \"b\"");
    EXPECT_TRUE(opt.would_accept("b"));
    EXPECT_TRUE(opt.would_accept("ab"));
    EXPECT_FALSE(opt.would_accept("aab"));

    GbnfMatcher bounded = make("root ::= [0-9]{2,3}");
    ASSERT_TRUE(bounded.update_text("12"));
    EXPECT_TRUE(bounded.is_done());
    ASSERT_TRUE(bounded.update_text("3"));
    EXPECT_TRUE(bounded.is_done());
    EXPECT_FALSE(bounded.would_accept("4")) << "three digits is the maximum";

    GbnfMatcher exact = make("root ::= [0-9]{2}");
    ASSERT_TRUE(exact.update_text("1"));
    EXPECT_FALSE(exact.is_done());
    ASSERT_TRUE(exact.update_text("2"));
    EXPECT_TRUE(exact.is_done());
    EXPECT_FALSE(exact.would_accept("3"));

    GbnfMatcher open = make("root ::= \"x\"{2,}");
    ASSERT_TRUE(open.update_text("x"));
    EXPECT_FALSE(open.is_done());
    ASSERT_TRUE(open.update_text("xxxxx"));
    EXPECT_TRUE(open.is_done());
}

// A postfix operator binds to the whole preceding element. Getting this wrong
// silently changes the language: `"ab"*` would become `a` followed by `b*`.
TEST(GbnfGrammar, RepetitionBindsTheWholeLiteral) {
    GbnfMatcher m = make("root ::= \"ab\"*");
    ASSERT_TRUE(m.update_text("abab"));
    EXPECT_TRUE(m.is_done());
    EXPECT_TRUE(m.would_accept("a"));
    EXPECT_FALSE(m.would_accept("b")) << "'b' would only be legal if * bound to it alone";

    GbnfMatcher half = make("root ::= \"ab\"*");
    ASSERT_TRUE(half.update_text("aba"));
    EXPECT_FALSE(half.is_done()) << "a half-written repetition is not a derivation";
    EXPECT_FALSE(half.would_accept("a"));
    EXPECT_TRUE(half.would_accept("b"));
}

TEST(GbnfGrammar, CharacterClassesAndAnyChar) {
    GbnfMatcher cls = make("root ::= [a-cx_]+");
    EXPECT_TRUE(cls.would_accept("b"));
    EXPECT_TRUE(cls.would_accept("_"));
    EXPECT_TRUE(cls.would_accept("x"));
    EXPECT_FALSE(cls.would_accept("d"));

    GbnfMatcher neg = make("root ::= \"\\\"\" [^\"]* \"\\\"\"");
    ASSERT_TRUE(neg.update_text("\"he said "));
    EXPECT_FALSE(neg.is_done());
    EXPECT_TRUE(neg.would_accept("anything at all"));
    ASSERT_TRUE(neg.update_text("\""));
    EXPECT_TRUE(neg.is_done());

    GbnfMatcher any = make("root ::= \"<\" . \">\"");
    ASSERT_TRUE(any.update_text("<"));
    EXPECT_TRUE(any.would_accept("q"));
    EXPECT_TRUE(any.would_accept("\n"));
    ASSERT_TRUE(any.update_text("q>"));
    EXPECT_TRUE(any.is_done());
}

TEST(GbnfGrammar, EscapesAndComments) {
    GbnfMatcher m = make(
        "# a leading comment\n"
        "root ::= \"a\\tb\" nl  # trailing comment\n"
        "nl ::= \"\\n\"\n");
    ASSERT_TRUE(m.update_text("a\tb\n"));
    EXPECT_TRUE(m.is_done());

    GbnfMatcher hexm = make("root ::= \"\\x41\" \"\\u00e9\"");
    ASSERT_TRUE(hexm.update_text("A\xc3\xa9"));
    EXPECT_TRUE(hexm.is_done());
}

// A rule may wrap across lines after `::=` or `|`; a newline otherwise ends it.
TEST(GbnfGrammar, AlternativesMayWrapAcrossLines) {
    GbnfMatcher m = make(
        "root ::=\n"
        "    \"alpha\" |\n"
        "    \"beta\"\n");
    EXPECT_TRUE(m.would_accept("beta"));
    EXPECT_FALSE(m.would_accept("gamma"));
}

TEST(GbnfGrammar, GroupsAndNestedAlternation) {
    GbnfMatcher m = make("root ::= (\"a\" | \"b\") \"-\" (\"1\" | \"2\")");
    EXPECT_TRUE(m.would_accept("a-2"));
    EXPECT_TRUE(m.would_accept("b-1"));
    EXPECT_FALSE(m.would_accept("c-1"));
    EXPECT_FALSE(m.would_accept("a-3"));
    ASSERT_TRUE(m.update_text("a-"));
    EXPECT_FALSE(m.is_done());
    ASSERT_TRUE(m.update_text("1"));
    EXPECT_TRUE(m.is_done());
}

// -----------------------------------------------------------------------------
// UTF-8: the grammar simulates codepoints, tokens are bytes, and a BPE token
// can end mid-character.
// -----------------------------------------------------------------------------

TEST(GbnfGrammar, MultiByteCharactersSplitAcrossTokens) {
    GbnfMatcher m = make("root ::= \"ä\"+");
    // "ä" is C3 A4 — a token boundary between the two bytes is normal.
    ASSERT_TRUE(m.update_text("\xc3"));
    EXPECT_FALSE(m.is_done()) << "half a character cannot be a complete derivation";
    ASSERT_TRUE(m.update_text("\xa4"));
    EXPECT_TRUE(m.is_done());
    EXPECT_TRUE(m.would_accept("\xc3\xa4"));
    EXPECT_FALSE(m.would_accept("a"));
}

// The discriminating case for the partial-codepoint guard: the STACKS accept
// here, so only the pending half-character keeps is_done() false. Without that
// check the server would report a finished reply whose last character is
// truncated — and the truncation only shows up in the client.
TEST(GbnfGrammar, HalfWrittenCharacterIsNotDoneEvenWhenTheStacksAccept) {
    GbnfMatcher star = make("root ::= \"ä\"*");
    EXPECT_TRUE(star.is_done()) << "zero repetitions accepts";
    ASSERT_TRUE(star.update_text("\xc3"));
    EXPECT_FALSE(star.is_done()) << "the stacks still accept, but a character is half-written";
    ASSERT_TRUE(star.update_text("\xa4"));
    EXPECT_TRUE(star.is_done());

    GbnfMatcher plus = make("root ::= \"ä\"+");
    ASSERT_TRUE(plus.update_text("\xc3\xa4"));
    EXPECT_TRUE(plus.is_done());
    ASSERT_TRUE(plus.update_text("\xc3"));
    EXPECT_FALSE(plus.is_done());
}

TEST(GbnfGrammar, RejectsBytesThatCannotCompleteLegally) {
    GbnfMatcher m = make("root ::= \"ä\"");
    // C3 opens a two-byte codepoint; only A4 completes it into 'ä'.
    ASSERT_TRUE(m.update_text("\xc3"));
    EXPECT_FALSE(m.would_accept("\xa5")) << "C3 A5 is 'å', which the grammar forbids";
    EXPECT_FALSE(m.would_accept("a")) << "a continuation byte was expected";
    EXPECT_TRUE(m.would_accept("\xa4"));

    GbnfMatcher ascii = make("root ::= [a-z]+");
    EXPECT_FALSE(ascii.would_accept("\xc3\xa4")) << "no multi-byte char is in [a-z]";
    EXPECT_FALSE(ascii.would_accept("\xc3")) << "and no lead byte can start one";
    EXPECT_FALSE(ascii.would_accept("\x80")) << "a bare continuation byte is never a start";
}

// An overlong encoding spells an allowed character with more bytes than UTF-8
// permits. Decoding it naively lets a token smuggle in a character the grammar
// may well forbid, so it is rejected at both lengths that can express one.
TEST(GbnfGrammar, RejectsOverlongEncodings) {
    GbnfMatcher m = make("root ::= [a-z]+");
    EXPECT_TRUE(m.would_accept("a"));
    EXPECT_FALSE(m.would_accept("\xe0\x81\xa1")) << "E0 81 A1 is a 3-byte overlong 'a'";
    EXPECT_FALSE(m.would_accept("\xe0\x81")) << "and its prefix is already dead";
    EXPECT_FALSE(m.would_accept("\xc1\xa1")) << "C1 A1 is a 2-byte overlong 'a'";
    EXPECT_FALSE(m.would_accept("\xf8\x80\x80\x81\xa1")) << "F8 is not a UTF-8 lead byte at all";

    // Legitimate multi-byte characters still pass, so this is not a blanket ban.
    GbnfMatcher wide = make("root ::= [\\u00e0-\\u00ff]+");
    EXPECT_TRUE(wide.would_accept("\xc3\xa4"));
}

// The mask prefilter must never reject a byte the simulator would accept —
// that would silently narrow the language.
TEST(GbnfGrammar, LeadByteFilterAgreesWithTheSimulator) {
    GbnfMatcher m = make("root ::= [0-9] | \"ä\" | \"€\"");
    uint8_t lead[256];
    m.lead_bytes(lead);
    for (int b = 0; b < 256; b++) {
        const std::string s(1, static_cast<char>(b));
        if (m.would_accept(s)) {
            EXPECT_TRUE(lead[b]) << "byte " << b << " is legal but the prefilter drops it";
        }
    }
    EXPECT_TRUE(lead[static_cast<unsigned char>('5')]);
    EXPECT_FALSE(lead[static_cast<unsigned char>('a')]);
    EXPECT_TRUE(lead[0xC3]) << "'ä' starts with C3";
    EXPECT_TRUE(lead[0xE2]) << "'€' starts with E2";
    EXPECT_FALSE(lead[0xF0]) << "no 4-byte codepoint is in this grammar";

    // Mid-character only continuation bytes may follow.
    ASSERT_TRUE(m.update_text("\xe2"));
    m.lead_bytes(lead);
    EXPECT_TRUE(lead[0x82]);
    EXPECT_FALSE(lead[static_cast<unsigned char>('5')]);
}

// -----------------------------------------------------------------------------
// A grammar of the shape agents actually send
// -----------------------------------------------------------------------------

TEST(GbnfGrammar, RecursiveJsonGrammar) {
    const std::string json =
        "root   ::= object\n"
        "object ::= \"{\" ws (pair (\",\" ws pair)*)? \"}\"\n"
        "pair   ::= string ws \":\" ws value\n"
        "value  ::= object | array | string | number | \"true\" | \"false\" | \"null\"\n"
        "array  ::= \"[\" ws (value (\",\" ws value)*)? \"]\"\n"
        "string ::= \"\\\"\" [^\"]* \"\\\"\"\n"
        "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n"
        "ws     ::= [ \\t\\n]*\n";

    GbnfMatcher m = make(json);
    ASSERT_TRUE(m.update_text("{\"a\": [1, {\"b\": -2.5}], \"c\": null"));
    EXPECT_FALSE(m.is_done());
    ASSERT_TRUE(m.update_text("}"));
    EXPECT_TRUE(m.is_done());

    GbnfMatcher bad = make(json);
    EXPECT_FALSE(bad.update_text("{\"a\": 1,, }")) << "a doubled comma is not in the language";

    GbnfMatcher unclosed = make(json);
    ASSERT_TRUE(unclosed.update_text("{\"a\": [1"));
    EXPECT_FALSE(unclosed.is_done()) << "an unclosed array must not read as finished";
    EXPECT_FALSE(unclosed.would_accept("}")) << "the array has to close first";
}

// The pooled-manager hazard: a ConstraintManager is reused across requests, so
// a second grammar is compiled into the SAME matcher. Everything derived from
// the first one — the interned stacks and their memoised transitions — indexes
// a rule table that no longer exists.
TEST(GbnfGrammar, RecompilingReplacesTheLanguageCompletely) {
    GbnfMatcher m;
    std::string err;
    ASSERT_TRUE(m.compile("root ::= \"aaa\"", &err)) << err;
    ASSERT_TRUE(m.update_text("aa"));
    EXPECT_TRUE(m.would_accept("a"));

    ASSERT_TRUE(m.compile("root ::= \"bbb\"", &err)) << err;
    EXPECT_FALSE(m.is_done()) << "compiling must restart at the new root";
    EXPECT_FALSE(m.would_accept("a")) << "the previous grammar must be gone";
    EXPECT_TRUE(m.would_accept("b"));
    ASSERT_TRUE(m.update_text("bbb"));
    EXPECT_TRUE(m.is_done());

    // A grammar with a different shape at the same rule indices is the case
    // that stale memoised transitions would silently mis-enforce.
    ASSERT_TRUE(m.compile("root ::= \"b\" [0-9]\n", &err)) << err;
    ASSERT_TRUE(m.update_text("b"));
    EXPECT_FALSE(m.would_accept("b")) << "the old grammar allowed a second 'b' here";
    EXPECT_TRUE(m.would_accept("7"));
}

TEST(GbnfGrammar, ResetReturnsToTheStart) {
    GbnfMatcher m = make("root ::= \"ab\"");
    ASSERT_TRUE(m.update_text("ab"));
    EXPECT_TRUE(m.is_done());
    m.reset();
    EXPECT_FALSE(m.is_done());
    EXPECT_TRUE(m.would_accept("a"));
    EXPECT_FALSE(m.would_accept("b"));
}
