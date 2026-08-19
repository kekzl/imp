// ===========================================================================
// Property tests for the schema-less json_object FSM (JsonConstrainer).
//
// The example-based batteries next door encode bugs we already shipped
// (#517, #650, #761, #850, #1014, #1067 — every one found by a symptom, none
// by a test). These tests attack the same surface generatively instead: a
// random document generator plus nlohmann/json as an INDEPENDENT oracle, so a
// grammar regression fails here without anyone having to imagine the shape
// first. GOAL.md release bar 7 ("valid, terminating JSON under any sampler
// state") is the contract under test.
//
// CPU-only by construction: the default-constructed JsonConstrainer touches no
// CUDA (its device pointers stay null without init(), so the destructor frees
// nothing), which puts these in the `unit` lane — the lane CI actually runs.
// That matters: the FSM lives in a .cu, so every bug above escaped CI, which
// has no GPU runner.
//
// Determinism: one fixed seed, and every failure message prints the exact
// document so a failure is reproducible without re-rolling the dice.
// ===========================================================================

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "compute/json_constrain.h"
#include "compute/constrain_common.h"

#include <random>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr uint32_t kSeed = 0x5EED1067;  // fixed: failures must reproduce

// --- Random valid-JSON generator -------------------------------------------
// Escape-free on purpose (escapes have their own example-based battery), but
// NOT ASCII-only. It used to be, and the comment here called non-ASCII "a known,
// deliberate limitation of the token classifier" — a sentence that turned out to
// describe #1197, where constrained German output silently lost every umlaut.
//
// Worth being precise about what this generator does and does not cover: these
// tests drive the FSM, and the FSM was never the broken part — it compares
// through `unsigned char` and accepted umlauts all along. The bug was one layer
// out, in classify_token()'s category pre-filter, which the mask consults BEFORE
// the FSM is asked (see TokenCategory.NonAsciiCountsAsStringContent, which is
// what actually fails when the fix is reverted). So this covers UTF-8 through
// the grammar; it does not cover the mask. Both are needed.

std::string gen_string(std::mt19937& rng) {
    static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789_ -";
    // Multi-byte UTF-8 of two, three and four bytes: the classifier must treat
    // every byte of a sequence as string content, not just the ASCII range.
    static const char* kNonAscii[] = {"ä", "ö", "ü", "ß", "é", "日", "本", "🐻", "→"};
    std::uniform_int_distribution<int> len(0, 8);
    std::uniform_int_distribution<size_t> ch(0, sizeof(kAlphabet) - 2);
    std::uniform_int_distribution<size_t> nonascii(0, sizeof(kNonAscii) / sizeof(kNonAscii[0]) - 1);
    std::uniform_int_distribution<int> pick_nonascii(0, 4);  // ~20% of characters
    std::string s = "\"";
    int n = len(rng);
    for (int i = 0; i < n; i++) {
        if (pick_nonascii(rng) == 0)
            s += kNonAscii[nonascii(rng)];
        else
            s += kAlphabet[ch(rng)];
    }
    return s + "\"";
}

std::string gen_number(std::mt19937& rng) {
    static const char* kNumbers[] = {"0", "1", "-1", "42", "-0.5", "2.5", "1e3", "1.5e-2", "1234567890"};
    std::uniform_int_distribution<size_t> pick(0, sizeof(kNumbers) / sizeof(kNumbers[0]) - 1);
    return kNumbers[pick(rng)];
}

std::string gen_value(std::mt19937& rng, int depth);

std::string gen_object(std::mt19937& rng, int depth) {
    std::uniform_int_distribution<int> count(0, 3);
    int n = count(rng);
    std::string s = "{";
    for (int i = 0; i < n; i++) {
        if (i)
            s += ",";
        // Keys must be unique — a duplicate key is legal JSON for nlohmann but
        // the FSM tracks emitted keys, so it would be an unfair comparison.
        s += "\"k" + std::to_string(i) + "\":" + gen_value(rng, depth + 1);
    }
    return s + "}";
}

std::string gen_array(std::mt19937& rng, int depth) {
    std::uniform_int_distribution<int> count(0, 3);
    int n = count(rng);
    std::string s = "[";
    for (int i = 0; i < n; i++) {
        if (i)
            s += ",";
        s += gen_value(rng, depth + 1);
    }
    return s + "]";
}

std::string gen_value(std::mt19937& rng, int depth) {
    // Past depth 3 only scalars, so documents stay bounded.
    std::uniform_int_distribution<int> pick(0, depth < 3 ? 6 : 4);
    switch (pick(rng)) {
        case 0:
            return gen_string(rng);
        case 1:
            return gen_number(rng);
        case 2:
            return "true";
        case 3:
            return "false";
        case 4:
            return "null";
        case 5:
            return gen_object(rng, depth);
        default:
            return gen_array(rng, depth);
    }
}

// Root of a JSON document is an object or an array.
std::string gen_document(std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, 1);
    return pick(rng) ? gen_object(rng, 0) : gen_array(rng, 0);
}

// ===========================================================================
// P1 — Soundness: every document the oracle calls valid must be accepted.
// This is the direction #1067 broke (valid nested docs were REJECTED:
// `{"a":{"b":1},"c":2}` hit a premature DONE).
// ===========================================================================
TEST(JsonConstrainPropertyTest, AcceptsEveryValidDocument) {
    std::mt19937 rng(kSeed);
    for (int i = 0; i < 2000; i++) {
        const std::string doc = gen_document(rng);
        ASSERT_TRUE(nlohmann::json::accept(doc)) << "generator produced invalid JSON: " << doc;
        JsonConstrainer c;
        EXPECT_TRUE(c.sim_token_valid(doc)) << "FSM rejected valid document: " << doc;
    }
}

// ===========================================================================
// P2 — Prefix closure: the FSM is a prefix machine. Every prefix of a valid
// document must stay legal, because that is exactly the state the model is in
// mid-generation. A rejected prefix means a dead end the sampler cannot escape.
// ===========================================================================
TEST(JsonConstrainPropertyTest, AcceptsEveryPrefixOfValidDocument) {
    std::mt19937 rng(kSeed + 1);
    for (int i = 0; i < 500; i++) {
        const std::string doc = gen_document(rng);
        for (size_t cut = 1; cut < doc.size(); cut++) {
            const std::string prefix = doc.substr(0, cut);
            JsonConstrainer c;
            EXPECT_TRUE(c.sim_token_valid(prefix))
                << "FSM rejected prefix [" << prefix << "] of valid document " << doc;
        }
    }
}

// ===========================================================================
// P3 — No dead ends: from any prefix of a valid document at least one ASCII
// continuation must be legal. This is release-bar 7 ("terminating JSON under
// any sampler state") stated as an invariant: a state with an all-masked
// vocabulary would leave the sampler with nothing to emit.
// ===========================================================================
TEST(JsonConstrainPropertyTest, NoDeadEndStates) {
    std::mt19937 rng(kSeed + 2);
    for (int i = 0; i < 300; i++) {
        const std::string doc = gen_document(rng);
        for (size_t cut = 1; cut < doc.size(); cut++) {
            const std::string prefix = doc.substr(0, cut);
            bool any_legal = false;
            for (int ch = 0x20; ch < 0x7F && !any_legal; ch++) {
                JsonConstrainer c;
                c.advance_text(prefix);
                if (c.sim_token_valid(std::string(1, static_cast<char>(ch))))
                    any_legal = true;
            }
            EXPECT_TRUE(any_legal) << "dead end: no legal continuation after [" << prefix << "] (document "
                                   << doc << ")";
        }
    }
}

// ===========================================================================
// P4 — Rejection: mutations that cannot be a prefix of ANY valid document must
// be rejected. Two families that are structurally guaranteed to be illegal:
//   (a) swapped closer  — '}' <-> ']' at the final position
//   (b) trailing content after the root value closes
// (a) is the shape #1067 accepted (`,"bare-string"` then `]]` while the FSM
// believed it was in array context).
// ===========================================================================
TEST(JsonConstrainPropertyTest, RejectsSwappedFinalCloser) {
    std::mt19937 rng(kSeed + 3);
    int checked = 0;
    for (int i = 0; i < 1000; i++) {
        std::string doc = gen_document(rng);
        const char last = doc.back();
        doc.back() = (last == '}') ? ']' : '}';
        ASSERT_FALSE(nlohmann::json::accept(doc)) << "oracle thinks swapped closer is valid: " << doc;
        JsonConstrainer c;
        EXPECT_FALSE(c.sim_token_valid(doc)) << "FSM accepted mismatched closer: " << doc;
        checked++;
    }
    EXPECT_GT(checked, 0);
}

// P4c - trailing commas (#1096). gen_document() never produces one, and the two
// rejection families above (swapped final closer, trailing content after the
// root) cannot create one either, so this shape had no coverage at all.
//
// Measured, so the comment does not overclaim: re-admitting CLOSE_BRACKET in
// ARRAY_NEED_VALUE and CLOSE_BRACE in OBJECT_NEED_KEY - i.e. undoing #1096 in
// compute_allowed_mask() - does NOT make this test fail. apply_mask() uses the
// mask only as a pre-filter and then runs sim_token_valid() on every candidate
// that passes it, and advance_char() enforces the rule independently. The mask
// half of #1096 is defence in depth; this test covers the half that decides.
TEST(JsonConstrainPropertyTest, RejectsTrailingCommas) {
    static const char* kDocs[] = {
        "[1,]", "[1, 2,]", "[[1],]", "[{\"a\":1},]",
        "{\"a\":1,}", "{\"a\":1, \"b\":2,}", "{\"a\":[1],}",
    };
    for (const char* d : kDocs) {
        ASSERT_FALSE(nlohmann::json::accept(d)) << "oracle thinks it is valid: " << d;
        JsonConstrainer c;
        EXPECT_FALSE(c.sim_token_valid(d)) << "FSM accepted a trailing comma: " << d;
    }
}

TEST(JsonConstrainPropertyTest, RejectsTrailingContentAfterRoot) {
    static const char* kTrailers[] = {"{", "[", "\"", "1", "}", "]", ",", ":", "x"};
    std::mt19937 rng(kSeed + 4);
    for (int i = 0; i < 1000; i++) {
        const std::string doc = gen_document(rng);
        for (const char* t : kTrailers) {
            const std::string mutated = doc + t;
            ASSERT_FALSE(nlohmann::json::accept(mutated)) << "oracle accepts trailing: " << mutated;
            JsonConstrainer c;
            EXPECT_FALSE(c.sim_token_valid(mutated)) << "FSM accepted trailing content: " << mutated;
        }
    }
}

// ===========================================================================
// P4b — Trailing commas (#1096). A comma before a closer is the single most
// common way a model writes almost-JSON, and the FSM used to allow it: after a
// value, `,` returned to the same state an OPENER produces — and that state
// legally accepts the closer, because an empty `[]` / `{}` is valid. So `[1,]`
// and `{"a":1,}` passed the mask and the reply did not parse, which is exactly
// the contract json_object exists to keep.
//
// Generative, because the failing shape found in the wild was three levels
// deep inside a larger document; an example-based test would have had to
// imagine that.
// ===========================================================================
TEST(JsonConstrainPropertyTest, RejectsTrailingCommaBeforeAnyCloser) {
    std::mt19937 rng(kSeed + 6);
    int mutated_docs = 0;
    for (int i = 0; i < 2000; i++) {
        const std::string doc = gen_document(rng);
        // Insert a comma before each closer in turn.
        for (size_t p = 0; p < doc.size(); p++) {
            if (doc[p] != ']' && doc[p] != '}')
                continue;
            // Only meaningful after a value — `[,]` / `{,}` are a different
            // (already covered) rejection.
            if (p == 0 || doc[p - 1] == '[' || doc[p - 1] == '{')
                continue;
            const std::string bad = doc.substr(0, p) + "," + doc.substr(p);
            ASSERT_FALSE(nlohmann::json::accept(bad)) << "oracle accepts trailing comma: " << bad;
            JsonConstrainer c;
            EXPECT_FALSE(c.sim_token_valid(bad)) << "FSM accepted a trailing comma: " << bad;
            mutated_docs++;
        }
    }
    EXPECT_GT(mutated_docs, 100) << "generator produced too few closers to be a real test";
}

// The two minimal spellings, kept explicit so a failure names the shape
// directly instead of printing a random 200-byte document.
TEST(JsonConstrainPropertyTest, RejectsMinimalTrailingComma) {
    for (const char* bad : {"[1,]", "{\"a\":1,}", "[[1,],2]", "{\"a\":[1,2,]}", "[\"x\",]", "[true,]",
                            "[null,]", "[{},]", "{\"a\":{},}"}) {
        ASSERT_FALSE(nlohmann::json::accept(bad)) << bad;
        JsonConstrainer c;
        EXPECT_FALSE(c.sim_token_valid(bad)) << "FSM accepted: " << bad;
    }
    // The empty forms stay legal — the fix must not ban `[]` / `{}`.
    for (const char* good : {"[]", "{}", "[[],{}]", "{\"a\":[],\"b\":{}}"}) {
        ASSERT_TRUE(nlohmann::json::accept(good)) << good;
        JsonConstrainer c;
        EXPECT_TRUE(c.sim_token_valid(good)) << "FSM rejected valid: " << good;
    }
}

// ===========================================================================
// P5 — Structural truncation: cutting a document short and closing it with the
// WRONG bracket must be rejected. Catches context confusion one level up, the
// exact failure mode of #1067 (close popped its own frame but resumed in the
// grandparent's state).
// ===========================================================================
TEST(JsonConstrainPropertyTest, RejectsWrongCloserAtNestedDepth) {
    std::mt19937 rng(kSeed + 5);
    int exercised = 0;
    for (int i = 0; i < 1500; i++) {
        const std::string doc = gen_document(rng);
        for (size_t cut = 1; cut < doc.size(); cut++) {
            // Only prefixes whose next real char is a closer are interesting:
            // append the OTHER closer and it must be refused.
            const char next = doc[cut];
            if (next != '}' && next != ']')
                continue;
            const std::string prefix = doc.substr(0, cut);
            const std::string wrong = prefix + (next == '}' ? ']' : '}');
            JsonConstrainer c;
            if (c.sim_token_valid(wrong)) {
                ADD_FAILURE() << "FSM accepted wrong closer: [" << wrong << "] (document " << doc << ")";
            }
            exercised++;
        }
    }
    EXPECT_GT(exercised, 0) << "generator produced no nested closers — test is vacuous";
}

// --- #1104 probe: does the number FSM reject a second decimal point? --------
// The live failure on Qwen3.6-35B-A3B-NVFP4 was `{"city":    3.5.5.5.5...`,
// i.e. the mask admitted `.` inside an already-fractional number. If the FSM
// itself accepts that, the bug is grammar-level and CPU-reproducible; if it
// rejects it, the mask is right and something bypasses it at decode time.
TEST(JsonConstrainFsm, RejectsSecondDecimalPointInNumber) {
    JsonConstrainer c;
    c.advance_text("{\"city\":");
    EXPECT_TRUE(c.sim_token_valid("3.5")) << "FSM rejected a plain fractional number";

    JsonConstrainer d;
    d.advance_text("{\"city\": 3.5");
    EXPECT_FALSE(d.sim_token_valid(".5"))
        << "FSM accepted a SECOND decimal point — this is the #1104 output shape";
    EXPECT_FALSE(d.sim_token_valid(".")) << "FSM accepted '.' after a fractional number";
}

// Full RFC 8259 number grammar, both directions. The permissive version
// accepted every one of these malformed forms.
TEST(JsonConstrainFsm, NumberGrammarMatchesRfc8259) {
    struct Case {
        const char* num;
        bool valid;
    };
    const Case cases[] = {
        {"0", true},
        {"-0", true},
        {"3", true},
        {"3.5", true},
        {"1e5", true},
        {"1E5", true},
        {"1e+5", true},
        {"1e-5", true},
        {"-2.5e-3", true},
        {"3.5.5", false},  // second decimal point — the #1104 shape
        {"1e5e5", false},  // second exponent
        {"1e+-5", false},  // double sign
        {"3-5", false},    // sign inside the mantissa
        {"1.2e", false},
        {"3.", false},
        {"-", false},  // incomplete: still owe a digit
        // Whitespace ends a number; it must never splice one back together.
        {"1 ", true},
        {"1.  1", false},
        {"1.  ", false},
    };
    for (const auto& c : cases) {
        JsonConstrainer fsm;
        // Wrap in an object so the number sits in a real value position, and
        // require the document to close — an incomplete number must not be
        // terminable by '}'.
        const std::string doc = std::string("{\"v\":") + c.num + "}";
        EXPECT_EQ(fsm.sim_token_valid(doc), c.valid)
            << "number '" << c.num << "' judged " << (c.valid ? "invalid" : "valid")
            << " — document: " << doc;
    }
}

// The simulator must not leak number sub-state onto the live FSM.
TEST(JsonConstrainFsm, SimulationDoesNotMutateNumberSubState) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"v\":3.5");
    EXPECT_FALSE(fsm.sim_token_valid(".5"));  // walks into the number, then unwinds
    EXPECT_TRUE(fsm.sim_token_valid("e5"));   // exponent still available afterwards
    EXPECT_TRUE(fsm.sim_token_valid("}"));    // and the number can still be closed
}
// --- #1104 part 2: force-close near the token budget ------------------------
// The grammar fix alone still let a wandering model run to max_tokens inside a
// long string, returning a truncated document. With the remaining allowance
// known, the mask must narrow to the closers.
TEST(JsonConstrainFsm, ForceCloseNarrowsMaskWhenBudgetIsSpent) {
    // Deep inside a string value: {"a":"xxxx  → open object + open string.
    // The narrowing is a MASK decision, so assert on the category mask —
    // sim_token_valid() answers grammar legality, and "y" is legal in a string
    // no matter how little budget is left.
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\":\"xxxx");

    fsm.set_remaining_budget(100);
    EXPECT_TRUE(fsm.allowed_categories_for_test() & CAT_STRING_CHAR)
        << "narrowing engaged while the budget was comfortable";

    fsm.set_remaining_budget(1);
    const uint16_t tight = fsm.allowed_categories_for_test();
    EXPECT_FALSE(tight & CAT_STRING_CHAR) << "string content still allowed with no budget left";
    EXPECT_TRUE(tight & CAT_QUOTE) << "closing quote must stay available";

    fsm.set_remaining_budget(-1);
    EXPECT_TRUE(fsm.allowed_categories_for_test() & CAT_STRING_CHAR)
        << "narrowing engaged with an unknown budget";
}

// Once the structures are closed the document must be completable, not stuck.
TEST(JsonConstrainFsm, ForceCloseStillReachesAValidDocument) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\":\"xx");
    fsm.set_remaining_budget(1);
    ASSERT_TRUE(fsm.allowed_categories_for_test() & CAT_QUOTE);
    fsm.advance_text("\"");  // close the string
    EXPECT_TRUE(fsm.allowed_categories_for_test() & CAT_CLOSE_BRACE)
        << "object closer unavailable after closing the string";
    EXPECT_TRUE(fsm.sim_token_valid("}")) << "grammar rejects the closer the mask offers";
}

// --- #1291: the force-close has to walk out of a state that owes something ---
//
// #1096 forbids a closer straight after a comma so `[1,]` cannot happen. #1104
// demands a closer once the budget is spent. Where they meet, the narrowing
// used to leave NOTHING legal, the empty-allow net retried with the ordinary
// mask, and the reply came back truncated anyway — the exact outcome the
// force-close exists to prevent. Measured on Qwen3.6-35B-A3B-NVFP4 at
// max_tokens=40: the mask narrowed to `}`/`]` in ARRAY_NEED_VALUE and the
// model emitted a quote instead (#1291).
//
// The mask must therefore offer the cheapest step OUT of the owing state, and
// the budget must cover the whole walk.

TEST(JsonConstrainFsm, ForceCloseOffersAValueAfterAnArrayComma) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\":[1,");  // ARRAY_NEED_VALUE — ']' is illegal here
    fsm.set_remaining_budget(1);
    const uint16_t m = fsm.allowed_categories_for_test();
    EXPECT_TRUE(m & CAT_NUMBER_START) << "no way out of ARRAY_NEED_VALUE — this is #1291";
    EXPECT_FALSE(m & CAT_CLOSE_BRACKET) << "offered a closer the grammar forbids after a comma";
}

TEST(JsonConstrainFsm, ForceCloseOffersAKeyAfterAnObjectComma) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\":1,");  // OBJECT_NEED_KEY — '}' is illegal here
    fsm.set_remaining_budget(1);
    const uint16_t m = fsm.allowed_categories_for_test();
    EXPECT_TRUE(m & CAT_QUOTE) << "no way out of OBJECT_NEED_KEY";
    EXPECT_FALSE(m & CAT_CLOSE_BRACE) << "offered a closer the grammar forbids after a comma";
}

TEST(JsonConstrainFsm, ForceCloseOffersAColonAfterAKey) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\"");  // AFTER_KEY — the grammar demands ':' next
    fsm.set_remaining_budget(1);
    const uint16_t m = fsm.allowed_categories_for_test();
    EXPECT_TRUE(m & CAT_COLON) << "no way out of AFTER_KEY";
    EXPECT_FALSE(m & CAT_CLOSE_BRACE) << "offered a closer before the value exists";
}

// The walk has to actually terminate, not just take one legal step. Follow the
// mask from the state that broke on the 35B and require a closable document.
TEST(JsonConstrainFsm, ForceCloseWalksOutOfAnArrayCommaToAClosableDocument) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\":[1,");
    fsm.set_remaining_budget(1);
    ASSERT_TRUE(fsm.allowed_categories_for_test() & CAT_NUMBER_START);
    fsm.advance_text("0");  // the value the mask offered
    EXPECT_TRUE(fsm.sim_token_valid("]")) << "array still not closable after the forced value";
    fsm.advance_text("]");
    EXPECT_TRUE(fsm.sim_token_valid("}")) << "object still not closable after the array";
}

// The walk needs one token more than the states it passes through suggest: the
// forced value itself enters a frame (a number lands in IN_NUMBER *inside* its
// container). Without that margin the e2e walk emits `-1]` and runs out before
// the `}` — measured on the #1291 repro. At `{"a":[1,` the stack owes 1, the
// array owes 1, the value owes 1, so the narrowing has to be live at 4.
TEST(JsonConstrainFsm, ForceCloseKeepsAMarginForTheForcedValuesOwnFrame) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\":[1,");
    fsm.set_remaining_budget(4);
    EXPECT_EQ(fsm.allowed_categories_for_test(), CAT_NUMBER_START)
        << "narrowing not yet live at 4 — the walk will land one token short";
}

// The narrowing must stay off while the budget is comfortable, in these states
// too — otherwise every array in a long reply gets a forced `0`.
TEST(JsonConstrainFsm, ForceCloseStaysOffInNeedStatesWithBudget) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"a\":[1,");
    fsm.set_remaining_budget(100);
    const uint16_t m = fsm.allowed_categories_for_test();
    EXPECT_TRUE(m & CAT_QUOTE) << "a string value is legal here and the budget is ample";
    EXPECT_TRUE(m & CAT_OPEN_BRACKET) << "a nested array is legal here and the budget is ample";
}

// #1104: raw control characters must never reach a string. The grammar has
// always rejected them; apply_mask's in-string fast path skipped the check for
// any token without a quote or a backslash, which is exactly what a newline
// token is. Pinned at the grammar level here — the fast path is a GPU path,
// but it now defers to the same rule.
TEST(JsonConstrainFsm, RejectsRawControlCharsInsideStrings) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"k\":\"abc");
    EXPECT_FALSE(fsm.sim_token_valid("\n")) << "raw newline accepted inside a string";
    EXPECT_FALSE(fsm.sim_token_valid("de\nf")) << "raw newline accepted mid-token";
    EXPECT_FALSE(fsm.sim_token_valid("\t")) << "raw tab accepted inside a string";
    EXPECT_TRUE(fsm.sim_token_valid("def")) << "ordinary string content rejected";
    EXPECT_TRUE(fsm.sim_token_valid("\\n")) << "ESCAPED newline must stay legal";
}

// ---------------------------------------------------------------------------
// #1197: constrained output lost every non-ASCII character — "Die Bären hören"
// came back as "Die Baren horen". The FSM was never the problem; it casts to
// unsigned char before comparing. classify_token() did not, and `char` is
// signed here, so every byte of a multi-byte UTF-8 sequence read as negative:
// the single-char path failed `first >= 32` and the multi-char path hit
// `c < 32` and cleared is_str. Tokens carrying an umlaut therefore lost
// CAT_STRING_CHAR and were masked out by category, before the FSM was ever
// asked. The model then spelled the nearest ASCII word it was allowed to.
// ---------------------------------------------------------------------------
TEST(TokenCategory, NonAsciiCountsAsStringContent) {
    // A whole word, the way a BPE vocabulary usually carries it.
    EXPECT_TRUE(classify_token("Bären") & CAT_STRING_CHAR) << "multi-byte word rejected";
    EXPECT_TRUE(classify_token("ä") & CAT_STRING_CHAR) << "two-byte character rejected";
    EXPECT_TRUE(classify_token("größte") & CAT_STRING_CHAR) << "sharp s rejected";
    EXPECT_TRUE(classify_token("日本語") & CAT_STRING_CHAR) << "three-byte characters rejected";
    EXPECT_TRUE(classify_token("🐻") & CAT_STRING_CHAR) << "four-byte character rejected";

    // A BPE token can be a single raw byte — every byte of a UTF-8 sequence
    // has to pass on its own, or the sequence can never be spelled.
    for (int b = 0x80; b <= 0xFF; b++) {
        const std::string one(1, static_cast<char>(b));
        EXPECT_TRUE(classify_token(one) & CAT_STRING_CHAR) << "lone byte 0x" << std::hex << b << " rejected";
    }

    // What must STAY rejected, so the fix does not open the door too wide.
    EXPECT_FALSE(classify_token("\x01") & CAT_STRING_CHAR) << "control char accepted";
    EXPECT_FALSE(classify_token("\n") & CAT_STRING_CHAR) << "raw newline accepted";
    EXPECT_FALSE(classify_token("\"") & CAT_STRING_CHAR) << "bare quote accepted";
    EXPECT_FALSE(classify_token("\\") & CAT_STRING_CHAR) << "bare backslash accepted";
    EXPECT_FALSE(classify_token("ab\nc") & CAT_STRING_CHAR) << "embedded newline accepted";
}

// ---------------------------------------------------------------------------
// #1199 follow-up: a free string value could not be CLOSED. The end of a string
// is spelled by a BPE vocabulary far more often as `."`, `n"`, `!"` than as a
// bare `"`. Those tokens got neither CAT_STRING_CHAR (is_str clears on any
// quote) nor CAT_QUOTE (it keyed on `first == '"'`), so their category was
// 0x0000 and the STRING_VALUE pre-filter — CAT_STRING_CHAR | CAT_QUOTE —
// dropped them before token_legal, which accepts them, was consulted.
//
// Measured on Qwen3-8B-NVFP4 at the position after `...rascheln`: unmasked, `."`
// is top-1 at logprob 0.0 and `.”` sits 13.4 nats behind it; masked, `."` is not
// in the top 5 at all and `.”` wins. The model then wrote a typographic quote —
// legal string content — and the value ran to max_tokens as invalid JSON.
// ---------------------------------------------------------------------------
TEST(TokenCategory, StringEndingTokensCarryQuote) {
    const uint16_t string_phase = CAT_STRING_CHAR | CAT_QUOTE;  // schema STRING_VALUE
    // The forms a vocabulary actually uses to end a string.
    for (const char* t : {".\"", "n\"", "!\"", "ln\"", ".\")", "\"}", "\"", "e\","}) {
        EXPECT_TRUE(classify_token(t) & string_phase)
            << "token " << t << " cannot pass the STRING_VALUE category pre-filter, "
            << "so a free string can never be closed with it";
    }
    // The quote is what earns CAT_QUOTE, wherever it sits in the token.
    EXPECT_TRUE(classify_token(".\"") & CAT_QUOTE) << "trailing quote not recognised";
    EXPECT_TRUE(classify_token("\"}") & CAT_QUOTE) << "leading quote not recognised";
    EXPECT_TRUE(classify_token("a\"b") & CAT_QUOTE) << "embedded quote not recognised";
    // And a token with no quote must NOT gain it — the pre-filter still has to
    // separate string content from string end.
    EXPECT_FALSE(classify_token(".\u201d") & CAT_QUOTE) << "typographic quote counted as a real one";
    EXPECT_FALSE(classify_token("rascheln") & CAT_QUOTE) << "plain word counted as a quote";
    EXPECT_TRUE(classify_token(".\u201d") & CAT_STRING_CHAR) << "typographic quote must stay string content";
}

TEST(JsonConstrainFsm, AcceptsUmlautsInsideStrings) {
    JsonConstrainer fsm;
    fsm.advance_text("{\"k\":\"Die ");
    EXPECT_TRUE(fsm.sim_token_valid("Bären")) << "umlaut word rejected inside a string";
    EXPECT_TRUE(fsm.sim_token_valid("ö")) << "lone umlaut rejected inside a string";
}

}  // namespace
}  // namespace imp
