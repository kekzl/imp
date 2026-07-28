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

#include <random>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr uint32_t kSeed = 0x5EED1067;  // fixed: failures must reproduce

// --- Random valid-JSON generator -------------------------------------------
// ASCII only and escape-free on purpose: non-ASCII in constrained strings is a
// known, deliberate limitation of the token classifier, not a grammar bug, and
// mixing it in here would produce false failures rather than findings.

std::string gen_string(std::mt19937& rng) {
    static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789_ -";
    std::uniform_int_distribution<int> len(0, 8);
    std::uniform_int_distribution<size_t> ch(0, sizeof(kAlphabet) - 2);
    std::string s = "\"";
    int n = len(rng);
    for (int i = 0; i < n; i++)
        s += kAlphabet[ch(rng)];
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
    const Case cases[] =
        {
            {"0", true},       {"-0", true},     {"3", true},    {"3.5", true},
            {"1e5", true},     {"1E5", true},    {"1e+5", true}, {"1e-5", true},
            {"-2.5e-3", true}, {"3.5.5", false},                // second decimal point — the #1104 shape
            {"1e5e5", false},                                   // second exponent
            {"1e+-5", false},                                   // double sign
            {"3-5", false},                                     // sign inside the mantissa
            {"1.2e", false},   {"3.", false},    {"-", false},  // incomplete: still owe a digit
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
}  // namespace
}  // namespace imp
