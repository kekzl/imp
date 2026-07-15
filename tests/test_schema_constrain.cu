#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/json_constrain.h"
#include "compute/constrain_common.h"
#include "compute/json_schema.h"
#include "compute/schema_constrain.h"
#include "model/tokenizer.h"
#include "runtime/constraint_manager.h"

#include <string>
#include <cfloat>

#include "test_cuda_skip.h"

// Schema-FSM side of constrained decoding: RegexNfa (`pattern`),
// SchemaConstrainer masking/update, $ref/$defs, and the jump-ahead
// forced_text probe (#844). The any-JSON constrainer + preamble gate live
// in test_json_constrain.cu.

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// RegexNfa — direct unit tests for JSON-schema `pattern` enforcement.
// Isolated from the model / preamble gate to pinpoint over-masking.
// ---------------------------------------------------------------------------

// Feed a whole string; return final state set ({} if it died).
static std::vector<int> nfa_run(const RegexNfa& n, const std::string& s) {
    std::vector<int> st = n.start_set();
    for (char c : s) {
        st = n.step(st, static_cast<unsigned char>(c));
        if (st.empty())
            return st;
    }
    return st;
}

TEST(RegexNfaTest, Literal) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("abc"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_TRUE(n.accepts(nfa_run(n, "abc")));
    EXPECT_TRUE(nfa_run(n, "abx").empty());        // diverges at 3rd char
    EXPECT_FALSE(n.accepts(nfa_run(n, "ab")));     // prefix alive but not accepting
    EXPECT_TRUE(nfa_run(n, "x").empty());          // wrong first char dies
}

TEST(RegexNfaTest, CharClassRange) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("[A-Z]"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_TRUE(n.accepts(nfa_run(n, "D")));
    EXPECT_TRUE(nfa_run(n, "d").empty());          // lowercase not in class
}

TEST(RegexNfaTest, CountedRepeat) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("[A-Z]{3}"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_FALSE(n.step(n.start_set(), 'D').empty());  // first char must survive
    EXPECT_TRUE(n.accepts(nfa_run(n, "DEU")));
    EXPECT_FALSE(n.accepts(nfa_run(n, "DE")));     // only 2 — not yet accepting
    EXPECT_TRUE(nfa_run(n, "DEUX").empty());       // 4th char dies
    EXPECT_TRUE(nfa_run(n, "Dx").empty());         // 2nd char wrong class
}

TEST(RegexNfaTest, Anchored) {
    RegexNfa n;
    ASSERT_TRUE(n.compile("^[A-Z]{3}$"));
    EXPECT_FALSE(n.start_set().empty());
    EXPECT_FALSE(n.step(n.start_set(), 'D').empty());
    EXPECT_TRUE(n.accepts(nfa_run(n, "DEU")));
    EXPECT_FALSE(n.accepts(nfa_run(n, "DE")));
}

// End-to-end SchemaConstrainer: a `pattern` value must allow pattern-valid tokens
// and mask the rest — and crucially must NOT mask everything (the over-masking
// regression that produced "!!!!"). No model / preamble gate involved.
TEST(SchemaConstrainTest, PatternEnforcementMasksCorrectly) {
    SKIP_IF_NO_CUDA();

    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "code",
                                     ":",     "}",   "D",    "DEU", "abc"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string","pattern":"^[A-Z]{3}$"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);

    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    // Walk the FSM to the string value: {  "code"  :  "
    for (int t : {3, 4, 5, 4, 6, 4})
        sc.update(t);

    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, vocab * sizeof(float));
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    auto allowed = [&](int i) { return h[i] > -1e30f; };
    int n_allowed = 0;
    for (int i = 0; i < vocab; i++)
        if (allowed(i)) n_allowed++;

    EXPECT_GT(n_allowed, 0) << "over-masking regression: every token forbidden in STRING_PATTERN";
    EXPECT_TRUE(allowed(8)) << "'D' (uppercase, pattern-alive) must be allowed";
    EXPECT_TRUE(allowed(9)) << "'DEU' (full ^[A-Z]{3}$ match) must be allowed";
    EXPECT_FALSE(allowed(10)) << "'abc' (lowercase) must be masked by the pattern";
}

// At OBJECT_OPEN, a multi-char token that begins with the opening quote
// (`"code`, `"Why`) opens the key string AND fills key chars in one step.
// Such tokens must be narrowed to valid key prefixes — otherwise a non-key
// token (`"Why`) slips through on its CAT_QUOTE bit and the FSM gets stuck
// mid-key, degenerating into "!!!!". No model / preamble gate involved.
TEST(SchemaConstrainTest, ObjectOpenQuotePrefixedKeyMasked) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4    5     6        7       8     9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "}", "\"", "\"code", "\"Why", ":", "code"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);

    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    sc.update(3);  // "{"  → OBJECT_OPEN

    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, vocab * sizeof(float));
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    auto allowed = [&](int i) { return h[i] > -1e30f; };

    EXPECT_TRUE(allowed(5)) << "bare opening quote must be allowed";
    EXPECT_TRUE(allowed(6)) << "'\"code' (quote + valid complete key) must be allowed";
    EXPECT_FALSE(allowed(7)) << "'\"Why' (quote + invalid key) must be masked — the OBJECT_OPEN hole";
    EXPECT_FALSE(allowed(9)) << "'code' without opening quote must be masked at OBJECT_OPEN (category)";
    EXPECT_FALSE(allowed(4)) << "'}' must be masked: required key 'code' not yet emitted";
}

// Key order in a JSON object is not significant: {"type":"string","enum":[...]}
// and the alphabetically-reordered {"enum":[...],"type":"string"} (what a
// request round-trip through a JSON library produces) must both parse to ENUM.
// A later "type":"string" must not demote the node back to a free string.
TEST(SchemaConstrainTest, EnumPrecedenceIsOrderIndependent) {
    auto a = parse_json_schema(R"({"type":"string","enum":["en","de","fr"]})");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->type, SchemaType::ENUM) << "type-then-enum must be ENUM";

    auto b = parse_json_schema(R"({"enum":["en","de","fr"],"type":"string"})");
    ASSERT_TRUE(b != nullptr);
    EXPECT_EQ(b->type, SchemaType::ENUM) << "enum-then-type must still be ENUM";
    EXPECT_EQ(b->enum_values.size(), 3u);
}

// Run apply_mask over a vocab of `n` tokens and return which token ids survive.
static std::vector<bool> schema_allowed(SchemaConstrainer& sc, int n) {
    std::vector<float> h(n, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
    std::vector<bool> out(n);
    for (int i = 0; i < n; i++)
        out[i] = h[i] > -1e30f;
    return out;
}

// A combined token like `{}` opens AND closes an object in one step; it must be
// rejected while required keys are unmet (the first-char category mask only
// sees CAT_OPEN_BRACE and would let it through → empty object, schema violated).
TEST(SchemaConstrainTest, PrematureObjectCloseRejected) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5     6     7
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "}", "{}", "\"", "code"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    // Fresh root: phase VALUE_START expecting the object to open.
    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(a[3]) << "'{' must open the object";
    EXPECT_FALSE(a[4]) << "bare '}' masked by category (not a value start)";
    EXPECT_FALSE(a[5]) << "'{}' combined token must be rejected — required 'code' unmet";
}

// #850: a backslash inside an object KEY was accepted and silently dropped
// (no phase change, no buffer append), so the NEXT char matched the property
// prefix while the emitted text carried the escape — `{"\number_x":5}`-style
// schema-invalid keys observed live on Qwen3-8B json_schema. Property names
// are matched on raw chars (escape sequences were never decoded), so no
// legal key needs an escape: reject `\` in keys outright, single-char and
// smuggled inside a multi-char token alike.
TEST(SchemaConstrainTest, BackslashInKeyRejected) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4     5    6     7      8
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "n", "\\", "\"\\n", "um"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"num":{"type":"string"}},"required":["num"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    // At OBJECT_OPEN: a combined token opening the key with an escape
    // (`"\n`) must be masked; the bare quote stays legal.
    sc.update(3);  // "{" -> OBJECT_OPEN
    auto open = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(open[4]) << "bare '\"' must open the key";
    EXPECT_FALSE(open[7]) << "'\"\\n' must be masked — escape smuggled into the key";

    // Inside OBJECT_KEY: the bare backslash must be masked; real key
    // prefixes stay legal.
    sc.update(4);  // '"' -> OBJECT_KEY
    auto key = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(key[5]) << "'n' is a valid prefix of 'num'";
    EXPECT_FALSE(key[6]) << "'\\' must be masked inside a key (#850)";
    EXPECT_FALSE(key[8]) << "'um' is not a valid prefix of 'num'";
}

// After the last property's value, a comma would dangle (no key can follow) —
// it must be masked, leaving only the closing brace. Prevents `{"a":"x",}`.
TEST(SchemaConstrainTest, TrailingCommaRejected) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5    6    7    8    9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "a", "x", ":", "}", ","};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    for (int t : {3, 4, 5, 4, 7, 4, 6, 4})  // { "a" : "x"  -> OBJECT_AFTER_VALUE
        sc.update(t);
    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(a[8]) << "'}' must close the object after the only property";
    EXPECT_FALSE(a[9]) << "',' must be masked — no further property can follow";
}

// JSON forbids leading zeros: after a single '0' the integer part is done, so
// another digit is illegal (also bounds `0999...` degeneration).
TEST(SchemaConstrainTest, IntegerLeadingZeroRejected) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5    6    7    8    9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "n", ":", "0", "5", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    for (int t : {3, 4, 5, 4, 6, 7})  // { "n" : 0  -> NUMBER_VALUE (leading zero)
        sc.update(t);
    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_FALSE(a[8]) << "digit after a leading '0' must be masked";
    EXPECT_TRUE(a[9]) << "'}' must be allowed to close the number/object";
}

// #751: an unbounded integer/number must not run digits forever — once the digit
// run hits the cap the continue-number category is dropped, forcing the number to
// close (valid, terminated JSON instead of a runaway to max_tokens).
TEST(SchemaConstrainTest, IntegerDigitRunCappedForcesClose) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5    6    7    8    9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "n", ":", "1", "0", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    for (int t : {3, 4, 5, 4, 6, 7})  // { "n" : 1  -> NUMBER_VALUE, digit_count=1
        sc.update(t);
    // Below the cap (after a handful of digits) more digits are still allowed.
    sc.update(8);  // "0" -> digit_count=2
    auto below = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(below[8]) << "digit must be allowed well below the cap";
    // Drive the digit run to the cap (kMaxNumberDigits=40); digit_count is now 2.
    for (int i = 0; i < 38; i++)
        sc.update(8);  // -> digit_count=40
    auto capped = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_FALSE(capped[8]) << "digit must be masked once the digit-run cap is hit";
    EXPECT_TRUE(capped[9]) << "'}' must be allowed to close the capped number";
}

// A combined value token must be validated against an enum/integer constraint,
// not just its opening quote/digit category.
TEST(SchemaConstrainTest, EnumAndIntegerComboTokensValidated) {
    SKIP_IF_NO_CUDA();
    //          0       1     2      3   4    5            6    7      8       9      10     11    12     13
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "sentiment", ":", "\"en\"", "\":\"", "\"x", "5", "5.", "5.0", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    // --- enum property ---
    {
        auto schema = parse_json_schema(
            R"({"type":"object","properties":{"sentiment":{"type":"string","enum":["en","de","fr"]}},"required":["sentiment"]})");
        ASSERT_TRUE(schema != nullptr);
        SchemaConstrainer sc;
        ASSERT_TRUE(sc.init(tok, std::move(schema)));
        for (int t : {3, 4, 5, 4, 6})  // { " sentiment " :  -> value frame (enum)
            sc.update(t);
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[7]) << "'\"en\"' (exact enum value in one token) must be allowed";
        EXPECT_TRUE(a[4]) << "bare opening '\"' must be allowed";
        EXPECT_FALSE(a[8]) << "'\":\"' (quote+colon+quote, not an enum value) must be masked";
        EXPECT_FALSE(a[9]) << "'\"x' (invalid enum prefix) must be masked";
    }

    // --- integer property: reject float-shaped combined tokens ---
    {
        auto schema = parse_json_schema(
            R"({"type":"object","properties":{"sentiment":{"type":"integer"}},"required":["sentiment"]})");
        ASSERT_TRUE(schema != nullptr);
        SchemaConstrainer sc;
        ASSERT_TRUE(sc.init(tok, std::move(schema)));
        for (int t : {3, 4, 5, 4, 6})  // { " sentiment " :  -> value frame (integer)
            sc.update(t);
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[10]) << "'5' must be allowed for an integer";
        EXPECT_FALSE(a[11]) << "'5.' must be masked for an integer";
        EXPECT_FALSE(a[12]) << "'5.0' must be masked for an integer";
    }
}

// ---------------------------------------------------------------------------
// minItems / maxItems (#1014) — the degeneration guard: a budget-force-closed
// reasoning model loops enum array items (`["tech","tech",...`) to max_tokens.
// Explicit maxItems must hard-stop the array; an enum-items array without one
// is capped at the enum's cardinality; minItems blocks premature close.
// ---------------------------------------------------------------------------

namespace {
// Shared vocab for the array-bounds tests:
//  0        1      2       3    4     5    6    7    8    9    10   11
std::vector<std::string> array_vocab() {
    return {"<unk>", "<s>", "</s>", "{", "\"", "t", ":", "[", "]", ",", "a", "}"};
}
constexpr int kTokOpenBrace = 3, kTokQuote = 4, kTokKey = 5, kTokColon = 6;
constexpr int kTokOpenBracket = 7, kTokCloseBracket = 8, kTokComma = 9, kTokItemA = 10;

// Drive `{"t":[` then `n` complete items ("a"), leaving the FSM in
// ARRAY_AFTER_ITEM (or ARRAY_OPEN for n=0).
void drive_array(SchemaConstrainer& sc, int n_items) {
    for (int t : {kTokOpenBrace, kTokQuote, kTokKey, kTokQuote, kTokColon, kTokOpenBracket})
        sc.update(t);
    for (int i = 0; i < n_items; i++) {
        if (i > 0)
            sc.update(kTokComma);
        sc.update(kTokQuote);
        sc.update(kTokItemA);
        sc.update(kTokQuote);
    }
}
}  // namespace

TEST(SchemaConstrainTest, MaxItemsMasksCommaAtCap) {
    SKIP_IF_NO_CUDA();
    auto toks = array_vocab();
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"t":{"type":"array","maxItems":2,)"
        R"("items":{"type":"string"}}},"required":["t"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    drive_array(sc, 1);
    auto one = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(one[kTokComma]) << "below maxItems the array may continue";
    EXPECT_TRUE(one[kTokCloseBracket]) << "no minItems — close always legal";
    // Item 2 on the same FSM: `,"a"` — now at the cap.
    sc.update(kTokComma), sc.update(kTokQuote), sc.update(kTokItemA), sc.update(kTokQuote);
    auto capped = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_FALSE(capped[kTokComma]) << "at maxItems the comma must be masked";
    EXPECT_TRUE(capped[kTokCloseBracket]) << "']' must stay legal at the cap";
}

TEST(SchemaConstrainTest, EnumItemsArrayCappedAtCardinality) {
    SKIP_IF_NO_CUDA();
    auto toks = array_vocab();
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    // Two-member enum, NO maxItems: effective cap = 2 (see effective_max_items).
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"t":{"type":"array",)"
        R"("items":{"type":"string","enum":["a","aa"]}}},"required":["t"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    drive_array(sc, 2);
    auto capped = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_FALSE(capped[kTokComma])
        << "enum-items array without maxItems must cap at the enum cardinality";
    EXPECT_TRUE(capped[kTokCloseBracket]) << "']' must stay legal at the cap";
}

TEST(SchemaConstrainTest, MinItemsBlocksPrematureClose) {
    SKIP_IF_NO_CUDA();
    auto toks = array_vocab();
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"t":{"type":"array","minItems":2,)"
        R"("items":{"type":"string"}}},"required":["t"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    drive_array(sc, 0);
    auto empty = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_FALSE(empty[kTokCloseBracket]) << "empty array below minItems may not close";
    // First item on the same FSM: `"a"`.
    sc.update(kTokQuote), sc.update(kTokItemA), sc.update(kTokQuote);
    auto one = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_FALSE(one[kTokCloseBracket]) << "one item below minItems=2 may not close";
    EXPECT_TRUE(one[kTokComma]) << "the array must be allowed to continue";
    sc.update(kTokComma), sc.update(kTokQuote), sc.update(kTokItemA), sc.update(kTokQuote);
    auto two = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(two[kTokCloseBracket]) << "minItems satisfied — close must be legal";
}

// ---------------------------------------------------------------------------
// TOOL_CALL enforcement (#1002): envelope literals forced, "name" before
// "arguments", name enum restricted to the tool set, arguments bound to the
// CHOSEN tool's parameter schema, EOS forced after the close literal.
// ---------------------------------------------------------------------------
TEST(SchemaConstrainTest, ToolCallEnvelopeAndNameBinding) {
    SKIP_IF_NO_CUDA();
    // Single-char vocab over the emission corpus + a negative probe 'x'.
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
    std::string chars = "<>tol_ca\n{\"nme:usrgd,1}/x";
    for (char c : chars)
        toks.push_back(std::string(1, c));
    auto id = [&](char c) {
        for (size_t i = 3; i < toks.size(); i++)
            if (toks[i][0] == c)
                return static_cast<int>(i);
        ADD_FAILURE() << "missing token for char " << c;
        return 0;
    };
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);

    // Two tools: add / sub, both {"a" hm... use params with property "d" ...
    std::vector<std::pair<std::string, std::string>> tools = {
        {"add", R"({"type":"object","properties":{"d":{"type":"number"}},"required":["d"]})"},
        {"sub", R"({"type":"object","properties":{"u":{"type":"number"}},"required":["u"]})"},
    };
    auto schema = build_tool_call_schema(tools);
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    sc.set_envelope("<tool_call>\n", "\n</tool_call>");
    sc.reset();

    // 1. Envelope first: '<' legal, '{' not.
    auto at_start = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_start[id('<')]) << "envelope open must be legal";
    EXPECT_FALSE(at_start[id('{')]) << "the body may not start before the envelope";

    auto feed = [&](const std::string& s) {
        for (char c : s)
            sc.update(id(c));
    };
    feed("<tool_call>\n{\"");
    // 2. Key order: only "name" may open.
    auto at_key = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_key[id('n')]) << "'name' must be available first";
    EXPECT_FALSE(at_key[id('a')]) << "'arguments' may not precede 'name'";

    feed("name\":\"");
    // 3. Name enum: 'a'(add)/'s'(sub) legal, 'x' not.
    auto at_enum = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_enum[id('a')]);
    EXPECT_TRUE(at_enum[id('s')]);
    EXPECT_FALSE(at_enum[id('x')]) << "non-tool names must be masked";

    feed("add\",\"arguments\":{\"");
    // 4. Binding: only add's parameter 'd' is a legal key — not sub's 'u'.
    auto at_args = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_args[id('d')]) << "chosen tool's parameter must be legal";
    EXPECT_FALSE(at_args[id('u')]) << "the OTHER tool's parameter must be masked";

    feed("d\":1}}");
    // 5. Close literal forced.
    auto at_close = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_close[id('\n')]) << "close literal must be legal after the body";
    EXPECT_FALSE(at_close[id('{')]);

    feed("\n</tool_call>");
    // 6. Stack drained: EOS forced.
    auto at_done = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(at_done[2]) << "EOS must be allowed after the envelope closes";
    EXPECT_FALSE(at_done[id('<')]) << "no trailing text after the close literal";
}

TEST(SchemaConstrainTest, ToolCallBuilderRejectsUnenforceable) {
    // Free-form parameters (no properties) → decline.
    EXPECT_TRUE(build_tool_call_schema({{"t", R"({"type":"object"})"}}) == nullptr);
    // $defs inside a parameter schema → decline (would resolve wrongly).
    EXPECT_TRUE(build_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"$ref":"#/$defs/I"}},)"
                           R"("$defs":{"I":{"type":"integer"}}})"}}) == nullptr);
    // Empty tool list → decline.
    EXPECT_TRUE(build_tool_call_schema({}) == nullptr);
    // Well-formed → builds.
    EXPECT_TRUE(build_tool_call_schema(
                    {{"t", R"({"type":"object","properties":{"i":{"type":"integer"}}})"}}) != nullptr);
}

// ---------------------------------------------------------------------------
// $ref / $defs (issue #555) — pydantic/zod emit $defs+$ref for EVERY nested
// model, so this is the agent-framework path, not an exotic corner.
// ---------------------------------------------------------------------------

// Parse-level: $defs collected, $ref resolves, unsupported/unresolvable refs
// fail the parse (decline constrained decoding instead of enforcing a wrong
// grammar). No CUDA needed.
TEST(SchemaConstrainTest, RefDefsParseAndResolve) {
    // pydantic-style nested model
    auto a = parse_json_schema(
        R"({"type":"object","properties":{"inner":{"$ref":"#/$defs/Inner"}},)"
        R"("required":["inner"],"$defs":{"Inner":{"type":"object",)"
        R"("properties":{"x":{"type":"integer"}},"required":["x"]}}})");
    ASSERT_TRUE(a != nullptr);
    ASSERT_EQ(a->properties.size(), 1u);
    const SchemaNode* inner_ref = a->properties[0].second.get();
    EXPECT_EQ(inner_ref->type, SchemaType::REF);
    const SchemaNode* inner = resolve_schema_ref(a.get(), inner_ref);
    ASSERT_TRUE(inner != nullptr);
    EXPECT_EQ(inner->type, SchemaType::OBJECT);
    ASSERT_EQ(inner->properties.size(), 1u);
    EXPECT_EQ(inner->properties[0].first, "x");

    // "definitions" spelling + clone preserves refs/defs
    auto b = parse_json_schema(
        R"({"definitions":{"S":{"type":"string"}},"type":"object",)"
        R"("properties":{"s":{"$ref":"#/definitions/S"}}})");
    ASSERT_TRUE(b != nullptr);
    auto b2 = b->clone();
    const SchemaNode* s_res = resolve_schema_ref(b2.get(), b2->properties[0].second.get());
    ASSERT_TRUE(s_res != nullptr);
    EXPECT_EQ(s_res->type, SchemaType::STRING);

    // root self-ref "#"
    auto c = parse_json_schema(
        R"({"type":"object","properties":{"next":{"$ref":"#"}},"required":[]})");
    ASSERT_TRUE(c != nullptr);
    EXPECT_EQ(resolve_schema_ref(c.get(), c->properties[0].second.get()), c.get());

    // unresolvable name → parse fails
    EXPECT_EQ(parse_json_schema(
                  R"({"type":"object","properties":{"a":{"$ref":"#/$defs/Missing"}}})"),
              nullptr);
    // external / deep-pointer refs → parse fails
    EXPECT_EQ(parse_json_schema(
                  R"({"properties":{"a":{"$ref":"https://example.com/s.json"}}})"),
              nullptr);
    EXPECT_EQ(parse_json_schema(
                  R"({"properties":{"a":{"$ref":"#/properties/b"}}})"),
              nullptr);
    // pure ref->ref cycle → parse fails (no structure to terminate resolution)
    EXPECT_EQ(parse_json_schema(
                  R"({"$ref":"#/$defs/A","$defs":{"A":{"$ref":"#/$defs/B"},)"
                  R"("B":{"$ref":"#/$defs/A"}}})"),
              nullptr);
}

// Enforcement through a $ref: the referenced object's keys/required are
// enforced exactly as an inline schema would be.
TEST(SchemaConstrainTest, RefNestedModelEnforced) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5        6    7    8    9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "inner", ":", "x", "5", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"inner":{"$ref":"#/$defs/Inner"}},)"
        R"("required":["inner"],"$defs":{"Inner":{"type":"object",)"
        R"("properties":{"x":{"type":"integer"}},"required":["x"]}}})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
    for (int t : {3, 4, 5, 4, 6})  // { "inner" :  → value frame is the REF target
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[3]) << "'{' must open the $ref'd inner object";
        EXPECT_FALSE(a[8]) << "bare digit masked — inner value is an object, not a number";
    }
    for (int t : {3, 4, 7, 4, 6})  // { "x" :  → inner integer value
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[8]) << "'5' must be allowed for Inner.x (integer)";
        EXPECT_FALSE(a[9]) << "'}' masked — required Inner.x not yet emitted";
    }
}

// True recursion: a tree node whose children are arrays of itself. The frame
// stack must follow $ref depth arbitrarily (here 2 levels) — the gap the old
// NFA backend could not represent.
TEST(SchemaConstrainTest, RecursiveSchemaEnforced) {
    SKIP_IF_NO_CUDA();
    //                                 0       1      2      3    4    5    6    7    8    9    10   11
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "v", ":", "1", ",", "kids", "[", "]"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, 1, 2);
    auto schema = parse_json_schema(
        R"({"$ref":"#/$defs/Node","$defs":{"Node":{"type":"object",)"
        R"("properties":{"v":{"type":"integer"},"kids":{"type":"array",)"
        R"("items":{"$ref":"#/$defs/Node"}}},"required":["v"]}}})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    // { "v" : 1 , "kids" : [
    for (int t : {3, 4, 5, 4, 6, 7, 8, 4, 9, 4, 6, 10})
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[3]) << "'{' must start a recursive child Node inside kids[]";
        EXPECT_TRUE(a[11]) << "']' must be allowed (empty kids)";
        EXPECT_FALSE(a[7]) << "bare digit masked — items are Node objects";
    }
    // open child: { "v" : 1 — the recursive frame enforces Node's grammar at
    // depth 2: a second "kids" key (','+key) must be offered, digits must not.
    for (int t : {3, 4, 5, 4, 6, 7})
        sc.update(t);
    {
        auto a = schema_allowed(sc, static_cast<int>(toks.size()));
        EXPECT_TRUE(a[8]) << "',' after child's v — 'kids' is still emittable at depth 2";
        EXPECT_FALSE(a[4]) << "bare quote masked mid-number at depth 2";
    }
}

// ===========================================================================
// Vocab-mismatch regression (SIGBUS 2026-06-09): SafeTensors models have
// MORE logits than tokenizer entries (Qwen3-8B-NVFP4: lm_head vocab 151936
// vs tokenizer.json 151669). apply_mask receives the LOGITS vocab size; the
// constrainer classified only the TOKENIZER vocab. The host validation loop
// then read token_texts_[i] out of bounds (SIGBUS — killed imp-server on the
// first json_mode request), and the mask kernels read the category/allow
// device buffers out of bounds. Contract under test: apply_mask with a
// larger vocab_size must (a) not crash and (b) mask every logit in the
// padding range [tokenizer_vocab, model_vocab) — padding ids are unknown to
// the grammar and untrained in the model, so they must never be sampleable.
// ===========================================================================

// ===========================================================================
// Raw-control-char regression (2026-06-10): JSON forbids unescaped U+0000–
// U+001F inside strings, but both FSMs accepted "any content char" there.
// Multi-char tokens whose FIRST char passes the category mask (`"<newline>`
// opens a string and smuggles a raw newline in one step) produced output that
// json.loads() rejects ("Invalid control character"). Observed live on
// Qwen3-8B-NVFP4 json_schema generation.
// ===========================================================================

TEST(SchemaConstrainTest, RawControlCharInStringMasked) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5       6    7     8        9
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "code", ":", "}", "\"\n", "\"ok"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    // Walk to OBJECT_COLON: {  "code"  :   — next token opens the string value.
    for (int t : {3, 4, 5, 4, 6})
        sc.update(t);

    auto a = schema_allowed(sc, static_cast<int>(toks.size()));
    EXPECT_TRUE(a[9]) << "'\"ok' (quote + printable content) must be allowed";
    EXPECT_FALSE(a[8]) << "'\"<newline>' must be masked — raw control char inside a string";
}

TEST(JsonConstrainTest, RawControlCharInStringMasked) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5      6
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "\"\n", "\"ok"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    JsonConstrainer jc;
    ASSERT_TRUE(jc.init(tok));
    jc.update(3);  // '{' → OBJECT_START, next token may open a key string

    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, vocab * sizeof(float));
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    jc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    EXPECT_GT(h[6], -1e30f) << "'\"ok' (quote + printable key chars) must be allowed";
    EXPECT_FLOAT_EQ(h[5], -FLT_MAX) << "'\"<newline>' must be masked — raw control char in string";
}

TEST(JsonConstrainTest, ModelVocabLargerThanTokenizerMasksPadding) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5    6
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "}", "0"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    JsonConstrainer jc;
    ASSERT_TRUE(jc.init(tok));

    const int tok_vocab = static_cast<int>(toks.size());
    const int model_vocab = tok_vocab + 9;  // simulated lm_head padding rows

    std::vector<float> h(model_vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, model_vocab * sizeof(float));
    cudaMemcpy(d, h.data(), model_vocab * sizeof(float), cudaMemcpyHostToDevice);
    jc.apply_mask(d, model_vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, model_vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    // '{' starts a JSON document — must stay alive.
    EXPECT_GT(h[3], -1e30f) << "'{' must be allowed at document start";
    // Every padding id (no tokenizer entry) must be masked.
    for (int i = tok_vocab; i < model_vocab; i++) {
        EXPECT_FLOAT_EQ(h[i], -FLT_MAX) << "padding id " << i << " leaked through the json mask";
    }
}

TEST(SchemaConstrainTest, ModelVocabLargerThanTokenizerMasksPadding) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5       6    7
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "code", ":", "}"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);

    auto schema = parse_json_schema(
        R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})");
    ASSERT_TRUE(schema != nullptr);
    SchemaConstrainer sc;
    ASSERT_TRUE(sc.init(tok, std::move(schema)));

    const int tok_vocab = static_cast<int>(toks.size());
    const int model_vocab = tok_vocab + 9;

    std::vector<float> h(model_vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, model_vocab * sizeof(float));
    cudaMemcpy(d, h.data(), model_vocab * sizeof(float), cudaMemcpyHostToDevice);
    sc.apply_mask(d, model_vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, model_vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    EXPECT_GT(h[3], -1e30f) << "'{' must be allowed at schema root";
    for (int i = tok_vocab; i < model_vocab; i++) {
        EXPECT_FLOAT_EQ(h[i], -FLT_MAX) << "padding id " << i << " leaked through the schema mask";
    }
}

// ---------------------------------------------------------------------------
// Jump-ahead (#844): forced_text — the characters every schema-legal
// continuation must spell next. Pure probe: never advances the FSM.
// ---------------------------------------------------------------------------

//                                                0        1      2      3    4     5    6    7    8    9    10   11   12
static const std::vector<std::string> kForcedToks = {
    "<unk>", "<s>", "</s>", "{", "\"", "c", "o", "d", "e", ":", "}", "x", "y"};

// In-place init (SchemaConstrainer owns raw device buffers — not movable).
static void init_forced_sc(Tokenizer& tok, SchemaConstrainer& sc, const char* schema_json) {
    std::vector<float> scores(kForcedToks.size(), 0.0f);
    tok.load_vocab(kForcedToks, scores, /*bos_id=*/1, /*eos_id=*/2);
    auto schema = parse_json_schema(schema_json);
    ASSERT_TRUE(schema != nullptr);
    ASSERT_TRUE(sc.init(tok, std::move(schema)));
}

static constexpr const char* kCodeStringSchema =
    R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})";

TEST(SchemaForcedTextTest, EmitsSchemaSkeleton) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc, kCodeStringSchema);

    std::string text;
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"code\":\"")
        << "the forced text must cover the full schema skeleton up to the free string";

    // Pure probe: the FSM must be untouched — probing again yields the same.
    std::string again;
    sc.forced_text(again, 96);
    EXPECT_EQ(again, text);
}

TEST(SchemaForcedTextTest, AdvancesWithFsmState) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc, kCodeStringSchema);

    // Walk { " c — the probe must continue from mid-key.
    for (int t : {3, 4, 5})
        sc.update(t);
    std::string text;
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "ode\":\"");
}

TEST(SchemaForcedTextTest, HonorsMaxChars) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc, kCodeStringSchema);

    std::string text;
    EXPECT_EQ(sc.forced_text(text, 3), 3);
    EXPECT_EQ(text, "{\"c");
}

TEST(SchemaForcedTextTest, CompletesFullyDeterminedDocument) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    // Single-value enum: the ENTIRE document {"code":"x"} is forced.
    SchemaConstrainer sc;
    init_forced_sc(tok, sc,
        R"({"type":"object","properties":{"code":{"enum":["x"]}},"required":["code"]})");

    std::string text;
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"code\":\"x\"}")
        << "a fully-determined document must be forced to completion (EOS stays a masked step)";
}

TEST(SchemaForcedTextTest, StopsAtEnumChoiceAfterCommonPrefix) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc,
        R"({"type":"object","properties":{"code":{"enum":["xxo","xxc"]}},"required":["code"]})");

    std::string text;
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"code\":\"xx")
        << "the common enum prefix is forced; the walk stops where values diverge";
}

TEST(SchemaForcedTextTest, StopsAtKeyChoice) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc,
        R"({"type":"object","properties":{"cx":{"type":"string"},"ox":{"type":"string"}},"required":["cx","ox"]})");

    std::string text;
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"") << "two candidate keys: the quote is forced, the first key char is not";
}

TEST(SchemaForcedTextTest, BooleanValueStopsAtChoice) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc,
        R"({"type":"object","properties":{"code":{"type":"boolean"}},"required":["code"]})");

    std::string text;
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"code\":") << "true/false is a real choice — the walk stops at the value";
}

TEST(SchemaForcedTextTest, NullLiteralForcedThroughClose) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc,
        R"({"type":"object","properties":{"code":{"type":"null"}},"required":["code"]})");

    std::string text;
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"code\":null}") << "a null literal (and the final close) is fully forced";
}

TEST(SchemaForcedTextTest, PreambleGateBlocksForcing) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    SchemaConstrainer sc;
    init_forced_sc(tok, sc, kCodeStringSchema);
    const int32_t think_close = 12;  // stand-in close token ('y', unused by the schema walk)
    sc.set_preamble(think_close, /*max_tokens=*/8192, /*thinking_open=*/true);

    std::string text;
    EXPECT_EQ(sc.forced_text(text, 96), 0) << "no forcing while the preamble gate is active";

    sc.update(think_close);  // </think> — gate exits to OFF, FSM enforcing
    EXPECT_GT(sc.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"code\":\"");
}

TEST(SchemaForcedTextTest, ConstraintManagerRouting) {
    SKIP_IF_NO_CUDA();
    Tokenizer tok;
    std::vector<float> scores(kForcedToks.size(), 0.0f);
    tok.load_vocab(kForcedToks, scores, /*bos_id=*/1, /*eos_id=*/2);

    // json_mode has no schema skeleton — must return 0.
    ConstraintManager jm;
    jm.prepare(/*json_mode=*/true, "", &tok);
    ASSERT_TRUE(jm.has_json());
    std::string text;
    EXPECT_EQ(jm.forced_text(text, 96), 0);

    // json_schema routes through to SchemaConstrainer::forced_text.
    ConstraintManager sm;
    sm.prepare(/*json_mode=*/false, kCodeStringSchema, &tok);
    ASSERT_TRUE(sm.has_schema());
    EXPECT_GT(sm.forced_text(text, 96), 0);
    EXPECT_EQ(text, "{\"code\":\"");
}

}  // namespace
}  // namespace imp
