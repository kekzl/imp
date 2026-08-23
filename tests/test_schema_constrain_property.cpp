// ===========================================================================
// Property tests for the json_schema FSM (SchemaConstrainer).
//
// Companion to test_json_constrain_property.cpp, which covers the schema-less
// json_object grammar. This file attacks the schema-driven side, where #761
// (runaway digit run), #850 (backslash in keys) and #1014 (minItems/maxItems
// unenforced) shipped — each found by a symptom, none by a test.
//
// No external oracle: nlohmann validates JSON syntax but not JSON Schema, so
// the generator emits a schema and a CONFORMING document together. Conformance
// is known by construction, which makes both directions testable — accept the
// conforming document, reject the constructed violations.
//
// CPU-only: init_grammar_for_test() installs the grammar without the tokenizer
// classification and device buffers that only apply_mask needs, so this runs in
// the `unit` lane. That is the point — CI has no GPU runner, so the GPU-lane
// batteries next door never guarded these bugs on a pull request.
// ===========================================================================

#include <gtest/gtest.h>

#include "compute/json_schema.h"
#include "compute/schema_constrain.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr uint32_t kSeed = 0x5C4E3A01;

struct Generated {
    std::string schema;  // JSON Schema text
    std::string doc;     // a document that conforms to it, by construction
};

// Every generated object makes ALL properties required and emits them in
// schema order: the tightest, least ambiguous shape, so a rejection is a real
// finding rather than the FSM exercising a legitimate freedom.
Generated gen_node(std::mt19937& rng, int depth) {
    std::uniform_int_distribution<int> pick(0, depth < 2 ? 6 : 4);
    switch (pick(rng)) {
        case 0:
            return {R"({"type":"string"})", "\"abc\""};
        case 1:
            return {R"({"type":"integer"})", "42"};
        case 2:
            return {R"({"type":"number"})", "2.5"};
        case 3:
            return {R"({"type":"boolean"})", "true"};
        case 4:
            return {R"({"type":"string","enum":["red","green"]})", "\"green\""};
        case 5: {  // object
            std::uniform_int_distribution<int> n(1, 3);
            const int count = n(rng);
            std::string props, req, doc = "{";
            for (int i = 0; i < count; i++) {
                Generated child = gen_node(rng, depth + 1);
                const std::string key = "p" + std::to_string(i);
                if (i) {
                    props += ",";
                    req += ",";
                    doc += ",";
                }
                props += "\"" + key + "\":" + child.schema;
                req += "\"" + key + "\"";
                doc += "\"" + key + "\":" + child.doc;
            }
            return {R"({"type":"object","properties":{)" + props + R"(},"required":[)" + req + "]}",
                    doc + "}"};
        }
        default: {  // array with an explicit minItems the document satisfies
            Generated item = gen_node(rng, depth + 1);
            std::uniform_int_distribution<int> n(1, 3);
            const int count = n(rng);
            std::string doc = "[";
            for (int i = 0; i < count; i++) {
                if (i)
                    doc += ",";
                doc += item.doc;
            }
            return {R"({"type":"array","items":)" + item.schema + R"(,"minItems":)" + std::to_string(count) +
                        "}",
                    doc + "]"};
        }
    }
}

// Root must be an object or array.
Generated gen_document(std::mt19937& rng) {
    for (;;) {
        Generated g = gen_node(rng, 0);
        if (!g.doc.empty() && (g.doc.front() == '{' || g.doc.front() == '['))
            return g;
    }
}

// Build a constrainer over `schema_json`; returns nullptr if the schema fails
// to parse (a generator bug, which the caller asserts on).
std::unique_ptr<SchemaConstrainer> make_fsm(const std::string& schema_json) {
    auto schema = parse_json_schema(schema_json);
    if (!schema)
        return nullptr;
    auto sc = std::make_unique<SchemaConstrainer>();
    if (!sc->init_grammar_for_test(std::move(schema)))
        return nullptr;
    return sc;
}

// ===========================================================================
// P1 — Soundness: a document built to satisfy its schema must be accepted.
// ===========================================================================
TEST(SchemaConstrainPropertyTest, AcceptsConformingDocument) {
    std::mt19937 rng(kSeed);
    for (int i = 0; i < 500; i++) {
        Generated g = gen_document(rng);
        auto sc = make_fsm(g.schema);
        ASSERT_NE(sc, nullptr) << "schema failed to parse: " << g.schema;
        EXPECT_TRUE(sc->token_legal(g.doc))
            << "FSM rejected conforming document\n  schema: " << g.schema << "\n  doc:    " << g.doc;
    }
}

// ===========================================================================
// P2 — Prefix closure: every prefix of a conforming document must stay legal,
// since that is the state the model occupies mid-generation.
// ===========================================================================
TEST(SchemaConstrainPropertyTest, AcceptsEveryPrefixOfConformingDocument) {
    std::mt19937 rng(kSeed + 1);
    for (int i = 0; i < 200; i++) {
        Generated g = gen_document(rng);
        auto sc = make_fsm(g.schema);
        ASSERT_NE(sc, nullptr);
        for (size_t cut = 1; cut < g.doc.size(); cut++) {
            EXPECT_TRUE(sc->token_legal(g.doc.substr(0, cut)))
                << "FSM rejected prefix [" << g.doc.substr(0, cut) << "]\n  schema: " << g.schema
                << "\n  doc:    " << g.doc;
        }
    }
}

// ===========================================================================
// P3 — required enforcement: closing an object before its required properties
// are emitted must be refused. `{}` against a schema with required keys is the
// minimal case (the shape PrematureObjectCloseRejected pins by example).
// ===========================================================================
TEST(SchemaConstrainPropertyTest, RejectsObjectMissingRequiredProperties) {
    std::mt19937 rng(kSeed + 2);
    int exercised = 0;
    for (int i = 0; i < 500; i++) {
        Generated g = gen_document(rng);
        if (g.doc.front() != '{')
            continue;
        auto sc = make_fsm(g.schema);
        ASSERT_NE(sc, nullptr);
        EXPECT_FALSE(sc->token_legal("{}"))
            << "FSM accepted empty object despite required properties\n  schema: " << g.schema;
        exercised++;
    }
    EXPECT_GT(exercised, 0) << "no object roots generated — test would be vacuous";
}

// ===========================================================================
// P4 — minItems enforcement (#1014): closing an array before minItems are
// emitted must be refused.
// ===========================================================================
TEST(SchemaConstrainPropertyTest, RejectsArrayBelowMinItems) {
    std::mt19937 rng(kSeed + 3);
    int exercised = 0;
    for (int i = 0; i < 500; i++) {
        Generated g = gen_document(rng);
        if (g.doc.front() != '[')
            continue;
        auto sc = make_fsm(g.schema);
        ASSERT_NE(sc, nullptr);
        // The generator always sets minItems >= 1, so the empty array violates.
        EXPECT_FALSE(sc->token_legal("[]"))
            << "FSM accepted empty array below minItems\n  schema: " << g.schema;
        exercised++;
    }
    EXPECT_GT(exercised, 0) << "no array roots generated — test would be vacuous";
}

// ===========================================================================
// P5 — no dead ends: from any prefix of a conforming document at least one
// ASCII continuation must remain legal (release-bar 7: the sampler must always
// have something valid to emit).
// ===========================================================================
TEST(SchemaConstrainPropertyTest, NoDeadEndStates) {
    std::mt19937 rng(kSeed + 4);
    for (int i = 0; i < 150; i++) {
        Generated g = gen_document(rng);
        auto sc = make_fsm(g.schema);
        ASSERT_NE(sc, nullptr);
        for (size_t cut = 1; cut < g.doc.size(); cut++) {
            const std::string prefix = g.doc.substr(0, cut);
            bool any_legal = false;
            for (int ch = 0x20; ch < 0x7F && !any_legal; ch++) {
                if (sc->token_legal(prefix + static_cast<char>(ch)))
                    any_legal = true;
            }
            EXPECT_TRUE(any_legal) << "dead end after [" << prefix << "]\n  schema: " << g.schema
                                   << "\n  doc:    " << g.doc;
        }
    }
}

// ===========================================================================
// P6 — enum enforcement: a string value outside the enum must be refused at
// the point it becomes unambiguous.
// ===========================================================================
TEST(SchemaConstrainPropertyTest, RejectsValueOutsideEnum) {
    auto sc = make_fsm(
        R"({"type":"object","properties":{"c":{"type":"string","enum":["red","green"]}},"required":["c"]})");
    ASSERT_NE(sc, nullptr);
    EXPECT_TRUE(sc->token_legal(R"({"c":"red"})"));
    EXPECT_TRUE(sc->token_legal(R"({"c":"green"})"));
    EXPECT_FALSE(sc->token_legal(R"({"c":"blue"})"));
    EXPECT_FALSE(sc->token_legal(R"({"c":"redd"})"));  // valid prefix, invalid whole
}

// ===========================================================================
// #1104 — the schema FSM carried the same permissive number grammar as
// JsonConstrainer: '.', 'e', 'E', '+', '-' were accepted unconditionally in
// NUMBER_VALUE, so "3.5.5.5…" was legal and a degenerating model could not be
// forced to close the number. Live symptom: the reply ran to max_tokens and
// came back as truncated, unparseable JSON.
// ===========================================================================
TEST(SchemaConstrainPropertyTest, NumberGrammarMatchesRfc8259) {
    const std::string schema = R"({"type":"object","properties":{"v":{"type":"number"}},)"
                               R"("required":["v"],"additionalProperties":false})";
    struct Case {
        const char* num;
        bool valid;
    };
    const Case cases[] = {
        {"0", true},      {"-0", true},   {"3", true},       {"3.5", true},    {"1e5", true},
        {"1E5", true},    {"1e+5", true}, {"-2.5e-3", true}, {"3.5.5", false},  // second decimal point — the
                                                                                // observed shape
        {"1e5e5", false},                                // second exponent
        {"1e+-5", false},                                // double sign
        {"3.", false},    {"1e", false},  {"-", false},  // incomplete: still owe a digit
    };
    for (const auto& c : cases) {
        auto sc = make_fsm(schema);
        ASSERT_NE(sc, nullptr);
        const std::string doc = std::string("{\"v\":") + c.num + "}";
        EXPECT_EQ(sc->token_legal(doc), c.valid) << "number '" << c.num << "' judged "
                                                 << (c.valid ? "invalid" : "valid") << " — document: " << doc;
    }
}

// ===========================================================================
// Parser-level cases: what the schema parser does with input it cannot handle.
//
// Every one of these used to return a non-null tree. The FSM then enforced
// something the caller did not ask for, at HTTP 200, which is the outcome
// docs/API.md excludes ("a constraint imp cannot compile is a 400").
// ===========================================================================

// #1564: parse_bool() returns false WITHOUT consuming when the value is not
// true/false, so `additionalProperties: {schema}` left pos_ on '{' and every
// key after it was dropped. With `properties` gone the node is an empty object
// schema, which constraint_manager.cpp routes to the any-JSON constrainer:
// the reply is JSON with arbitrary keys and nothing says so.
TEST(SchemaParserDesync, AdditionalPropertiesAsObjectDoesNotTruncateTheSchema) {
    const std::string schema = R"({"type":"object","additionalProperties":{"type":"number"},)"
                               R"("properties":{"a":{"type":"string"}},"required":["a"]})";
    auto node = parse_json_schema(schema);
    ASSERT_NE(node, nullptr) << "the object form is legal JSON Schema and must parse";
    EXPECT_EQ(node->type, SchemaType::OBJECT);
    ASSERT_EQ(node->properties.size(), 1u)
        << "keys after additionalProperties were dropped, so the request silently "
           "downgrades to json_object";
    EXPECT_EQ(node->properties[0].first, "a");
    ASSERT_EQ(node->required.size(), 1u);
    EXPECT_EQ(node->required[0], "a");
}

TEST(SchemaParserDesync, AdditionalPropertiesFalseStillParses) {
    // Negative control: the boolean form is the common one and must be untouched.
    auto node = parse_json_schema(R"({"type":"object","properties":{"v":{"type":"string"}},"required":["v"],)"
                                  R"("additionalProperties":false})");
    ASSERT_NE(node, nullptr);
    EXPECT_FALSE(node->additional_properties);
    EXPECT_EQ(node->properties.size(), 1u);
}

// #1564: parse_string() has the same non-consuming default, so {"enum":[1,2,3]}
// produced enum_values == {""} and constrained the model to the empty string.
// The FSM emits an enum as quoted string content, so there is no representation
// for a numeric member: refusing is the only outcome that is not wrong.
TEST(SchemaParserDesync, NonStringEnumIsRefused) {
    EXPECT_EQ(parse_json_schema(R"({"type":"integer","enum":[1,2,3]})"), nullptr);
    EXPECT_EQ(parse_json_schema(R"({"enum":[true,false]})"), nullptr);
    EXPECT_EQ(parse_json_schema(R"({"enum":[null]})"), nullptr);
}

TEST(SchemaParserDesync, StringEnumStillParses) {
    auto node = parse_json_schema(R"({"type":"string","enum":["red","green"]})");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->type, SchemaType::ENUM);
    ASSERT_EQ(node->enum_values.size(), 2u);
    EXPECT_EQ(node->enum_values[0], "red");
}

TEST(SchemaParserDesync, TrailingInputIsAnError) {
    EXPECT_EQ(parse_json_schema(R"({"type":"string"} {"type":"integer"})"), nullptr);
}

TEST(SchemaParserDesync, UnclosedObjectIsAnError) {
    EXPECT_EQ(parse_json_schema(R"({"type":"object","properties":{"a":{"type":"string"})"), nullptr);
}

// #1540: an unconstrained `integer` had no digit bound. At the server's default
// temperature the sampler stayed in the digit state and emitted
// 1020000000000000000000000000000000000000 for a population field - a value no
// int64 consumer can read back. Measured on Qwen3.8-27B-NVFP4 at temperature
// 0.6; at temperature 0 the same request answered 13528079.
TEST(SchemaIntegerBound, DigitsStopAtInt64Width) {
    auto sc = make_fsm(R"({"type":"object","properties":{"pop":{"type":"integer"}},)"
                       R"("required":["pop"]})");
    ASSERT_NE(sc, nullptr);

    // 19 digits is int64's width and stays legal.
    EXPECT_TRUE(sc->token_legal(R"({"pop": 9223372036854775807})"));
    // 20 does not: the FSM masks the digit, so the model has to close the value.
    EXPECT_FALSE(sc->token_legal(R"({"pop": 92233720368547758070})"));
    // The reported output, at 40 digits.
    EXPECT_FALSE(sc->token_legal(R"({"pop": 1020000000000000000000000000000000000000})"));
}

TEST(SchemaIntegerBound, ShortIntegersAreUnaffected) {
    auto sc = make_fsm(R"({"type":"object","properties":{"pop":{"type":"integer"}},)"
                       R"("required":["pop"]})");
    ASSERT_NE(sc, nullptr);
    for (const char* doc :
         {R"({"pop": 0})", R"({"pop": 42})", R"({"pop": -13528079})", R"({"pop": 3691000})"}) {
        EXPECT_TRUE(sc->token_legal(doc)) << doc;
    }
}

// The bound is on `integer`. A `number` keeps its JSON-legal mantissa, because
// there the digits carry precision rather than magnitude.
TEST(SchemaIntegerBound, NumberIsNotBounded) {
    auto sc = make_fsm(R"({"type":"object","properties":{"x":{"type":"number"}},)"
                       R"("required":["x"]})");
    ASSERT_NE(sc, nullptr);
    EXPECT_TRUE(sc->token_legal(R"({"x": 1.02000000000000000000000000000000000000001})"));
}

// #1567: thirteen standard assertion keywords were accepted and dropped by
// skip_value(). A caller that bounds its output was answered as if it had not.
TEST(SchemaUnenforceableKeywords, AssertionKeywordsAreRefused) {
    const char* refused[] = {
        R"({"type":"integer","minimum":1,"maximum":5})",
        R"({"type":"integer","exclusiveMinimum":0})",
        R"({"type":"number","multipleOf":2})",
        R"({"allOf":[{"type":"string"}]})",
        R"({"not":{"type":"string"}})",
        R"({"type":"array","uniqueItems":true})",
        R"({"type":"object","patternProperties":{"^a":{"type":"string"}}})",
        R"({"type":"object","propertyNames":{"pattern":"^a"}})",
        R"({"type":"array","prefixItems":[{"type":"string"}]})",
        R"({"type":"object","minProperties":1})",
    };
    for (const char* s : refused) {
        EXPECT_EQ(parse_json_schema(s), nullptr) << "must be refused, not silently dropped: " << s;
    }
}

TEST(SchemaUnenforceableKeywords, AnnotationsAreStillIgnored) {
    // format is an annotation in Draft 2020-12 unless the format-assertion
    // vocabulary is in use, and these four change no legal value. Refusing them
    // would break working clients for nothing.
    auto node = parse_json_schema(R"({"$schema":"https://json-schema.org/draft/2020-12/schema",)"
                                  R"("title":"T","description":"D","examples":["x"],"default":"x",)"
                                  R"("type":"string","format":"date-time"})");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->type, SchemaType::STRING);
}

TEST(SchemaUnenforceableKeywords, ConstIsEnforcedAsASingleValueEnum) {
    auto node = parse_json_schema(R"({"const":"fixed"})");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->type, SchemaType::ENUM);
    ASSERT_EQ(node->enum_values.size(), 1u);
    EXPECT_EQ(node->enum_values[0], "fixed");
    // Same string-only limit as enum, and for the same reason.
    EXPECT_EQ(parse_json_schema(R"({"const":42})"), nullptr);
}

// #1609: recursive descent over a request body. 10^5 nested "items" objects
// overflow the worker thread's stack; the cap turns that into a parse error,
// which the admission path already renders as a 400.
TEST(SchemaDepthCap, DeeplyNestedSchemaIsRefusedInsteadOfOverflowing) {
    std::string deep;
    const int kLevels = 5000;
    for (int i = 0; i < kLevels; i++)
        deep += R"({"type":"array","items":)";
    deep += R"({"type":"string"})";
    for (int i = 0; i < kLevels; i++)
        deep += "}";
    EXPECT_EQ(parse_json_schema(deep), nullptr);
}

TEST(SchemaDepthCap, ModeratelyNestedSchemaStillParses) {
    // Negative control: 8 levels is deeper than anything a real generator
    // emits and must keep working.
    std::string s;
    const int kLevels = 8;
    for (int i = 0; i < kLevels; i++)
        s += R"({"type":"array","items":)";
    s += R"({"type":"string"})";
    for (int i = 0; i < kLevels; i++)
        s += "}";
    auto node = parse_json_schema(s);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->type, SchemaType::ARRAY);
}

}  // namespace
}  // namespace imp
