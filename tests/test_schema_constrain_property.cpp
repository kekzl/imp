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

}  // namespace
}  // namespace imp
