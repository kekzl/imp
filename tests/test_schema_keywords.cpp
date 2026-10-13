// JSON-Schema keywords Pydantic / Zod emit that the schema FSM enforces: anyOf / oneOf / type
// arrays / nullable (Optional[T]), untyped schemas (Any), arrays without items, additionalProperties
// as a schema (Dict[str, T]), min/maxProperties, prefixItems (tuple), allOf, minimum / maximum /
// exclusive bounds. CPU-only: init_grammar_for_test() skips tokenizer classification.
#include <gtest/gtest.h>

#include "compute/json_schema.h"
#include "compute/schema_constrain.h"

#include <memory>
#include <random>
#include <string>

namespace imp {
namespace {

std::unique_ptr<SchemaConstrainer> fsm(const std::string& schema_json) {
    auto schema = parse_json_schema(schema_json);
    if (!schema)
        return nullptr;
    auto sc = std::make_unique<SchemaConstrainer>();
    if (!sc->init_grammar_for_test(std::move(schema)))
        return nullptr;
    return sc;
}

// {"x": <schema>} with x required.
std::string field(const std::string& schema) {
    return R"({"type":"object","properties":{"x":)" + schema + R"(},"required":["x"]})";
}

void expect_complete(const std::string& schema, const std::string& doc) {
    auto sc = fsm(schema);
    ASSERT_NE(sc, nullptr) << schema;
    EXPECT_TRUE(sc->completes_for_test(doc)) << "schema " << schema << "\n  doc " << doc;
}

void expect_illegal(const std::string& schema, const std::string& doc) {
    auto sc = fsm(schema);
    ASSERT_NE(sc, nullptr) << schema;
    EXPECT_FALSE(sc->token_legal(doc)) << "schema " << schema << "\n  doc " << doc;
}

void expect_incomplete(const std::string& schema, const std::string& doc) {
    auto sc = fsm(schema);
    ASSERT_NE(sc, nullptr) << schema;
    EXPECT_FALSE(sc->completes_for_test(doc)) << "schema " << schema << "\n  doc " << doc;
}

// Pydantic Optional[int]: was a free string, so the answer was "5" instead of 5 or null.
TEST(SchemaKeywords, AnyOfOptionalTakesTheValueNotAString) {
    const std::string s = field(R"({"anyOf":[{"type":"integer"},{"type":"null"}]})");
    expect_complete(s, R"({"x":5})");
    expect_complete(s, R"({"x":null})");
    expect_complete(s, R"({"x": -12})");
    expect_illegal(s, R"({"x":")");
    expect_illegal(s, R"({"x":5.5})");
}

TEST(SchemaKeywords, AnyOfAcrossTypesAndNumberKinds) {
    const std::string s = field(R"({"oneOf":[{"type":"string"},{"type":"integer"},{"type":"number"}]})");
    expect_complete(s, R"({"x":"a"})");
    expect_complete(s, R"({"x":7})");
    expect_complete(s, R"({"x":7.25})");
    expect_illegal(s, R"({"x":t)");
}

TEST(SchemaKeywords, AnyOfObjectsSplitOnADiscriminator) {
    const std::string s = field(
        R"({"anyOf":[)"
        R"({"type":"object","properties":{"kind":{"const":"cat"},"lives":{"type":"integer"}},"required":["kind","lives"]},)"
        R"({"type":"object","properties":{"kind":{"const":"dog"},"name":{"type":"string"}},"required":["kind","name"]}]})");
    expect_complete(s, R"({"x":{"kind":"cat","lives":9}})");
    expect_complete(s, R"({"x":{"kind":"dog","name":"Rex"}})");
    expect_illegal(s, R"({"x":{"kind":"cat","name")");
    expect_incomplete(s, R"({"x":{"kind":"dog"}})");
}

TEST(SchemaKeywords, AnyOfInsideArraysAndNested) {
    expect_complete(R"({"type":"array","items":{"anyOf":[{"type":"string"},{"type":"null"}]}})",
                    R"(["a",null,"b"])");
    expect_complete(field(R"({"anyOf":[{"anyOf":[{"type":"boolean"},{"type":"null"}]},{"type":"string"}]})"),
                    R"({"x":false})");
    // An Optional object behind a $ref, the Pydantic shape.
    expect_complete(
        R"({"$defs":{"P":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}},)"
        R"("type":"object","properties":{"p":{"anyOf":[{"$ref":"#/$defs/P"},{"type":"null"}]}},)"
        R"("required":["p"]})",
        R"({"p":{"n":1}})");
}

TEST(SchemaKeywords, TypeArraysAndNullable) {
    const std::string s = field(R"({"type":["string","null"],"maxLength":3})");
    expect_complete(s, R"({"x":"abc"})");
    expect_complete(s, R"({"x":null})");
    expect_illegal(s, R"({"x":"abcd")");
    expect_illegal(s, R"({"x":1)");
    const std::string n = field(R"({"type":"integer","nullable":true})");
    expect_complete(n, R"({"x":3})");
    expect_complete(n, R"({"x":null})");
}

// Pydantic Any / a bare {}: was forced to a string.
TEST(SchemaKeywords, UntypedSchemaIsAnyValue) {
    const std::string s = field(R"({"description":"anything"})");
    expect_complete(s, R"({"x":{"a":[1,"b",null]}})");
    expect_complete(s, R"({"x":42})");
    expect_complete(s, R"({"x":"s"})");
    expect_complete(s, R"({"x":[]})");
}

// An array without items took one char per item: [12] was illegal.
TEST(SchemaKeywords, ArrayWithoutItemsTakesAnyValues) {
    expect_complete(R"({"type":"array"})", R"([12,"ab",{"k":[1]},true])");
    expect_complete(R"({"type":"array"})", R"([])");
}

TEST(SchemaKeywords, AdditionalPropertiesSchemaAndPropertyCounts) {
    const std::string dict = R"({"type":"object","additionalProperties":{"type":"integer"},)"
                             R"("minProperties":1,"maxProperties":2})";
    expect_complete(dict, R"({"a":1})");
    expect_complete(dict, R"({"a":1,"b":22})");
    expect_illegal(dict, R"({"a":"x")");
    expect_illegal(dict, R"({})");
    expect_illegal(dict, R"({"a":1,"b":2,)");
    expect_illegal(dict, R"({"a":1,"a")");  // duplicate key
    const std::string mixed = R"({"type":"object","properties":{"id":{"type":"string"}},"required":["id"],)"
                              R"("additionalProperties":{"type":"boolean"}})";
    expect_complete(mixed, R"({"id":"q","flag":true})");
    expect_illegal(mixed, R"({"id":"q","flag":1)");
}

TEST(SchemaKeywords, PrefixItemsTuples) {
    const std::string pair = R"({"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}],)"
                             R"("minItems":2,"maxItems":2})";
    expect_complete(pair, R"([1,"a"])");
    expect_illegal(pair, R"(["a")");
    expect_illegal(pair, R"([1,"a",)");
    expect_incomplete(pair, R"([1])");
    const std::string closed = R"({"type":"array","prefixItems":[{"type":"integer"}],"items":false})";
    expect_complete(closed, R"([1])");
    expect_illegal(closed, R"([1,)");
    const std::string tail =
        R"({"type":"array","prefixItems":[{"type":"string"}],"items":{"type":"integer"}})";
    expect_complete(tail, R"(["h",1,2,3])");
    expect_illegal(tail, R"(["h","i")");
}

TEST(SchemaKeywords, AllOfMergesObjectsAndUnwrapsOneRef) {
    expect_complete(
        R"({"$defs":{"P":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}},)"
        R"("type":"object","properties":{"p":{"allOf":[{"$ref":"#/$defs/P"}],"description":"pet"}},)"
        R"("required":["p"]})",
        R"({"p":{"n":1}})");
    const std::string merged =
        R"({"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},)"
        R"({"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}]})";
    expect_complete(merged, R"({"a":1,"b":"x"})");
    expect_incomplete(merged, R"({"a":1})");
    expect_illegal(merged, R"({"a":1,"c")");
    EXPECT_EQ(parse_json_schema(R"({"allOf":[{"type":"object","properties":{"a":{"type":"integer"}}},)"
                                R"({"type":"object","properties":{"a":{"type":"string"}}}]})"),
              nullptr);
}

TEST(SchemaKeywords, IntegerBounds) {
    const std::string s = field(R"({"type":"integer","minimum":1,"maximum":5})");
    expect_complete(s, R"({"x":3})");
    expect_complete(s, R"({"x":5})");
    expect_illegal(s, R"({"x":0)");
    expect_illegal(s, R"({"x":6)");
    expect_illegal(s, R"({"x":12)");
    expect_illegal(s, R"({"x":-)");
    const std::string neg = field(R"({"type":"integer","minimum":-10,"maximum":-2})");
    expect_complete(neg, R"({"x":-10})");
    expect_illegal(neg, R"({"x":-1})");  // "-1" can still become -10, but cannot end
    const std::string draft4 = field(R"({"type":"integer","minimum":0,"exclusiveMinimum":true})");
    expect_illegal(draft4, R"({"x":0})");
    expect_complete(draft4, R"({"x":1})");
}

TEST(SchemaKeywords, NumberBounds) {
    const std::string s = field(R"({"type":"number","exclusiveMinimum":0,"maximum":1})");
    expect_complete(s, R"({"x":0.5})");
    expect_complete(s, R"({"x":1})");
    expect_complete(s, R"({"x":1.0})");
    expect_illegal(s, R"({"x":0})");
    expect_illegal(s, R"({"x":1.5)");
    expect_illegal(s, R"({"x":2)");
    expect_illegal(s, R"({"x":1e)");  // bounded numbers stay plain decimals
    // A bound implies a number when no type is given.
    expect_complete(field(R"({"minimum":10})"), R"({"x":12.5})");
}

// --- Property battery over the new forms: conforming documents complete, every prefix stays
// legal, and no prefix is a dead end.

struct Gen {
    std::string schema;
    std::string doc;
};

Gen gen(std::mt19937& rng, int depth) {
    std::uniform_int_distribution<int> pick(0, depth < 2 ? 11 : 6);
    auto coin = [&] { return std::uniform_int_distribution<int>(0, 1)(rng) == 1; };
    switch (pick(rng)) {
        case 0:
            return {R"({"type":"string"})", "\"abc\""};
        case 1: {
            const int lo = std::uniform_int_distribution<int>(-50, 50)(rng);
            const int v = lo + std::uniform_int_distribution<int>(0, 30)(rng);
            return {R"({"type":"integer","minimum":)" + std::to_string(lo) + R"(,"maximum":)" +
                        std::to_string(lo + 30) + "}",
                    std::to_string(v)};
        }
        case 2:
            return {R"({"type":"number","exclusiveMinimum":0,"maximum":10})", "2.75"};
        case 3:
            return {R"({"type":["integer","null"]})", coin() ? "7" : "null"};
        case 4:
            return {R"({"title":"Any"})", coin() ? R"({"k":[1,"x",null]})" : "3"};
        case 5:
            return {R"({"type":"boolean","nullable":true})", coin() ? "true" : "null"};
        case 6:
            return {R"({"type":"string","enum":["red","green"]})", "\"green\""};
        case 7: {  // Optional[child]
            Gen child = gen(rng, depth + 1);
            return {R"({"anyOf":[)" + child.schema + R"(,{"type":"null"}]})", coin() ? child.doc : "null"};
        }
        case 8: {  // object, all required
            std::string props, req, doc = "{";
            const int n = std::uniform_int_distribution<int>(1, 3)(rng);
            for (int i = 0; i < n; ++i) {
                Gen child = gen(rng, depth + 1);
                const std::string key = "p" + std::to_string(i);
                props += (i ? "," : "") + ("\"" + key + "\":" + child.schema);
                req += (i ? "," : "") + ("\"" + key + "\"");
                doc += (i ? "," : "") + ("\"" + key + "\":" + child.doc);
            }
            return {R"({"type":"object","properties":{)" + props + R"(},"required":[)" + req + "]}",
                    doc + "}"};
        }
        case 9: {  // Dict[str, child] with minProperties
            Gen child = gen(rng, depth + 1);
            const int n = std::uniform_int_distribution<int>(1, 3)(rng);
            std::string doc = "{";
            for (int i = 0; i < n; ++i)
                doc += (i ? "," : "") + ("\"k" + std::to_string(i) + "\":" + child.doc);
            return {R"({"type":"object","additionalProperties":)" + child.schema + R"(,"minProperties":)" +
                        std::to_string(n) + "}",
                    doc + "}"};
        }
        case 10: {  // tuple
            Gen a = gen(rng, depth + 1), b = gen(rng, depth + 1);
            return {R"({"type":"array","prefixItems":[)" + a.schema + "," + b.schema +
                        R"(],"minItems":2,"maxItems":2})",
                    "[" + a.doc + "," + b.doc + "]"};
        }
        default: {  // list
            Gen item = gen(rng, depth + 1);
            const int n = std::uniform_int_distribution<int>(1, 3)(rng);
            std::string doc = "[";
            for (int i = 0; i < n; ++i)
                doc += (i ? "," : "") + item.doc;
            return {R"({"type":"array","items":)" + item.schema + R"(,"minItems":)" + std::to_string(n) + "}",
                    doc + "]"};
        }
    }
}

// Object root: a number at the root never completes (no terminator).
Gen gen_root(std::mt19937& rng) {
    Gen child = gen(rng, 0);
    return {field(child.schema), R"({"x":)" + child.doc + "}"};
}

TEST(SchemaKeywordsProperty, ConformingDocumentsComplete) {
    std::mt19937 rng(0x5C4E3A99);
    for (int i = 0; i < 400; ++i) {
        const Gen g = gen_root(rng);
        expect_complete(g.schema, g.doc);
    }
}

TEST(SchemaKeywordsProperty, EveryPrefixIsLegalAndNoneIsADeadEnd) {
    std::mt19937 rng(0x5C4E3A9A);
    for (int i = 0; i < 120; ++i) {
        const Gen g = gen_root(rng);
        auto sc = fsm(g.schema);
        ASSERT_NE(sc, nullptr) << g.schema;
        for (size_t cut = 1; cut < g.doc.size(); ++cut) {
            const std::string prefix = g.doc.substr(0, cut);
            EXPECT_TRUE(sc->token_legal(prefix)) << "prefix " << prefix << "\n  schema " << g.schema;
            bool any = false;
            for (int ch = 0x20; ch < 0x7F && !any; ++ch)
                any = sc->token_legal(prefix + static_cast<char>(ch));
            EXPECT_TRUE(any) << "dead end after " << prefix << "\n  schema " << g.schema;
        }
    }
}

}  // namespace
}  // namespace imp
