#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace imp {

// Regex -> NFA (host-side) for JSON-Schema "pattern" token-level constraining. NOT a full
// regex engine. Supported: literals; . (any char but newline); \d \D \w \W \s \S and
// negations; backslash-escaped metachars; [...] / [^...] classes with a-z ranges and \d/\w/\s
// shorthands; * + ? (greedy, but matching is set-based so greediness is irrelevant); {n} {n,}
// {n,m}; ( ); a|b alternation; ^ $ anchors (accepted as no-ops since JSON-Schema pattern is
// unanchored by spec; we always anchor the whole string; mid-pattern ^/$ unsupported).
// NOT supported: backreferences, lookaround, named groups, non-greedy ?, unicode property
// escapes, \b. compile() returns false for patterns it can't represent; callers then skip
// pattern enforcement.

// A single NFA transition: an edge that consumes one input char if `is_epsilon`
// is false (matching when char_class[ch]==true), or a free epsilon move.
struct NfaEdge {
    int to = -1;
    bool is_epsilon = true;
    // 256-entry membership table for the byte that this edge accepts.
    std::vector<uint8_t> char_class;  // size 256 when !is_epsilon
};

struct NfaState {
    std::vector<NfaEdge> edges;
    bool accepting = false;
};

class RegexNfa {
public:
    // Compile `pattern` into a Thompson NFA. Returns false on unsupported
    // syntax (caller should treat as "no pattern constraint").
    bool compile(const std::string& pattern);

    bool compiled() const { return compiled_; }

    // Epsilon-closure of the start state.
    std::vector<int> start_set() const;

    // Step the active state set with one input byte; returns the new set
    // (may be empty -> the prefix is now dead).
    std::vector<int> step(const std::vector<int>& states, unsigned char c) const;

    // True if any state in `states` is accepting.
    bool accepts(const std::vector<int>& states) const;

    // Bounds on what one pattern may cost: request-supplied text on an unauthenticated endpoint,
    // both limits below were unbounded (#1608, #1609). kMaxRepeat mirrors the GBNF parser's own
    // bound (gbnf_parser.cpp:18): {n,m} clones the atom n times, so n is an allocation count.
    // kMaxStates bounds the nested form, which multiplies rather than adds:
    // (((a{100}){100}){100}){100} is 28 bytes and asks for ~10^8 states, each with a 256-entry
    // char class.
    static constexpr long kMaxRepeat = 1024;
    static constexpr size_t kMaxStates = 100000;
    static constexpr int kMaxDepth = 64;

private:
    std::vector<NfaState> states_;
    int start_ = -1;
    int accept_ = -1;
    bool compiled_ = false;

    // Parser scratch
    const std::string* src_ = nullptr;
    size_t pos_ = 0;
    bool error_ = false;
    int depth_ = 0;

    int new_state();
    void add_epsilon(int from, int to);
    void add_edge(int from, int to, const std::vector<uint8_t>& cls);
    void epsilon_closure(std::vector<int>& set) const;

    // Recursive-descent Thompson construction. Each returns a (start, accept)
    // fragment pair via out params; returns false on error.
    struct Frag {
        int start;
        int accept;
    };
    bool parse_alt(Frag& out);
    bool parse_concat(Frag& out);
    bool parse_repeat(Frag& out);
    bool parse_atom(Frag& out);
    bool parse_class(Frag& out);  // [...]
    static bool make_shorthand(char esc, std::vector<uint8_t>& cls);  // \d \w \s ...
};

enum class SchemaType {
    STRING,
    NUMBER,
    INTEGER,
    BOOLEAN,
    NULL_TYPE,
    OBJECT,
    ARRAY,
    ENUM,
    ANY_OF,
    REF,  // $ref to a $defs/definitions entry (or "#" = schema root)
    // Tool-call body {"name":"<tool>","arguments":{...}} (#1002): an OBJECT-like node with
    // ORDERED keys (name before arguments) whose "arguments" schema resolves dynamically to the
    // chosen tool's parameter schema. properties = [(name, ENUM over tool names), (arguments, free
    // placeholder)]; defs = [(tool name, parameter schema)]. Built via build_tool_call_schema(),
    // never parsed from JSON.
    TOOL_CALL,
    // Qwen-Coder XML tool-call body: <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...
    // </function>. Name lives in the tag (enum_values=tool names), parameter keys come from the
    // chosen tool's properties (defs=[(tool name, parameter schema)], same layout as TOOL_CALL);
    // values are raw text delimited by "\n</parameter>", never JSON. Built via
    // build_xml_tool_call_schema(), never parsed from JSON.
    XML_TOOL_CALL,
};

struct SchemaNode {
    SchemaType type = SchemaType::STRING;

    // OBJECT
    std::vector<std::pair<std::string, std::unique_ptr<SchemaNode>>> properties;
    std::vector<std::string> required;
    bool additional_properties = false;

    // ARRAY
    std::unique_ptr<SchemaNode> items;
    // ARRAY constraints (JSON Schema "minItems" / "maxItems"). -1 = unset.
    int min_items = -1;
    int max_items = -1;

    // ENUM
    std::vector<std::string> enum_values;

    // ANY_OF
    std::vector<std::unique_ptr<SchemaNode>> any_of;

    // REF: name of the referenced definition ("#" = schema root). Resolved against the ROOT node's
    // `defs` table at constrain time, so recursive/mutually-recursive schemas are representable
    // without ownership cycles (frames hold non-owning resolved pointers).
    std::string ref_name;

    // Definitions table ($defs / definitions). Collected from anywhere in the
    // schema document and attached to the ROOT node by parse_json_schema().
    std::vector<std::pair<std::string, std::unique_ptr<SchemaNode>>> defs;

    // STRING constraints (JSON Schema "pattern" / "minLength" / "maxLength").
    std::string pattern;       // raw regex source; empty = no pattern
    int min_length = -1;       // -1 = unset
    int max_length = -1;       // -1 = unset
    // Compiled NFA for `pattern`, lazily built at schema-load time. Shared via
    // shared_ptr so clone() doesn't recompile. Null if no/unsupported pattern.
    std::shared_ptr<RegexNfa> pattern_nfa;

    // Deep copy
    std::unique_ptr<SchemaNode> clone() const;
};

// Parses a JSON Schema string into a SchemaNode tree. Supports $defs/definitions +
// "$ref": "#/$defs/<name>" | "#/definitions/<name>" | "#", including recursive and
// mutually-recursive definitions (pydantic/zod emit $defs+$ref for nested models). Returns
// nullptr on parse failure or an unresolved $ref (callers then decline constrained decoding).
std::unique_ptr<SchemaNode> parse_json_schema(const std::string& json);

// Resolves a (possibly REF) node against the root's defs table. Returns the node itself when
// not a REF; nullptr on a missing target or a pure ref->ref cycle (parse_json_schema validates
// both, so runtime hits are defensive only).
const SchemaNode* resolve_schema_ref(const SchemaNode* root, const SchemaNode* node);

// Builds a TOOL_CALL root node (#1002) from (tool name, parameter-schema JSON) pairs. Every
// parameter schema must parse AND carry enforceable structure (an object with >=1 property, or
// an enum); a free-form/opaque schema dead-ends the key phase, so the builder returns nullptr
// and the caller falls back to prompt-hint-only tool choice.
std::unique_ptr<SchemaNode> build_tool_call_schema(
    const std::vector<std::pair<std::string, std::string>>& tools);

// Same contract for the Qwen-Coder XML dialect: an XML_TOOL_CALL root whose enum_values are
// the tool names and whose defs carry each tool's parameter schema (keys+required drive the
// FSM; values are raw text). Same enforceability gates as build_tool_call_schema.
std::unique_ptr<SchemaNode> build_xml_tool_call_schema(
    const std::vector<std::pair<std::string, std::string>>& tools);

}  // namespace imp
