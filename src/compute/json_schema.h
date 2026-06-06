#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// Regex -> NFA (host-side) for JSON-Schema "pattern" token-level constraining.
//
// Supported subset (documented; this is NOT a full regex engine):
//   - literal characters
//   - .            any char except newline
//   - \d \D \w \W \s \S   digit / word / whitespace classes (and negations)
//   - \\ \. \* ... backslash-escaped metacharacters (literal)
//   - [...] [^...] character classes with a-z style ranges and the \d/\w/\s
//                  shorthands inside the class
//   - * + ?        greedy quantifiers (matching is set-based, so greediness
//                  is irrelevant for prefix-liveness)
//   - {n} {n,} {n,m}  bounded repetition
//   - ( ... )      grouping
//   - a|b          alternation
//   - ^ $          anchors. Because JSON-Schema "pattern" is an *unanchored*
//                  search by spec, a leading ^ / trailing $ are accepted but
//                  treated as no-ops here; we always anchor the whole string
//                  (token-level masking can only enforce a fully-matching
//                  string). Mid-pattern ^/$ are not supported.
//
// NOT supported: backreferences, lookaround, named groups, non-greedy ?,
// unicode property escapes, \b word boundaries. compile() returns false for
// patterns it cannot represent; callers should then skip pattern enforcement.
// ---------------------------------------------------------------------------

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

    bool empty_set_dead(const std::vector<int>& states) const { return states.empty(); }

private:
    std::vector<NfaState> states_;
    int start_ = -1;
    int accept_ = -1;
    bool compiled_ = false;

    // Parser scratch
    const std::string* src_ = nullptr;
    size_t pos_ = 0;
    bool error_ = false;

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
    bool make_shorthand(char esc, std::vector<uint8_t>& cls);  // \d \w \s ...
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
};

struct SchemaNode {
    SchemaType type = SchemaType::STRING;

    // OBJECT
    std::vector<std::pair<std::string, std::unique_ptr<SchemaNode>>> properties;
    std::vector<std::string> required;
    bool additional_properties = false;

    // ARRAY
    std::unique_ptr<SchemaNode> items;

    // ENUM
    std::vector<std::string> enum_values;

    // ANY_OF
    std::vector<std::unique_ptr<SchemaNode>> any_of;

    // REF: name of the referenced definition ("#" = schema root). Resolution
    // happens against the ROOT node's `defs` table at constrain time, so
    // recursive and mutually-recursive schemas are representable without
    // ownership cycles (frames hold non-owning resolved pointers).
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

// Parse a JSON Schema string into a SchemaNode tree. Supports $defs /
// definitions + "$ref": "#/$defs/<name>" | "#/definitions/<name>" | "#",
// including recursive and mutually-recursive definitions (agent frameworks —
// pydantic, zod — emit $defs+$ref for every nested model).
// Returns nullptr on parse failure or when a $ref cannot be resolved (callers
// then decline constrained decoding rather than enforcing a wrong grammar).
std::unique_ptr<SchemaNode> parse_json_schema(const std::string& json);

// Resolve a (possibly REF) node against the root's defs table. Returns the
// node itself when it is not a REF; nullptr on a missing target or a pure
// ref->ref cycle (parse_json_schema validates both, so runtime hits are
// defensive only).
const SchemaNode* resolve_schema_ref(const SchemaNode* root, const SchemaNode* node);

// ---------------------------------------------------------------------------
// GBNF-style grammar loader (Part B — PARTIAL / non-recursive subset).
//
// Compiles a llama.cpp-flavoured GBNF grammar into a single byte-level NFA
// (the same RegexNfa engine used for "pattern"), so the existing token-mask
// FSM machinery applies unchanged. Because an NFA cannot represent unbounded
// recursion, the supported subset is the *non-recursive* fragment:
//
//   Supported:
//     root ::= <expr>                rule definition (root rule must be "root")
//     "literal"                      double-quoted literal strings
//     [a-z0-9_] / [^...]             character classes (RegexNfa [...] subset)
//     seq                            space-separated concatenation
//     a | b                          alternation
//     x* x+ x?                       repetition
//     ( ... )                        grouping
//     rulename                       reference to another (non-recursive) rule
//     # comment                      line comments
//
//   NOT supported (rejected with a clear error; see TODO in .cpp):
//     - recursive / mutually-recursive rules (e.g. JSON value ::= object ...)
//     - {n,m} repetition counts
//     - char-class escapes beyond what RegexNfa's [...] handles
//
// compile_gbnf_grammar() returns a compiled RegexNfa on success, or nullptr
// (with a logged error naming the unsupported construct) on failure. Callers
// should fall back to unconstrained generation when it returns nullptr.
// ---------------------------------------------------------------------------
std::shared_ptr<RegexNfa> compile_gbnf_grammar(const std::string& gbnf);

}  // namespace imp
