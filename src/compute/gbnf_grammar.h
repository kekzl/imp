#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// GBNF (llama.cpp grammar dialect) -> rule table + nondeterministic pushdown simulator for
// token-level constrained decoding (docs/roadmap.md gap 8). A context-free grammar needs a
// STACK, which RegexNfa (#1091) structurally cannot have.
// Supported syntax: root ::= alternatives (entry rule); name ::= a | b; "literal"; [a-z0-9_]
// [^\n] classes/ranges/negation; . (any char); rule-ref; ( ... ) grouping; x* x+ x?;
// x{m} x{m,} x{m,n}; # comment to EOL. Escapes: \n \r \t \" \' \\ \[ \] \xNN \uNNNN \UNNNNNNNN.
// NOT supported, refused at compile time: left recursion, undefined rule refs, missing `root`.
// compile() returns false with a message; caller declines constrained decoding rather than
// enforcing an ungrammar.
// Works on UNICODE CODEPOINTS; GbnfMatcher does UTF-8 assembly and carries a partial codepoint
// across tokens, so callers feed it raw token text.

// Inclusive codepoint ranges; `negated` flips membership ([^...]).
struct GbnfCharSet {
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    bool negated = false;

    bool matches(uint32_t cp) const;
    // True if ANY codepoint in [lo,hi] is a member. Used for partial-UTF-8
    // liveness: a token may end mid-codepoint, and the question is then whether
    // any completion of it could be legal.
    bool intersects(uint32_t lo, uint32_t hi) const;
};

// One element of an alternative: either consumes a character, or descends into
// another rule. Repetition and grouping are desugared into synthetic rules at
// parse time, so the runtime only ever sees these two kinds.
struct GbnfItem {
    bool is_rule = false;
    int32_t rule = -1;  // is_rule
    GbnfCharSet chars;  // !is_rule
};

struct GbnfAlt {
    std::vector<GbnfItem> items;
};

struct GbnfRule {
    std::string name;  // empty for synthetic rules
    std::vector<GbnfAlt> alts;
};

// A position inside one alternative: where input is consumed. Chained through
// `parent` these form a parse continuation — the stack of a pushdown automaton.
struct GbnfPos {
    int32_t rule = 0;
    int32_t alt = 0;
    int32_t idx = 0;
    bool operator==(const GbnfPos& o) const { return rule == o.rule && alt == o.alt && idx == o.idx; }
};

// Live state: every parse continuation still alive, as INTERNED STACK IDS (arena indices),
// sorted/deduplicated; -1 = empty stack (derivation complete, stopping here is legal).
// Ids not vectors of frames: computing one token mask simulates the whole vocabulary, so
// hash-consing stacks makes a step an integer op and the state set a cheap flat vector to
// copy/compare/cache.
using GbnfStackSet = std::vector<int32_t>;

// Parses GBNF source into a rule table, reporting root's index. Returns false with a one-line
// reason on syntax error, undefined rule, or missing root. Implemented in gbnf_parser.cpp,
// kept a separate TU since it shares nothing with the simulator but these structs.
bool parse_gbnf(const std::string& src, std::vector<GbnfRule>& rules, int32_t& root, std::string* err);

class GbnfGrammar {
public:
    // Compile GBNF source. On failure returns false and, when `err` is given,
    // fills it with a one-line human-readable reason.
    bool compile(const std::string& src, std::string* err = nullptr);
    bool compiled() const { return compiled_; }

    // Expanded stack set for the start of `root`.
    GbnfStackSet start_set() const;

    // Advance every stack by one codepoint; the result may be empty (dead).
    GbnfStackSet step(const GbnfStackSet& stacks, uint32_t cp) const;
    // Same, into a caller-owned buffer (`out` must not alias `stacks`) — the
    // vocabulary walk runs this once per byte per token, so the allocation the
    // by-value form costs is worth avoiding.
    void step_into(const GbnfStackSet& stacks, uint32_t cp, GbnfStackSet& out) const;

    // The derivation is complete iff some stack is empty.
    static bool accepts(const GbnfStackSet& stacks);

    // Can the next codepoint be anything in [lo,hi]? (partial-UTF-8 liveness)
    bool can_consume_range(const GbnfStackSet& stacks, uint32_t lo, uint32_t hi) const;

    // Bytes that may START the next codepoint. This is the mask prefilter: a
    // vocabulary walk that had to simulate every token would cost far more than
    // decoding does, and the first byte kills the overwhelming majority.
    void lead_bytes(const GbnfStackSet& stacks, uint8_t out[256]) const;

    const std::vector<GbnfRule>& rules() const { return rules_; }

    // Guard rails. A grammar needing more than this is pathological; exceeding
    // them drops continuations, which can only make the constraint STRICTER
    // (the caller's empty-mask guard then lets generation stop cleanly).
    static constexpr size_t kMaxStackDepth = 128;
    static constexpr size_t kMaxStacks = 4096;

private:
    // One interned stack frame: a position plus the continuation below it.
    struct StackNode {
        GbnfPos pos;
        int32_t parent;  // -1 = bottom of the stack
        int32_t depth;
    };
    struct StackKey {
        GbnfPos pos;
        int32_t parent;
        bool operator==(const StackKey& o) const { return pos == o.pos && parent == o.parent; }
    };
    struct StackKeyHash {
        size_t operator()(const StackKey& k) const;
    };

    int32_t intern(GbnfPos pos, int32_t parent) const;
    // Where a stack goes once its pending character is consumed - independent of which character it
    // was (the charset only decides whether the transition happens), so it is computed once per
    // stack and reused for every codepoint and token afterwards.
    const GbnfStackSet& successors(int32_t stack) const;
    // Starts a new visited-marking generation for expand(); O(1), no clearing.
    void begin_visit() const;
    // Push a stack through rule descents / completed alternatives until its top
    // waits for input (or it is empty), appending the results to `out`.
    void expand(int32_t stack, GbnfStackSet& out) const;
    const GbnfItem* top_item(int32_t stack) const;

    // Fixed point over rules that can derive the empty string.
    void compute_nullable();
    // Refuses `a ::= a "x"` and mutual variants; see the header comment.
    bool check_left_recursion(std::string* err) const;

    std::vector<GbnfRule> rules_;
    std::vector<uint8_t> nullable_;
    int32_t root_ = -1;
    bool compiled_ = false;

    // Interning arena. Mutable because it is a memo: two identical stacks are
    // the same id no matter which request reached it first, so sharing it makes
    // every later step cheaper without changing any observable behaviour.
    mutable std::vector<StackNode> arena_;
    mutable std::unordered_map<StackKey, int32_t, StackKeyHash> intern_;
    mutable std::vector<uint32_t> visited_;
    mutable uint32_t visit_epoch_ = 0;
    mutable std::vector<int32_t> work_;
    mutable std::vector<GbnfStackSet> next_cache_;
    mutable std::vector<uint8_t> next_ready_;
};

// A codepoint only half-decoded (last token ended mid-character). `min` is the smallest value
// the sequence may legally encode (0x80/0x800/0x10000) - without it an overlong encoding would
// smuggle a forbidden character past the mask as a shorter, allowed codepoint.
struct GbnfPartial {
    uint32_t value = 0;
    uint32_t min = 0;
    int remaining = 0;
};

// Byte-level matcher over a compiled grammar: carries the live stack set and partial codepoint
// so callers can feed it raw token text. Deliberately CUDA-free - every past constrained-decoding
// bug in this tree escaped CI because its test needed a GPU.
class GbnfMatcher {
public:
    bool compile(const std::string& src, std::string* err = nullptr);
    bool compiled() const { return grammar_.compiled(); }
    const GbnfGrammar& grammar() const { return grammar_; }

    // Back to the start of `root`, keeping the compiled grammar.
    void reset();

    // Would `text` keep the output inside the language? Does not advance.
    bool would_accept(const std::string& text) const;
    // Same, but commits the resulting state when the text is legal.
    bool update_text(const std::string& text);

    // True when the derivation is complete and no character is half-written,
    // i.e. stopping here is legal.
    bool is_done() const;

    // Bytes that may start the next token (mask prefilter). Mid-codepoint this
    // is the continuation-byte range.
    void lead_bytes(uint8_t out[256]) const;

    // Cache key for the current state, including the partial codepoint.
    std::vector<int32_t> state_key() const;

private:
    bool run(const std::string& text, GbnfStackSet& stacks, GbnfPartial& partial) const;

    GbnfGrammar grammar_;
    GbnfStackSet stacks_;
    // A BPE token can end mid-character; the grammar only ever sees whole
    // codepoints, so the tail is held here until the next token completes it.
    GbnfPartial partial_;
};

}  // namespace imp
