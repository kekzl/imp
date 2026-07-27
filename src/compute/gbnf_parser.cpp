#include "compute/gbnf_grammar.h"

#include <map>
#include <set>
#include <string>

// GBNF source -> rule table. Split from the simulator so a parser change does
// not re-ptxas the simulator's TU and vice versa; the two share nothing but the
// rule structs.

namespace imp {

namespace {

constexpr uint32_t kMaxCodepoint = 0x10FFFF;
// A repetition bound larger than this is a grammar bomb, not an intent: {0,100000}
// would materialise 100k synthetic rules before decoding a single token.
constexpr int kMaxRepeat = 1024;

bool is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
}

// Recursive-descent GBNF parser. Groups and repetitions are desugared into
// synthetic (unnamed) rules here, so the simulator only handles two item kinds.
struct Parser {
    const std::string& src;
    std::vector<GbnfRule>& rules;
    std::map<std::string, int32_t> ids;
    std::set<std::string> defined;
    size_t pos = 0;
    std::string err;

    Parser(const std::string& s, std::vector<GbnfRule>& r) : src(s), rules(r) {}

    bool eof() const { return pos >= src.size(); }
    char peek(size_t off = 0) const { return pos + off < src.size() ? src[pos + off] : '\0'; }

    bool fail(const std::string& msg) {
        if (err.empty()) {
            // Byte offset, not line number: a grammar arrives as one JSON string
            // and the caller has no line-split view of it either.
            err = msg + " (at offset " + std::to_string(pos) + ")";
        }
        return false;
    }

    // `allow_newline` false = a newline terminates the current sequence, which
    // is what separates one rule from the next. Inside ( ... ) and right after
    // `::=` / `|` a newline is just whitespace, so grammars can wrap lines.
    void skip_space(bool allow_newline) {
        while (!eof()) {
            char c = peek();
            if (c == '#') {
                while (!eof() && peek() != '\n')
                    pos++;
            } else if (c == ' ' || c == '\t' || c == '\r') {
                pos++;
            } else if (c == '\n' && allow_newline) {
                pos++;
            } else {
                return;
            }
        }
    }

    int32_t rule_id(const std::string& name) {
        auto it = ids.find(name);
        if (it != ids.end())
            return it->second;
        int32_t id = static_cast<int32_t>(rules.size());
        rules.push_back(GbnfRule{name, {}});
        ids[name] = id;
        return id;
    }

    int32_t add_synthetic(std::vector<GbnfAlt> alts) {
        int32_t id = static_cast<int32_t>(rules.size());
        rules.push_back(GbnfRule{"", std::move(alts)});
        return id;
    }

    static GbnfItem rule_item(int32_t id) {
        GbnfItem it;
        it.is_rule = true;
        it.rule = id;
        return it;
    }

    static GbnfItem char_item(uint32_t cp) {
        GbnfItem it;
        it.chars.ranges.emplace_back(cp, cp);
        return it;
    }

    // ---- lexical -----------------------------------------------------------

    bool hex(int digits, uint32_t& out) {
        out = 0;
        for (int i = 0; i < digits; i++) {
            char c = peek();
            uint32_t v;
            if (c >= '0' && c <= '9')
                v = static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')
                v = static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                v = static_cast<uint32_t>(c - 'A' + 10);
            else
                return fail("bad hex escape");
            out = (out << 4) | v;
            pos++;
        }
        return true;
    }

    // One character of a literal or class: an escape, or a UTF-8 codepoint.
    bool read_char(uint32_t& cp) {
        if (eof())
            return fail("unterminated literal");
        if (peek() != '\\') {
            unsigned char b = static_cast<unsigned char>(src[pos++]);
            if (b < 0x80) {
                cp = b;
                return true;
            }
            int extra = (b >= 0xF0) ? 3 : (b >= 0xE0) ? 2 : (b >= 0xC0) ? 1 : -1;
            if (extra < 0)
                return fail("invalid UTF-8 in grammar");
            cp = b & static_cast<unsigned char>(0x3F >> extra);
            for (int i = 0; i < extra; i++) {
                if (eof() || (static_cast<unsigned char>(src[pos]) & 0xC0) != 0x80)
                    return fail("truncated UTF-8 in grammar");
                cp = (cp << 6) | (static_cast<unsigned char>(src[pos++]) & 0x3F);
            }
            return true;
        }
        pos++;  // backslash
        if (eof())
            return fail("trailing backslash");
        char e = src[pos++];
        switch (e) {
            case 'n':
                cp = '\n';
                return true;
            case 'r':
                cp = '\r';
                return true;
            case 't':
                cp = '\t';
                return true;
            case '0':
                cp = 0;
                return true;
            case 'x':
                return hex(2, cp);
            case 'u':
                return hex(4, cp);
            case 'U':
                return hex(8, cp);
            // Every other escape is the character itself — this is how a
            // grammar writes a literal quote, bracket or backslash.
            default:
                cp = static_cast<unsigned char>(e);
                return true;
        }
    }

    // ---- grammar -----------------------------------------------------------

    bool parse_string(std::vector<GbnfItem>& items) {
        pos++;  // opening quote
        while (!eof() && peek() != '"') {
            uint32_t cp;
            if (!read_char(cp))
                return false;
            items.push_back(char_item(cp));
        }
        if (eof())
            return fail("unterminated string literal");
        pos++;  // closing quote
        return true;
    }

    bool parse_class(GbnfItem& out) {
        pos++;  // [
        if (peek() == '^') {
            out.chars.negated = true;
            pos++;
        }
        while (!eof() && peek() != ']') {
            uint32_t lo;
            if (!read_char(lo))
                return false;
            uint32_t hi = lo;
            if (peek() == '-' && peek(1) != ']' && peek(1) != '\0') {
                pos++;
                if (!read_char(hi))
                    return false;
            }
            if (hi < lo)
                return fail("inverted character range");
            out.chars.ranges.emplace_back(lo, hi);
        }
        if (eof())
            return fail("unterminated character class");
        pos++;  // ]
        return true;
    }

    // X?  ->  S ::= | X
    int32_t make_opt(const GbnfItem& x, int32_t tail) {
        GbnfAlt empty;
        GbnfAlt one;
        one.items.push_back(x);
        if (tail >= 0)
            one.items.push_back(rule_item(tail));
        return add_synthetic({std::move(empty), std::move(one)});
    }

    // X*  ->  S ::= | X S      (right-recursive: a left-recursive spelling is
    // exactly what this simulator cannot expand)
    int32_t make_star(const GbnfItem& x) {
        int32_t id = static_cast<int32_t>(rules.size());
        rules.push_back(GbnfRule{"", {}});
        GbnfAlt empty;
        GbnfAlt loop;
        loop.items.push_back(x);
        loop.items.push_back(rule_item(id));
        rules[static_cast<size_t>(id)].alts = {std::move(empty), std::move(loop)};
        return id;
    }

    bool parse_repeat_bounds(int& lo, int& hi) {
        pos++;  // {
        std::string a, b;
        while (peek() >= '0' && peek() <= '9')
            a += src[pos++];
        bool has_comma = peek() == ',';
        if (has_comma) {
            pos++;
            while (peek() >= '0' && peek() <= '9')
                b += src[pos++];
        }
        if (peek() != '}')
            return fail("unterminated { } repetition");
        pos++;
        if (a.empty() && b.empty())
            return fail("empty { } repetition");
        lo = a.empty() ? 0 : std::stoi(a);
        hi = has_comma ? (b.empty() ? -1 : std::stoi(b)) : lo;
        if (lo > kMaxRepeat || hi > kMaxRepeat)
            return fail("repetition bound over " + std::to_string(kMaxRepeat));
        if (hi >= 0 && hi < lo)
            return fail("inverted { } repetition");
        return true;
    }

    // Postfix operators bind to the element just parsed — the WHOLE element:
    // `"abc"*` repeats the string, not its last character. A string literal
    // expands to one item per character, so `start` marks where it began and
    // everything from there is folded into a synthetic rule first.
    // Applied immediately (no whitespace skip) so `a *` is a rule ref followed
    // by a syntax error rather than a silent repetition.
    bool apply_postfix(std::vector<GbnfItem>& items, size_t start) {
        while (!eof()) {
            char c = peek();
            if (c != '*' && c != '+' && c != '?' && c != '{')
                return true;
            if (items.size() <= start)
                return fail("repetition without a preceding element");
            GbnfItem x;
            if (items.size() - start == 1) {
                x = items[start];
            } else {
                GbnfAlt alt;
                alt.items.assign(items.begin() + static_cast<long>(start), items.end());
                x = rule_item(add_synthetic({std::move(alt)}));
            }
            items.erase(items.begin() + static_cast<long>(start), items.end());
            if (c == '*') {
                pos++;
                items.push_back(rule_item(make_star(x)));
            } else if (c == '?') {
                pos++;
                items.push_back(rule_item(make_opt(x, -1)));
            } else if (c == '+') {
                pos++;
                items.push_back(x);
                items.push_back(rule_item(make_star(x)));
            } else {
                int lo = 0, hi = 0;
                if (!parse_repeat_bounds(lo, hi))
                    return false;
                for (int i = 0; i < lo; i++)
                    items.push_back(x);
                if (hi < 0) {
                    items.push_back(rule_item(make_star(x)));
                } else if (hi > lo) {
                    // Nested optionals, innermost first: {2,4} of X becomes
                    // X X (X (X)?)?
                    int32_t tail = -1;
                    for (int i = 0; i < hi - lo; i++)
                        tail = make_opt(x, tail);
                    items.push_back(rule_item(tail));
                }
            }
        }
        return true;
    }

    bool parse_alternates(std::vector<GbnfAlt>& out, bool nested);

    bool parse_sequence(GbnfAlt& out, bool nested) {
        while (true) {
            skip_space(nested);
            char c = peek();
            if (eof() || c == '|' || c == ')' || c == '\n')
                return true;
            const size_t elem_start = out.items.size();
            if (c == '"') {
                if (!parse_string(out.items))
                    return false;
            } else if (c == '[') {
                GbnfItem it;
                if (!parse_class(it))
                    return false;
                out.items.push_back(std::move(it));
            } else if (c == '.') {
                pos++;
                GbnfItem it;
                it.chars.negated = true;  // empty negated set = any codepoint
                out.items.push_back(std::move(it));
            } else if (c == '(') {
                pos++;
                std::vector<GbnfAlt> alts;
                if (!parse_alternates(alts, /*nested=*/true))
                    return false;
                skip_space(true);
                if (peek() != ')')
                    return fail("expected )");
                pos++;
                out.items.push_back(rule_item(add_synthetic(std::move(alts))));
            } else if (is_word_char(c)) {
                std::string name;
                while (!eof() && is_word_char(peek()))
                    name += src[pos++];
                out.items.push_back(rule_item(rule_id(name)));
            } else {
                return fail(std::string("unexpected character '") + c + "'");
            }
            if (!apply_postfix(out.items, elem_start))
                return false;
        }
    }

    bool parse_rule() {
        std::string name;
        while (!eof() && is_word_char(peek()))
            name += src[pos++];
        if (name.empty())
            return fail("expected a rule name");
        skip_space(false);
        if (src.compare(pos, 3, "::=") != 0)
            return fail("expected ::= after rule '" + name + "'");
        pos += 3;
        skip_space(true);
        if (!defined.insert(name).second)
            return fail("rule '" + name + "' defined twice");
        int32_t id = rule_id(name);
        std::vector<GbnfAlt> alts;
        if (!parse_alternates(alts, /*nested=*/false))
            return false;
        rules[static_cast<size_t>(id)].alts = std::move(alts);
        return true;
    }

    bool parse_grammar() {
        skip_space(true);
        while (!eof()) {
            if (!parse_rule())
                return false;
            skip_space(true);
        }
        return true;
    }
};

bool Parser::parse_alternates(std::vector<GbnfAlt>& out, bool nested) {
    GbnfAlt first;
    if (!parse_sequence(first, nested))
        return false;
    out.push_back(std::move(first));
    while (true) {
        skip_space(nested);
        if (peek() != '|')
            return true;
        pos++;
        skip_space(true);  // `|` may be followed by a line break
        GbnfAlt alt;
        if (!parse_sequence(alt, nested))
            return false;
        out.push_back(std::move(alt));
    }
}

}  // namespace

bool parse_gbnf(const std::string& src, std::vector<GbnfRule>& rules, int32_t& root, std::string* err) {
    Parser p(src, rules);
    if (!p.parse_grammar()) {
        if (err)
            *err = p.err;
        return false;
    }
    // A named rule with no alternatives was only ever referenced, never defined.
    for (const auto& r : rules) {
        if (!r.name.empty() && r.alts.empty()) {
            if (err)
                *err = "undefined rule '" + r.name + "'";
            return false;
        }
    }
    auto it = p.ids.find("root");
    if (it == p.ids.end()) {
        if (err)
            *err = "grammar has no 'root' rule";
        return false;
    }
    root = it->second;
    return true;
}

}  // namespace imp
