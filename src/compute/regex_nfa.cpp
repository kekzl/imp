// RegexNfa (compute/json_schema.h): the JSON-Schema "pattern" compiler and executor, moved
// verbatim out of json_schema.cpp.
#include "compute/json_schema.h"

#include <algorithm>
#include <string>
#include <vector>

namespace imp {
// RegexNfa: Thompson-construction NFA over bytes for the supported regex
// subset. All host-side; never compiled into device code.

int RegexNfa::new_state() {
    // #1608: {n,m} clones multiply state count, the resource a hostile pattern
    // spends. Past kMaxStates the parse errors; returning the last valid index
    // keeps in-flight add_edge()/add_epsilon() calls in bounds.
    if (states_.size() >= kMaxStates) {
        error_ = true;
        return states_.empty() ? 0 : static_cast<int>(states_.size()) - 1;
    }
    states_.emplace_back();
    return static_cast<int>(states_.size()) - 1;
}

void RegexNfa::add_epsilon(int from, int to) {
    NfaEdge e;
    e.to = to;
    e.is_epsilon = true;
    states_[from].edges.push_back(std::move(e));
}

void RegexNfa::add_edge(int from, int to, const std::vector<uint8_t>& cls) {
    NfaEdge e;
    e.to = to;
    e.is_epsilon = false;
    e.char_class = cls;
    states_[from].edges.push_back(std::move(e));
}

bool RegexNfa::make_shorthand(char esc, std::vector<uint8_t>& cls) {
    cls.assign(256, 0);
    auto set_digit = [&](bool neg) {
        for (int c = 0; c < 256; c++) {
            bool d = (c >= '0' && c <= '9');
            cls[c] = (d != neg) ? 1 : 0;
        }
    };
    auto set_word = [&](bool neg) {
        for (int c = 0; c < 256; c++) {
            bool w = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
            cls[c] = (w != neg) ? 1 : 0;
        }
    };
    auto set_space = [&](bool neg) {
        for (int c = 0; c < 256; c++) {
            bool s = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
            cls[c] = (s != neg) ? 1 : 0;
        }
    };
    switch (esc) {
        case 'd':
            set_digit(false);
            return true;
        case 'D':
            set_digit(true);
            return true;
        case 'w':
            set_word(false);
            return true;
        case 'W':
            set_word(true);
            return true;
        case 's':
            set_space(false);
            return true;
        case 'S':
            set_space(true);
            return true;
        default:
            return false;
    }
}

// atom := '(' alt ')' | '[' class ']' | '.' | escape | literal
bool RegexNfa::parse_atom(Frag& out) {
    if (pos_ >= src_->size()) {
        error_ = true;
        return false;
    }
    char c = (*src_)[pos_];

    if (c == '(') {
        pos_++;  // consume '('
        // `(?:...)` is non-capturing; nothing here captures and backreferences are
        // refused upstream, so skip the marker and parse the body as an ordinary
        // group. A silently wrong compiled pattern is the one failure mode to avoid.
        if (pos_ + 1 < src_->size() && (*src_)[pos_] == '?' && (*src_)[pos_ + 1] == ':')
            pos_ += 2;
        if (!parse_alt(out))
            return false;
        if (pos_ >= src_->size() || (*src_)[pos_] != ')') {
            error_ = true;
            return false;
        }
        pos_++;  // consume ')'
        return true;
    }

    if (c == '[') {
        return parse_class(out);
    }

    if (c == ')' || c == '|') {
        // empty atom — caller handles
        error_ = true;
        return false;
    }

    std::vector<uint8_t> cls(256, 0);

    if (c == '.') {
        pos_++;
        for (int i = 0; i < 256; i++)
            cls[i] = (i == '\n') ? 0 : 1;  // any except newline
    } else if (c == '\\') {
        pos_++;
        if (pos_ >= src_->size()) {
            error_ = true;
            return false;
        }
        char esc = (*src_)[pos_++];
        std::vector<uint8_t> sc;
        if (make_shorthand(esc, sc)) {
            cls = std::move(sc);
        } else {
            // escaped literal (\. \\ \+ \{ etc.) — also map common control escapes
            unsigned char lit;
            switch (esc) {
                case 'n':
                    lit = '\n';
                    break;
                case 't':
                    lit = '\t';
                    break;
                case 'r':
                    lit = '\r';
                    break;
                case 'f':
                    lit = '\f';
                    break;
                case 'v':
                    lit = '\v';
                    break;
                default:
                    lit = static_cast<unsigned char>(esc);
                    break;
            }
            cls[lit] = 1;
        }
    } else if (c == '^' || c == '$') {
        // Anchors: JSON-Schema pattern matching is treated as whole-string here
        // (token masking cannot enforce sub-string matches), so a leading ^ /
        // trailing $ are no-ops. Emit an epsilon fragment.
        pos_++;
        int s = new_state();
        int a = new_state();
        add_epsilon(s, a);
        out.start = s;
        out.accept = a;
        return true;
    } else {
        // literal byte
        pos_++;
        cls[static_cast<unsigned char>(c)] = 1;
    }

    int s = new_state();
    int a = new_state();
    add_edge(s, a, cls);
    out.start = s;
    out.accept = a;
    return true;
}

// class := '[' '^'? ( range | shorthand | char )+ ']'
bool RegexNfa::parse_class(Frag& out) {
    pos_++;  // consume '['
    bool negate = false;
    if (pos_ < src_->size() && (*src_)[pos_] == '^') {
        negate = true;
        pos_++;
    }
    std::vector<uint8_t> cls(256, 0);
    bool any = false;

    while (pos_ < src_->size() && (*src_)[pos_] != ']') {
        char c = (*src_)[pos_];
        if (c == '\\') {
            pos_++;
            if (pos_ >= src_->size()) {
                error_ = true;
                return false;
            }
            char esc = (*src_)[pos_++];
            std::vector<uint8_t> sc;
            if (make_shorthand(esc, sc)) {
                for (int i = 0; i < 256; i++)
                    if (sc[i])
                        cls[i] = 1;
                any = true;
                continue;
            }
            unsigned char lit;
            switch (esc) {
                case 'n':
                    lit = '\n';
                    break;
                case 't':
                    lit = '\t';
                    break;
                case 'r':
                    lit = '\r';
                    break;
                default:
                    lit = static_cast<unsigned char>(esc);
                    break;
            }
            cls[lit] = 1;
            any = true;
            continue;
        }
        // range a-z ?
        if (pos_ + 2 < src_->size() && (*src_)[pos_ + 1] == '-' && (*src_)[pos_ + 2] != ']') {
            unsigned char lo = static_cast<unsigned char>(c);
            unsigned char hi = static_cast<unsigned char>((*src_)[pos_ + 2]);
            pos_ += 3;
            if (lo > hi) {
                error_ = true;
                return false;
            }
            for (int i = lo; i <= hi; i++)
                cls[i] = 1;
            any = true;
        } else {
            cls[static_cast<unsigned char>(c)] = 1;
            pos_++;
            any = true;
        }
    }

    if (pos_ >= src_->size() || (*src_)[pos_] != ']' || !any) {
        error_ = true;
        return false;
    }
    pos_++;  // consume ']'

    if (negate) {
        for (int i = 0; i < 256; i++)
            cls[i] = cls[i] ? 0 : 1;
    }

    int s = new_state();
    int a = new_state();
    add_edge(s, a, cls);
    out.start = s;
    out.accept = a;
    return true;
}

// repeat := atom ('*' | '+' | '?' | '{n}' | '{n,}' | '{n,m}')?
bool RegexNfa::parse_repeat(Frag& out) {
    // Snapshot the atom's source start here so {n,m} can re-parse the exact
    // source span of *this* atom (parse_atom may advance pos_ arbitrarily for
    // groups / classes).
    size_t atom_begin = pos_;
    Frag atom{};
    if (!parse_atom(atom))
        return false;

    if (pos_ >= src_->size())
        return (out = atom, true);

    char q = (*src_)[pos_];

    if (q == '*' || q == '+' || q == '?') {
        pos_++;
        int s = new_state();
        int a = new_state();
        if (q == '*') {
            add_epsilon(s, atom.start);
            add_epsilon(s, a);
            add_epsilon(atom.accept, atom.start);
            add_epsilon(atom.accept, a);
        } else if (q == '+') {
            add_epsilon(s, atom.start);
            add_epsilon(atom.accept, atom.start);
            add_epsilon(atom.accept, a);
        } else {  // '?'
            add_epsilon(s, atom.start);
            add_epsilon(s, a);
            add_epsilon(atom.accept, a);
        }
        out.start = s;
        out.accept = a;
        return true;
    }

    if (q == '{') {
        // {n} {n,} {n,m}
        size_t save = pos_;
        pos_++;  // consume '{'
        long n = 0, m = -1;
        auto scan_int = [&](long& out_val) -> bool {
            long v = 0;
            bool any = false;
            while (pos_ < src_->size() && (*src_)[pos_] >= '0' && (*src_)[pos_] <= '9') {
                // #1608: `v = v*10 + d` over an unbounded digit run is signed
                // overflow, i.e. UB, before it is ever compared to a limit.
                // Saturate at the repeat cap; the caller rejects from there.
                if (v <= kMaxRepeat)
                    v = (v * 10) + ((*src_)[pos_] - '0');
                pos_++;
                any = true;
            }
            if (any)
                out_val = v;
            return any;
        };
        bool has_n = scan_int(n);
        bool comma = false;
        if (pos_ < src_->size() && (*src_)[pos_] == ',') {
            comma = true;
            pos_++;
            long mm = 0;
            if (scan_int(mm))
                m = mm;  // {n,m}; else {n,} -> m stays -1 (unbounded)
        }
        if (!has_n || pos_ >= src_->size() || (*src_)[pos_] != '}') {
            // Not a valid quantifier — treat '{' as literal: rewind.
            pos_ = save;
            return (out = atom, true);
        }
        pos_++;  // consume '}'
        if (!comma)
            m = n;  // {n}

        // #1608: n is a clone count; unbounded n costs one parse+alloc per clone
        // on an HTTP worker thread at admission time. Capped at kMaxRepeat.
        if (n > kMaxRepeat || m > kMaxRepeat) {
            error_ = true;
            return false;
        }

        // Build: n mandatory copies, then either unbounded (*) or (m-n) optional.
        int s = new_state();
        int cur = s;
        // We need fresh copies of the atom sub-pattern. Re-parse its exact
        // source span [atom_begin, save) for each additional copy. `save` is
        // the position right after the atom (the '{').
        std::string atom_src = src_->substr(atom_begin, save - atom_begin);

        auto clone_atom = [&](Frag& f) -> bool {
            const std::string* prev_src = src_;
            size_t prev_pos = pos_;
            bool prev_err = error_;
            // Parse the captured atom text in isolation.
            std::string local = atom_src;
            src_ = &local;
            pos_ = 0;
            error_ = false;
            bool ok = parse_repeat(f);  // atom may itself be a repeat-free atom
            // #1608: the clone's error must survive the restore, else new_state()'s
            // state-budget signal is lost and nested `(((a{100}){100}){100})` isn't stopped.
            bool inner_err = error_;
            src_ = prev_src;
            pos_ = prev_pos;
            error_ = prev_err || inner_err;
            return ok && !inner_err;
        };

        // first mandatory copy is the already-built `atom`
        long built = 0;
        if (n >= 1) {
            add_epsilon(cur, atom.start);
            cur = atom.accept;
            built = 1;
        } else {
            // n == 0: atom is optional/unbounded from the start
        }

        for (long i = built; i < n; i++) {
            Frag f{};
            if (!clone_atom(f))
                return false;
            add_epsilon(cur, f.start);
            cur = f.accept;
        }

        int a = new_state();
        if (m < 0) {
            // {n,} -> after n mandatory, a Kleene star of the atom
            Frag f{};
            if (!clone_atom(f))
                return false;
            int ls = new_state();
            add_epsilon(cur, ls);
            add_epsilon(ls, f.start);
            add_epsilon(ls, a);
            add_epsilon(f.accept, f.start);
            add_epsilon(f.accept, a);
        } else {
            // {n,m} -> (m-n) optional copies
            for (long i = n; i < m; i++) {
                Frag f{};
                if (!clone_atom(f))
                    return false;
                add_epsilon(cur, f.start);
                add_epsilon(cur, a);  // optional: skip the rest
                cur = f.accept;
            }
            add_epsilon(cur, a);
        }
        out.start = s;
        out.accept = a;
        return true;
    }

    out = atom;
    return true;
}

// concat := repeat*
bool RegexNfa::parse_concat(Frag& out) {
    int s = new_state();
    int cur = s;
    bool any = false;
    while (pos_ < src_->size() && (*src_)[pos_] != '|' && (*src_)[pos_] != ')') {
        Frag f{};
        if (!parse_repeat(f))
            return false;
        add_epsilon(cur, f.start);
        cur = f.accept;
        any = true;
    }
    if (!any) {
        // empty concatenation matches empty string
        int a = new_state();
        add_epsilon(s, a);
        out.start = s;
        out.accept = a;
        return true;
    }
    out.start = s;
    out.accept = cur;
    return true;
}

// alt := concat ('|' concat)*
bool RegexNfa::parse_alt(Frag& out) {
    // #1609: parse_atom() recurses into parse_alt() for a group, so '(' costs
    // one stack frame per byte. parse_alt is the single point closing the
    // mutual recursion, so one depth guard here covers all four functions.
    if (depth_ >= kMaxDepth) {
        error_ = true;
        return false;
    }
    depth_++;
    struct Pop {
        int& d;
        ~Pop() { d--; }
    } pop{depth_};

    Frag first{};
    if (!parse_concat(first))
        return false;
    if (pos_ >= src_->size() || (*src_)[pos_] != '|') {
        out = first;
        return true;
    }
    int s = new_state();
    int a = new_state();
    add_epsilon(s, first.start);
    add_epsilon(first.accept, a);
    while (pos_ < src_->size() && (*src_)[pos_] == '|') {
        pos_++;  // consume '|'
        Frag next{};
        if (!parse_concat(next))
            return false;
        add_epsilon(s, next.start);
        add_epsilon(next.accept, a);
    }
    out.start = s;
    out.accept = a;
    return true;
}

bool RegexNfa::compile(const std::string& pattern) {
    states_.clear();
    compiled_ = false;
    error_ = false;
    depth_ = 0;
    src_ = &pattern;
    pos_ = 0;

    Frag root{};
    if (!parse_alt(root)) {
        src_ = nullptr;
        return false;
    }
    // Must have consumed the whole pattern.
    if (pos_ != pattern.size() || error_) {
        src_ = nullptr;
        return false;
    }
    src_ = nullptr;

    start_ = root.start;
    accept_ = root.accept;
    states_[accept_].accepting = true;
    compiled_ = true;
    return true;
}

void RegexNfa::epsilon_closure(std::vector<int>& set) const {
    std::vector<int> stack(set.begin(), set.end());
    std::vector<uint8_t> in(states_.size(), 0);
    for (int s : set)
        in[s] = 1;
    while (!stack.empty()) {
        int s = stack.back();
        stack.pop_back();
        for (const auto& e : states_[s].edges) {
            if (e.is_epsilon && !in[e.to]) {
                in[e.to] = 1;
                set.push_back(e.to);
                stack.push_back(e.to);
            }
        }
    }
    std::sort(set.begin(), set.end());
}

std::vector<int> RegexNfa::start_set() const {
    std::vector<int> set;
    if (!compiled_)
        return set;
    set.push_back(start_);
    epsilon_closure(set);
    return set;
}

std::vector<int> RegexNfa::step(const std::vector<int>& states, unsigned char c) const {
    std::vector<int> next;
    if (!compiled_)
        return next;
    std::vector<uint8_t> in(states_.size(), 0);
    for (int s : states) {
        for (const auto& e : states_[s].edges) {
            if (!e.is_epsilon && e.char_class[c] && !in[e.to]) {
                in[e.to] = 1;
                next.push_back(e.to);
            }
        }
    }
    epsilon_closure(next);
    return next;
}

bool RegexNfa::accepts(const std::vector<int>& states) const {
    for (int s : states)
        if (states_[s].accepting)
            return true;
    return false;
}

}  // namespace imp
