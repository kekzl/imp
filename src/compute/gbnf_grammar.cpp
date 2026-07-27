#include "compute/gbnf_grammar.h"

#include <algorithm>

namespace imp {

namespace {

constexpr uint32_t kMaxCodepoint = 0x10FFFF;

void dedup(GbnfStackSet& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

}  // namespace

// ---------------------------------------------------------------------------
// Character sets
// ---------------------------------------------------------------------------

bool GbnfCharSet::matches(uint32_t cp) const {
    bool in = false;
    for (const auto& r : ranges) {
        if (cp >= r.first && cp <= r.second) {
            in = true;
            break;
        }
    }
    return negated ? !in : in;
}

bool GbnfCharSet::intersects(uint32_t lo, uint32_t hi) const {
    if (negated) {
        // Any codepoint in [lo,hi] outside every range? Walk the ranges in
        // order and look for a hole.
        std::vector<std::pair<uint32_t, uint32_t>> rs = ranges;
        std::sort(rs.begin(), rs.end());
        uint32_t cur = lo;
        for (const auto& r : rs) {
            if (r.second < cur)
                continue;
            if (r.first > cur)
                return true;  // hole at `cur`
            if (r.second >= hi)
                return false;
            cur = r.second + 1;
        }
        return cur <= hi;
    }
    for (const auto& r : ranges)
        if (r.first <= hi && lo <= r.second)
            return true;
    return false;
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

void GbnfGrammar::compute_nullable() {
    nullable_.assign(rules_.size(), 0);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t r = 0; r < rules_.size(); r++) {
            if (nullable_[r])
                continue;
            for (const auto& alt : rules_[r].alts) {
                bool all_null = true;
                for (const auto& it : alt.items) {
                    if (!it.is_rule || !nullable_[static_cast<size_t>(it.rule)]) {
                        all_null = false;
                        break;
                    }
                }
                if (all_null) {
                    nullable_[r] = 1;
                    changed = true;
                    break;
                }
            }
        }
    }
}

bool GbnfGrammar::check_left_recursion(std::string* err) const {
    // Edge r -> s when s can be entered at the FIRST input position of r, i.e.
    // everything before it in the alternative is nullable. A cycle there means
    // expand() would push frames forever.
    std::vector<std::vector<int32_t>> adj(rules_.size());
    for (size_t r = 0; r < rules_.size(); r++) {
        for (const auto& alt : rules_[r].alts) {
            for (const auto& it : alt.items) {
                if (!it.is_rule)
                    break;  // a character consumes input: no longer first position
                adj[r].push_back(it.rule);
                if (!nullable_[static_cast<size_t>(it.rule)])
                    break;
            }
        }
    }
    std::vector<uint8_t> color(rules_.size(), 0);  // 0 unvisited, 1 on stack, 2 done
    std::vector<std::pair<int32_t, size_t>> stack;
    for (size_t start = 0; start < rules_.size(); start++) {
        if (color[start])
            continue;
        stack.push_back({static_cast<int32_t>(start), 0});
        color[start] = 1;
        while (!stack.empty()) {
            auto& [r, i] = stack.back();
            if (i < adj[static_cast<size_t>(r)].size()) {
                int32_t next = adj[static_cast<size_t>(r)][i++];
                if (color[static_cast<size_t>(next)] == 1) {
                    if (err) {
                        const std::string& n = rules_[static_cast<size_t>(next)].name;
                        *err = "left recursion in rule '" + (n.empty() ? std::string("(group)") : n) + "'";
                    }
                    return false;
                }
                if (color[static_cast<size_t>(next)] == 0) {
                    color[static_cast<size_t>(next)] = 1;
                    stack.push_back({next, 0});
                }
            } else {
                color[static_cast<size_t>(r)] = 2;
                stack.pop_back();
            }
        }
    }
    return true;
}

bool GbnfGrammar::compile(const std::string& src, std::string* err) {
    rules_.clear();
    nullable_.clear();
    root_ = -1;
    compiled_ = false;
    // The arena's ids index THIS rule table — a recompile invalidates all of it,
    // memoised successors included. A ConstraintManager is POOLED, so a second
    // grammar lands in the same object and would otherwise be decoded with the
    // previous grammar's transitions.
    arena_.clear();
    intern_.clear();
    visited_.clear();
    visit_epoch_ = 0;
    next_cache_.clear();
    next_ready_.clear();

    if (!parse_gbnf(src, rules_, root_, err))
        return false;

    compute_nullable();
    if (!check_left_recursion(err))
        return false;

    compiled_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

size_t GbnfGrammar::StackKeyHash::operator()(const StackKey& k) const {
    uint64_t h = 1469598103934665603ull;
    for (int32_t v : {k.pos.rule, k.pos.alt, k.pos.idx, k.parent}) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(v));
        h *= 1099511628211ull;
    }
    return static_cast<size_t>(h);
}

int32_t GbnfGrammar::intern(GbnfPos pos, int32_t parent) const {
    StackKey key{pos, parent};
    auto it = intern_.find(key);
    if (it != intern_.end())
        return it->second;
    const int32_t id = static_cast<int32_t>(arena_.size());
    const int32_t depth = parent < 0 ? 1 : arena_[static_cast<size_t>(parent)].depth + 1;
    arena_.push_back(StackNode{pos, parent, depth});
    visited_.push_back(0);
    intern_.emplace(key, id);
    return id;
}

void GbnfGrammar::begin_visit() const {
    if (++visit_epoch_ == 0) {  // wrapped: clear the marks once, then carry on
        std::fill(visited_.begin(), visited_.end(), 0u);
        visit_epoch_ = 1;
    }
}

const GbnfItem* GbnfGrammar::top_item(int32_t stack) const {
    if (stack < 0)
        return nullptr;
    const GbnfPos& p = arena_[static_cast<size_t>(stack)].pos;
    const GbnfAlt& alt = rules_[static_cast<size_t>(p.rule)].alts[static_cast<size_t>(p.alt)];
    if (p.idx >= static_cast<int32_t>(alt.items.size()))
        return nullptr;
    return &alt.items[static_cast<size_t>(p.idx)];
}

void GbnfGrammar::expand(int32_t stack, GbnfStackSet& out) const {
    work_.clear();
    work_.push_back(stack);
    while (!work_.empty()) {
        const int32_t s = work_.back();
        work_.pop_back();
        if (s < 0) {
            out.push_back(-1);  // derivation complete
            continue;
        }
        if (visited_[static_cast<size_t>(s)] == visit_epoch_)
            continue;
        visited_[static_cast<size_t>(s)] = visit_epoch_;
        // Copy before interning: growing the arena invalidates references.
        const GbnfPos p = arena_[static_cast<size_t>(s)].pos;
        const int32_t parent = arena_[static_cast<size_t>(s)].parent;
        const int32_t depth = arena_[static_cast<size_t>(s)].depth;
        const GbnfAlt& alt = rules_[static_cast<size_t>(p.rule)].alts[static_cast<size_t>(p.alt)];
        if (p.idx >= static_cast<int32_t>(alt.items.size())) {
            // Alternative finished: return to the caller and step it past the
            // rule reference that got us here.
            if (parent < 0) {
                work_.push_back(-1);
            } else {
                const GbnfPos pp = arena_[static_cast<size_t>(parent)].pos;
                const int32_t pparent = arena_[static_cast<size_t>(parent)].parent;
                work_.push_back(intern(GbnfPos{pp.rule, pp.alt, pp.idx + 1}, pparent));
            }
            continue;
        }
        const GbnfItem& item = alt.items[static_cast<size_t>(p.idx)];
        if (!item.is_rule) {
            out.push_back(s);  // waiting for input
            continue;
        }
        if (static_cast<size_t>(depth) >= kMaxStackDepth)
            continue;  // pathological nesting: drop this continuation
        const size_t n_alts = rules_[static_cast<size_t>(item.rule)].alts.size();
        const int32_t target = item.rule;
        for (size_t a = 0; a < n_alts; a++)
            work_.push_back(intern(GbnfPos{target, static_cast<int32_t>(a), 0}, s));
    }
}

GbnfStackSet GbnfGrammar::start_set() const {
    GbnfStackSet out;
    if (!compiled_)
        return out;
    begin_visit();
    const size_t n_alts = rules_[static_cast<size_t>(root_)].alts.size();
    for (size_t a = 0; a < n_alts; a++)
        expand(intern(GbnfPos{root_, static_cast<int32_t>(a), 0}, -1), out);
    dedup(out);
    return out;
}

const GbnfStackSet& GbnfGrammar::successors(int32_t stack) const {
    const size_t s = static_cast<size_t>(stack);
    if (next_ready_.size() < arena_.size()) {
        next_ready_.resize(arena_.size(), 0);
        next_cache_.resize(arena_.size());
    }
    if (next_ready_[s])
        return next_cache_[s];

    const GbnfPos p = arena_[s].pos;
    const int32_t parent = arena_[s].parent;
    GbnfStackSet out;
    begin_visit();
    expand(intern(GbnfPos{p.rule, p.alt, p.idx + 1}, parent), out);
    dedup(out);
    if (out.size() > kMaxStacks)
        out.resize(kMaxStacks);
    // expand() interns, so the arena (and with it the required cache size) may
    // have grown since the check above.
    if (next_ready_.size() < arena_.size()) {
        next_ready_.resize(arena_.size(), 0);
        next_cache_.resize(arena_.size());
    }
    next_cache_[s] = std::move(out);
    next_ready_[s] = 1;
    return next_cache_[s];
}

void GbnfGrammar::step_into(const GbnfStackSet& stacks, uint32_t cp, GbnfStackSet& out) const {
    out.clear();
    for (int32_t s : stacks) {
        const GbnfItem* item = top_item(s);
        if (!item || item->is_rule || !item->chars.matches(cp))
            continue;  // -1 (completed) consumes nothing more, either
        const GbnfStackSet& nxt = successors(s);
        out.insert(out.end(), nxt.begin(), nxt.end());
    }
    dedup(out);
    if (out.size() > kMaxStacks)
        out.resize(kMaxStacks);
}

GbnfStackSet GbnfGrammar::step(const GbnfStackSet& stacks, uint32_t cp) const {
    GbnfStackSet out;
    step_into(stacks, cp, out);
    return out;
}

bool GbnfGrammar::accepts(const GbnfStackSet& stacks) {
    // The set is sorted, so the empty stack (-1) can only be first.
    return !stacks.empty() && stacks.front() == -1;
}

bool GbnfGrammar::can_consume_range(const GbnfStackSet& stacks, uint32_t lo, uint32_t hi) const {
    for (int32_t s : stacks) {
        const GbnfItem* item = top_item(s);
        if (item && !item->is_rule && item->chars.intersects(lo, hi))
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Byte-level matcher (UTF-8 assembly + live state)
// ---------------------------------------------------------------------------

namespace {

// Bytes remaining after this lead byte; -1 marks an invalid lead. C0/C1 are
// overlong two-byte leads and F5+ is past U+10FFFF: both are rejected outright,
// or a token could smuggle a forbidden character in as an overlong encoding.
inline int utf8_extra(unsigned char b) {
    if (b < 0x80)
        return 0;
    if (b < 0xC2)
        return -1;  // bare continuation byte, or an overlong lead
    if (b < 0xE0)
        return 1;
    if (b < 0xF0)
        return 2;
    if (b < 0xF5)
        return 3;
    return -1;
}

// Smallest codepoint a sequence with `extra` continuation bytes may encode.
inline uint32_t utf8_min(int extra) { return extra == 1 ? 0x80u : extra == 2 ? 0x800u : 0x10000u; }

}  // namespace

bool GbnfMatcher::compile(const std::string& src, std::string* err) {
    partial_ = GbnfPartial{};
    stacks_.clear();
    if (!grammar_.compile(src, err))
        return false;
    stacks_ = grammar_.start_set();
    if (stacks_.empty()) {
        if (err)
            *err = "grammar has no reachable start state";
        return false;
    }
    return true;
}

void GbnfMatcher::reset() {
    if (!grammar_.compiled())
        return;
    stacks_ = grammar_.start_set();
    partial_ = GbnfPartial{};
}

bool GbnfMatcher::run(const std::string& text, GbnfStackSet& stacks, GbnfPartial& partial) const {
    // Scratch for step_into; reused across the whole text so a multi-byte token
    // costs no allocation per character.
    GbnfStackSet next;
    for (unsigned char b : text) {
        if (partial.remaining > 0) {
            if ((b & 0xC0) != 0x80)
                return false;  // expected a continuation byte
            partial.value = (partial.value << 6) | (b & 0x3Fu);
            if (--partial.remaining == 0) {
                if (partial.value < partial.min || partial.value > 0x10FFFFu)
                    return false;  // overlong encoding, or past the last codepoint
                grammar_.step_into(stacks, partial.value, next);
                if (next.empty())
                    return false;
                stacks.swap(next);
            }
            continue;
        }
        const int extra = utf8_extra(b);
        if (extra < 0)
            return false;
        if (extra == 0) {
            grammar_.step_into(stacks, b, next);
            if (next.empty())
                return false;
            stacks.swap(next);
            continue;
        }
        partial.value = b & static_cast<uint32_t>(0x3F >> extra);
        partial.min = utf8_min(extra);
        partial.remaining = extra;
    }
    if (partial.remaining > 0) {
        // The text ended mid-character. It stays legal as long as SOME
        // completion of it is; the exact check runs once the next token
        // finishes the codepoint.
        const uint32_t shift = static_cast<uint32_t>(6 * partial.remaining);
        const uint32_t lo = std::max(partial.value << shift, partial.min);
        const uint32_t hi = std::min((partial.value << shift) | ((1u << shift) - 1u), 0x10FFFFu);
        if (lo > hi || !grammar_.can_consume_range(stacks, lo, hi))
            return false;
    }
    return true;
}

bool GbnfMatcher::would_accept(const std::string& text) const {
    GbnfStackSet stacks = stacks_;
    GbnfPartial partial = partial_;
    return run(text, stacks, partial);
}

bool GbnfMatcher::update_text(const std::string& text) {
    GbnfStackSet stacks = stacks_;
    GbnfPartial partial = partial_;
    if (!run(text, stacks, partial))
        return false;
    stacks_ = std::move(stacks);
    partial_ = partial;
    return true;
}

bool GbnfMatcher::is_done() const {
    // A pending partial codepoint means the last token cut a character in half:
    // the derivation cannot be complete there even if the stacks say so.
    return partial_.remaining == 0 && GbnfGrammar::accepts(stacks_);
}

void GbnfMatcher::lead_bytes(uint8_t out[256]) const {
    if (partial_.remaining > 0) {
        for (int i = 0; i < 256; i++)
            out[i] = (i & 0xC0) == 0x80 ? 1 : 0;
        return;
    }
    grammar_.lead_bytes(stacks_, out);
}

std::vector<int32_t> GbnfMatcher::state_key() const {
    std::vector<int32_t> key;
    key.push_back(static_cast<int32_t>(partial_.remaining));
    key.push_back(static_cast<int32_t>(partial_.value));
    key.push_back(static_cast<int32_t>(partial_.min));
    key.insert(key.end(), stacks_.begin(), stacks_.end());
    return key;
}

void GbnfGrammar::lead_bytes(const GbnfStackSet& stacks, uint8_t out[256]) const {
    for (int i = 0; i < 256; i++)
        out[i] = 0;
    for (int32_t s : stacks) {
        const GbnfItem* item = top_item(s);
        if (!item || item->is_rule)
            continue;
        const GbnfCharSet& cs = item->chars;
        for (uint32_t b = 0; b < 0x80; b++)
            if (!out[b] && cs.matches(b))
                out[b] = 1;
        // Multi-byte lead: allow the byte if the codepoint block it opens
        // intersects the set at all. The block's lower bound is clamped to the
        // shortest legal encoding for that length (C0/C1, and E0/F0 below their
        // minimum, are overlong — see utf8_extra), so a lead byte is never
        // allowed on the strength of codepoints it cannot legally encode.
        // Over-permissive only inside one block, and the exact check still runs
        // when the codepoint completes.
        auto mark_lead = [&](uint32_t b, uint32_t base, uint32_t span, uint32_t min_cp) {
            if (out[b])
                return;
            const uint32_t lo = std::max(base, min_cp);
            const uint32_t hi = std::min(base | span, kMaxCodepoint);
            if (lo <= hi && cs.intersects(lo, hi))
                out[b] = 1;
        };
        for (uint32_t b = 0xC2; b <= 0xDF; b++)
            mark_lead(b, (b & 0x1F) << 6, 0x3F, 0x80);
        for (uint32_t b = 0xE0; b <= 0xEF; b++)
            mark_lead(b, (b & 0x0F) << 12, 0xFFF, 0x800);
        for (uint32_t b = 0xF0; b <= 0xF4; b++)
            mark_lead(b, (b & 0x07) << 18, 0x3FFFF, 0x10000);
    }
}

}  // namespace imp
