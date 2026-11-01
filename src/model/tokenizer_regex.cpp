// tokenizer.json pre-tokenizer on PCRE2 with HF tokenizers' split semantics (pattern.rs find_matches,
// normalizer.rs split). Regex text is Oniguruma (Ruby syntax); onig_to_pcre2 rewrites the escapes the
// two engines read differently.
#include "model/tokenizer_regex.h"

#include "model/json_util.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

namespace imp {

const char* const kGpt2PretokRegex =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

struct RegexPretokenizer::Step {
    std::shared_ptr<pcre2_code> code;
    Behavior behavior = Behavior::kIsolated;
    bool invert = false;
};

namespace {

using Behavior = RegexPretokenizer::Behavior;
using Step = RegexPretokenizer::Step;

// Oniguruma \s: Unicode White_Space, which has U+0085; \w: L, M, N, Pc (onig unicode.c ONIGENC_CTYPE_WORD).
constexpr const char* kSpaceIn = R"(\s\x{85})";
constexpr const char* kWordIn = R"(\p{L}\p{M}\p{N}\p{Pc})";

std::string word_boundary(bool at) {
    const std::string w = std::string("[") + kWordIn + "]";
    return at ? "(?:(?<=" + w + ")(?!" + w + ")|(?<!" + w + ")(?=" + w + "))"
              : "(?:(?<=" + w + ")(?=" + w + ")|(?<!" + w + ")(?!" + w + "))";
}

// Rewrite of one escape letter outside ([) or inside a class. Returns false for a form that has no
// in-class equivalent (negated shorthand inside a class).
bool rewrite_escape(char e, bool in_class, std::string& out) {
    switch (e) {
        case 'h':
            out += in_class ? "0-9a-fA-F" : "[0-9a-fA-F]";
            return true;
        case 's':
            out += in_class ? std::string(kSpaceIn) : std::string("[") + kSpaceIn + "]";
            return true;
        case 'w':
            out += in_class ? std::string(kWordIn) : std::string("[") + kWordIn + "]";
            return true;
        case 'H':
        case 'S':
        case 'W':
            if (in_class)
                return false;
            out += std::string("[^") + (e == 'H' ? "0-9a-fA-F" : e == 'S' ? kSpaceIn : kWordIn) + "]";
            return true;
        case 'b':
        case 'B':
            if (in_class)
                return false;
            out += word_boundary(e == 'b');
            return true;
        default:
            out += '\\';
            out += e;
            return true;
    }
}

// Class intersection `&&[^B]]` / `&&[B]]` closing the class opened at out[cls]: PCRE2 has no `&&`,
// so the one code point is tested by a lookahead, [A&&[^B]] -> (?:(?![B])[A]). i at the first '&';
// returns the index of the outer ']' or 0 for a form without that rewrite.
size_t rewrite_intersection(const std::string& rx, size_t i, size_t cls, std::string& out) {
    size_t j = i + 2;
    if (j >= rx.size() || rx[j] != '[')
        return 0;
    ++j;
    const bool neg = j < rx.size() && rx[j] == '^';
    j += neg ? 1 : 0;
    std::string inner = "[";
    if (j < rx.size() && rx[j] == ']')
        inner += rx[j++];  // a leading ']' is literal
    for (; j < rx.size() && rx[j] != ']'; ++j) {
        if (rx[j] == '[' || (rx[j] == '&' && j + 1 < rx.size() && rx[j + 1] == '&'))
            return 0;
        if (rx[j] == '\\' && j + 1 < rx.size()) {
            if (!rewrite_escape(rx[++j], true, inner))
                return 0;
            continue;
        }
        inner += rx[j];
    }
    if (j + 1 >= rx.size() || rx[j + 1] != ']')
        return 0;
    const std::string outer = out.substr(cls) + "]";
    out.resize(cls);
    out += std::string("(?:(?") + (neg ? "!" : "=") + inner + "])" + outer + ")";
    return j + 1;
}

}  // namespace

std::string onig_to_pcre2(const std::string& rx) {
    std::string out;
    out.reserve(rx.size() + 32);
    int depth = 0;   // class nesting; Oniguruma nests, PCRE2 does not
    size_t cls = 0;  // out index of the open class's '['
    bool cls_neg = false;
    for (size_t i = 0; i < rx.size(); ++i) {
        const char c = rx[i];
        if (c == '\\' && i + 1 < rx.size()) {
            if (!rewrite_escape(rx[i + 1], depth > 0, out))
                return {};
            ++i;
            continue;
        }
        if (c == '[') {
            if (depth > 0)
                return {};  // nested class
            ++depth;
            cls = out.size();
            out += c;
            cls_neg = i + 1 < rx.size() && rx[i + 1] == '^';
            if (cls_neg)
                out += rx[++i];
            if (i + 1 < rx.size() && rx[i + 1] == ']')
                out += rx[++i];  // a leading ']' is literal
            continue;
        }
        if (c == ']' && depth > 0)
            --depth;
        else if (c == '&' && depth > 0 && i + 1 < rx.size() && rx[i + 1] == '&') {
            const size_t end = cls_neg ? 0 : rewrite_intersection(rx, i, cls, out);
            if (end == 0)
                return {};
            i = end;
            depth = 0;
            continue;
        }
        out += c;
    }
    return out;
}

namespace {

std::shared_ptr<pcre2_code> compile(const std::string& pattern, bool literal, std::string* why) {
    int err = 0;
    PCRE2_SIZE off = 0;
    // PCRE2_LITERAL refuses UCP; Ruby ^ and $ are line anchors (PCRE2_MULTILINE).
    const uint32_t opts = PCRE2_UTF | (literal ? PCRE2_LITERAL : PCRE2_UCP | PCRE2_MULTILINE);
    pcre2_code* code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), opts, &err,
                                     &off, nullptr);
    if (!code) {
        PCRE2_UCHAR msg[160];
        pcre2_get_error_message(err, msg, sizeof(msg));
        if (why)
            *why = "regex does not compile at " + std::to_string(off) + ": " + reinterpret_cast<char*>(msg);
        return nullptr;
    }
    return {code, pcre2_code_free};
}

bool regex_step(const std::string& onig, Behavior b, bool invert, std::vector<Step>& out, std::string* why) {
    const std::string rx = onig_to_pcre2(onig);
    if (rx.empty()) {
        if (why)
            *why = "regex form without a PCRE2 equivalent: " + onig;
        return false;
    }
    auto code = compile(rx, false, why);
    if (!code)
        return false;
    out.push_back({std::move(code), b, invert});
    return true;
}

bool parse_behavior(const JValue& pt, Behavior def, Behavior& b) {
    std::string s;
    if (!jobj_get_string(pt, "behavior", s)) {
        b = def;
        return true;
    }
    static const std::pair<const char*, Behavior> kNames[] = {
        {"Removed", Behavior::kRemoved},
        {"Isolated", Behavior::kIsolated},
        {"MergedWithPrevious", Behavior::kMergedWithPrevious},
        {"MergedWithNext", Behavior::kMergedWithNext},
        {"Contiguous", Behavior::kContiguous},
    };
    for (const auto& [name, v] : kNames)
        if (s == name) {
            b = v;
            return true;
        }
    return false;
}

bool json_true(const JValue* v, bool def) { return v && v->type == JType::NUMBER ? v->num_val != 0.0 : def; }

bool add_split(const JValue& pt, std::vector<Step>& out, std::string* why) {
    Behavior b;
    if (!parse_behavior(pt, Behavior::kIsolated, b)) {
        if (why)
            *why = "unknown Split behavior";
        return false;
    }
    const bool invert = json_true(jobj_find(pt, "invert"), false);
    const JValue* pat = jobj_find(pt, "pattern");
    std::string s;
    if (pat && pat->type == JType::OBJECT && jobj_get_string(*pat, "Regex", s))
        return regex_step(s, b, invert, out, why);
    if (pat && pat->type == JType::OBJECT && jobj_get_string(*pat, "String", s)) {
        auto code = compile(s, true, why);
        if (!code)
            return false;
        out.push_back({std::move(code), b, invert});
        return true;
    }
    if (why)
        *why = "Split without a Regex or String pattern";
    return false;
}

// HF char-predicate pre-tokenizers as single-codepoint regex steps (each match is one char).
bool add_char_step(const std::string& type, const JValue& pt, std::vector<Step>& out, std::string* why) {
    if (type == "Digits")
        return regex_step(R"(\p{N})",
                          json_true(jobj_find(pt, "individual_digits"), false) ? Behavior::kIsolated
                                                                               : Behavior::kContiguous,
                          false, out, why);
    if (type == "Punctuation") {
        Behavior b;
        if (!parse_behavior(pt, Behavior::kIsolated, b))
            return false;
        // char::is_ascii_punctuation || char::is_punctuation (Unicode P*)
        return regex_step(R"([\p{P}!-\/:-@\[-`{-~])", b, false, out, why);
    }
    if (type == "Whitespace")
        return regex_step(R"(\w+|[^\w\s]+)", Behavior::kRemoved, true, out, why);
    if (type == "WhitespaceSplit")
        return regex_step(R"(\s)", Behavior::kRemoved, false, out, why);
    std::string delim;
    if (type == "CharDelimiterSplit" && jobj_get_string(pt, "delimiter", delim)) {
        auto code = compile(delim, true, why);
        if (!code)
            return false;
        out.push_back({std::move(code), Behavior::kRemoved, false});
        return true;
    }
    if (why)
        *why = "pre-tokenizer type not covered: " + type;
    return false;
}

bool add_steps(const JValue& pt, std::vector<Step>& out, std::string* why, int depth) {
    std::string type;
    if (pt.type != JType::OBJECT || !jobj_get_string(pt, "type", type) || depth > 8) {
        if (why)
            *why = "malformed pre_tokenizer";
        return false;
    }
    if (type == "Sequence") {
        const JValue* list = jobj_find(pt, "pretokenizers");
        if (!list || list->type != JType::ARRAY)
            return false;
        for (const JValue& inner : list->arr)
            if (!add_steps(inner, out, why, depth + 1))
                return false;
        return true;
    }
    if (type == "Split")
        return add_split(pt, out, why);
    if (type == "ByteLevel")
        return !json_true(jobj_find(pt, "use_regex"), true) ||
               regex_step(kGpt2PretokRegex, Behavior::kIsolated, false, out, why);
    return add_char_step(type, pt, out, why);
}

struct Span {
    size_t begin, end;
    bool flag;  // find_matches: is a match; after the behavior: remove
};

size_t utf8_len_at(const std::string& s, size_t i) {
    const auto c = static_cast<unsigned char>(s[i]);
    const size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    return i + n <= s.size() ? n : 1;
}

// pattern.rs find_matches over onig FindMatches: an empty match right after the previous match is
// skipped one char on. A matcher error (invalid UTF-8) ends the scan: the rest stays one piece.
void find_matches(const pcre2_code* code, pcre2_match_data* md, const std::string& p,
                  std::vector<Span>& out) {
    out.clear();
    const auto* subj = reinterpret_cast<PCRE2_SPTR>(p.data());
    size_t last_end = 0, prev = 0, last_match = 0;
    bool has_last = false;
    // pcre2_match validates the whole subject's UTF-8 on every call: once per piece, not per match
    // (per match made the 45 KB corpus quadratic, 2.3 -> 68 ms).
    uint32_t opts = 0;
    while (last_end <= p.size()) {
        if (pcre2_match(code, subj, p.size(), last_end, opts, md, nullptr) < 0)
            break;
        opts = PCRE2_NO_UTF_CHECK;
        const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
        const size_t s = ov[0], e = ov[1];
        if (s == e && has_last && s == last_match) {
            if (last_end >= p.size())
                break;
            last_end += utf8_len_at(p, last_end);
            continue;
        }
        last_end = e;
        last_match = e;
        has_last = true;
        if (prev != s)
            out.push_back({prev, s, false});
        out.push_back({s, e, true});
        prev = e;
    }
    if (prev != p.size())
        out.push_back({prev, p.size(), false});
}

// normalizer.rs split: the behavior turns (offsets, is_match) into (offsets, remove).
void apply_behavior(Behavior b, std::vector<Span>& m, std::vector<Span>& out) {
    out.clear();
    if (b == Behavior::kIsolated || b == Behavior::kRemoved) {
        for (Span s : m)
            out.push_back({s.begin, s.end, b == Behavior::kRemoved && s.flag});
        return;
    }
    if (b == Behavior::kMergedWithNext) {
        bool prev_match = false;
        for (size_t k = m.size(); k-- > 0;) {
            if (m[k].flag && !prev_match && !out.empty())
                out.back().begin = m[k].begin;
            else
                out.push_back({m[k].begin, m[k].end, false});
            prev_match = m[k].flag;
        }
        std::reverse(out.begin(), out.end());
        return;
    }
    bool prev_match = false;  // kMergedWithPrevious, kContiguous
    for (Span s : m) {
        const bool merge = b == Behavior::kContiguous ? s.flag == prev_match : s.flag && !prev_match;
        if (merge && !out.empty())
            out.back().end = s.end;
        else
            out.push_back({s.begin, s.end, false});
        prev_match = s.flag;
    }
}

}  // namespace

// tokenizer.json Split regex of the HF source each GGUF name stands for; any other name keeps the
// GPT-2 ByteLevel regex, the fallback before the table existed.
std::vector<std::string> gguf_pre_regexes(const std::string& name) {
    static const char* const kLlama3 =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const char* const kQwen2 =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const char* const kQwen35 =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const char* const kO200k =
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const char* const kNemotron =
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const std::pair<const char*, const char*> kNames[] = {
        {"llama3", kLlama3}, {"llama-v3", kLlama3},        {"llama-bpe", kLlama3}, {"cl100k", kLlama3},
        {"qwen2", kQwen2},   {"deepseek-r1-qwen", kQwen2}, {"qwen35", kQwen35},    {"o200k", kO200k},
        {"gpt-4o", kO200k},  {"nemotron", kNemotron},      {"tekken", kNemotron},
    };
    for (const auto& [n, rx] : kNames)
        if (name == n)
            return {rx};
    return {kGpt2PretokRegex};
}

RegexPretokenizer::RegexPretokenizer() = default;
RegexPretokenizer::~RegexPretokenizer() = default;

size_t RegexPretokenizer::n_steps() const { return steps_.size(); }

std::shared_ptr<const RegexPretokenizer> RegexPretokenizer::from_json(const JValue& pre_tok,
                                                                      std::string* why) {
    auto pt = std::make_shared<RegexPretokenizer>();
    if (!add_steps(pre_tok, pt->steps_, why, 0))
        return nullptr;
    return pt;
}

std::shared_ptr<const RegexPretokenizer> RegexPretokenizer::from_regexes(
    const std::vector<std::string>& regexes, std::string* why) {
    auto pt = std::make_shared<RegexPretokenizer>();
    for (const auto& rx : regexes)
        if (!regex_step(rx, Behavior::kIsolated, false, pt->steps_, why))
            return nullptr;
    return regexes.empty() ? nullptr : pt;
}

std::vector<std::string> RegexPretokenizer::split(const std::string& text) const {
    std::vector<std::string> pieces, next;
    if (text.empty())
        return pieces;
    pieces.push_back(text);
    std::unique_ptr<pcre2_match_data, void (*)(pcre2_match_data*)> md(pcre2_match_data_create(1, nullptr),
                                                                      pcre2_match_data_free);
    std::vector<Span> matches, spans;
    for (const Step& st : steps_) {
        next.clear();
        for (const std::string& p : pieces) {
            find_matches(st.code.get(), md.get(), p, matches);
            if (st.invert)
                for (Span& s : matches)
                    s.flag = !s.flag;
            apply_behavior(st.behavior, matches, spans);
            for (const Span& s : spans)
                if (!s.flag && s.end > s.begin)
                    next.push_back(p.substr(s.begin, s.end - s.begin));
        }
        pieces.swap(next);
    }
    return pieces;
}

}  // namespace imp
