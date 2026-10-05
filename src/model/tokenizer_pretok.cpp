// Regex pre-tokenizers of the byte-level BPE families, scanned by hand (no Unicode regex engine).
// One scanner, parameterized by PretokSpec; classes come from unicode_class.h (#657).
#include "model/tokenizer.h"
#include "model/unicode_class.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imp {

namespace {

using unicode::cp_class;
using unicode::decode_utf8_at;

enum class Contractions { kNone, kPrefix, kSuffix };

// The alternatives each family's tokenizer.json regex is built from:
//   contractions  (?i:'s|'t|'re|'ve|'m|'ll|'d)   own alternative (kPrefix) or a letter-run suffix
//   letters       [^\r\n\p{L}\p{N}]?\p{L}+        marks=true: [\p{L}\p{M}]+; cased: o200k's
//                 [^\r\n\p{L}\p{N}]?[Lu Lt Lm Lo M]*[Ll Lm Lo M]+ | ...[Lu Lt Lm Lo M]+[Ll Lm Lo M]*
//   digits        \p{N}{1,max_digits}
//   symbols        ?[^\s\p{L}\p{N}]+[\r\n]*       marks=true also excludes \p{M}; slash=true [\r\n/]*
//   whitespace    \s*[\r\n]+ | \s+(?!\S) | \s+
struct PretokSpec {
    Contractions contractions;
    bool cased;
    bool marks;
    int max_digits;
    bool slash;
};

class Scanner {
public:
    Scanner(const std::string& text, const PretokSpec& spec) : t_(text), n_(text.size()), spec_(spec) {}

    std::vector<std::string> run() {
        std::vector<std::string> out;
        size_t i = 0;
        while (i < n_) {
            size_t end = match_at(i);
            out.push_back(t_.substr(i, end - i));
            i = end;
        }
        return out;
    }

private:
    const std::string& t_;
    const size_t n_;
    const PretokSpec spec_;

    static constexpr size_t kNone = std::string::npos;

    uint8_t cls(size_t k, size_t* len) const { return cp_class(decode_utf8_at(t_, k, len)); }
    size_t step(size_t k) const {
        size_t len;
        decode_utf8_at(t_, k, &len);
        return len;
    }
    bool is_nl(size_t k) const { return t_[k] == '\n' || t_[k] == '\r'; }
    bool is_space(size_t k) const {
        size_t len;
        return (cls(k, &len) & unicode::kSpace) != 0;
    }

    size_t contraction_end(size_t k) const {
        if (k + 1 >= n_ || t_[k] != '\'')
            return kNone;
        const char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(t_[k + 1])));
        const char c2 = k + 2 < n_ ? static_cast<char>(std::tolower(static_cast<unsigned char>(t_[k + 2])))
                                   : 0;
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
            return k + 3;
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
            return k + 2;
        return kNone;
    }

    // Greedy run of codepoints whose class intersects `mask`, from k.
    size_t scan(size_t k, uint8_t mask) const {
        size_t len;
        while (k < n_ && (cls(k, &len) & mask))
            k += len;
        return k;
    }

    // Letter body at k (no prefix), or kNone.
    size_t letter_body(size_t k) const {
        if (!spec_.cased) {
            const uint8_t m = unicode::kLetter | (spec_.marks ? unicode::kMark : 0);
            const size_t e = scan(k, m);
            return e > k ? e : kNone;
        }
        constexpr uint8_t kUp = unicode::kUpper | unicode::kOtherLetter | unicode::kMark;
        constexpr uint8_t kLo = unicode::kLower | unicode::kOtherLetter | unicode::kMark;
        size_t end = kNone;
        // Alt 1: up* lo+. Greedy up*, then lo+; if lo+ is empty, up* backtracks to its last
        // codepoint that is also in lo (Lm, Lo, M), which then ends the match.
        const size_t u = scan(k, kUp);
        const size_t l = scan(u, kLo);
        if (l > u) {
            end = l;
        } else {
            size_t len;
            for (size_t p = k; p < u; p += len)
                if (cls(p, &len) & kLo)
                    end = p + len;
        }
        // Alt 2: up+ lo* (only reached when alt 1 fails).
        if (end == kNone && u > k)
            end = u;
        if (end != kNone && spec_.contractions == Contractions::kSuffix) {
            const size_t c = contraction_end(end);
            if (c != kNone)
                end = c;
        }
        return end;
    }

    // [^\r\n\p{L}\p{N}]? + letter body; the optional prefix is tried first, then without it.
    size_t letters(size_t i) const {
        size_t len;
        const uint8_t c = cls(i, &len);
        if (!(c & (unicode::kLetter | unicode::kNumber)) && !is_nl(i) && i + len < n_) {
            const size_t e = letter_body(i + len);
            if (e != kNone)
                return e;
        }
        return letter_body(i);
    }

    size_t digits(size_t i) const {
        size_t k = i, len;
        for (int d = 0; d < spec_.max_digits && k < n_ && (cls(k, &len) & unicode::kNumber); d++)
            k += len;
        return k > i ? k : kNone;
    }

    size_t symbols(size_t i) const {
        const uint8_t stop = unicode::kSpace | unicode::kLetter | unicode::kNumber |
                             (spec_.marks ? unicode::kMark : 0);
        const size_t j = i + (t_[i] == ' ' ? 1 : 0);
        size_t k = j, len;
        while (k < n_ && !(cls(k, &len) & stop))
            k += len;
        if (k == j)
            return kNone;
        while (k < n_ && (is_nl(k) || (spec_.slash && t_[k] == '/')))
            k++;
        return k;
    }

    size_t whitespace(size_t i) const {
        size_t k = i, last_nl_end = kNone, last_start = i;
        while (k < n_ && is_space(k)) {
            last_start = k;
            if (is_nl(k))
                last_nl_end = k + 1;
            k += step(k);
        }
        if (k == i)
            return kNone;
        if (last_nl_end != kNone)
            return last_nl_end;  // \s*[\r\n]+: up to the last newline of the run
        if (k >= n_ || last_start == i)
            return k;       // \s+(?!\S) at end of text, or a single \s
        return last_start;  // \s+(?!\S): leave the last codepoint to prefix the next chunk
    }

    size_t match_at(size_t i) const {
        if (spec_.contractions == Contractions::kPrefix)
            if (const size_t e = contraction_end(i); e != kNone)
                return e;
        if (const size_t e = letters(i); e != kNone)
            return e;
        if (const size_t e = digits(i); e != kNone)
            return e;
        if (const size_t e = symbols(i); e != kNone)
            return e;
        if (const size_t e = whitespace(i); e != kNone)
            return e;
        return i + step(i);  // unreachable: every codepoint is L, N, \s or a symbol
    }
};

std::vector<std::string> scan(const std::string& text, const PretokSpec& spec) {
    return Scanner(text, spec).run();
}

}  // namespace

std::vector<std::string> qwen2_pre_tokenize(const std::string& text) {
    return scan(text, {Contractions::kPrefix, false, false, 1, false});
}

std::vector<std::string> qwen35_pre_tokenize(const std::string& text) {
    return scan(text, {Contractions::kPrefix, false, true, 1, false});
}

std::vector<std::string> cl100k_pre_tokenize(const std::string& text) {
    return scan(text, {Contractions::kPrefix, false, false, 3, false});
}

std::vector<std::string> o200k_pre_tokenize(const std::string& text) {
    return scan(text, {Contractions::kSuffix, true, false, 3, true});
}

std::vector<std::string> nemotron_pre_tokenize(const std::string& text) {
    return scan(text, {Contractions::kNone, true, false, 1, true});
}

// ---- Split sequence (DeepSeek-V2): each step re-splits every piece, matches isolated ----

struct SplitSequence {
    enum class Kind { kClass, kTrailingSpace, kDigits };
    struct Step {
        Kind kind;
        bool opt_space = false;                             // leading \s?
        bool plus = false;                                  // class+ (else one codepoint)
        bool individual = false;                            // Digits: one piece per digit
        std::vector<std::pair<uint32_t, uint32_t>> ranges;  // sorted, inclusive
    };
    std::vector<Step> steps;
};

namespace {

using Step = SplitSequence::Step;
using Kind = SplitSequence::Kind;

// "[...]" with literal codepoints, ranges and \r \n \t \\ \- \[ \] escapes; no negation, no classes.
bool parse_class(const std::string& rx, size_t& i, std::vector<std::pair<uint32_t, uint32_t>>& out) {
    if (i >= rx.size() || rx[i] != '[' || (i + 1 < rx.size() && rx[i + 1] == '^'))
        return false;
    i++;
    auto next_cp = [&](uint32_t& cp) {
        size_t len;
        if (rx[i] == '\\') {
            if (i + 1 >= rx.size())
                return false;
            const char e = rx[i + 1];
            cp = e == 'r' ? '\r' : e == 'n' ? '\n' : e == 't' ? '\t' : static_cast<unsigned char>(e);
            if (!(e == 'r' || e == 'n' || e == 't' || e == '\\' || e == '-' || e == '[' || e == ']'))
                return false;
            i += 2;
            return true;
        }
        cp = decode_utf8_at(rx, i, &len);
        i += len;
        return true;
    };
    while (i < rx.size() && rx[i] != ']') {
        uint32_t lo, hi;
        if (!next_cp(lo))
            return false;
        hi = lo;
        if (i + 1 < rx.size() && rx[i] == '-' && rx[i + 1] != ']') {
            i++;
            if (!next_cp(hi) || hi < lo)
                return false;
        }
        out.emplace_back(lo, hi);
    }
    if (i >= rx.size() || out.empty())
        return false;
    i++;  // ']'
    std::sort(out.begin(), out.end());
    return true;
}

bool parse_step(const std::string& spec, Step& st) {
    if (spec.rfind("digits:", 0) == 0) {
        st.kind = Kind::kDigits;
        st.individual = spec == "digits:1";
        return true;
    }
    if (spec.rfind("re:", 0) != 0)
        return false;
    const std::string rx = spec.substr(3);
    if (rx == "\\s+$") {
        st.kind = Kind::kTrailingSpace;
        return true;
    }
    st.kind = Kind::kClass;
    size_t i = 0;
    if (rx.rfind("\\s?", 0) == 0) {
        st.opt_space = true;
        i = 3;
    }
    if (!parse_class(rx, i, st.ranges))
        return false;
    if (i < rx.size() && rx[i] == '+') {
        st.plus = true;
        i++;
    }
    return i == rx.size();
}

bool in_ranges(const Step& st, uint32_t cp) {
    auto it = std::upper_bound(st.ranges.begin(), st.ranges.end(), std::make_pair(cp, UINT32_MAX));
    return it != st.ranges.begin() && cp <= (it - 1)->second;
}

bool space_at(const std::string& s, size_t k, size_t* len) {
    return (cp_class(decode_utf8_at(s, k, len)) & unicode::kSpace) != 0;
}

constexpr size_t kNoMatch = std::string::npos;

size_t step_len(const std::string& t, size_t k) {
    size_t len;
    decode_utf8_at(t, k, &len);
    return len;
}

// Leftmost-first match [start, end) at or after `from`, or false.
bool find_match(const Step& st, const std::string& t, size_t from, size_t& start, size_t& end) {
    const size_t n = t.size();
    for (size_t i = from, len; i < n; i += len) {
        const uint32_t cp = decode_utf8_at(t, i, &len);
        if (st.kind == Kind::kDigits) {
            if (!(cp_class(cp) & unicode::kNumber))
                continue;
            start = i;
            end = i + len;
            size_t l2;
            while (!st.individual && end < n && (cp_class(decode_utf8_at(t, end, &l2)) & unicode::kNumber))
                end += l2;
            return true;
        }
        if (st.kind == Kind::kTrailingSpace) {
            // \s+$ with Ruby $ (end of text or before "\n"): the longest such run from i.
            size_t k = i, best = kNoMatch, l2;
            while (k < n && space_at(t, k, &l2)) {
                k += l2;
                if (k == n || t[k] == '\n')
                    best = k;
            }
            if (best == kNoMatch) {
                if (k > i)
                    len = k - i;  // every start inside this run fails the same way
                continue;
            }
            start = i;
            end = best;
            return true;
        }
        size_t j = i, l2;
        if (st.opt_space && (cp_class(cp) & unicode::kSpace) && i + len < n &&
            in_ranges(st, decode_utf8_at(t, i + len, &l2)))
            j = i + len;
        else if (!in_ranges(st, cp))
            continue;
        start = i;
        end = j + step_len(t, j);
        while (st.plus && end < n && in_ranges(st, decode_utf8_at(t, end, &l2)))
            end += l2;
        return true;
    }
    return false;
}

}  // namespace

std::shared_ptr<const SplitSequence> compile_split_sequence(const std::vector<std::string>& steps) {
    auto seq = std::make_shared<SplitSequence>();
    for (const auto& s : steps) {
        Step st;
        if (!parse_step(s, st))
            return nullptr;
        seq->steps.push_back(std::move(st));
    }
    return seq->steps.empty() ? nullptr : seq;
}

std::vector<std::string> split_sequence_pre_tokenize(const SplitSequence& seq, const std::string& text) {
    std::vector<std::string> pieces{text}, next;
    for (const Step& st : seq.steps) {
        next.clear();
        for (const std::string& p : pieces) {
            size_t last = 0, start, end;
            while (find_match(st, p, last, start, end)) {
                if (start > last)
                    next.push_back(p.substr(last, start - last));
                next.push_back(p.substr(start, end - start));
                last = end;
            }
            if (last < p.size())
                next.push_back(p.substr(last));
        }
        pieces.swap(next);
    }
    return pieces;
}

std::vector<std::string> digit_triples_right_split(const std::string& text) {
    // \w as Oniguruma's \b sees it (HF tokenizers): L, M, every N (2½ stays whole), Pc.
    const auto is_word = [](uint32_t cp) {
        constexpr uint32_t kPc[] = {'_',    0x203F, 0x2040, 0x2054, 0xFE33,
                                    0xFE34, 0xFE4D, 0xFE4E, 0xFE4F, 0xFF3F};
        return (cp_class(cp) & (unicode::kLetter | unicode::kMark | unicode::kNumber)) != 0 ||
               std::find(std::begin(kPc), std::end(kPc), cp) != std::end(kPc);
    };
    std::vector<std::string> out;
    size_t last = 0, k = 0, len = 0;
    while (k < text.size()) {
        if (!(cp_class(decode_utf8_at(text, k, &len)) & unicode::kDecimal)) {
            k += len;
            continue;
        }
        std::vector<size_t> starts;  // codepoint offsets of the \d run
        size_t e = k;
        while (e < text.size() && (cp_class(decode_utf8_at(text, e, &len)) & unicode::kDecimal)) {
            starts.push_back(e);
            e += len;
        }
        if (e < text.size() && is_word(decode_utf8_at(text, e, &len))) {
            k = e;  // no \b after the run: no match anywhere inside it
            continue;
        }
        if (k > last)
            out.push_back(text.substr(last, k - last));
        const size_t n = starts.size();
        for (size_t g = (n % 3 == 0 ? 3 : n % 3); k < e; g += 3) {
            const size_t end = g < n ? starts[g] : e;
            out.push_back(text.substr(k, end - k));
            k = end;
        }
        last = e;
    }
    if (last < text.size())
        out.push_back(text.substr(last));
    return out;
}

}  // namespace imp
