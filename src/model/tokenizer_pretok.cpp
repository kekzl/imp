// Regex pre-tokenizers of the byte-level BPE families, scanned by hand (no Unicode regex engine).
// One scanner, parameterized by PretokSpec; classes come from unicode_class.h (#657).
#include "model/tokenizer.h"
#include "model/unicode_class.h"

#include <cctype>
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
        size_t e;
        if (spec_.contractions == Contractions::kPrefix && (e = contraction_end(i)) != kNone)
            return e;
        if ((e = letters(i)) != kNone)
            return e;
        if ((e = digits(i)) != kNone)
            return e;
        if ((e = symbols(i)) != kNone)
            return e;
        if ((e = whitespace(i)) != kNone)
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

}  // namespace imp
