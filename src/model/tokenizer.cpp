#include "model/tokenizer.h"
#include "core/logging.h"
#include "model/json_util.h"  // shared JValue/JsonParser/jobj_find (was duplicated here)
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <climits>
#include <cstring>
#include <queue>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace imp {

// ---- UTF-8 helpers ----

int Tokenizer::utf8_char_len(uint8_t c) {
    if ((c & 0x80) == 0)
        return 1;
    if ((c & 0xE0) == 0xC0)
        return 2;
    if ((c & 0xF0) == 0xE0)
        return 3;
    if ((c & 0xF8) == 0xF0)
        return 4;
    return 1;  // invalid byte, treat as single
}

static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0x10FFFF) {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// ---- GPT2 byte-level encoding tables ----
//
// GPT2 maps each byte (0-255) to a unique Unicode codepoint:
// - Printable ASCII (33-126): identity mapping
// - Latin-1 supplement (161-172, 174-255): identity mapping
// - All other bytes (0-32, 127-160, 173): mapped to 256+ range
//
// This ensures every byte has a visible Unicode representation.

static const uint32_t BYTE_TO_CODEPOINT[256] = {
    // 0-32: mapped to 256-288
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    285,
    286,
    287,
    288,
    // 33-126: identity (! to ~)
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    // 127-160: mapped to 289-322
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
    299,
    300,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    318,
    319,
    320,
    321,
    322,
    // 161-172: identity
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    // 173: mapped to 323
    323,
    // 174-255: identity
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
};

// Reverse mapping: codepoint -> byte value (built once)
static uint8_t CODEPOINT_TO_BYTE[324];
static bool CODEPOINT_TABLE_INIT = false;

static void init_codepoint_table() {
    if (CODEPOINT_TABLE_INIT)
        return;
    for (int b = 0; b < 256; b++) {
        CODEPOINT_TO_BYTE[BYTE_TO_CODEPOINT[b]] = static_cast<uint8_t>(b);
    }
    CODEPOINT_TABLE_INIT = true;
}

// Convert a single byte to its GPT2 Unicode character (UTF-8 encoded)
static std::string byte_to_gpt2(uint8_t byte) { return codepoint_to_utf8(BYTE_TO_CODEPOINT[byte]); }

// Convert a UTF-8 character (from GPT2 encoding) back to the original byte
// Returns -1 if not a valid GPT2 byte-encoded character
static int gpt2_to_byte(const char* s, int len) {
    init_codepoint_table();
    uint32_t cp = 0;
    if (len == 1) {
        cp = static_cast<uint8_t>(s[0]);
    } else if (len == 2) {
        cp = ((static_cast<uint32_t>(s[0]) & 0x1F) << 6) | (static_cast<uint32_t>(s[1]) & 0x3F);
    } else if (len == 3) {
        cp = ((static_cast<uint32_t>(s[0]) & 0x0F) << 12) | ((static_cast<uint32_t>(s[1]) & 0x3F) << 6) |
             (static_cast<uint32_t>(s[2]) & 0x3F);
    } else {
        return -1;
    }
    if (cp < 324) {
        return CODEPOINT_TO_BYTE[cp];
    }
    return -1;
}

// ---- GPT2 pre-tokenization ----
//
// Splits input text into chunks before applying BPE to each independently.
// This is a simplified version of the cl100k_base / Qwen2 pre-tokenizer.
// Key rules:
// - Spaces attach to the following word
// - Letter sequences form chunks
// - Digit sequences (up to 3) form chunks
// - Individual punctuation chars form chunks
// - Newlines group together

static std::vector<std::string> gpt2_pre_tokenize(const std::string& text) {
    std::vector<std::string> result;
    if (text.empty())
        return result;

    size_t i = 0;
    while (i < text.size()) {
        std::string chunk;

        // Collect leading spaces/tabs (attach to next non-whitespace chunk)
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) {
            chunk += text[i++];
        }

        if (i >= text.size()) {
            if (!chunk.empty())
                result.push_back(chunk);
            break;
        }

        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c == '\n' || c == '\r') {
            // Newlines: collect consecutive newlines
            while (i < text.size() && (text[i] == '\n' || text[i] == '\r')) {
                chunk += text[i++];
            }
        } else if (std::isalpha(c) || c >= 128) {
            // Letters (ASCII + multi-byte UTF-8 treated as letters)
            while (i < text.size()) {
                unsigned char cc = static_cast<unsigned char>(text[i]);
                if (std::isalpha(cc) || cc >= 128) {
                    int len = 1;
                    if ((cc & 0xE0) == 0xC0)
                        len = 2;
                    else if ((cc & 0xF0) == 0xE0)
                        len = 3;
                    else if ((cc & 0xF8) == 0xF0)
                        len = 4;
                    for (int j = 0; j < len && i < text.size(); j++)
                        chunk += text[i++];
                } else {
                    break;
                }
            }
        } else if (std::isdigit(c)) {
            // Digits: groups of up to 3
            int count = 0;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])) && count < 3) {
                chunk += text[i++];
                count++;
            }
        } else {
            // Punctuation/other: single character
            chunk += text[i++];
        }

        if (!chunk.empty()) {
            result.push_back(chunk);
        }
    }

    return result;
}

// Lightweight non-ASCII classifier for the regex-faithful pre-tokenizers:
// returns true when the UTF-8 sequence starting at text[i] is a PUNCTUATION/
// SYMBOL codepoint (NOT \p{L}/\p{N}) — common typographic and technical
// blocks only. Everything else ≥0x80 keeps the letter approximation. Without
// this, " →\n" / " —\n" took the letter rule (no trailing-newline absorption)
// instead of the symbol run and diverged from canonical segmentation (#657).
static bool utf8_punct_symbol(const std::string& text, size_t i) {
    const unsigned char c0 = static_cast<unsigned char>(text[i]);
    uint32_t cp = 0;
    if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
        cp = ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
    } else if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
        cp = ((c0 & 0x0F) << 12) | ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(text[i + 2]) & 0x3F);
    } else {
        return false;  // 4-byte (emoji etc.) and malformed: keep letter approx
    }
    // Latin-1 punctuation/symbols: ¡ « ° ± § © … (excluding letters À-ÿ)
    if ((cp >= 0xA0 && cp <= 0xBF) || cp == 0xD7 || cp == 0xF7)
        return true;
    // General punctuation … bullets, dashes, quotes (U+2000-206F), super/sub-
    // scripts, currency, letterlike, arrows, math, technical, box drawing,
    // geometric shapes, misc symbols, dingbats (… U+2BFF).
    if (cp >= 0x2000 && cp <= 0x2BFF)
        return true;
    // CJK punctuation 。、「」 (U+3000-303F).
    if (cp >= 0x3000 && cp <= 0x303F)
        return true;
    return false;
}

// ---- Qwen2 pre-tokenization ----
//
// Faithful hand-rolled scan of the Qwen2 pre-tokenizer regex (the canonical
// segmentation Qwen2/Qwen3 were trained with; llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2):
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)            contractions
//   | [^\r\n\p{L}\p{N}]?\p{L}+              one optional prefix char + letter run
//   | \p{N}                                 SINGLE digit
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*             optional space + SYMBOL RUN + newlines
//   | \s*[\r\n]+                            whitespace ending in newlines
//   | \s+(?!\S)                             trailing whitespace
//   | \s+                                   other whitespace
//
// The previous routing sent qwen2 through gpt2_pre_tokenize, whose
// "punctuation = single character" rule makes cross-symbol BPE merges
// impossible ("->" became "-", ">"; "(x):" four chunks) and whose
// digit-groups-of-3 rule is non-canonical for Qwen (single digits). On
// code/markdown text that produced ~20% more tokens than canonical
// (llama.cpp control: 3084 vs 3690 on a 10 KB corpus) and +70% teacher-forced
// NLL on matched text (#657) — and every production prompt containing code
// was segmented non-canonically. \p{L} is approximated as ASCII isalpha() or
// any non-ASCII byte (same approximation as the other pre-tokenizers here).
// Shared scanner for the qwen2/cl100k regex family — identical except for the
// digit rule: qwen2 matches single digits (\p{N}), cl100k groups up to three
// (\p{N}{1,3}; Phi-4 and the GPT-4/tiktoken cl100k lineage).
static std::vector<std::string> qwen2_like_pre_tokenize(const std::string& text, int max_digit_run) {
    std::vector<std::string> result;
    const size_t n = text.size();
    auto utf8_step = [&](size_t k) -> size_t {
        const unsigned char ck = static_cast<unsigned char>(text[k]);
        size_t len = 1;
        if ((ck & 0xE0) == 0xC0)
            len = 2;
        else if ((ck & 0xF0) == 0xE0)
            len = 3;
        else if ((ck & 0xF8) == 0xF0)
            len = 4;
        return (len <= n - k) ? len : 1;
    };
    // Position-aware \p{L} approximation: ASCII alpha, or a non-ASCII
    // codepoint that is not a known punctuation/symbol block (→ — “ etc.).
    auto is_letter_at = [&](size_t k) {
        const unsigned char ck = static_cast<unsigned char>(text[k]);
        if (ck < 128)
            return std::isalpha(ck) != 0;
        return !utf8_punct_symbol(text, k);
    };
    auto is_dig = [](unsigned char c) { return std::isdigit(c) != 0; };
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    auto is_nl = [](unsigned char c) { return c == '\n' || c == '\r'; };

    size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(text[i]);

        // 1. Contractions 's 't 're 've 'm 'll 'd (case-insensitive).
        if (c == '\'' && i + 1 < n) {
            const char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[i + 1])));
            const char c2 =
                (i + 2 < n) ? static_cast<char>(std::tolower(static_cast<unsigned char>(text[i + 2]))) : 0;
            size_t clen = 0;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
                clen = 2;
            else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
                clen = 3;
            if (clen > 0) {
                result.push_back(text.substr(i, clen));
                i += clen;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+ — one optional non-letter/digit/newline
        //    prefix CODEPOINT (space, tab, or punctuation) glued to a letter run.
        {
            size_t j = i;
            if (!is_letter_at(i) && !is_dig(c) && !is_nl(c))
                j = i + utf8_step(i);  // candidate prefix codepoint
            if (j < n && is_letter_at(j)) {
                size_t k = j;
                while (k < n && is_letter_at(k))
                    k += utf8_step(k);
                result.push_back(text.substr(i, k - i));
                i = k;
                continue;
            }
        }

        // 3. Digit rule: \p{N} (qwen2) or \p{N}{1,3} (cl100k).
        if (is_dig(c)) {
            size_t k = i;
            while (k < n && k - i < static_cast<size_t>(max_digit_run) &&
                   is_dig(static_cast<unsigned char>(text[k])))
                k++;
            result.push_back(text.substr(i, k - i));
            i = k;
            continue;
        }

        // 4. ' '?[^\s\p{L}\p{N}]+[\r\n]* — optional space + symbol run + newlines.
        {
            size_t j = i + (text[i] == ' ' ? 1 : 0);
            size_t k = j;
            while (k < n) {
                const unsigned char ck = static_cast<unsigned char>(text[k]);
                if (is_ws(ck) || is_dig(ck) || is_letter_at(k))
                    break;
                k += utf8_step(k);  // ASCII symbol or non-ASCII punct/symbol cp
            }
            if (k > j) {
                while (k < n && is_nl(static_cast<unsigned char>(text[k])))
                    k++;
                result.push_back(text.substr(i, k - i));
                i = k;
                continue;
            }
        }

        // 5.-7. Whitespace rules.
        if (is_ws(c)) {
            size_t k = i;
            size_t last_nl = std::string::npos;
            while (k < n && is_ws(static_cast<unsigned char>(text[k]))) {
                if (is_nl(static_cast<unsigned char>(text[k])))
                    last_nl = k;
                k++;
            }
            if (last_nl != std::string::npos) {
                // \s*[\r\n]+ — greedy up to the LAST newline in the run; any
                // trailing spaces/tabs stay for the next match (they become
                // the ' ?' / prefix of the following chunk).
                result.push_back(text.substr(i, last_nl + 1 - i));
                i = last_nl + 1;
                continue;
            }
            if (k >= n) {
                // \s+(?!\S) — trailing whitespace at end of text.
                result.push_back(text.substr(i, k - i));
                i = k;
                continue;
            }
            if (k - i > 1) {
                // \s+(?!\S) with backtracking: leave ONE space for the next
                // chunk ("    return" → ["   ", " return"]).
                result.push_back(text.substr(i, k - i - 1));
                i = k - 1;
                continue;
            }
            // \s+ — single whitespace char the letter/symbol rules didn't take
            // (e.g. the space in " 5": digits don't absorb a leading space).
            result.push_back(text.substr(i, 1));
            i += 1;
            continue;
        }

        // Unreachable in practice (every byte class is handled above).
        result.push_back(text.substr(i, 1));
        i += 1;
    }
    return result;
}

std::vector<std::string> qwen2_pre_tokenize(const std::string& text) {
    return qwen2_like_pre_tokenize(text, /*max_digit_run=*/1);
}

std::vector<std::string> cl100k_pre_tokenize(const std::string& text) {
    return qwen2_like_pre_tokenize(text, /*max_digit_run=*/3);
}

// ---- o200k pre-tokenization (gpt-oss / GPT-4o family) ----
//
// Faithful hand-rolled scan of the o200k_harmony pre-tokenizer regex
// (gpt-oss tokenizer.json; llama.cpp pre type "gpt-4o"):
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   | \p{N}{1,3}                          digit runs of up to THREE
//   |  ?[^\s\p{L}\p{N}]+[\r\n/]*          symbol run + trailing newlines/slashes
//   | \s*[\r\n]+ | \s+(?!\S) | \s+        whitespace (same as qwen2)
//
// Differences vs qwen2: CASE-AWARE letter runs (upper-run then lower-run, so
// "camelCase" splits "camel|Case" but "HTTPSession" stays whole), contractions
// attach as a SUFFIX ("don't" is one chunk), digits group 1-3 (not single),
// and the symbol-run trailing class additionally includes '/'. Verified
// chunk-by-chunk against HF tokenizers pre_tokenize_str on gpt-oss-20b.
// \p{Lu} is approximated as ASCII uppercase; non-ASCII bytes count as
// lowercase-class letters (both regex letter classes include Lm/Lo/M, so the
// union segmentation is unaffected).
std::vector<std::string> o200k_pre_tokenize(const std::string& text) {
    std::vector<std::string> result;
    const size_t n = text.size();
    auto is_upper = [](unsigned char c) { return c >= 'A' && c <= 'Z'; };
    // Position-aware lowercase-class approximation: ASCII a-z, or a non-ASCII
    // codepoint outside the known punctuation/symbol blocks.
    auto is_lower_at = [&](size_t k) {
        const unsigned char ck = static_cast<unsigned char>(text[k]);
        if (ck < 128)
            return ck >= 'a' && ck <= 'z';
        return !utf8_punct_symbol(text, k);
    };
    auto is_letter_at = [&](size_t k) {
        return is_upper(static_cast<unsigned char>(text[k])) || is_lower_at(k);
    };
    auto is_dig = [](unsigned char c) { return std::isdigit(c) != 0; };
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    auto is_nl = [](unsigned char c) { return c == '\n' || c == '\r'; };
    auto utf8_step = [&](size_t k) -> size_t {
        const unsigned char ck = static_cast<unsigned char>(text[k]);
        size_t len = 1;
        if ((ck & 0xE0) == 0xC0)
            len = 2;
        else if ((ck & 0xF0) == 0xE0)
            len = 3;
        else if ((ck & 0xF8) == 0xF0)
            len = 4;
        return (len <= n - k) ? len : 1;
    };
    // Optional contraction SUFFIX after a letter run: 's 't 'm 'd 're 've 'll.
    auto contraction_len = [&](size_t k) -> size_t {
        if (k >= n || text[k] != '\'' || k + 1 >= n)
            return 0;
        const char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[k + 1])));
        const char c2 =
            (k + 2 < n) ? static_cast<char>(std::tolower(static_cast<unsigned char>(text[k + 2]))) : 0;
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
            return 3;
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
            return 2;
        return 0;
    };

    size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(text[i]);

        // 1.+2. Case-aware letter rules: optional non-letter/digit/newline
        //       prefix CODEPOINT, then upper-run + lower-run (≥1 letter
        //       total), then an optional contraction suffix.
        {
            size_t j = i;
            if (!is_letter_at(i) && !is_dig(c) && !is_nl(c))
                j = i + utf8_step(i);  // candidate prefix codepoint
            size_t k = j;
            while (k < n && is_upper(static_cast<unsigned char>(text[k])))
                k++;
            while (k < n && is_lower_at(k))
                k += utf8_step(k);
            if (k > j) {
                k += contraction_len(k);
                result.push_back(text.substr(i, k - i));
                i = k;
                continue;
            }
        }

        // 3. \p{N}{1,3} — digit runs of up to three.
        if (is_dig(c)) {
            size_t k = i;
            while (k < n && k - i < 3 && is_dig(static_cast<unsigned char>(text[k])))
                k++;
            result.push_back(text.substr(i, k - i));
            i = k;
            continue;
        }

        // 4. ' '?[^\s\p{L}\p{N}]+[\r\n/]* — symbol run + trailing newlines/slashes.
        {
            size_t j = i + (text[i] == ' ' ? 1 : 0);
            size_t k = j;
            while (k < n) {
                const unsigned char ck = static_cast<unsigned char>(text[k]);
                if (is_ws(ck) || is_dig(ck) || is_letter_at(k))
                    break;
                k += utf8_step(k);  // ASCII symbol or non-ASCII punct/symbol cp
            }
            if (k > j) {
                while (k < n) {
                    const unsigned char ck = static_cast<unsigned char>(text[k]);
                    if (!is_nl(ck) && ck != '/')
                        break;
                    k++;
                }
                result.push_back(text.substr(i, k - i));
                i = k;
                continue;
            }
        }

        // 5.-7. Whitespace rules (identical to qwen2_pre_tokenize).
        if (is_ws(c)) {
            size_t k = i;
            size_t last_nl = std::string::npos;
            while (k < n && is_ws(static_cast<unsigned char>(text[k]))) {
                if (is_nl(static_cast<unsigned char>(text[k])))
                    last_nl = k;
                k++;
            }
            if (last_nl != std::string::npos) {
                result.push_back(text.substr(i, last_nl + 1 - i));
                i = last_nl + 1;
                continue;
            }
            if (k >= n) {
                result.push_back(text.substr(i, k - i));
                i = k;
                continue;
            }
            if (k - i > 1) {
                result.push_back(text.substr(i, k - i - 1));
                i = k - 1;
                continue;
            }
            result.push_back(text.substr(i, 1));
            i += 1;
            continue;
        }

        // Unreachable in practice (every byte class is handled above).
        result.push_back(text.substr(i, 1));
        i += 1;
    }
    return result;
}

// ---- Load vocabulary ----

bool Tokenizer::load(const std::string& path) {
    // Read file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return false;
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        return false;
    }
    std::string file_data(st.st_size, '\0');
    ssize_t n = ::read(fd, file_data.data(), st.st_size);
    close(fd);
    if (n != st.st_size)
        return false;

    // Parse JSON
    JsonParser parser(file_data);
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        IMP_LOG_WARN("failed to parse tokenizer.json: %s", path.c_str());
        return false;
    }

    // Extract model object
    const JValue* model = jobj_find(root, "model");
    if (!model || model->type != JType::OBJECT) {
        IMP_LOG_WARN("tokenizer.json missing 'model' object");
        return false;
    }

    // Model type
    std::string model_type;
    jobj_get_string(*model, "type", model_type);

    // Extract vocabulary from model.vocab
    const JValue* vocab = jobj_find(*model, "vocab");
    if (vocab && vocab->type == JType::OBJECT) {
        // Find max id to size the vocab vector
        int max_id = 0;
        for (const auto& [token, val] : vocab->obj) {
            if (val.type == JType::NUMBER) {
                int id = static_cast<int>(val.num_val);
                if (id > max_id)
                    max_id = id;
            }
        }
        vocab_.resize(max_id + 1);
        scores_.resize(max_id + 1, 0.0f);

        token_to_id_.clear();
        token_to_id_.reserve(vocab->obj.size());
        for (const auto& [token, val] : vocab->obj) {
            if (val.type != JType::NUMBER)
                continue;
            int id = static_cast<int>(val.num_val);
            vocab_[id] = token;
            token_to_id_[token] = id;
        }

        IMP_LOG_INFO("tokenizer.json: loaded %zu vocab entries (type=%s)", vocab->obj.size(),
                     model_type.c_str());
    }

    // Extract merges (BPE only)
    // Supports both string format ("a b") and array format (["a", "b"])
    const JValue* merges = jobj_find(*model, "merges");
    if (merges && merges->type == JType::ARRAY) {
        std::vector<std::string> merge_strs;
        merge_strs.reserve(merges->arr.size());
        for (const auto& m : merges->arr) {
            if (m.type == JType::STRING) {
                merge_strs.push_back(m.str_val);
            } else if (m.type == JType::ARRAY && m.arr.size() == 2 && m.arr[0].type == JType::STRING &&
                       m.arr[1].type == JType::STRING) {
                merge_strs.push_back(m.arr[0].str_val + " " + m.arr[1].str_val);
            }
        }
        load_merges(merge_strs);
        IMP_LOG_INFO("tokenizer.json: loaded %zu merges", merge_strs.size());
    }

    // Extract added_tokens — may extend vocab and mark special tokens
    const JValue* added = jobj_find(root, "added_tokens");
    if (added && added->type == JType::ARRAY) {
        token_types_.resize(vocab_.size(), 1);  // default NORMAL=1
        added_token_ids_.resize(vocab_.size(), false);

        for (const auto& tok : added->arr) {
            if (tok.type != JType::OBJECT)
                continue;
            const JValue* id_v = jobj_find(tok, "id");
            const JValue* content_v = jobj_find(tok, "content");
            const JValue* special_v = jobj_find(tok, "special");

            if (!id_v || !content_v)
                continue;
            if (id_v->type != JType::NUMBER || content_v->type != JType::STRING)
                continue;
            int id = static_cast<int>(id_v->num_val);
            const std::string& content = content_v->str_val;
            bool is_special = special_v && special_v->type == JType::NUMBER && special_v->num_val != 0.0;
            // HF semantics: an added token with `normalized=false` is matched
            // ATOMICALLY against the raw input, regardless of `special` (which
            // only governs decode-skipping / add_special_tokens). imp previously
            // keyed atomic matching on `special` alone, so Qwen3's non-special
            // markers — `<think>`/`</think>` (151667/151668), `<tool_call>`,
            // `<|fim_*|>` — were BPE-split into "<","think",">" pieces. That broke
            // the `<think>`-as-stop-token guard and no-think suppression (the
            // closed `<think></think>` prompt block was just text the model
            // re-opened) and mis-tokenised tool-call markers. Promote them to
            // USER_DEFINED so build_special_pieces() pre-splits them atomically;
            // unlike CONTROL(3) they stay visible in decode (special=false).
            const JValue* normalized_v = jobj_find(tok, "normalized");
            bool is_normalized_false = normalized_v && normalized_v->type == JType::NUMBER &&
                                       normalized_v->num_val == 0.0;

            // Ensure vectors are large enough
            if (id >= static_cast<int>(vocab_.size())) {
                vocab_.resize(id + 1);
                scores_.resize(id + 1, 0.0f);
                token_types_.resize(id + 1, 1);
            }
            if (id >= static_cast<int>(added_token_ids_.size()))
                added_token_ids_.resize(id + 1, false);

            vocab_[id] = content;
            token_to_id_[content] = id;
            added_token_ids_[id] = true;
            if (is_special)
                token_types_[id] = 3;  // CONTROL (atomic-match + decode-skippable)
            else if (is_normalized_false)
                token_types_[id] = 4;  // USER_DEFINED (atomic-match, decode-visible)

            // Detect BOS/EOS tokens
            if (content == "<s>" || content == "<|begin_of_text|>" || content == "<|startoftext|>") {
                bos_id_ = id;
            }
            if (content == "</s>" || content == "<|end_of_text|>" || content == "<|endoftext|>" ||
                content == "<|eot_id|>") {
                if (eos_ids_.size() == 1 && eos_ids_[0] == 2) {
                    // Replace default
                    eos_ids_ = {static_cast<int32_t>(id)};
                } else {
                    add_eos_id(static_cast<int32_t>(id));
                }
            }
        }

        build_special_pieces();
    }

    // Detect pre-tokenizer type
    const JValue* pre_tok = jobj_find(root, "pre_tokenizer");
    if (pre_tok && pre_tok->type == JType::OBJECT) {
        std::string pt_type;
        jobj_get_string(*pre_tok, "type", pt_type);

        if (pt_type == "ByteLevel") {
            type_ = "gpt2";
            const JValue* prefix = jobj_find(*pre_tok, "add_prefix_space");
            if (prefix && prefix->type == JType::NUMBER)
                add_space_prefix_ = (prefix->num_val != 0.0);
            else
                add_space_prefix_ = false;
        } else if (pt_type == "Metaspace") {
            type_ = "spm";
            const JValue* prefix = jobj_find(*pre_tok, "add_prefix_space");
            if (prefix && prefix->type == JType::NUMBER)
                add_space_prefix_ = (prefix->num_val != 0.0);
        } else if (pt_type == "Sequence") {
            // Check inner pre-tokenizers for ByteLevel or Metaspace
            const JValue* pretoks = jobj_find(*pre_tok, "pretokenizers");
            if (pretoks && pretoks->type == JType::ARRAY) {
                for (const auto& pt : pretoks->arr) {
                    if (pt.type != JType::OBJECT)
                        continue;
                    std::string inner_type;
                    jobj_get_string(pt, "type", inner_type);
                    if (inner_type == "Split") {
                        // HF tokenizer.json carries the literal pre-tokenizer
                        // regex in a Split step. Discriminate the families by
                        // their regex fingerprints and route to the faithful
                        // scanners — without this, SafeTensors models fell to
                        // the gpt2 fallback (per-char punctuation,
                        // non-canonical segmentation, #657):
                        //  - "{1,3}" digit grouping → o200k (gpt-oss/GPT-4o;
                        //    its regex ALSO contains the contraction list, so
                        //    this check must come first)
                        //  - contraction alternation → qwen2
                        const JValue* pattern = jobj_find(pt, "pattern");
                        std::string rx;
                        if (pattern && pattern->type == JType::OBJECT)
                            jobj_get_string(*pattern, "Regex", rx);
                        // Fingerprints, most specific first: only o200k uses
                        // case classes (\p{Lu}); cl100k shares qwen2's rules
                        // except digit triples ({1,3}); plain contraction
                        // alternation → qwen2.
                        if (rx.find("\\p{Lu}") != std::string::npos)
                            pre_tokenizer_ = "o200k";
                        else if (rx.find("{1,3}") != std::string::npos)
                            pre_tokenizer_ = "cl100k";
                        else if (rx.find("'s|'t|'re|'ve|'m|'ll|'d") != std::string::npos)
                            pre_tokenizer_ = "qwen2";
                        continue;
                    }
                    if (inner_type == "ByteLevel") {
                        type_ = "gpt2";
                        const JValue* prefix = jobj_find(pt, "add_prefix_space");
                        if (prefix && prefix->type == JType::NUMBER)
                            add_space_prefix_ = (prefix->num_val != 0.0);
                        else
                            add_space_prefix_ = false;
                        break;
                    }
                    if (inner_type == "Metaspace") {
                        type_ = "spm";
                        const JValue* prefix = jobj_find(pt, "add_prefix_space");
                        if (prefix && prefix->type == JType::NUMBER)
                            add_space_prefix_ = (prefix->num_val != 0.0);
                        break;
                    }
                }
            }
        }
    } else if (model_type == "BPE") {
        // No pre_tokenizer specified but model is BPE — default to gpt2
        type_ = "gpt2";
        add_space_prefix_ = false;
    } else if (model_type == "Unigram") {
        type_ = "spm";
    }

    // For Unigram models, populate scores from model.vocab (array of [token, score])
    // Some Unigram tokenizer.json have vocab as array instead of object
    if (model_type == "Unigram") {
        const JValue* uni_vocab = jobj_find(*model, "vocab");
        if (uni_vocab && uni_vocab->type == JType::ARRAY) {
            int max_id = static_cast<int>(uni_vocab->arr.size()) - 1;
            vocab_.resize(max_id + 1);
            scores_.resize(max_id + 1, 0.0f);
            token_to_id_.clear();
            token_to_id_.reserve(uni_vocab->arr.size());
            for (size_t i = 0; i < uni_vocab->arr.size(); i++) {
                const auto& entry = uni_vocab->arr[i];
                if (entry.type == JType::ARRAY && entry.arr.size() >= 2) {
                    vocab_[i] = entry.arr[0].str_val;
                    scores_[i] = static_cast<float>(entry.arr[1].num_val);
                    token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
                }
            }
            IMP_LOG_INFO("tokenizer.json: loaded %zu Unigram vocab entries", uni_vocab->arr.size());
        }
    }

    IMP_LOG_INFO("tokenizer.json: type=%s vocab_size=%d bos=%d eos=%d add_prefix=%s", type_.c_str(),
                 static_cast<int>(vocab_.size()), bos_id_, eos_ids_.empty() ? -1 : eos_ids_[0],
                 add_space_prefix_ ? "true" : "false");
    return true;
}

bool Tokenizer::load_vocab(const std::vector<std::string>& tokens, const std::vector<float>& scores,
                           int bos_id, int eos_id) {
    if (tokens.empty())
        return false;

    vocab_ = tokens;
    scores_ = scores;
    scores_.resize(vocab_.size(), 0.0f);
    bos_id_ = bos_id;
    eos_ids_ = {eos_id};

    token_to_id_.clear();
    token_to_id_.reserve(vocab_.size());
    for (size_t i = 0; i < vocab_.size(); i++) {
        token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
    }

    return true;
}

void Tokenizer::load_merges(const std::vector<std::string>& merges) {
    merge_ranks_.clear();
    merge_ranks_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); i++) {
        merge_ranks_[merges[i]] = static_cast<int>(i);
    }
}

// ---- BPE Encode (SentencePiece style) ----

static const std::string SPIECE_SPACE = "\xe2\x96\x81";

// Cache the list of CONTROL-class added tokens (e.g. <|tool_call>, <|im_start|>,
// <|channel>) sorted by length descending so that longest-match wins. encode_*
// uses this list to split the input text on those literals before running BPE,
// so the rendered chat-template markers round-trip as their assigned token IDs.
void Tokenizer::build_special_pieces() {
    special_pieces_.clear();
    if (token_types_.empty())
        return;
    for (size_t id = 0; id < token_types_.size() && id < vocab_.size(); ++id) {
        // CONTROL (3) markers AND USER_DEFINED (4) symbols. SentencePiece
        // semantics: user-defined symbols match the RAW input literally and
        // atomically, before normalization/BPE. gemma-3 stores its multi-space
        // run tokens ('  ', '   ', …, 27×' ') and HTML tags (<code>, <i>, …)
        // as USER_DEFINED with literal-space pieces — the ▁-substituting BPE
        // body can never reproduce them ("▁▁" is not in the vocab), so imp
        // emitted N single-space tokens per indentation run: +1.3% tokens and
        // a large share of the +37.5% NLL gap vs llama.cpp on code/markdown
        // (#657). llama.cpp matches these via its user-defined trie.
        if (token_types_[id] != 3 && token_types_[id] != 4)
            continue;
        const std::string& s = vocab_[id];
        if (s.empty())
            continue;
        // Skip plain ASCII identifiers — they would shadow normal BPE matches
        // (e.g. a model that flagged "the" as control would break encoding).
        // All real special markers contain at least one non-alnum char.
        bool has_marker = false;
        for (unsigned char c : s) {
            if (!std::isalnum(c) && c != '_') {
                has_marker = true;
                break;
            }
        }
        if (!has_marker)
            continue;
        special_pieces_.emplace_back(s, static_cast<int32_t>(id));
    }
    std::sort(special_pieces_.begin(), special_pieces_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    IMP_LOG_INFO("Tokenizer: %zu special pieces cached for pre-split", special_pieces_.size());
}

namespace {
// Split text on cached special pieces. Returns a flat vector where each entry
// is either {.special_id=-1, .text=..} (BPE-this) or {.special_id=N, .text=""}
// (emit token N directly). Empty text chunks are dropped.
struct PieceChunk {
    std::string text;
    int32_t special_id = -1;
};
std::vector<PieceChunk> split_on_special(const std::string& text,
                                         const std::vector<std::pair<std::string, int32_t>>& specials) {
    std::vector<PieceChunk> out;
    if (specials.empty()) {
        if (!text.empty())
            out.push_back({text, -1});
        return out;
    }
    size_t i = 0;
    std::string cur;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& [piece, id] : specials) {
            size_t L = piece.size();
            if (L == 0 || i + L > text.size())
                continue;
            if (std::memcmp(text.data() + i, piece.data(), L) == 0) {
                if (!cur.empty()) {
                    out.push_back({std::move(cur), -1});
                    cur.clear();
                }
                out.push_back({"", id});
                i += L;
                matched = true;
                break;
            }
        }
        if (!matched) {
            cur += text[i++];
        }
    }
    if (!cur.empty())
        out.push_back({std::move(cur), -1});
    return out;
}
}  // namespace

std::vector<int32_t> Tokenizer::encode_spm(const std::string& text, bool no_prefix) const {
    if (text.empty() || vocab_.empty())
        return {};

    // Pre-split on registered control / added tokens. Without this an HF
    // tokenizer.json with type=spm + special added_tokens (Gemma-4 family,
    // some Mistral variants) silently BPEs the marker text as raw UTF-8 and
    // the model never sees the trained single-token id. Recurse once with
    // the marker stripped so the BPE body below runs on a clean substring.
    if (!special_pieces_.empty()) {
        auto pieces = split_on_special(text, special_pieces_);
        bool any_special = false;
        for (const auto& p : pieces)
            if (p.special_id >= 0) {
                any_special = true;
                break;
            }
        if (any_special) {
            std::vector<int32_t> out;
            out.reserve(text.size());
            bool first = true;
            for (const auto& p : pieces) {
                if (p.special_id >= 0) {
                    out.push_back(p.special_id);
                    first = false;
                    continue;
                }
                if (p.text.empty())
                    continue;
                // Suppress add_space_prefix_ on chunks that don't start at the
                // very beginning of the original text (matches HF behavior).
                auto sub = encode_spm(p.text, /*no_prefix=*/!first || no_prefix);
                out.insert(out.end(), sub.begin(), sub.end());
                first = false;
            }
            return out;
        }
    }

    // Pre-process: SentencePiece convention - replace spaces with ▁
    // add_space_prefix_: prepend ▁ at start (true for LLaMA/Mistral, false for Gemma)
    // no_prefix: skip the leading ▁ (for chat template pieces after special tokens)
    std::string processed;
    processed.reserve(text.size() + 4);
    if (add_space_prefix_ && !no_prefix) {
        processed += SPIECE_SPACE;
    }

    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            processed += SPIECE_SPACE;
        } else {
            processed += text[i];
        }
    }

    // Split into UTF-8 characters as initial symbols
    std::vector<std::string> symbols;
    symbols.reserve(processed.size());

    for (size_t i = 0; i < processed.size();) {
        int len = utf8_char_len(static_cast<uint8_t>(processed[i]));
        if (i + len > processed.size())
            len = 1;
        symbols.push_back(processed.substr(i, len));
        i += len;
    }

    // BPE merge loop using priority queue: O(n log n) instead of O(n²).
    // Linked list of symbols with prev/next pointers; deleted nodes are skipped.
    int n = static_cast<int>(symbols.size());
    std::vector<int> prev(n), next(n);
    std::vector<bool> deleted(n, false);
    for (int i = 0; i < n; i++) {
        prev[i] = i - 1;
        next[i] = i + 1;
    }

    // Max-heap: highest score first, then lowest position for tie-breaking
    struct MergeCand {
        float score;
        int pos;   // left symbol index
        int seq;   // left sequence number at insertion (for invalidation)
        int rseq;  // right sequence number at insertion
    };
    auto cmp = [](const MergeCand& a, const MergeCand& b) {
        if (a.score != b.score)
            return a.score < b.score;
        return a.pos > b.pos;
    };
    std::priority_queue<MergeCand, std::vector<MergeCand>, decltype(cmp)> pq(cmp);

    // Sequence counters per position: incremented on merge to invalidate stale entries
    std::vector<int> seq(n, 0);

    // Seed the queue with all valid adjacent pairs
    for (int i = 0; i < n - 1; i++) {
        std::string merged = symbols[i] + symbols[next[i]];
        auto it = token_to_id_.find(merged);
        if (it != token_to_id_.end()) {
            pq.push({scores_[it->second], i, seq[i], seq[next[i]]});
        }
    }

    while (!pq.empty()) {
        auto [score, pos, s, rs] = pq.top();
        (void)score;  // ordering key only
        pq.pop();

        // Validate: both symbols still exist and haven't been modified since insertion
        if (deleted[pos] || seq[pos] != s)
            continue;
        int right = next[pos];
        if (right >= n || deleted[right])
            continue;
        if (seq[right] != rs)
            continue;  // right symbol was modified

        // Merge: symbols[pos] absorbs symbols[right]
        symbols[pos] = symbols[pos] + symbols[right];
        deleted[right] = true;
        seq[pos]++;  // invalidate stale entries for this position

        // Update linked list
        next[pos] = next[right];
        if (next[right] < n)
            prev[next[right]] = pos;

        // Try new pair with left neighbor
        if (prev[pos] >= 0) {
            int lp = prev[pos];
            std::string m = symbols[lp] + symbols[pos];
            auto it = token_to_id_.find(m);
            if (it != token_to_id_.end()) {
                pq.push({scores_[it->second], lp, seq[lp], seq[pos]});
            }
        }
        // Try new pair with right neighbor
        if (next[pos] < n) {
            std::string m = symbols[pos] + symbols[next[pos]];
            auto it = token_to_id_.find(m);
            if (it != token_to_id_.end()) {
                pq.push({scores_[it->second], pos, seq[pos], seq[next[pos]]});
            }
        }
    }

    // Collect non-deleted symbols → token IDs
    std::vector<int32_t> ids;
    ids.reserve(n);

    for (int i = 0; i < n; i++) {
        if (deleted[i])
            continue;
        const auto& sym = symbols[i];
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Byte fallback
            for (unsigned char byte : sym) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", byte);
                auto byte_it = token_to_id_.find(buf);
                if (byte_it != token_to_id_.end()) {
                    ids.push_back(byte_it->second);
                }
            }
        }
    }

    return ids;
}

// ---- BPE Encode (GPT2 byte-level style) ----

// ---- Llama3 pre-tokenizer ----
// Key differences from default:
//  - Contractions like 's, 't, 're etc. split separately
//  - Spaces are individual tokens (not attached to next word)
//  - Digits are split individually (not groups of 3)

static std::vector<std::string> llama3_pre_tokenize(const std::string& text) {
    std::vector<std::string> result;
    if (text.empty())
        return result;

    // Common English contractions that get their own tokens
    static const char* contractions[] = {
        "'s",
        "'t",
        "'re",
        "'ve",
        "'m",
        "'ll",
        "'d",
        "\xe2\x80\x99s",
        "\xe2\x80\x99t",
        "\xe2\x80\x99re",
        "\xe2\x80\x99"
        "ve",
        "\xe2\x80\x99"
        "m",
        "\xe2\x80\x99"
        "ll",
        "\xe2\x80\x99"
        "d",
    };

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Check for contractions
        bool found_contraction = false;
        if (c == '\'' ||
            (c == 0xe2 && i + 2 < text.size() && text[i + 1] == '\x80' && text[i + 2] == '\x99')) {
            for (const char* ctr : contractions) {
                size_t len = std::strlen(ctr);
                if (i + len <= text.size() && text.compare(i, len, ctr) == 0) {
                    result.push_back(text.substr(i, len));
                    i += len;
                    found_contraction = true;
                    break;
                }
            }
        }
        if (found_contraction)
            continue;

        if (c == ' ' || c == '\t') {
            // Space: attach to following word (like GPT2)
            std::string chunk;
            chunk += text[i++];
            while (i < text.size()) {
                unsigned char cc = static_cast<unsigned char>(text[i]);
                if (cc == ' ' || cc == '\t' || cc == '\n' || cc == '\r')
                    break;
                if (std::ispunct(cc) && cc != '\'')
                    break;
                int len = 1;
                if ((cc & 0xE0) == 0xC0)
                    len = 2;
                else if ((cc & 0xF0) == 0xE0)
                    len = 3;
                else if ((cc & 0xF8) == 0xF0)
                    len = 4;
                for (int j = 0; j < len && i < text.size(); j++)
                    chunk += text[i++];
            }
            result.push_back(std::move(chunk));
        } else if (c == '\n' || c == '\r') {
            std::string chunk;
            while (i < text.size() && (text[i] == '\n' || text[i] == '\r'))
                chunk += text[i++];
            result.push_back(std::move(chunk));
        } else if (std::isalpha(c) || c >= 128) {
            std::string chunk;
            while (i < text.size()) {
                unsigned char cc = static_cast<unsigned char>(text[i]);
                if (!std::isalpha(cc) && cc < 128)
                    break;
                int len = 1;
                if ((cc & 0xE0) == 0xC0)
                    len = 2;
                else if ((cc & 0xF0) == 0xE0)
                    len = 3;
                else if ((cc & 0xF8) == 0xF0)
                    len = 4;
                for (int j = 0; j < len && i < text.size(); j++)
                    chunk += text[i++];
            }
            result.push_back(std::move(chunk));
        } else if (std::isdigit(c)) {
            // Digits: one at a time (llama3 splits individual digits)
            result.push_back(std::string(1, text[i++]));
        } else {
            result.push_back(std::string(1, text[i++]));
        }
    }
    return result;
}

// ---- Gemma-4 encode: SPM-style ▁ escaping + BPE merge ranks ----
// Gemma-4 uses SentencePiece-style BPE: spaces→▁, no word-level pre-splitting
// (only split on newlines), raw UTF-8 characters, BPE merges by rank.
std::vector<int32_t> Tokenizer::encode_gemma4(const std::string& text) const {
    if (text.empty() || vocab_.empty())
        return {};

    // 0. Pre-split on registered control / added tokens (e.g. <|tool_call>,
    //    <|channel>, <|tool>) so each is emitted as its single token id
    //    instead of being BPE'd as raw UTF-8.
    auto pieces = split_on_special(text, special_pieces_);
    std::vector<int32_t> out_ids;
    out_ids.reserve(text.size());
    for (const auto& p : pieces) {
        if (p.special_id >= 0) {
            out_ids.push_back(p.special_id);
            continue;
        }
        if (p.text.empty())
            continue;
        // Recursive call: re-enter encode_gemma4 with the BPE-only chunk.
        // (special_pieces_ now matches nothing inside p.text by construction.)
        // To avoid infinite recursion we call the BPE body directly; we emulate
        // that by calling this function once specials are stripped — the
        // first-call splitter has already removed them, so the recursion ends
        // immediately at the no-match path below.
        // For simplicity we just inline the BPE path here.
        const std::string& bpe_text = p.text;

        // 1. Escape spaces → ▁
        std::string processed;
        processed.reserve(bpe_text.size() + 4);
        for (size_t i = 0; i < bpe_text.size(); i++) {
            if (bpe_text[i] == ' ') {
                processed += SPIECE_SPACE;  // ▁ (U+2581)
            } else {
                processed += bpe_text[i];
            }
        }

        // 2. Split on newlines only (gemma4 pre-tokenizer regex: [^\n]+|[\n]+)
        std::vector<std::string> chunks;
        size_t start = 0;
        while (start < processed.size()) {
            if (processed[start] == '\n') {
                size_t end = start;
                while (end < processed.size() && processed[end] == '\n')
                    end++;
                chunks.push_back(processed.substr(start, end - start));
                start = end;
            } else {
                size_t end = start;
                while (end < processed.size() && processed[end] != '\n')
                    end++;
                chunks.push_back(processed.substr(start, end - start));
                start = end;
            }
        }

        std::vector<int32_t> all_ids;
        all_ids.reserve(text.size());

        for (const auto& chunk : chunks) {
            // 3. Split into UTF-8 characters (raw, no byte encoding)
            std::vector<std::string> symbols;
            symbols.reserve(chunk.size());
            for (size_t i = 0; i < chunk.size();) {
                int len = utf8_char_len(static_cast<uint8_t>(chunk[i]));
                if (i + len > chunk.size())
                    len = 1;
                symbols.push_back(chunk.substr(i, len));
                i += len;
            }

            // 4. BPE merge loop using merge ranks (lower rank = merge first)
            int ns = static_cast<int>(symbols.size());
            std::vector<int> sprev(ns), snext(ns);
            std::vector<bool> sdel(ns, false);
            for (int i = 0; i < ns; i++) {
                sprev[i] = i - 1;
                snext[i] = i + 1;
            }

            struct BPEMerge {
                int rank;
                int pos;
                int seq;
            };
            auto cmp = [](const BPEMerge& a, const BPEMerge& b) {
                if (a.rank != b.rank)
                    return a.rank > b.rank;  // min-heap: lower rank first
                return a.pos > b.pos;
            };
            std::priority_queue<BPEMerge, std::vector<BPEMerge>, decltype(cmp)> pq(cmp);
            std::vector<int> sseq(ns, 0);

            for (int i = 0; i < ns - 1; i++) {
                std::string key = symbols[i] + " " + symbols[snext[i]];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end()) {
                    pq.push({it->second, i, sseq[i]});
                }
            }

            while (!pq.empty()) {
                auto [rank, pos, s] = pq.top();
                pq.pop();
                if (sdel[pos] || sseq[pos] != s)
                    continue;
                int right = snext[pos];
                if (right >= ns || sdel[right])
                    continue;

                // Re-validate: the pair at this position may have changed since
                // the merge was enqueued (the right neighbor may have merged
                // with ITS neighbor). Check that the current pair still has
                // the popped rank.
                std::string merged = symbols[pos] + symbols[right];
                {
                    std::string cur_key = symbols[pos] + " " + symbols[right];
                    auto vit = merge_ranks_.find(cur_key);
                    if (vit == merge_ranks_.end() || vit->second != rank)
                        continue;
                }
                // Vocab-existence guard: merge_ranks_ contains rules whose output
                // is not in the final vocab (intermediate merge steps, e.g.
                // "Lin u → Linu" where "Linu" is not a token). Applying such a
                // merge produces a symbol that fails vocab lookup and byte-
                // falls back the entire word. Skipping these keeps sub-parts
                // that ARE tokens intact. Matches llama.cpp behavior.
                if (token_to_id_.find(merged) == token_to_id_.end())
                    continue;

                symbols[pos] = merged;
                sdel[right] = true;
                sseq[pos]++;
                snext[pos] = snext[right];
                if (snext[right] < ns)
                    sprev[snext[right]] = pos;

                if (sprev[pos] >= 0) {
                    int lp = sprev[pos];
                    std::string key = symbols[lp] + " " + symbols[pos];
                    auto it = merge_ranks_.find(key);
                    if (it != merge_ranks_.end()) {
                        pq.push({it->second, lp, sseq[lp]});
                    }
                }
                if (snext[pos] < ns) {
                    std::string key = symbols[pos] + " " + symbols[snext[pos]];
                    auto it = merge_ranks_.find(key);
                    if (it != merge_ranks_.end()) {
                        pq.push({it->second, pos, sseq[pos]});
                    }
                }
            }

            // 5. Convert symbols to token IDs
            for (int i = 0; i < ns; i++) {
                if (sdel[i])
                    continue;
                auto it = token_to_id_.find(symbols[i]);
                if (it != token_to_id_.end()) {
                    all_ids.push_back(it->second);
                } else {
                    // Fallback: the merge-guard above prevents producing symbols
                    // that aren't in vocab via merging, so we only land here for
                    // initial UTF-8 characters that the vocab doesn't cover. Try
                    // per-character vocab lookup first, then byte fallback.
                    const std::string& sym = symbols[i];
                    for (size_t ci = 0; ci < sym.size();) {
                        int clen = utf8_char_len(static_cast<uint8_t>(sym[ci]));
                        if (ci + clen > sym.size())
                            clen = 1;
                        std::string ch = sym.substr(ci, clen);
                        auto ch_it = token_to_id_.find(ch);
                        if (ch_it != token_to_id_.end()) {
                            all_ids.push_back(ch_it->second);
                        } else {
                            for (int bi = 0; bi < clen; bi++) {
                                char buf[8];
                                snprintf(buf, sizeof(buf), "<0x%02X>",
                                         static_cast<unsigned char>(sym[ci + bi]));
                                auto bit = token_to_id_.find(buf);
                                if (bit != token_to_id_.end()) {
                                    all_ids.push_back(bit->second);
                                }
                            }
                        }
                        ci += clen;
                    }
                }
            }
        }
        out_ids.insert(out_ids.end(), all_ids.begin(), all_ids.end());
    }  // end for piece

    return out_ids;
}

std::vector<int32_t> Tokenizer::encode_gpt2(const std::string& text) const {
    if (text.empty() || vocab_.empty())
        return {};

    // 0. Pre-split on registered control tokens. Models like Qwen3.6 / Hermes
    //    rely on multi-character markers (e.g. <|im_start|>, <|tool_call>)
    //    that the pre-tokenizer regex doesn't always isolate cleanly; an
    //    explicit longest-match pass guarantees the marker round-trips as
    //    its assigned token id.
    auto pieces = split_on_special(text, special_pieces_);
    std::vector<int32_t> out_ids;
    out_ids.reserve(text.size());
    for (const auto& piece : pieces) {
        if (piece.special_id >= 0) {
            out_ids.push_back(piece.special_id);
            continue;
        }
        if (piece.text.empty())
            continue;
        const std::string& bpe_text = piece.text;

        // 1. Pre-tokenize into chunks (dispatch based on pre-tokenizer type)
        std::vector<std::string> chunks;
        if (pre_tokenizer_ == "llama3" || pre_tokenizer_ == "llama-v3" || pre_tokenizer_ == "llama-bpe") {
            chunks = llama3_pre_tokenize(bpe_text);
        } else if (pre_tokenizer_ == "qwen2" || pre_tokenizer_ == "qwen35") {
            // Qwen2/Qwen3 family: canonical regex incl. symbol RUNS and single
            // digits — the gpt2 fallback's per-char punctuation made canonical
            // merges ("->", "():") impossible (#657). "qwen35" (Qwen3.5/3.6
            // GGUFs) differs from qwen2 only by adding \p{M} to the letter
            // run, which is_letter_at already treats as letters — falling
            // through to gpt2 instead over-split symbol runs (+13% tokens on
            // the 35B hero corpus).
            chunks = qwen2_pre_tokenize(bpe_text);
        } else if (pre_tokenizer_ == "o200k" || pre_tokenizer_ == "gpt-4o") {
            // gpt-oss / GPT-4o family (o200k_harmony): case-aware letter runs,
            // digit triples, slash-aware symbol runs (#657).
            chunks = o200k_pre_tokenize(bpe_text);
        } else if (pre_tokenizer_ == "cl100k") {
            // GPT-4/tiktoken cl100k lineage (Phi-4): qwen2 rules with digit
            // triples (#657).
            chunks = cl100k_pre_tokenize(bpe_text);
        } else {
            chunks = gpt2_pre_tokenize(bpe_text);
        }

        std::vector<int32_t> all_ids;
        all_ids.reserve(bpe_text.size());  // rough estimate

        for (const auto& chunk : chunks) {
            // 2. Convert each byte to GPT2 unicode character
            std::vector<std::string> symbols;
            symbols.reserve(chunk.size());
            for (unsigned char byte : chunk) {
                symbols.push_back(byte_to_gpt2(byte));
            }

            // 3. BPE merge loop using priority queue: O(n log n)
            int ns = static_cast<int>(symbols.size());
            std::vector<int> sprev(ns), snext(ns);
            std::vector<bool> sdel(ns, false);
            for (int i = 0; i < ns; i++) {
                sprev[i] = i - 1;
                snext[i] = i + 1;
            }

            // Min-heap: lowest rank first, then lowest position
            struct GPT2Merge {
                int rank;
                int pos;
                int seq;
            };
            auto gcmp = [](const GPT2Merge& a, const GPT2Merge& b) {
                if (a.rank != b.rank)
                    return a.rank > b.rank;
                return a.pos > b.pos;
            };
            std::priority_queue<GPT2Merge, std::vector<GPT2Merge>, decltype(gcmp)> gpq(gcmp);

            std::vector<int> sseq(ns, 0);

            for (int i = 0; i < ns - 1; i++) {
                std::string key = symbols[i] + " " + symbols[snext[i]];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end()) {
                    gpq.push({it->second, i, sseq[i]});
                }
            }

            while (!gpq.empty()) {
                auto [rank, pos, s] = gpq.top();
                gpq.pop();

                if (sdel[pos] || sseq[pos] != s)
                    continue;
                int right = snext[pos];
                if (right >= ns || sdel[right])
                    continue;

                // Re-validate: the pair at this position may have changed since
                // the merge was enqueued (e.g., the right neighbor was merged
                // with ITS right neighbor, changing the symbol). Check that the
                // current pair still maps to the same rank.
                {
                    std::string cur_key = symbols[pos] + " " + symbols[right];
                    auto vit = merge_ranks_.find(cur_key);
                    if (vit == merge_ranks_.end() || vit->second != rank)
                        continue;
                }

                symbols[pos] = symbols[pos] + symbols[right];
                sdel[right] = true;
                sseq[pos]++;

                snext[pos] = snext[right];
                if (snext[right] < ns)
                    sprev[snext[right]] = pos;

                if (sprev[pos] >= 0) {
                    int lp = sprev[pos];
                    std::string key = symbols[lp] + " " + symbols[pos];
                    auto it = merge_ranks_.find(key);
                    if (it != merge_ranks_.end()) {
                        gpq.push({it->second, lp, sseq[lp]});
                    }
                }
                if (snext[pos] < ns) {
                    std::string key = symbols[pos] + " " + symbols[snext[pos]];
                    auto it = merge_ranks_.find(key);
                    if (it != merge_ranks_.end()) {
                        gpq.push({it->second, pos, sseq[pos]});
                    }
                }
            }

            // 4. Look up token IDs
            for (int i = 0; i < ns; i++) {
                if (sdel[i])
                    continue;
                const auto& sym = symbols[i];
                auto it = token_to_id_.find(sym);
                if (it != token_to_id_.end()) {
                    all_ids.push_back(it->second);
                } else {
                    // Fallback: try individual GPT2 byte tokens
                    for (size_t ci = 0; ci < sym.size();) {
                        int len = utf8_char_len(static_cast<uint8_t>(sym[ci]));
                        if (ci + len > sym.size())
                            len = 1;
                        std::string ch = sym.substr(ci, len);
                        auto ch_it = token_to_id_.find(ch);
                        if (ch_it != token_to_id_.end()) {
                            all_ids.push_back(ch_it->second);
                        }
                        ci += len;
                    }
                }
            }
        }
        out_ids.insert(out_ids.end(), all_ids.begin(), all_ids.end());
    }  // end for piece

    return out_ids;
}

// ---- NFC Normalization ----
// Handles the most common combining sequences for Latin scripts.
// Covers: accented Latin characters (é, ñ, ü, etc.) which are the vast
// majority of NFC normalization cases in real-world text.

namespace {

// Composition table: (base_codepoint, combining_codepoint) → composed_codepoint
struct NfcEntry {
    uint32_t base;
    uint32_t combining;
    uint32_t composed;
};

// Most common Latin composition pairs (base + combining mark → precomposed)
// Combining marks: 0x0300 (grave), 0x0301 (acute), 0x0302 (circumflex),
//   0x0303 (tilde), 0x0304 (macron), 0x0308 (diaeresis), 0x030C (caron)
static const NfcEntry kNfcTable[] = {
    // Grave accent (0x0300)
    {0x0041, 0x0300, 0x00C0},  // À
    {0x0045, 0x0300, 0x00C8},  // È
    {0x0049, 0x0300, 0x00CC},  // Ì
    {0x004F, 0x0300, 0x00D2},  // Ò
    {0x0055, 0x0300, 0x00D9},  // Ù
    {0x0061, 0x0300, 0x00E0},  // à
    {0x0065, 0x0300, 0x00E8},  // è
    {0x0069, 0x0300, 0x00EC},  // ì
    {0x006F, 0x0300, 0x00F2},  // ò
    {0x0075, 0x0300, 0x00F9},  // ù

    // Acute accent (0x0301)
    {0x0041, 0x0301, 0x00C1},  // Á
    {0x0043, 0x0301, 0x0106},  // Ć
    {0x0045, 0x0301, 0x00C9},  // É
    {0x0049, 0x0301, 0x00CD},  // Í
    {0x004C, 0x0301, 0x0139},  // Ĺ
    {0x004E, 0x0301, 0x0143},  // Ń
    {0x004F, 0x0301, 0x00D3},  // Ó
    {0x0052, 0x0301, 0x0154},  // Ŕ
    {0x0053, 0x0301, 0x015A},  // Ś
    {0x0055, 0x0301, 0x00DA},  // Ú
    {0x0059, 0x0301, 0x00DD},  // Ý
    {0x005A, 0x0301, 0x0179},  // Ź
    {0x0061, 0x0301, 0x00E1},  // á
    {0x0063, 0x0301, 0x0107},  // ć
    {0x0065, 0x0301, 0x00E9},  // é
    {0x0069, 0x0301, 0x00ED},  // í
    {0x006C, 0x0301, 0x013A},  // ĺ
    {0x006E, 0x0301, 0x0144},  // ń
    {0x006F, 0x0301, 0x00F3},  // ó
    {0x0072, 0x0301, 0x0155},  // ŕ
    {0x0073, 0x0301, 0x015B},  // ś
    {0x0075, 0x0301, 0x00FA},  // ú
    {0x0079, 0x0301, 0x00FD},  // ý
    {0x007A, 0x0301, 0x017A},  // ź

    // Circumflex (0x0302)
    {0x0041, 0x0302, 0x00C2},  // Â
    {0x0043, 0x0302, 0x0108},  // Ĉ
    {0x0045, 0x0302, 0x00CA},  // Ê
    {0x0047, 0x0302, 0x011C},  // Ĝ
    {0x0048, 0x0302, 0x0124},  // Ĥ
    {0x0049, 0x0302, 0x00CE},  // Î
    {0x004A, 0x0302, 0x0134},  // Ĵ
    {0x004F, 0x0302, 0x00D4},  // Ô
    {0x0053, 0x0302, 0x015C},  // Ŝ
    {0x0055, 0x0302, 0x00DB},  // Û
    {0x0057, 0x0302, 0x0174},  // Ŵ
    {0x0059, 0x0302, 0x0176},  // Ŷ
    {0x0061, 0x0302, 0x00E2},  // â
    {0x0063, 0x0302, 0x0109},  // ĉ
    {0x0065, 0x0302, 0x00EA},  // ê
    {0x0067, 0x0302, 0x011D},  // ĝ
    {0x0068, 0x0302, 0x0125},  // ĥ
    {0x0069, 0x0302, 0x00EE},  // î
    {0x006A, 0x0302, 0x0135},  // ĵ
    {0x006F, 0x0302, 0x00F4},  // ô
    {0x0073, 0x0302, 0x015D},  // ŝ
    {0x0075, 0x0302, 0x00FB},  // û
    {0x0077, 0x0302, 0x0175},  // ŵ
    {0x0079, 0x0302, 0x0177},  // ŷ

    // Tilde (0x0303)
    {0x0041, 0x0303, 0x00C3},  // Ã
    {0x004E, 0x0303, 0x00D1},  // Ñ
    {0x004F, 0x0303, 0x00D5},  // Õ
    {0x0061, 0x0303, 0x00E3},  // ã
    {0x006E, 0x0303, 0x00F1},  // ñ
    {0x006F, 0x0303, 0x00F5},  // õ

    // Diaeresis/Umlaut (0x0308)
    {0x0041, 0x0308, 0x00C4},  // Ä
    {0x0045, 0x0308, 0x00CB},  // Ë
    {0x0049, 0x0308, 0x00CF},  // Ï
    {0x004F, 0x0308, 0x00D6},  // Ö
    {0x0055, 0x0308, 0x00DC},  // Ü
    {0x0059, 0x0308, 0x0178},  // Ÿ
    {0x0061, 0x0308, 0x00E4},  // ä
    {0x0065, 0x0308, 0x00EB},  // ë
    {0x0069, 0x0308, 0x00EF},  // ï
    {0x006F, 0x0308, 0x00F6},  // ö
    {0x0075, 0x0308, 0x00FC},  // ü
    {0x0079, 0x0308, 0x00FF},  // ÿ

    // Caron/Háček (0x030C)
    {0x0043, 0x030C, 0x010C},  // Č
    {0x0044, 0x030C, 0x010E},  // Ď
    {0x0045, 0x030C, 0x011A},  // Ě
    {0x004E, 0x030C, 0x0147},  // Ň
    {0x0052, 0x030C, 0x0158},  // Ř
    {0x0053, 0x030C, 0x0160},  // Š
    {0x0054, 0x030C, 0x0164},  // Ť
    {0x005A, 0x030C, 0x017D},  // Ž
    {0x0063, 0x030C, 0x010D},  // č
    {0x0064, 0x030C, 0x010F},  // ď
    {0x0065, 0x030C, 0x011B},  // ě
    {0x006E, 0x030C, 0x0148},  // ň
    {0x0072, 0x030C, 0x0159},  // ř
    {0x0073, 0x030C, 0x0161},  // š
    {0x0074, 0x030C, 0x0165},  // ť
    {0x007A, 0x030C, 0x017E},  // ž

    // Cedilla (0x0327)
    {0x0043, 0x0327, 0x00C7},  // Ç
    {0x0063, 0x0327, 0x00E7},  // ç
    {0x0053, 0x0327, 0x015E},  // Ş
    {0x0073, 0x0327, 0x015F},  // ş

    // Ring above (0x030A)
    {0x0041, 0x030A, 0x00C5},  // Å
    {0x0061, 0x030A, 0x00E5},  // å
    {0x0055, 0x030A, 0x016E},  // Ů
    {0x0075, 0x030A, 0x016F},  // ů

    // Macron (0x0304)
    {0x0041, 0x0304, 0x0100},  // Ā
    {0x0045, 0x0304, 0x0112},  // Ē
    {0x0049, 0x0304, 0x012A},  // Ī
    {0x004F, 0x0304, 0x014C},  // Ō
    {0x0055, 0x0304, 0x016A},  // Ū
    {0x0061, 0x0304, 0x0101},  // ā
    {0x0065, 0x0304, 0x0113},  // ē
    {0x0069, 0x0304, 0x012B},  // ī
    {0x006F, 0x0304, 0x014D},  // ō
    {0x0075, 0x0304, 0x016B},  // ū
};

static constexpr int kNfcTableSize = sizeof(kNfcTable) / sizeof(kNfcTable[0]);

// Decode one UTF-8 codepoint from text at position pos, advance pos.
// On truncated input (multi-byte sequence cut short at end of string),
// returns U+FFFD and advances pos to end-of-string, rather than returning
// a partial codepoint and advancing past the end.
static uint32_t nfc_decode_utf8(const std::string& s, size_t& pos) {
    uint8_t c = static_cast<uint8_t>(s[pos]);
    uint32_t cp;
    int len;
    if ((c & 0x80) == 0) {
        cp = c;
        len = 1;
    } else if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07;
        len = 4;
    } else {
        pos++;
        return 0xFFFD;
    }
    if (pos + static_cast<size_t>(len) > s.size()) {
        pos = s.size();
        return 0xFFFD;
    }
    for (int i = 1; i < len; i++) {
        cp = (cp << 6) | (static_cast<uint8_t>(s[pos + i]) & 0x3F);
    }
    pos += len;
    return cp;
}

// Encode a Unicode codepoint to UTF-8 and append to result
static void nfc_encode_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Check if a codepoint is a combining mark (Unicode General Category Mn/Mc/Me)
// Simplified: only checks the combining diacritical marks block (0x0300-0x036F)
// which covers the vast majority of combining marks in practice.
static bool is_combining_mark(uint32_t cp) { return (cp >= 0x0300 && cp <= 0x036F); }

// Look up composition in table
static uint32_t try_compose(uint32_t base, uint32_t combining) {
    for (int i = 0; i < kNfcTableSize; i++) {
        if (kNfcTable[i].base == base && kNfcTable[i].combining == combining) {
            return kNfcTable[i].composed;
        }
    }
    return 0;  // no composition found
}

// Normalize a UTF-8 string to NFC form (basic Latin coverage)
static std::string normalize_nfc(const std::string& text) {
    if (text.empty())
        return text;

    // Quick check: if no bytes in the combining mark range (0xCC-0xCD in UTF-8),
    // the text has no combining marks and is already NFC.
    bool has_combining = false;
    for (size_t i = 0; i + 1 < text.size(); i++) {
        uint8_t c = static_cast<uint8_t>(text[i]);
        if (c == 0xCC || c == 0xCD) {
            has_combining = true;
            break;
        }
    }
    if (!has_combining)
        return text;

    // Decode to codepoints, compose adjacent base+combining pairs
    std::vector<uint32_t> codepoints;
    size_t pos = 0;
    while (pos < text.size()) {
        codepoints.push_back(nfc_decode_utf8(text, pos));
    }

    // Compose: scan for base + combining mark pairs
    std::string result;
    result.reserve(text.size());

    size_t i = 0;
    while (i < codepoints.size()) {
        uint32_t cp = codepoints[i];

        // Try to compose with following combining marks
        while (i + 1 < codepoints.size() && is_combining_mark(codepoints[i + 1])) {
            uint32_t composed = try_compose(cp, codepoints[i + 1]);
            if (composed != 0) {
                cp = composed;
                i++;
            } else {
                break;  // can't compose further
            }
        }

        nfc_encode_utf8(result, cp);
        i++;
    }

    return result;
}

}  // anonymous namespace

// ---- Encode dispatch ----

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool no_prefix) const {
    // NFC normalization: compose decomposed Unicode sequences
    std::string normalized = normalize_nfc(text);
    if (type_ == "gpt2") {
        return encode_gpt2(normalized);
    }
    if (type_ == "gemma4") {
        return encode_gemma4(normalized);
    }
    if (type_ == "bert") {
        return encode_wordpiece(normalized);
    }
    return encode_spm(normalized, no_prefix);
}

// ---- BERT WordPiece (#836) ----
// Uncased basic tokenizer (ASCII lowercase, whitespace split, punctuation
// isolated) followed by greedy longest-match WordPiece: the first piece of a
// word matches the raw vocab entry, continuations match "##"-prefixed
// entries; a word with no full segmentation becomes [UNK]. Accent stripping
// (NFD + combining-mark removal) is not implemented — the uncased vocab is
// effectively ASCII and the HF-oracle cosine check gates correctness.
std::vector<int32_t> Tokenizer::encode_wordpiece(const std::string& text) const {
    std::vector<int32_t> out;
    auto it_unk = token_to_id_.find("[UNK]");
    const int32_t unk_id = (it_unk != token_to_id_.end()) ? it_unk->second : 100;

    std::vector<std::string> words;
    std::string cur;
    auto flush = [&] {
        if (!cur.empty()) {
            words.push_back(cur);
            cur.clear();
        }
    };
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            const char lc = static_cast<char>(std::tolower(c));
            if (std::isspace(static_cast<unsigned char>(lc))) {
                flush();
                ++i;
            } else if (std::ispunct(static_cast<unsigned char>(lc))) {
                flush();
                words.emplace_back(1, lc);
                ++i;
            } else {
                cur += lc;
                ++i;
            }
        } else {
            // Multibyte codepoint: keep as-is inside the current word.
            const size_t len = (c >= 0xF0) ? 4u : (c >= 0xE0) ? 3u : 2u;
            cur += text.substr(i, std::min(len, text.size() - i));
            i += len;
        }
    }
    flush();

    for (const auto& w : words) {
        std::vector<int32_t> pieces;
        size_t start = 0;
        bool bad = false;
        while (start < w.size()) {
            size_t end = w.size();
            int32_t id = -1;
            while (end > start) {
                // llama.cpp's BERT-GGUF vocab stores WordPiece in SPM
                // convention: word-initial pieces carry "▁", continuations
                // are bare (HF "##xyz" -> "xyz", "xyz" -> "▁xyz").
                std::string sub =
                    (start == 0 ? std::string("\xE2\x96\x81") : std::string()) +
                    w.substr(start, end - start);
                auto it = token_to_id_.find(sub);
                if (it != token_to_id_.end()) {
                    id = it->second;
                    break;
                }
                --end;
            }
            if (id < 0) {
                bad = true;
                break;
            }
            pieces.push_back(id);
            start = end;
        }
        if (bad)
            out.push_back(unk_id);
        else
            out.insert(out.end(), pieces.begin(), pieces.end());
    }
    return out;
}

// ---- Decode (SentencePiece) ----

std::string Tokenizer::decode_spm(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t tok : tokens) {
        result += decode_spm_token(tok);
    }
    // Second pass: byte-fallback tokens contribute single raw bytes that
    // together may form ▁ (U+2581 = 0xE2 0x96 0x81). The per-token replace
    // in decode_spm_token can't see across token boundaries, so catch any
    // remaining ▁ here and convert to ASCII space.
    size_t pos = 0;
    while ((pos = result.find(SPIECE_SPACE, pos)) != std::string::npos) {
        result.replace(pos, SPIECE_SPACE.size(), " ");
        pos += 1;
    }
    return result;
}

std::string Tokenizer::decode_spm_token(int32_t token) const {
    if (token < 0 || token >= static_cast<int32_t>(vocab_.size()))
        return "";

    std::string piece = vocab_[token];

    // Replace SentencePiece space marker with actual space
    size_t pos = 0;
    while ((pos = piece.find(SPIECE_SPACE, pos)) != std::string::npos) {
        piece.replace(pos, SPIECE_SPACE.size(), " ");
        pos += 1;
    }

    // Handle byte tokens: <0xHH> -> single byte
    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        unsigned int byte_val = 0;
        if (std::sscanf(piece.c_str(), "<0x%02X>", &byte_val) == 1) {
            return std::string(1, static_cast<char>(byte_val));
        }
    }

    return piece;
}

// ---- Decode (GPT2 byte-level) ----

std::string Tokenizer::decode_gpt2(const std::vector<int32_t>& tokens) const {
    init_codepoint_table();
    std::string result;
    for (int32_t tok : tokens) {
        result += decode_gpt2_token(tok);
    }
    return result;
}

std::string Tokenizer::decode_gpt2_token(int32_t token) const {
    if (token < 0 || token >= static_cast<int32_t>(vocab_.size()))
        return "";

    const std::string& piece = vocab_[token];
    std::string decoded;

    // Each UTF-8 character in piece represents a byte via GPT2 encoding
    for (size_t i = 0; i < piece.size();) {
        int len = utf8_char_len(static_cast<uint8_t>(piece[i]));
        if (i + len > piece.size())
            len = 1;

        int byte_val = gpt2_to_byte(piece.data() + i, len);
        if (byte_val >= 0) {
            decoded += static_cast<char>(byte_val);
        } else {
            // Not a GPT2 byte-encoded char, pass through
            decoded += piece.substr(i, len);
        }
        i += len;
    }

    return decoded;
}

// ---- Decode dispatch ----

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    if (type_ == "gpt2") {
        return decode_gpt2(tokens);
    }
    return decode_spm(tokens);
}

std::string Tokenizer::decode_token(int32_t token) const {
    if (type_ == "gpt2") {
        return decode_gpt2_token(token);
    }
    return decode_spm_token(token);
}

// ---- Accessors ----

int Tokenizer::vocab_size() const { return static_cast<int>(vocab_.size()); }

int Tokenizer::bos_id() const { return bos_id_; }

// eos_id() is now inline in tokenizer.h

int32_t Tokenizer::find_token(const std::string& text) const {
    auto it = token_to_id_.find(text);
    if (it != token_to_id_.end())
        return it->second;
    return -1;
}

}  // namespace imp
