// NFC composition for Tokenizer::encode (called only under an NFC normalizer, tokenizer.cpp).
#include "model/tokenizer.h"
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// NFC normalization: handles the most common combining sequences for Latin scripts
// (accented characters), which cover the vast majority of real-world NFC cases.

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

// Decodes one UTF-8 codepoint at pos, advances pos. On truncated input (multi-byte sequence
// cut short at end of string), returns U+FFFD and advances to end-of-string rather than
// returning a partial codepoint and reading past the end.
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

    // Quick check: combining marks U+0300-036F lead with 0xCC/0xCD, conjoining Hangul jamo
    // U+1100-11FF with 0xE1 0x84-0x87. Neither present: already NFC for this implementation.
    bool has_combining = false;
    for (size_t i = 0; i + 1 < text.size(); i++) {
        uint8_t c = static_cast<uint8_t>(text[i]);
        uint8_t c1 = static_cast<uint8_t>(text[i + 1]);
        if (c == 0xCC || c == 0xCD || (c == 0xE1 && c1 >= 0x84 && c1 <= 0x87)) {
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

        // Hangul (Unicode 3.12): L U+1100-1112 + V U+1161-1175 -> LV, LV + T U+11A8-11C2 -> LVT.
        constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
        constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28;
        if (cp >= kLBase && cp < kLBase + kLCount && i + 1 < codepoints.size() &&
            codepoints[i + 1] >= kVBase && codepoints[i + 1] < kVBase + kVCount) {
            cp = kSBase + ((cp - kLBase) * kVCount + (codepoints[i + 1] - kVBase)) * kTCount;
            i++;
        }
        if (cp >= kSBase && cp < kSBase + kLCount * kVCount * kTCount && (cp - kSBase) % kTCount == 0 &&
            i + 1 < codepoints.size() && codepoints[i + 1] > kTBase && codepoints[i + 1] < kTBase + kTCount) {
            cp += codepoints[i + 1] - kTBase;
            i++;
        }

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

std::string nfc_normalize(const std::string& text) { return normalize_nfc(text); }

}  // namespace imp
