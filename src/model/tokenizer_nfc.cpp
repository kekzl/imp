// NFC for Tokenizer::encode (called only under an NFC normalizer, tokenizer.cpp): UAX #15
// canonical decomposition, canonical ordering, canonical composition. Data: unicode_nfc_data.cpp.
#include "model/tokenizer.h"
#include "model/unicode_class.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

namespace {

using unicode::combining_class;
using unicode::decode_utf8_at;

constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kSCount = kLCount * kVCount * kTCount;

// A malformed byte stays a byte: raw units are starters that never decompose or compose.
struct Unit {
    uint32_t cp;
    bool raw;
};

const uint64_t* find_decomp(uint32_t cp) {
    const auto table = unicode::decomposition_table();
    const uint64_t key = static_cast<uint64_t>(cp) << 43;
    auto it = std::lower_bound(table.begin(), table.end(), key);
    return (it != table.end() && (*it >> 43) == cp) ? &*it : nullptr;
}

void decompose(uint32_t cp, std::vector<Unit>& out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const uint32_t s = cp - kSBase;
        out.push_back({kLBase + s / (kVCount * kTCount), false});
        out.push_back({kVBase + (s % (kVCount * kTCount)) / kTCount, false});
        if (s % kTCount)
            out.push_back({kTBase + s % kTCount, false});
        return;
    }
    const uint64_t* d = find_decomp(cp);
    if (!d) {
        out.push_back({cp, false});
        return;
    }
    decompose(static_cast<uint32_t>((*d >> 22) & 0x1FFFFF), out);
    const auto second = static_cast<uint32_t>((*d >> 1) & 0x1FFFFF);
    if (second)
        decompose(second, out);
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase &&
        b < kTBase + kTCount)
        return a + (b - kTBase);
    static const auto* pairs = [] {
        auto* m = new std::unordered_map<uint64_t, uint32_t>();
        for (uint64_t e : unicode::decomposition_table())
            if (e & 1)
                (*m)[((e >> 22) & 0x1FFFFF) << 21 | ((e >> 1) & 0x1FFFFF)] = static_cast<uint32_t>(e >> 43);
        return m;
    }();
    auto it = pairs->find(static_cast<uint64_t>(a) << 21 | b);
    return it == pairs->end() ? 0 : it->second;
}

void append_utf8(std::string& out, uint32_t cp) {
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

}  // namespace

std::string nfc_normalize(const std::string& text) {
    // Quick check: no codepoint NFC can change -> unchanged (ASCII and most precomposed text).
    bool stable = true;
    for (size_t i = 0, len = 1; i < text.size() && stable; i += len) {
        if (static_cast<unsigned char>(text[i]) < 0x80) {
            len = 1;
            continue;
        }
        const uint32_t cp = decode_utf8_at(text, i, &len);
        stable = len == 1 || !(unicode::cp_class(cp) & unicode::kNfcUnstable);
    }
    if (stable)
        return text;

    std::vector<Unit> units;
    units.reserve(text.size());
    for (size_t i = 0, len = 1; i < text.size(); i += len) {
        const uint32_t cp = decode_utf8_at(text, i, &len);
        if (len == 1 && cp >= 0x80)
            units.push_back({cp, true});
        else
            decompose(cp, units);
    }

    // Canonical ordering: stable sort of each run of non-starters by combining class.
    auto ccc = [](const Unit& u) { return u.raw ? 0 : combining_class(u.cp); };
    for (size_t i = 0; i < units.size();) {
        if (ccc(units[i]) == 0) {
            i++;
            continue;
        }
        size_t j = i;
        while (j < units.size() && ccc(units[j]) != 0)
            j++;
        std::stable_sort(units.begin() + i, units.begin() + j,
                         [&](const Unit& a, const Unit& b) { return ccc(a) < ccc(b); });
        i = j;
    }

    // Canonical composition: a mark joins the last starter unless blocked by a class >= its own.
    std::vector<Unit> out;
    out.reserve(units.size());
    size_t starter = std::string::npos;
    int last_ccc = 0;
    for (const Unit& u : units) {
        const int c = ccc(u);
        if (starter != std::string::npos && !u.raw && !out[starter].raw &&
            (last_ccc < c || (last_ccc == 0 && starter == out.size() - 1))) {
            if (const uint32_t comp = compose_pair(out[starter].cp, u.cp)) {
                out[starter].cp = comp;
                continue;
            }
        }
        if (c == 0)
            starter = out.size();
        last_ccc = c;
        out.push_back(u);
    }

    std::string result;
    result.reserve(text.size());
    for (const Unit& u : out) {
        if (u.raw)
            result += static_cast<char>(u.cp);
        else
            append_utf8(result, u.cp);
    }
    return result;
}

}  // namespace imp
