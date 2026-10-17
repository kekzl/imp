#include "runtime/phrase_regex.h"

#include "runtime/phrase_ban.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <cstdint>
#include <memory>
#include <string>

namespace imp {

namespace {

constexpr uint32_t kMatchLimit = 100000;  // backtracking steps per pcre2_match; past it: no match

pcre2_code* code_of(const std::shared_ptr<void>& p) { return static_cast<pcre2_code*>(p.get()); }

// One pattern's match data and backtracking limit.
struct Matcher {
    pcre2_code* code;
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> md;
    std::unique_ptr<pcre2_match_context, decltype(&pcre2_match_context_free)> mc;
    explicit Matcher(pcre2_code* c)
        : code(c),
          md(pcre2_match_data_create_from_pattern(c, nullptr), &pcre2_match_data_free),
          mc(pcre2_match_context_create(nullptr), &pcre2_match_context_free) {
        if (mc)
            pcre2_set_match_limit(mc.get(), kMatchLimit);
    }
    // Match from byte `off`: true and [start, end) on a match.
    bool at(std::string_view text, size_t off, size_t& start, size_t& end) {
        if (!md || !mc)
            return false;
        if (pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), off, 0, md.get(),
                        mc.get()) < 0)
            return false;
        const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md.get());
        start = ov[0];
        end = ov[1];
        return true;
    }
};

}  // namespace

std::shared_ptr<const PhraseRegex> PhraseRegex::compile(const std::string& pattern, std::string& why) {
    int err = 0;
    PCRE2_SIZE at = 0;
    pcre2_code* code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
                                     PCRE2_UTF | PCRE2_MATCH_INVALID_UTF, &err, &at, nullptr);
    if (code == nullptr) {
        PCRE2_UCHAR msg[256];
        pcre2_get_error_message(err, msg, sizeof(msg));
        why = "pattern does not compile at offset " + std::to_string(at) + ": " +
              reinterpret_cast<const char*>(msg);
        return nullptr;
    }
    auto r = std::shared_ptr<PhraseRegex>(new PhraseRegex());
    r->code_ = std::shared_ptr<void>(code, [](void* c) { pcre2_code_free(static_cast<pcre2_code*>(c)); });
    size_t s = 0, e = 0;
    if (Matcher(code).at("", 0, s, e)) {
        why = "pattern matches the empty string";
        return nullptr;
    }
    return r;
}

std::optional<size_t> PhraseRegex::match_start(std::string_view text, size_t from, size_t prev_len) const {
    Matcher m(code_of(code_));
    size_t off = from;
    while (off < text.size()) {
        size_t s = 0, e = 0;
        if (!m.at(text, off, s, e))
            return std::nullopt;
        if (e > s && e > prev_len)
            return s;
        // Next candidate start: one character after this match's start.
        off = s + 1;
        while (off < text.size() && (static_cast<unsigned char>(text[off]) & 0xC0) == 0x80)
            off++;
    }
    return std::nullopt;
}

std::optional<size_t> phrase_hit(const PhraseBanState& s, std::string_view text, size_t prev_len) {
    std::optional<size_t> best = phrase_match_start(text, prev_len, s.phrases);
    const size_t from = text.size() > kMaxBannedPhraseBytes ? text.size() - kMaxBannedPhraseBytes : 0;
    for (const auto& re : s.regex)
        if (const auto at = re->match_start(text, from, prev_len); at && (!best || *at < *best))
            best = at;
    return best;
}

}  // namespace imp
