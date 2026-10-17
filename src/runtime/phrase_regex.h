#pragma once

// Banned regex patterns (roadmap row 118): PCRE2 (UTF, invalid UTF tolerated: a token boundary may
// split a character), compiled once per request by the server. phrase_hit combines them with the
// literal phrases of runtime/phrase_ban.h.

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace imp {

struct PhraseBanState;

inline constexpr int kMaxBannedRegex = 16;

class PhraseRegex {
public:
    // nullptr with `why` set: PCRE2 rejects the pattern, or it matches the empty string.
    static std::shared_ptr<const PhraseRegex> compile(const std::string& pattern, std::string& why);
    // Start byte of the leftmost non-empty match at or after byte `from` that ends after byte prev_len.
    [[nodiscard]] std::optional<size_t> match_start(std::string_view text, size_t from,
                                                    size_t prev_len) const;

private:
    std::shared_ptr<void> code_;  // pcre2_code, freed with pcre2_code_free
};

// Earliest start of a literal phrase or a pattern match (inside the last kMaxBannedPhraseBytes)
// that ends after byte prev_len.
[[nodiscard]] std::optional<size_t> phrase_hit(const PhraseBanState& s, std::string_view text,
                                               size_t prev_len);

}  // namespace imp
