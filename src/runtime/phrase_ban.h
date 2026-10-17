#pragma once

// Banned phrases (roadmap row 118): a text match on the decoded output rewinds the request to
// the phrase's first token and bans that token at that position. Pure helpers and the per-request
// state here; the engine side is runtime/engine_phrase_ban.cpp.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace imp {

inline constexpr int kMaxBannedPhrases = 64;
inline constexpr size_t kMaxBannedPhraseBytes = 256;
inline constexpr int kMaxPhraseRewinds = 64;  // per request; past it a phrase is let through

struct PhraseBanState {
    std::vector<std::string> phrases;
    size_t max_len = 0;
    std::string text;                                  // decoded output_tokens
    std::vector<uint32_t> tok_end;                     // byte end of output token i in text
    std::vector<std::pair<int, int32_t>> bans;         // (output position, banned token id)
    std::vector<std::pair<int32_t, float>> user_bias;  // the request's own logit_bias
    std::vector<float> first_logits;                   // raw logits of output position 0 (host)
    size_t safe = 0;                                   // output tokens no rewind can reach
    int rewinds = 0;
    // Think state at add_request: a rewind replays track_think_state from here.
    bool in_think0 = false;
    int think_exit0 = -1;
    bool content_after0 = false;
    std::string think_tail0;
};

// Earliest start byte of a phrase occurrence that ends after byte prev_len (only new text matches).
inline std::optional<size_t> phrase_match_start(std::string_view text, size_t prev_len,
                                                const std::vector<std::string>& phrases) {
    std::optional<size_t> best;
    for (const auto& p : phrases) {
        if (p.empty() || p.size() > text.size())
            continue;
        const size_t from = prev_len + 1 > p.size() ? prev_len + 1 - p.size() : 0;
        const size_t at = text.find(p, from);
        if (at != std::string_view::npos && (!best || at < *best))
            best = at;
    }
    return best;
}

// Output token holding byte b: the first i with tok_end[i] > b.
inline size_t phrase_token_at(const std::vector<uint32_t>& tok_end, size_t b) {
    return static_cast<size_t>(std::upper_bound(tok_end.begin(), tok_end.end(), static_cast<uint32_t>(b)) -
                               tok_end.begin());
}

// Tokens whose text ends at or before text_len - (max_len - 1): no later match can start in them.
inline size_t phrase_safe_tokens(const std::vector<uint32_t>& tok_end, size_t text_len, size_t max_len) {
    const size_t keep = max_len > 0 ? max_len - 1 : 0;
    if (text_len < keep)
        return 0;
    const size_t limit = text_len - keep;
    return static_cast<size_t>(
        std::upper_bound(tok_end.begin(), tok_end.end(), static_cast<uint32_t>(limit)) - tok_end.begin());
}

}  // namespace imp
