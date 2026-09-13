#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace imp {

// A draft and where it came from. `start` is the history index of tokens[0]
// (one past the matched n-gram), or -1 when nothing was drafted; the caller
// classifies the draft's source region (prompt / prediction / prior output)
// from it. Kept together: an out-parameter could let a caller read a stale
// start after the draft was discarded.
struct NgramDraft {
    std::vector<int32_t> tokens;
    int start = -1;

    bool empty() const { return tokens.empty(); }
    size_t size() const { return tokens.size(); }
};

// Prompt-lookup draft: find the most recent earlier occurrence of the
// longest suffix n-gram of `hist` (match length in [min_match, max_match])
// and return up to `k` tokens that followed it. Empty when no suffix of at
// least min_match tokens recurs, or the match is the suffix itself.
//
// Tie-breaking: longer match wins; among equal lengths, most recent wins
// (tracks local phrasing better than distant repeats).
[[nodiscard]] NgramDraft ngram_draft(std::span<const int32_t> hist, int k, int min_match, int max_match);

}  // namespace imp
