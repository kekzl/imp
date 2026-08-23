#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace imp {

// A draft and where it came from. `start` is the history index of tokens[0]
// (one past the matched n-gram), or -1 when nothing was drafted; the caller
// classifies the draft's source region (prompt / prediction / prior output)
// from it for accept accounting. The two travel together because a start
// without its tokens says nothing, and an out-parameter let a caller read a
// stale one after a draft was later discarded.
struct NgramDraft {
    std::vector<int32_t> tokens;
    int start = -1;

    bool empty() const { return tokens.empty(); }
    size_t size() const { return tokens.size(); }
};

// Prompt-lookup draft: find the most recent earlier occurrence of the
// longest suffix n-gram of `hist` (match length in [min_match, max_match])
// and return up to `k` tokens that followed that occurrence. Returns an
// empty draft when no suffix of at least min_match tokens recurs in the
// history, or when the match is the suffix itself.
//
// Tie-breaking: longer match wins; among equal lengths the most recent
// occurrence wins (recency tracks the model's local phrasing better than
// distant repeats).
[[nodiscard]] NgramDraft ngram_draft(std::span<const int32_t> hist, int k, int min_match, int max_match);

}  // namespace imp
