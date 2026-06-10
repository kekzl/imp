#pragma once

#include <cstdint>
#include <vector>

namespace imp {

// Prompt-lookup draft: find the most recent earlier occurrence of the
// longest suffix n-gram of `hist[0..n)` (match length in [min_match,
// max_match]) and return up to `k` tokens that followed that occurrence.
// Returns an empty vector when no suffix of at least min_match tokens
// recurs in the history, or when the match is the suffix itself.
//
// Tie-breaking: longer match wins; among equal lengths the most recent
// occurrence wins (recency tracks the model's local phrasing better than
// distant repeats).
std::vector<int32_t> ngram_draft(const int32_t* hist, int n, int k, int min_match, int max_match);

}  // namespace imp
