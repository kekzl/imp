#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace imp {

// Suffix-indexed prompt-lookup drafting (SuffixDecoding-style, arXiv
// 2411.04975). Same query contract as ngram_draft() — "longest suffix
// match → continuation" — with three upgrades:
//
//   1. O(1) amortized matching: every min_match-gram of the history is
//      hash-indexed as tokens arrive, instead of an O(n) backward scan
//      per verify step (the scan also required rebuilding the full
//      input++prediction++output vector each step).
//   2. Frequency-voted continuations: the draft follows the majority
//      next-token across ALL occurrences of the matched suffix, not the
//      single most recent one (ties: longer context match, then recency).
//   3. Adaptive draft length: a continuation backed by strong evidence —
//      multiple agreeing occurrences, or a maximal-length (max_match)
//      context match such as the OpenAI `prediction` region — extends
//      past the base k up to k_max.
//
// The index owns a copy of the history (append-only; the engine feeds
// input ++ prediction ++ output incrementally). Host memory only:
// ~4 B/token history + ~16 B/token index — a few MiB at 128k context.
class SuffixDraftIndex {
public:
    SuffixDraftIndex(int min_match, int max_match);

    // Append tokens to the indexed history.
    void append(const int32_t* toks, int n);
    int size() const { return static_cast<int>(hist_.size()); }

    // Draft up to k (base) / k_max (evidence-backed) tokens continuing the
    // current history suffix. Returns empty when no min_match suffix gram
    // recurs. draft_start (optional) receives the history index the winning
    // continuation was copied from — one past the matched occurrence — for
    // source-region classification (prompt / prediction / prior output).
    std::vector<int32_t> draft(int k, int k_max, int* draft_start = nullptr) const;

private:
    uint64_t gram_hash_at_(int end) const;  // hash of hist_[end - min_match_, end)

    int min_match_;
    int max_match_;
    std::vector<int32_t> hist_;
    // gram hash → end positions (one past the gram), most recent last.
    // Capped per key (most recent kept) to bound vote cost on degenerate
    // histories (whitespace runs, repeated separators).
    std::unordered_map<uint64_t, std::vector<int32_t>> index_;
};

}  // namespace imp
