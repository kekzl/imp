#pragma once

#include "runtime/ngram_draft.h"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace imp {

// Suffix-indexed prompt-lookup drafting (SuffixDecoding-style, arXiv
// 2411.04975). Same contract as ngram_draft(): longest suffix match ->
// continuation. O(1) amortized hash-indexed matching, majority-vote
// continuation across occurrences (ties: longer match, then recency),
// adaptive length up to k_max on strong evidence. History is append-only,
// fed as input ++ prediction ++ output. Memory: ~4B/token history +
// ~16B/token index (a few MiB at 128k context).
class SuffixDraftIndex {
public:
    SuffixDraftIndex(int min_match, int max_match);

    // Append tokens to the indexed history.
    void append(std::span<const int32_t> toks);
    int size() const { return static_cast<int>(hist_.size()); }

    // Draft up to k (base) / k_max (evidence-backed) tokens continuing the
    // current history suffix. Returns an empty draft when no min_match suffix
    // gram recurs. `start` is the history index the winning continuation was
    // copied from, one past the matched occurrence, for source-region
    // classification (prompt / prediction / prior output).
    [[nodiscard]] NgramDraft draft(int k, int k_max) const;
    // Same, continuing the suffix of `query` from this index (a corpus, #2421). `start` = -1;
    // negative history tokens separate documents and end a continuation.
    [[nodiscard]] NgramDraft draft_from(std::span<const int32_t> query, int k, int k_max) const;
    [[nodiscard]] std::span<const int32_t> tokens() const { return hist_; }

private:
    [[nodiscard]] NgramDraft draft_impl_(const int32_t* q, int qn, int k, int k_max, bool self) const;
    uint64_t gram_hash_at_(int end) const;  // hash of hist_[end - min_match_, end)

    int min_match_;
    int max_match_;
    std::vector<int32_t> hist_;
    // gram hash → end positions (one past the gram), most recent last.
    // Capped per key (most recent kept) to bound vote cost on degenerate
    // histories (whitespace runs, repeated separators).
    std::unordered_map<uint64_t, std::vector<int32_t>> index_;
};

// Corpus index key in the engine's request-id -> index map (request ids count up from 0, #2421).
inline constexpr int kNgramCorpusKey = -1;

// idx's own draft, else the corpus continuation of idx's history (empty without a corpus).
[[nodiscard]] NgramDraft with_corpus_fallback(const SuffixDraftIndex& idx,
                                              const std::unordered_map<int, SuffixDraftIndex>& indexes, int k,
                                              int k_max);

}  // namespace imp
