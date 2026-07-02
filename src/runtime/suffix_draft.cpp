#include "runtime/suffix_draft.h"

#include <algorithm>

namespace imp {

namespace {
// Occurrence-list cap per gram key. Voting cost per drafted token is
// O(survivors); degenerate histories (whitespace runs, separator-heavy
// tables) would otherwise accumulate thousands of occurrences of the same
// gram. Most recent occurrences are kept — they track local phrasing best.
constexpr int kMaxOccurrences = 64;
}  // namespace

SuffixDraftIndex::SuffixDraftIndex(int min_match, int max_match)
    : min_match_(std::max(1, min_match)), max_match_(std::max(max_match, min_match_)) {}

uint64_t SuffixDraftIndex::gram_hash_at_(int end) const {
    // FNV-1a over the min_match_ tokens ending at `end`, finalized with a
    // splitmix64-style mix. Collisions are guarded by token comparison in
    // draft(), so hash quality only affects bucket balance.
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = end - min_match_; i < end; ++i) {
        h ^= static_cast<uint32_t>(hist_[i]);
        h *= 0x100000001b3ULL;
    }
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    return h;
}

void SuffixDraftIndex::append(const int32_t* toks, int n) {
    if (toks == nullptr || n <= 0)
        return;
    hist_.insert(hist_.end(), toks, toks + n);
    const int total = static_cast<int>(hist_.size());
    // A gram window [end - min_match, end) is new iff it covers at least one
    // appended token, i.e. end > total - n (windows straddling the boundary
    // included — they were not indexable before this append).
    for (int end = std::max(min_match_, total - n + 1); end <= total; ++end) {
        auto& occ = index_[gram_hash_at_(end)];
        if (static_cast<int>(occ.size()) >= kMaxOccurrences)
            occ.erase(occ.begin());
        occ.push_back(end);
    }
}

std::vector<int32_t> SuffixDraftIndex::draft(int k, int k_max, int* draft_start) const {
    if (draft_start)
        *draft_start = -1;
    const int n = static_cast<int>(hist_.size());
    if (k <= 0 || n < min_match_ + 1)
        return {};
    k_max = std::max(k, k_max);

    const auto it = index_.find(gram_hash_at_(n));
    if (it == index_.end())
        return {};

    // Candidate occurrences of the current suffix gram (collision-checked),
    // each with its backward context-match length in [min_match_, max_match_].
    const int32_t* suffix = hist_.data() + n - min_match_;
    struct Candidate {
        int end;  // one past the matched gram
        int len;  // backward match length
    };
    std::vector<Candidate> cands;
    cands.reserve(it->second.size());
    for (const int end : it->second) {
        if (end >= n)
            continue;  // the suffix itself
        if (!std::equal(suffix, suffix + min_match_, hist_.data() + end - min_match_))
            continue;  // hash collision
        int len = min_match_;
        while (len < max_match_ && end - len - 1 >= 0 && n - len - 1 >= 0 &&
               hist_[end - len - 1] == hist_[n - len - 1])
            ++len;
        cands.push_back({end, len});
    }
    if (cands.empty())
        return {};

    // Frequency-voted forward walk. Survivors are occurrences whose
    // continuation matched every drafted token so far.
    std::vector<int32_t> out;
    out.reserve(k);
    int rep_end = -1;  // representative survivor (longest len, then most recent)
    for (int i = 0; i < k_max; ++i) {
        int32_t best_tok = -1;
        int best_votes = 0, best_len = 0, best_end = -1;
        int voters = 0;
        for (const auto& c : cands) {
            if (c.end + i >= n)
                continue;  // exhausted
            ++voters;
            const int32_t tok = hist_[c.end + i];
            int votes = 0, longest = 0, recent = -1;
            for (const auto& d : cands) {
                if (d.end + i < n && hist_[d.end + i] == tok) {
                    ++votes;
                    longest = std::max(longest, d.len);
                    recent = std::max(recent, d.end);
                }
            }
            if (votes > best_votes || (votes == best_votes && longest > best_len) ||
                (votes == best_votes && longest == best_len && recent > best_end)) {
                best_tok = tok;
                best_votes = votes;
                best_len = longest;
                best_end = recent;
            }
        }
        if (voters == 0)
            break;
        // Past the base k, extend only on strong evidence: multiple
        // agreeing occurrences, or a maximal-length context match (e.g.
        // the prediction region tracking the completion token-exact).
        const bool unanimous = best_votes == voters;
        if (i >= k && !(unanimous && (best_votes >= 2 || best_len >= max_match_)))
            break;
        out.push_back(best_tok);
        if (i == 0)
            rep_end = best_end;  // source region of the draft's first token
        // Drop disagreeing/exhausted occurrences.
        cands.erase(std::remove_if(cands.begin(), cands.end(),
                                   [&](const Candidate& c) {
                                       return c.end + i >= n || hist_[c.end + i] != best_tok;
                                   }),
                    cands.end());
    }
    if (out.empty())
        return {};
    if (draft_start)
        *draft_start = rep_end;
    return out;
}

}  // namespace imp
