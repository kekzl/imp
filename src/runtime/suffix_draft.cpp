#include "runtime/suffix_draft.h"

#include <algorithm>
#include <tuple>

namespace imp {

namespace {
// Occurrence-list cap per gram key. Voting cost per drafted token is
// O(survivors); degenerate histories (whitespace runs, separator-heavy
// tables) would otherwise accumulate thousands of occurrences of the same
// gram. Most recent occurrences are kept: they track local phrasing best.
constexpr int kMaxOccurrences = 64;
}  // namespace

SuffixDraftIndex::SuffixDraftIndex(int min_match, int max_match)
    : min_match_(std::max(1, min_match)), max_match_(std::max(max_match, min_match_)) {}

// FNV-1a over the `len` tokens ending at `end`, finalized with a splitmix64-style mix.
// Collisions are guarded by token comparison in draft(), so hash quality only affects
// bucket balance.
static uint64_t gram_hash(const int32_t* end, int len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const int32_t* p = end - len; p < end; ++p) {
        h ^= static_cast<uint32_t>(*p);
        h *= 0x100000001b3ULL;
    }
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    return h;
}

uint64_t SuffixDraftIndex::gram_hash_at_(int end) const { return gram_hash(hist_.data() + end, min_match_); }

void SuffixDraftIndex::append(std::span<const int32_t> toks) {
    if (toks.empty())
        return;
    const int n = static_cast<int>(toks.size());
    hist_.insert(hist_.end(), toks.begin(), toks.end());
    const int total = static_cast<int>(hist_.size());
    // A gram window [end - min_match, end) is new iff it covers at least one
    // appended token, i.e. end > total - n (windows straddling the boundary
    // included: they were not indexable before this append).
    for (int end = std::max(min_match_, total - n + 1); end <= total; ++end) {
        auto& occ = index_[gram_hash_at_(end)];
        if (static_cast<int>(occ.size()) >= kMaxOccurrences)
            occ.erase(occ.begin());
        occ.push_back(end);
    }
}

NgramDraft SuffixDraftIndex::draft(int k, int k_max) const {
    return draft_impl_(hist_.data(), static_cast<int>(hist_.size()), k, k_max, /*self=*/true);
}

NgramDraft SuffixDraftIndex::draft_from(std::span<const int32_t> query, int k, int k_max) const {
    NgramDraft d = draft_impl_(query.data(), static_cast<int>(query.size()), k, k_max, /*self=*/false);
    d.start = -1;
    return d;
}

namespace {

struct Candidate {
    int end;  // one past the matched gram
    int len;  // backward match length
};

// Occurrences `occ` of q's suffix gram in hist (collision-checked), each with its backward
// context-match length in [min_match, max_match]. Ends at n (the suffix itself) are skipped.
std::vector<Candidate> collect_candidates(const std::vector<int32_t>& hist, const std::vector<int32_t>& occ,
                                          const int32_t* q, int qn, int min_match, int max_match) {
    const int n = static_cast<int>(hist.size());
    const int32_t* suffix = q + qn - min_match;
    std::vector<Candidate> cands;
    cands.reserve(occ.size());
    for (const int end : occ) {
        if (end >= n || !std::equal(suffix, suffix + min_match, hist.data() + end - min_match))
            continue;  // the suffix itself / no continuation, or a hash collision
        int len = min_match;
        while (len < max_match && end - len - 1 >= 0 && qn - len - 1 >= 0 &&
               hist[end - len - 1] == q[qn - len - 1])
            ++len;
        cands.push_back({end, len});
    }
    return cands;
}

struct Vote {
    int32_t tok = -1;
    int votes = 0, len = 0, end = -1, voters = 0;
};

// Majority token at continuation offset i (ties: longer match, then recency); `live` excludes
// exhausted occurrences.
template <class Live>
Vote vote_at(const std::vector<int32_t>& hist, const std::vector<Candidate>& cands, int i, const Live& live) {
    Vote best;
    for (const auto& c : cands) {
        if (!live(c.end + i))
            continue;  // exhausted
        ++best.voters;
        const int32_t tok = hist[c.end + i];
        int votes = 0, longest = 0, recent = -1;
        for (const auto& d : cands) {
            if (live(d.end + i) && hist[d.end + i] == tok) {
                ++votes;
                longest = std::max(longest, d.len);
                recent = std::max(recent, d.end);
            }
        }
        if (std::tie(votes, longest, recent) > std::tie(best.votes, best.len, best.end)) {
            best.tok = tok;
            best.votes = votes;
            best.len = longest;
            best.end = recent;
        }
    }
    return best;
}

}  // namespace

// q / qn: the history whose suffix is continued (hist_ itself when self). An occurrence's
// continuation ends at the history end or at a negative separator token.
NgramDraft SuffixDraftIndex::draft_impl_(const int32_t* q, int qn, int k, int k_max, bool self) const {
    const int n = static_cast<int>(hist_.size());
    if (k <= 0 || qn < min_match_ + (self ? 1 : 0) || n == 0)
        return {};
    k_max = std::max(k, k_max);
    const auto it = index_.find(gram_hash(q + qn, min_match_));
    if (it == index_.end())
        return {};
    std::vector<Candidate> cands = collect_candidates(hist_, it->second, q, qn, min_match_, max_match_);

    // Frequency-voted forward walk. Survivors are occurrences whose
    // continuation matched every drafted token so far.
    auto live = [&](int pos) { return pos < n && hist_[pos] >= 0; };
    std::vector<int32_t> out;
    out.reserve(k);
    int rep_end = -1;  // representative survivor (longest len, then most recent)
    for (int i = 0; i < k_max && !cands.empty(); ++i) {
        const Vote v = vote_at(hist_, cands, i, live);
        if (v.voters == 0)
            break;
        // Past the base k, extend only on strong evidence: multiple
        // agreeing occurrences, or a maximal-length context match (e.g.
        // the prediction region tracking the completion token-exact).
        const bool unanimous = v.votes == v.voters;
        if (i >= k && !(unanimous && (v.votes >= 2 || v.len >= max_match_)))
            break;
        out.push_back(v.tok);
        if (i == 0)
            rep_end = v.end;  // source region of the draft's first token
        // Drop disagreeing/exhausted occurrences.
        std::erase_if(cands,
                      [&](const Candidate& c) { return !live(c.end + i) || hist_[c.end + i] != v.tok; });
    }
    if (out.empty())
        return {};
    return {std::move(out), rep_end};
}

NgramDraft with_corpus_fallback(const SuffixDraftIndex& idx,
                                const std::unordered_map<int, SuffixDraftIndex>& indexes, int k, int k_max) {
    NgramDraft nd = idx.draft(k, k_max);
    if (!nd.empty())
        return nd;
    const auto it = indexes.find(kNgramCorpusKey);
    return it == indexes.end() ? nd : it->second.draft_from(idx.tokens(), k, k_max);
}

}  // namespace imp
