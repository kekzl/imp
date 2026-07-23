#pragma once

#include <cstdint>
#include <vector>

namespace imp {

// Token-Recycling adjacency drafter (Token Recycling, ACL 2025, arXiv
// 2408.08696; plan: docs/plans/2026-07-22-token-recycling-spec-tree.md).
//
// Engine-scoped, cross-request table `token -> top-M likely successors`
// (MRU/rank ordered, -1 = empty). Fed from (a) emitted-token bigrams and
// (b) the model's own per-step top-K logit ids harvested in the verify
// chunk. Unlike suffix/n-gram prompt-lookup it fires on unigram context —
// the last emitted token has almost always been seen — so it drafts on
// fresh reasoning text where suffix matching finds nothing. Drafts are
// verified losslessly (greedy argmax accept), so a wrong draft can only
// cost speed, never token identity.
//
// Host memory: vocab_size * slots * 4 B (150k vocab @ M=8 ≈ 4.8 MiB).
class TokenRecycleTable {
public:
    TokenRecycleTable(int vocab_size, int slots);

    // Record that `next` followed `prev` in emitted text. Promotes `next`
    // to the front slot of `prev` (MRU, deduplicated). Out-of-range ids
    // are ignored.
    void observe_pair(int32_t prev, int32_t next);

    // Record the model's top-K successor candidates for `token` (best
    // first, e.g. from the verify-chunk logits). Establishes rank order:
    // ids[0] lands in slot 0. Out-of-range ids are skipped.
    void observe_topk(int32_t token, const int32_t* ids, int n);

    // Follow the front-slot successor chain from t0 for up to k tokens.
    // Stops early when a token has no successors. Cycles are allowed —
    // bounded by k; the lossless verify is the safety net.
    std::vector<int32_t> draft_linear(int32_t t0, int k) const;

    // Multi-candidate draft (route (a) of the spec-tree plan): up to
    // `width` candidates, one per recorded successor of t0 (rank order);
    // candidate i starts with successor(t0, i) and continues along
    // front-slot chains up to `depth` tokens. Empty when t0 is unseen.
    std::vector<std::vector<int32_t>> draft_candidates(int32_t t0, int width, int depth) const;

    // True when `token` has at least one recorded successor.
    bool has(int32_t token) const;

    // Successor id in `slot` (0 = most recent / best), -1 when empty or
    // out of range. Exposed for tests and the tree drafter.
    int32_t successor(int32_t token, int slot) const;

private:
    bool valid_(int32_t tok) const { return tok >= 0 && tok < vocab_; }
    int32_t* row_(int32_t tok) { return succ_.data() + static_cast<size_t>(tok) * slots_; }
    const int32_t* row_(int32_t tok) const {
        return succ_.data() + static_cast<size_t>(tok) * slots_;
    }

    int vocab_;
    int slots_;
    std::vector<int32_t> succ_;  // vocab_ * slots_, -1 = empty
};

}  // namespace imp
