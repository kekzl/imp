#pragma once

// KV eviction block selection (kv_cache.evict_keep_ratio, docs/plans/2026-11-01-kv-eviction.md).
// Pure host function: per-token scores in, logical block indices to free out. Unit-tested on the
// CPU (tests/test_kv_evict_select.cpp).

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

namespace imp {

struct KvEvictParams {
    float keep_ratio = 0.0f;  // fraction of the context's blocks kept (sinks + recent included)
    int sink_tokens = 0;      // blocks covering [0, sink_tokens) always kept
    int recent_tokens = 0;    // blocks covering [ctx_len - recent_tokens, ctx_len) always kept
    int block_size = 16;
};

// Token range [begin, end) holding the eviction candidates: full blocks between the sink blocks
// and the first block the recent window touches. begin == end when nothing is evictable.
struct KvEvictRange {
    int begin = 0;
    int end = 0;
};
inline KvEvictRange kv_evict_candidates(int ctx_len, const KvEvictParams& p) {
    const int bs = std::max(1, p.block_size);
    if (ctx_len <= 0)
        return {};
    const int sink_blocks = (std::max(0, p.sink_tokens) + bs - 1) / bs;
    const int recent_start = std::min(std::max(0, ctx_len - std::max(1, p.recent_tokens)) / bs, ctx_len / bs);
    if (recent_start <= sink_blocks)
        return {};
    return {sink_blocks * bs, recent_start * bs};
}

// Block score = max of its token scores; NaN reads as -inf. Kept budget =
// ceil(keep_ratio * n_blocks) blocks; forced blocks (sinks, recent window, the partial tail)
// count against it. The top-scoring candidates fill the rest, ties keep the earlier block.
// Returns the evicted logical block indices in ascending order (only full blocks).
inline std::vector<int> kv_evict_select(std::span<const float> token_scores, int ctx_len,
                                        const KvEvictParams& p) {
    std::vector<int> evict;
    const int bs = std::max(1, p.block_size);
    if (ctx_len <= 0 || p.keep_ratio <= 0.0f || p.keep_ratio >= 1.0f)
        return evict;
    const int n_blocks = (ctx_len + bs - 1) / bs;
    // The partial tail block is always inside the recent window.
    const KvEvictRange r = kv_evict_candidates(ctx_len, p);
    if (r.end <= r.begin)
        return evict;
    const int sink_blocks = r.begin / bs;
    const int recent_start = r.end / bs;
    const int budget = static_cast<int>(std::ceil(static_cast<double>(p.keep_ratio) * n_blocks));
    const int forced = sink_blocks + (n_blocks - recent_start);
    const int n_cand = recent_start - sink_blocks;
    const int slots = std::clamp(budget - forced, 0, n_cand);
    if (slots >= n_cand)
        return evict;

    std::vector<float> score(static_cast<size_t>(n_cand), -INFINITY);
    for (int b = 0; b < n_cand; ++b) {
        const int t0 = (sink_blocks + b) * bs;
        const int t1 = std::min(t0 + bs, std::min(ctx_len, static_cast<int>(token_scores.size())));
        float m = -INFINITY;
        for (int t = t0; t < t1; ++t) {
            const float v = token_scores[static_cast<size_t>(t)];
            if (!std::isnan(v))
                m = std::max(m, v);
        }
        score[static_cast<size_t>(b)] = m;
    }
    std::vector<int> order(static_cast<size_t>(n_cand));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return score[static_cast<size_t>(a)] > score[static_cast<size_t>(b)];
    });
    std::vector<char> keep(static_cast<size_t>(n_cand), 0);
    for (int i = 0; i < slots; ++i)
        keep[static_cast<size_t>(order[static_cast<size_t>(i)])] = 1;
    for (int b = 0; b < n_cand; ++b)
        if (!keep[static_cast<size_t>(b)])
            evict.push_back(sink_blocks + b);
    return evict;
}

}  // namespace imp
