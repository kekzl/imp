#include "runtime/token_recycle_draft.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <ranges>

namespace imp {

TokenRecycleTable::TokenRecycleTable(int vocab_size, int slots)
    : vocab_(vocab_size),
      slots_(slots),
      succ_(static_cast<size_t>(vocab_size) * slots, -1),
      streak_(static_cast<size_t>(vocab_size), 0),
      rank_hits_(static_cast<size_t>(std::max(0, slots)), 0) {}

// MRU slot shuffle only: returns true when `next` was already recorded
// (a RE-observation). Streak accounting stays with the callers.
bool TokenRecycleTable::promote_(int32_t prev, int32_t next) {
    int32_t* r = row_(prev);
    // Find existing occurrence (or the end of the used region).
    int pos = slots_ - 1;
    bool existed = false;
    for (int i = 0; i < slots_; ++i) {
        if (r[i] == next) {
            pos = i;
            existed = true;
            break;
        }
        if (r[i] == -1) {
            pos = i;
            break;
        }
    }
    // Shift [0, pos) down one and put `next` at the front (MRU).
    for (int i = pos; i > 0; --i)
        r[i] = r[i - 1];
    r[0] = next;
    return existed;
}

void TokenRecycleTable::observe_pair(int32_t prev, int32_t next) {
    if (!valid_(prev) || !valid_(next))
        return;
    // Streak = the front slot is a RE-observed pair (seen before, not
    // necessarily consecutively: consecutive-only misses most fresh
    // reasoning text). A brand-new successor resets it.
    streak_[prev] = promote_(prev, next)
                        ? static_cast<uint8_t>(std::min(255, streak_[prev] + 1))
                        : uint8_t{0};
}

void TokenRecycleTable::observe_topk(int32_t token, std::span<const int32_t> ids) {
    if (!valid_(token) || ids.empty())
        return;
    // Streak follows the model's rank-0 candidate: re-observed -> confirm,
    // brand-new -> reset. Lower ranks only refresh the slot pool.
    bool front_existed = false;
    if (valid_(ids[0])) {
        const int32_t* r = row_(token);
        for (int i = 0; i < slots_; ++i)
            if (r[i] == ids[0]) {
                front_existed = true;
                break;
            }
    }
    // Promote in reverse rank order so ids[0] ends up in slot 0.
    for (const int32_t id : std::views::reverse(ids))
        if (valid_(id))
            (void)promote_(token, id);  // return = "pair existed", streak uses front_existed
    streak_[token] = front_existed
                         ? static_cast<uint8_t>(std::min(255, streak_[token] + 1))
                         : uint8_t{0};
}

std::vector<int32_t> TokenRecycleTable::draft_linear(int32_t t0, int k, int min_streak) const {
    std::vector<int32_t> out;
    int32_t cur = t0;
    for (int i = 0; i < k; ++i) {
        if (!valid_(cur))
            break;
        if (streak_[cur] < min_streak)
            break;
        int32_t s = row_(cur)[0];
        if (s < 0)
            break;
        out.push_back(s);
        cur = s;
    }
    return out;
}

std::vector<std::vector<int32_t>> TokenRecycleTable::draft_candidates(int32_t t0, int width,
                                                                      int depth) const {
    std::vector<std::vector<int32_t>> out;
    if (!valid_(t0) || width <= 0 || depth <= 0)
        return out;
    const int n = std::min(width, slots_);
    for (int c = 0; c < n; ++c) {
        const int32_t first = row_(t0)[c];
        if (first < 0)
            break;  // slots are packed front-first
        std::vector<int32_t> cand;
        cand.push_back(first);
        int32_t cur = first;
        for (int j = 1; j < depth; ++j) {
            const int32_t s = row_(cur)[0];
            if (s < 0)
                break;
            cand.push_back(s);
            cur = s;
        }
        out.push_back(std::move(cand));
    }
    return out;
}

std::vector<std::vector<int32_t>> TokenRecycleTable::draft_tree(int32_t t0, int width, int depth) const {
    std::vector<std::vector<int32_t>> out;
    if (!valid_(t0) || width <= 0 || depth <= 0 || row_(t0)[0] < 0)
        return out;
    // Cost of a hop through slot r = log2(rate(0) / rate(r)): slot 0 is free, so path length is not
    // penalised (prior: cost = r, the rank sum). A deviation at depth j costs j / 1024
    // more than at depth 0: a miss is likeliest at the first hop, so an early alternative pays more.
    std::vector<double> hop(static_cast<size_t>(slots_));
    for (int r = 0; r < slots_; ++r)
        hop[static_cast<size_t>(r)] = std::max(0.0, std::log2(rank_hit_rate(0) / rank_hit_rate(r)));
    struct Node {
        double cost;
        int64_t seq;  // FIFO among equal costs: deterministic order
        std::vector<int32_t> toks;
    };
    auto worse = [](const Node& a, const Node& b) {
        return a.cost != b.cost ? a.cost > b.cost : a.seq > b.seq;
    };
    std::priority_queue<Node, std::vector<Node>, decltype(worse)> open(worse);
    int64_t seq = 0;
    open.push({0.0, seq++, {}});
    // Each pop pushes at most slots_ children: the bound keeps a sparse table from looping.
    for (int pops = 0;
         !open.empty() && static_cast<int>(out.size()) < width && pops < width * depth * slots_ + 1; ++pops) {
        Node n = open.top();
        open.pop();
        const int32_t cur = n.toks.empty() ? t0 : n.toks.back();
        const int d = static_cast<int>(n.toks.size());
        if (d == depth || !valid_(cur) || row_(cur)[0] < 0) {
            if (!n.toks.empty())
                out.push_back(std::move(n.toks));
            continue;
        }
        for (int r = 0; r < slots_; ++r) {
            const int32_t s = row_(cur)[r];
            if (s < 0)
                break;  // slots are packed front-first
            Node c{n.cost + hop[static_cast<size_t>(r)] + (r > 0 ? d / 1024.0 : 0.0), seq++, n.toks};
            c.toks.push_back(s);
            open.push(std::move(c));
        }
    }
    return out;
}

void TokenRecycleTable::observe_rank(int32_t prev, int32_t next) {
    if (!valid_(prev) || !valid_(next) || row_(prev)[0] < 0)
        return;  // no draft could have come from an empty row
    ++rank_samples_;
    for (int r = 0; r < slots_; ++r)
        if (row_(prev)[r] == next) {
            ++rank_hits_[static_cast<size_t>(r)];
            return;
        }
}

double TokenRecycleTable::rank_hit_rate(int rank) const {
    if (rank < 0 || rank >= slots_)
        return 0.0;
    constexpr double kPriorSamples = 16.0;
    const double prior = std::ldexp(1.0, -(rank + 1));
    return (static_cast<double>(rank_hits_[static_cast<size_t>(rank)]) + kPriorSamples * prior) /
           (static_cast<double>(rank_samples_) + kPriorSamples);
}

void observe_candidate_ranks(TokenRecycleTable& tr, bool enabled, int32_t t0,
                             const std::vector<std::vector<int32_t>>& mc, const int32_t* argmax) {
    if (!enabled)
        return;
    size_t rows_per_cand = 1;  // the engine's mc_rows_per_cand: 1 + the longest candidate
    for (const auto& cand : mc)
        rows_per_cand = std::max(rows_per_cand, 1 + cand.size());
    for (size_t c = 0; c < mc.size(); ++c)
        for (size_t r = 0; r <= mc[c].size(); ++r) {
            bool seen = false;
            for (size_t e = 0; e < c && !seen; ++e)
                seen = mc[e].size() >= r && std::equal(mc[c].begin(), mc[c].begin() + r, mc[e].begin());
            if (!seen)
                tr.observe_rank(r == 0 ? t0 : mc[c][r - 1], argmax[c * rows_per_cand + r]);
        }
}

bool TokenRecycleTable::has(int32_t token) const {
    return valid_(token) && row_(token)[0] >= 0;
}

int32_t TokenRecycleTable::successor(int32_t token, int slot) const {
    if (!valid_(token) || slot < 0 || slot >= slots_)
        return -1;
    return row_(token)[slot];
}

}  // namespace imp
