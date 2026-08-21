#include "runtime/token_recycle_draft.h"

#include <algorithm>
#include <ranges>

namespace imp {

TokenRecycleTable::TokenRecycleTable(int vocab_size, int slots)
    : vocab_(vocab_size), slots_(slots),
      succ_(static_cast<size_t>(vocab_size) * slots, -1),
      streak_(static_cast<size_t>(vocab_size), 0) {}

// MRU slot shuffle only — returns true when `next` was already recorded
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
    // necessarily consecutively — consecutive-only measured ~zero recall on
    // fresh reasoning text: 2 drafts in 1024 tokens). A brand-new successor
    // resets it.
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
            promote_(token, id);
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

bool TokenRecycleTable::has(int32_t token) const {
    return valid_(token) && row_(token)[0] >= 0;
}

int32_t TokenRecycleTable::successor(int32_t token, int slot) const {
    if (!valid_(token) || slot < 0 || slot >= slots_)
        return -1;
    return row_(token)[slot];
}

}  // namespace imp
