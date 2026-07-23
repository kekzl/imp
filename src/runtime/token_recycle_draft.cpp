#include "runtime/token_recycle_draft.h"

#include <algorithm>

namespace imp {

TokenRecycleTable::TokenRecycleTable(int vocab_size, int slots)
    : vocab_(vocab_size), slots_(slots),
      succ_(static_cast<size_t>(vocab_size) * slots, -1) {}

void TokenRecycleTable::observe_pair(int32_t prev, int32_t next) {
    if (!valid_(prev) || !valid_(next))
        return;
    int32_t* r = row_(prev);
    // Find existing occurrence (or the end of the used region).
    int pos = slots_ - 1;
    for (int i = 0; i < slots_; ++i) {
        if (r[i] == next || r[i] == -1) {
            pos = i;
            break;
        }
    }
    // Shift [0, pos) down one and put `next` at the front (MRU).
    for (int i = pos; i > 0; --i)
        r[i] = r[i - 1];
    r[0] = next;
}

void TokenRecycleTable::observe_topk(int32_t token, const int32_t* ids, int n) {
    if (!valid_(token) || !ids)
        return;
    // Promote in reverse rank order so ids[0] ends up in slot 0.
    for (int i = n - 1; i >= 0; --i)
        observe_pair(token, ids[i]);
}

std::vector<int32_t> TokenRecycleTable::draft_linear(int32_t t0, int k) const {
    std::vector<int32_t> out;
    int32_t cur = t0;
    for (int i = 0; i < k; ++i) {
        if (!valid_(cur))
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
