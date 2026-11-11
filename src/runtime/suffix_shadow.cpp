#include "runtime/suffix_shadow.h"

#include "core/logging.h"

#include <cstdio>
#include <utility>

namespace imp {

namespace {
// Requests that left without a last step (cancel) keep their records: past this many ids the map restarts.
constexpr size_t kMaxRequests = 256;
}  // namespace

void SuffixShadow::drafted(int id, int pos, std::vector<int32_t> draft, int match) {
    if (recs_.size() > kMaxRequests)
        recs_.clear();
    auto& recs = recs_[id];
    if (recs.empty() || recs.back().stepped)
        recs.emplace_back();
    Rec& r = recs.back();
    r.pos = pos;
    r.match = draft.empty() ? 0 : match;
    r.draft = std::move(draft);
}

void SuffixShadow::stepped(int id, int n, int rows, int nd, int matched, int emitted) {
    const auto it = recs_.find(id);
    if (it == recs_.end() || it->second.empty() || it->second.back().stepped)
        return;
    Rec& r = it->second.back();
    r.n = n;
    r.rows = rows;
    r.nd = nd;
    r.matched = matched;
    r.emitted = emitted;
    r.stepped = true;
}

std::vector<std::string> SuffixShadow::settle(int id, std::span<const int32_t> out, bool last) {
    std::vector<std::string> lines;
    const auto it = recs_.find(id);
    if (it == recs_.end())
        return lines;
    auto& recs = it->second;
    const int n_out = static_cast<int>(out.size());
    size_t keep = 0;
    for (size_t i = 0; i < recs.size(); ++i) {
        Rec& r = recs[i];
        const int len = static_cast<int>(r.draft.size());
        int agree = 0;
        while (agree < len && r.pos + agree < n_out && r.draft[agree] == out[r.pos + agree])
            ++agree;
        // Decided: a mismatch inside the emitted tokens, or the whole draft compared.
        const bool decided = agree == len || r.pos + agree < n_out;
        if (!r.stepped || (!decided && !last)) {
            if (keep != i)
                recs[keep] = std::move(r);
            ++keep;
            continue;
        }
        char buf[160];
        std::snprintf(buf, sizeof(buf), "spec-sfx-shadow id=%d pos=%d n=%d T=%d mtp=%d:%d:%d sfx=%d:%d:%d",
                      id, r.pos, r.n, r.rows, r.nd, r.matched, r.emitted, len, r.match, agree);
        lines.emplace_back(buf);
    }
    recs.resize(keep);
    if (last)
        recs_.erase(it);
    return lines;
}

void SuffixShadow::trace_draft(int id, std::span<const int32_t> out, std::vector<int32_t> draft, int match) {
    for (const auto& line : settle(id, out, /*last=*/false))
        IMP_LOG_INFO("%s", line.c_str());
    drafted(id, static_cast<int>(out.size()), std::move(draft), match);
}

void SuffixShadow::trace_step(int id, int n, int rows, int nd, int matched, int emitted,
                              std::span<const int32_t> out, bool last) {
    stepped(id, n, rows, nd, matched, emitted);
    if (!last)
        return;
    for (const auto& line : settle(id, out, /*last=*/true))
        IMP_LOG_INFO("%s", line.c_str());
}

}  // namespace imp
