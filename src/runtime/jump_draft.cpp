#include "runtime/jump_draft.h"

#include <algorithm>

namespace imp {

std::vector<int32_t> jump_draft_from_history(const std::vector<int32_t>& out,
                                             const std::function<std::string(int32_t)>& piece,
                                             const std::string& text, int window, int min_tokens) {
    if (text.empty() || out.empty())
        return {};
    const int n = static_cast<int>(out.size());
    const int lo = std::max(0, n - window);
    std::vector<std::string> pieces(static_cast<size_t>(n - lo));
    for (int i = lo; i < n; ++i)
        pieces[static_cast<size_t>(i - lo)] = piece(out[static_cast<size_t>(i)]);
    // Most recent start first, longest coverage wins; a full cover of text ends the search.
    std::vector<int32_t> best;
    size_t best_pos = 0;
    for (int s = n - 1; s >= lo; --s) {
        std::vector<int32_t> draft;
        size_t pos = 0;
        for (int i = s; i < n; ++i) {
            const std::string& pc = pieces[static_cast<size_t>(i - lo)];
            if (pc.empty() || pos + pc.size() > text.size() || text.compare(pos, pc.size(), pc) != 0)
                break;
            draft.push_back(out[static_cast<size_t>(i)]);
            pos += pc.size();
            if (pos == text.size())
                break;
        }
        if (pos > best_pos) {
            best = std::move(draft);
            best_pos = pos;
            if (best_pos == text.size())
                break;
        }
    }
    return static_cast<int>(best.size()) >= min_tokens ? best : std::vector<int32_t>{};
}

}  // namespace imp
