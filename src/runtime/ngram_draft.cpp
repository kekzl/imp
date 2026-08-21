#include "runtime/ngram_draft.h"

#include <algorithm>

namespace imp {

NgramDraft ngram_draft(std::span<const int32_t> hist, int k, int min_match, int max_match) {
    const int n = static_cast<int>(hist.size());
    // No nullptr check: a span carries its own length, so "a pointer with a
    // length that does not belong to it" is not a callable state any more.
    if (k <= 0 || min_match < 1 || n < min_match + 1)
        return {};
    if (max_match < min_match)
        max_match = min_match;

    // Single backward pass over candidate end positions of the matched
    // n-gram. `end` is the index one past the candidate occurrence, i.e.
    // hist[end-m .. end) is compared against the suffix hist[n-m .. n).
    // We first test the cheap min_match window, then extend backwards up
    // to max_match to rank candidates by match length.
    const int32_t* suffix = hist.data() + n - min_match;
    int best_end = -1;
    int best_len = 0;
    for (int end = n - 1; end >= min_match; --end) {
        if (!std::equal(suffix, suffix + min_match, hist.data() + end - min_match))
            continue;
        int len = min_match;
        while (len < max_match && end - len - 1 >= 0 && n - len - 1 >= 0 &&
               hist[end - len - 1] == hist[n - len - 1]) {
            ++len;
        }
        if (len > best_len) {
            best_len = len;
            best_end = end;
            if (best_len >= max_match)
                break;  // can't do better; most recent due to scan order
        }
    }
    if (best_end < 0)
        return {};

    int avail = n - best_end;
    int take = std::min(k, avail);
    if (take <= 0)
        return {};
    const auto taken = hist.subspan(static_cast<size_t>(best_end), static_cast<size_t>(take));
    return {std::vector<int32_t>(taken.begin(), taken.end()), best_end};
}

}  // namespace imp
