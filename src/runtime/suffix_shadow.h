#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// Drafts the shadow asks the suffix index for: past the deepest step a policy replay prices (32 rows).
inline constexpr int kSuffixShadowDrafts = 32;

// diagnostics.spec_trace on the bound verify: what a suffix draft would have earned at each step, scored
// against the tokens the request then emitted. One line per request step once the outcome is known:
// "spec-sfx-shadow id=I pos=P n=N T=T mtp=nd:m:e sfx=len:match:agree". Read by tools/analysis/sfx_shadow.py.
class SuffixShadow {
public:
    // Before the step: the suffix draft at output position pos (empty = no match). A draft whose step
    // never ran (declined) is replaced.
    void drafted(int id, int pos, std::vector<int32_t> draft, int match);
    // After the step: requests and rows per request, the verify's own drafts, matches, emitted tokens.
    void stepped(int id, int n, int rows, int nd, int matched, int emitted);
    // Lines of the records `out` (the request's output tokens) decides, all of them when last; they leave.
    [[nodiscard]] std::vector<std::string> settle(int id, std::span<const int32_t> out, bool last);

    // The engine's two calls: settle + log + drafted, and stepped (+ settle + log on the last step).
    void trace_draft(int id, std::span<const int32_t> out, std::vector<int32_t> draft, int match);
    void trace_step(int id, int n, int rows, int nd, int matched, int emitted, std::span<const int32_t> out,
                    bool last);

private:
    struct Rec {
        int pos = 0, match = 0, n = 0, rows = 0, nd = 0, matched = 0, emitted = 0;
        bool stepped = false;
        std::vector<int32_t> draft;
    };
    std::unordered_map<int, std::vector<Rec>> recs_;
};

}  // namespace imp
