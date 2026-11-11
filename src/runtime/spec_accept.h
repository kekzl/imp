#pragma once

// Accept loop of a verified speculative chunk, shared by the single-stream verify and the bound
// batched verify. The engine passes its side effects in; tests/test_spec_gates.cpp runs it on a CPU.

#include <cstdint>

namespace imp {

struct SpecAccept {
    int matched = 0;  // accepted drafts: KV rows p0 + 1 .. p0 + matched stay
    int emitted = 0;  // tokens appended to the output
};

// Row j's argmax is emitted; it joins the accepted prefix only if it equals draft[j], crossed no think
// boundary (stop_at_think_edge), the chunk emitted fewer than think_left tokens and extends(j) allows
// row j + 1. emit(j, tok) returns true = finished.
// Unfinished: emitted == matched + 1, the last token sits in no kept KV row and the next step forwards it.
// think_left = think_logic::think_tokens_left before the chunk: emitted never exceeds it (B-69.1).
template <class Emit, class Extends>
SpecAccept spec_accept_rows(const int32_t* argmax, const int32_t* draft, int n_draft, int n_rows,
                            bool stop_at_think_edge, const bool& in_think, int think_left, Emit&& emit,
                            Extends&& extends) {
    SpecAccept r;
    const bool think_at_start = in_think;
    for (int j = 0; j < n_rows; ++j) {
        const int32_t tok = argmax[j];
        r.emitted++;
        if (emit(j, tok))
            break;
        if (j >= n_draft || tok != draft[j])
            break;  // bonus reached, or the draft diverged
        if (stop_at_think_edge && in_think != think_at_start)
            break;  // tok stays the bonus: the eager step forwards it once
        if (r.emitted >= think_left)
            break;  // budget reached: tok is the bonus, the eager step forces the closer next
        if (!extends(j))
            break;
        r.matched++;
    }
    return r;
}

}  // namespace imp
