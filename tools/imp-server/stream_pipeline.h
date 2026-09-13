#pragma once

// Pure (no-engine, no-HTTP) decision logic for the streaming stop-sequence holdback pipeline -
// the buffering arithmetic where the max_stop_len=0 NUL-terminator regression lived (see id 657).
// Extracted so the math is unit-tested on the CPU against hand-derived expectations.

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace imp::stream {

// utf8_complete_len: longest prefix of `s` ending on a codepoint boundary. A token piece can
// split a multi-byte codepoint across boundaries, and streaming must only emit whole codepoints
// (a half one corrupts the client's string). Invalid lead byte: emits up to (not including) it.
inline size_t utf8_complete_len(const std::string& s) {
    size_t len = s.size();
    if (len == 0)
        return 0;
    // Walk back from the end to the start of the last codepoint.
    size_t i = len - 1;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80)
        --i;
    unsigned char lead = static_cast<unsigned char>(s[i]);
    int expected;
    if (lead < 0x80)
        expected = 1;
    else if ((lead & 0xE0) == 0xC0)
        expected = 2;
    else if ((lead & 0xF0) == 0xE0)
        expected = 3;
    else if ((lead & 0xF8) == 0xF0)
        expected = 4;
    else
        return i;  // invalid byte — emit up to it
    if (i + static_cast<size_t>(expected) <= len)
        return len;  // complete
    return i;        // incomplete — emit up to start of this sequence
}

// Result of inspecting the holdback buffer after appending a piece.
struct HoldbackDecision {
    bool complete_match = false;  // a full stop sequence is present in the buffer
    size_t flush_len = 0;         // bytes safe to emit now (always <= buffer size)
    // matched_index: which stop_sequences entry matched, or -1. Anthropic reports which stop ended
    // the turn; previously the match was a bare bool with nothing to report (#1550).
    int matched_index = -1;
};

// holdback_decision contract: (1) any stop sequence present in `pending` -> flush up to the
// FIRST occurrence, drop the stop and everything after. (2) Otherwise hold back the last
// (max_stop_len-1) bytes as a possible partial prefix, flushing only when the buffer exceeds
// max_stop_len; flush_len = size-max_stop_len+1, clamped to buffer size (the NUL-terminator bug,
// see id 657) so max_stop_len==0 safely collapses to "flush everything".
inline HoldbackDecision holdback_decision(const std::string& pending, size_t max_stop_len,
                                          const std::vector<std::string>& stop_sequences) {
    HoldbackDecision d;
    // Finds the EARLIEST occurrence in `pending`, not the first sequence in the caller's list order:
    // stops {"B","A"} on "xAyB" used to cut at "B" (offset 3) though "A" (offset 1) occurs first -
    // the contract always said first occurrence, only the loop disagreed.
    size_t best = std::string::npos;
    for (size_t i = 0; i < stop_sequences.size(); i++) {
        const std::string& stop = stop_sequences[i];
        if (stop.empty())
            continue;
        size_t pos = pending.find(stop);
        if (pos != std::string::npos && (best == std::string::npos || pos < best)) {
            best = pos;
            d.matched_index = static_cast<int>(i);
        }
    }
    if (best != std::string::npos) {
        d.complete_match = true;
        d.flush_len = best;  // best <= size, no clamp needed
        return d;
    }
    if (pending.size() > max_stop_len) {
        size_t safe = pending.size() - max_stop_len + 1;
        if (safe > pending.size())
            safe = pending.size();  // max_stop_len == 0 -> never escape the buffer
        // The stop-match cut is a byte offset and can land mid-character even in well-formed input; pull
        // it back to the last codepoint boundary or the delta ships half a character (dump_safe -> U+FFFD).
        d.flush_len = utf8_complete_len(pending.substr(0, safe));
    }
    return d;
}

// TokenSpans (#1588): tracks which decoded token produced which bytes of a holdback buffer,
// since a flush boundary from the stop matcher is not a token boundary - a live token counter
// can't attribute held-back bytes once it has moved on. Tracks post-filter bytes (what the
// client receives), not raw token text. Header-only, pure, CPU-tested.
class TokenSpans {
public:
    struct Emit {
        size_t offset;    // start of this piece within the flushed prefix
        size_t length;    // bytes
        int token_index;  // -1 when the bytes span no single token
    };

    // Record that `bytes` bytes were appended to the buffer by token `index`.
    void append(size_t bytes, int index) {
        if (bytes == 0)
            return;
        end_ += bytes;
        spans_.push_back({end_, index});
    }

    // Split the first `up_to` bytes into per-token pieces and drop them from
    // the tracker. A trailing piece that no complete span covers is returned
    // with token_index -1: the honest answer, rather than the nearest index.
    std::vector<Emit> flush(size_t up_to) {
        std::vector<Emit> out;
        size_t emitted = 0;
        size_t consumed = 0;
        while (consumed < spans_.size() && spans_[consumed].end <= up_to) {
            const size_t end = spans_[consumed].end;
            if (end > emitted) {
                out.push_back({emitted, end - emitted, spans_[consumed].index});
                emitted = end;
            }
            ++consumed;
        }
        if (emitted < up_to)
            out.push_back({emitted, up_to - emitted, -1});

        spans_.erase(spans_.begin(), spans_.begin() + static_cast<long>(consumed));
        for (auto& sp : spans_)
            sp.end -= std::min(sp.end, up_to);
        end_ -= std::min(end_, up_to);
        return out;
    }

    void clear() {
        spans_.clear();
        end_ = 0;
    }

    [[nodiscard]] bool empty() const { return spans_.empty(); }

private:
    struct Span {
        size_t end;  // one past this token's last byte, relative to the buffer start
        int index;
    };
    std::vector<Span> spans_;
    size_t end_ = 0;
};

}  // namespace imp::stream
