#pragma once

// Pure (no-engine, no-HTTP) decision logic for the streaming text pipeline.
//
// The streaming handlers in handlers.cpp hold text back until they can prove it
// is not the prefix of a stop sequence, then flush the "safe" portion as an SSE
// content delta. The buffering arithmetic — and its UTF-8-boundary cousin in
// utils.cpp — is where the max_stop_len=0 NUL-terminator regression lived
// (`size - max_stop_len + 1` flushed one byte PAST pending_text's end, emitting
// the std::string '\0' terminator into every delta). Extracted here so the math
// can be unit-tested on the CPU against hand-derived expectations.

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace imp::stream {

// Length of the longest prefix of `s` that ends on a UTF-8 codepoint boundary.
// A token piece may split a multi-byte codepoint across token boundaries; the
// streaming handlers must only emit complete codepoints (a half-codepoint in an
// SSE delta corrupts the client's string). Returns s.size() when the buffer ends
// cleanly, else the byte index at which the trailing incomplete sequence starts.
// On an invalid lead byte, emits up to (not including) that byte.
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
    // Index into `stop_sequences` of the sequence that matched, or -1. The
    // Anthropic wire format reports which stop ended the turn
    // (`stop_reason: "stop_sequence"`, `stop_sequence: "<text>"`), and it had
    // nothing to report because the match was a bool (#1550).
    int matched_index = -1;
};

// Decide how much of `pending` may be flushed.
//
// Contract (mirrors handlers.cpp):
//   1. If any stop sequence occurs in `pending`, report a complete match and the
//      flush length = byte offset of the FIRST such occurrence (text before the
//      stop is user-visible; the stop and everything after is dropped).
//   2. Otherwise hold back the last (max_stop_len - 1) bytes as a possible
//      partial stop prefix and flush the rest — but ONLY when the buffer is
//      longer than max_stop_len. flush_len = size - max_stop_len + 1.
//      With max_stop_len == 0 (no stop sequences) this collapses to "flush
//      everything"; the +1 must NOT escape the buffer (the bug), so flush_len
//      is clamped to the buffer size.
//
// flush_len is guaranteed <= pending.size() so callers can erase(0, flush_len)
// without ever touching the NUL terminator.
inline HoldbackDecision holdback_decision(const std::string& pending, size_t max_stop_len,
                                          const std::vector<std::string>& stop_sequences) {
    HoldbackDecision d;
    // The EARLIEST occurrence, not the first sequence in the list that happens
    // to occur anywhere: with stops {"B", "A"} on "xAyB" the list order used to
    // cut at "B" (offset 3) and report "B", while the text the model produced
    // ended at "A" (offset 1). The contract above always said "first
    // occurrence"; only the loop disagreed.
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
        // The cut is a byte offset, so it can land inside a multi-byte character
        // even when `pending` itself is well-formed — pull it back to the last
        // codepoint boundary, or the delta ships half a character (which
        // dump_safe then turns into U+FFFD: "größer" -> "gr??ßer").
        d.flush_len = utf8_complete_len(pending.substr(0, safe));
    }
    return d;
}

// Which decoded token produced which bytes of a holdback buffer (#1588).
//
// The streaming paths buffer text before emitting it, so a flush boundary is
// not a token boundary: the stop matcher decides how many bytes are safe, and
// that cut can land in the middle of a token's contribution. A per-token
// logprob therefore cannot be attached from a live "current token" counter -
// by the time held-back bytes go out, the counter has moved on. That is why
// the stop-sequence path shipped no logprobs at all rather than wrong ones.
//
// What is tracked is the bytes a token contributed AFTER the think-split and
// tool-call filters, not the token's raw text, because those are the bytes the
// client receives.
//
// Header-only and pure: no engine, no HTTP, no JSON. Tested in the CPU lane.
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
