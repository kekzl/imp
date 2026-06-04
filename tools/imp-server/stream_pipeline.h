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
    for (const auto& stop : stop_sequences) {
        if (stop.empty())
            continue;
        size_t pos = pending.find(stop);
        if (pos != std::string::npos) {
            d.complete_match = true;
            d.flush_len = pos;  // pos <= size, no clamp needed
            return d;
        }
    }
    if (pending.size() > max_stop_len) {
        size_t safe = pending.size() - max_stop_len + 1;
        if (safe > pending.size())
            safe = pending.size();  // max_stop_len == 0 -> never escape the buffer
        d.flush_len = safe;
    }
    return d;
}

}  // namespace imp::stream
