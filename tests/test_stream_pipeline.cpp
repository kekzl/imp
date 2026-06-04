// CPU unit tests for the server streaming text pipeline (Test-Audit Phase 2,
// Risk #5). Targets the pure holdback + UTF-8-boundary logic in
// tools/imp-server/stream_pipeline.h, which is the single source of truth used
// by all three streaming handlers in handlers.cpp.
//
// The bug this guards: with max_stop_len == 0 (no stop sequences) the holdback
// math `pending.size() - max_stop_len + 1` evaluated to size + 1 and was fed to
// flush_text, which read ONE byte past pending_text — emitting the std::string
// '\0' terminator into every SSE content delta and silently disabling
// cross-token stop matching. Mock-server contract tests stayed green throughout.
//
// Ground truth is hand-derived from the holdback contract (documented in
// stream_pipeline.h) and from the UTF-8 spec; each case states the reasoning.

#include "stream_pipeline.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using imp::stream::holdback_decision;
using imp::stream::utf8_complete_len;

// A faithful re-implementation of the handler flush loop, driven purely by
// holdback_decision, so we can assert stream==non-stream equality end to end.
// It mirrors handlers.cpp: append piece -> decide -> flush prefix -> on a
// complete match, drop the stop and everything after it.
struct StreamSim {
    std::string pending;
    std::string emitted;  // concatenation of all SSE content deltas
    bool stopped = false;

    void feed(const std::string& piece, size_t max_stop_len, const std::vector<std::string>& stops) {
        if (stopped)
            return;
        pending += piece;
        auto d = holdback_decision(pending, max_stop_len, stops);
        // flush_text contract: emit pending[0:flush_len], erase it.
        ASSERT_LE(d.flush_len, pending.size());  // never escapes the buffer (the bug)
        emitted.append(pending, 0, d.flush_len);
        pending.erase(0, d.flush_len);
        if (d.complete_match) {
            stopped = true;
            pending.clear();  // stop + tail are dropped, never emitted
        }
    }

    // Final trailing flush (handlers emit leftover pending_text at loop end).
    void finish() {
        if (!stopped)
            emitted += pending;
        pending.clear();
    }
};

// ---------------------------------------------------------------------------
// The NUL regression: max_stop_len == 0 must flush exactly the buffer, no +1.
// ---------------------------------------------------------------------------

TEST(Holdback, NoStopSequencesFlushesWholeBufferNoOverrun) {
    // No stop sequences -> max_stop_len 0. The buffer "4 " (the regression's
    // observed payload) must flush all 2 bytes and NOTHING more. The pre-fix
    // code returned 3 (= 2 - 0 + 1), reading the '\0' terminator.
    std::string pending = "4 ";
    auto d = holdback_decision(pending, /*max_stop_len=*/0, /*stops=*/{});
    EXPECT_FALSE(d.complete_match);
    EXPECT_EQ(d.flush_len, 2u);  // == size, not size+1
}

TEST(Holdback, EmptyBufferFlushesNothing) {
    auto d = holdback_decision("", 0, {});
    EXPECT_FALSE(d.complete_match);
    EXPECT_EQ(d.flush_len, 0u);
}

// ---------------------------------------------------------------------------
// Stop-sequence holdback across chunk boundaries.
// ---------------------------------------------------------------------------

TEST(Holdback, HoldsBackPotentialPartialPrefix) {
    // stop="STOP" (len 4) -> max_stop_len 4. Buffer "helloST" could still grow
    // into "...STOP", so keep the last (4-1)=3 bytes ("lST"... actually the
    // last 3: "oST") and flush the rest. flush_len = 7 - 4 + 1 = 4 -> "hell".
    std::vector<std::string> stops = {"STOP"};
    auto d = holdback_decision("helloST", /*max_stop_len=*/4, stops);
    EXPECT_FALSE(d.complete_match);
    EXPECT_EQ(d.flush_len, 4u);  // emits "hell", holds "oST"
}

TEST(Holdback, ShortBufferUnderMaxStopLenHeldEntirely) {
    // Buffer "ST" (len 2) <= max_stop_len 4 -> nothing is safe yet, hold all.
    std::vector<std::string> stops = {"STOP"};
    auto d = holdback_decision("ST", 4, stops);
    EXPECT_FALSE(d.complete_match);
    EXPECT_EQ(d.flush_len, 0u);
}

TEST(Holdback, CompleteMatchFlushesPrefixAndSignalsStop) {
    // "abcSTOPxyz": first occurrence of STOP at byte 3 -> flush "abc", stop.
    std::vector<std::string> stops = {"STOP"};
    auto d = holdback_decision("abcSTOPxyz", 4, stops);
    EXPECT_TRUE(d.complete_match);
    EXPECT_EQ(d.flush_len, 3u);  // "abc"; the stop and "xyz" are dropped
}

TEST(Holdback, FirstStopInListOrderWins) {
    // Contract (mirrors handlers.cpp's loop): stops are tried in LIST order and
    // the first one that occurs anywhere wins — NOT the earliest position. Here
    // "STOP" (list index 0) is found at byte 7 and reported even though "END"
    // sits earlier at byte 2. This is intentional: it matches the production
    // loop `for (stop : stops) { if (find(stop)) break; }`.
    std::vector<std::string> stops = {"STOP", "END"};
    auto d = holdback_decision("hiENDxxSTOP", 4, stops);
    EXPECT_TRUE(d.complete_match);
    EXPECT_EQ(d.flush_len, 7u);  // flush up to "STOP" at byte 7 -> "hiENDxx"
}

TEST(Holdback, EarlierListEntryMatchesAtItsPosition) {
    // When the FIRST list entry ("END") does occur, its position (byte 2) is the
    // flush length, regardless of a later-listed stop. Confirms the loop returns
    // on the first list entry, at that entry's own match position.
    std::vector<std::string> stops = {"END", "STOP"};
    auto d = holdback_decision("hiENDxxSTOP", 4, stops);
    EXPECT_TRUE(d.complete_match);
    EXPECT_EQ(d.flush_len, 2u);  // "hi"
}

TEST(StreamSim, StopMatchSplitAcrossChunksIsCaught) {
    // The stop arrives in pieces "ST"+"OP" — the holdback must not emit "ST"
    // early (it could be a stop prefix) and must catch the match once complete.
    StreamSim s;
    std::vector<std::string> stops = {"STOP"};
    s.feed("hi ", 4, stops);
    s.feed("ST", 4, stops);
    s.feed("OP done", 4, stops);
    s.finish();
    // Everything before STOP ("hi ") is visible; STOP and after are dropped.
    EXPECT_EQ(s.emitted, "hi ");
    EXPECT_TRUE(s.stopped);
}

// ---------------------------------------------------------------------------
// UTF-8 boundary handling: never emit a half codepoint.
// ---------------------------------------------------------------------------

TEST(Utf8, CompleteAsciiAndMultibyteEmitFully) {
    EXPECT_EQ(utf8_complete_len("hello"), 5u);
    // "é" = 0xC3 0xA9 (2 bytes), complete.
    EXPECT_EQ(utf8_complete_len("a\xC3\xA9"), 3u);
    // "€" = 0xE2 0x82 0xAC (3 bytes), complete.
    EXPECT_EQ(utf8_complete_len("\xE2\x82\xAC"), 3u);
}

TEST(Utf8, TruncatedTrailingSequenceHeldBack) {
    // "a" + lead byte of a 2-byte codepoint (0xC3) with no continuation:
    // emit only "a" (1), hold the dangling lead byte.
    EXPECT_EQ(utf8_complete_len("a\xC3"), 1u);
    // 3-byte "€" missing its last byte: lead 0xE2 + one continuation 0x82.
    // Start of the sequence is index 0 -> hold all, emit 0.
    EXPECT_EQ(utf8_complete_len("\xE2\x82"), 0u);
    // 4-byte codepoint (U+1F600 "😀" = F0 9F 98 80) missing the last byte after
    // a leading 'x': emit "x" (1), hold the 3 partial bytes.
    EXPECT_EQ(utf8_complete_len("x\xF0\x9F\x98"), 1u);
}

TEST(Utf8, EmptyAndInvalidLead) {
    EXPECT_EQ(utf8_complete_len(""), 0u);
    // A stray continuation byte (0x80) preceded by valid ASCII: the walk-back
    // lands on index 0 ('a', a valid 1-byte lead) which IS complete, so the
    // whole 2-byte string is reported (the stray 0x80 is emitted, matching the
    // production fallback "emit what we have rather than stall").
    EXPECT_EQ(utf8_complete_len("a\x80"), 2u);
}

TEST(StreamMultibyteSim, SplitCodepointNeverEmittedHalf) {
    // Simulate the UTF-8 buffering the handler does on top of holdback: a
    // 2-byte "é" split as 0xC3 then 0xA9 must never appear half-emitted.
    // We model the handler's utf8_buf flush (no stop sequences).
    std::string utf8_buf;
    std::string out;
    auto push = [&](const std::string& piece) {
        utf8_buf += piece;
        size_t c = utf8_complete_len(utf8_buf);
        out.append(utf8_buf, 0, c);
        utf8_buf.erase(0, c);
    };
    push("a\xC3");  // "a" emits, 0xC3 held
    EXPECT_EQ(out, "a");
    push("\xA9z");  // now "éz" complete
    EXPECT_EQ(out, "a\xC3\xA9z");
    EXPECT_TRUE(utf8_buf.empty());
}

// ---------------------------------------------------------------------------
// Stream concatenation == non-stream result for the same token sequence.
// ---------------------------------------------------------------------------

TEST(StreamEquality, NoStopSequencesConcatEqualsFullDecode) {
    // Non-stream result for a token decode would be the full concatenation of
    // all pieces. Streaming with no stop sequences (max_stop_len 0) must emit
    // exactly that, byte-for-byte (this is the path the NUL bug corrupted).
    std::vector<std::string> pieces = {"The ", "quick ", "brown ", "fox"};
    std::string nonstream;
    for (auto& p : pieces)
        nonstream += p;

    StreamSim s;
    for (auto& p : pieces)
        s.feed(p, 0, {});
    s.finish();

    EXPECT_EQ(s.emitted, nonstream);
    EXPECT_EQ(s.emitted, "The quick brown fox");
}

TEST(StreamEquality, WithStopMatchesNonStreamTruncation) {
    // Non-stream handler truncates output_text at the first stop occurrence
    // (handlers.cpp: output_text.substr(0, pos)). The streamed concatenation
    // must equal that same truncation.
    std::vector<std::string> pieces = {"answer: 42", "\nHuman: more"};
    std::vector<std::string> stops = {"\nHuman"};
    size_t msl = stops[0].size();

    // Non-stream reference: build full text, cut at first stop.
    std::string full;
    for (auto& p : pieces)
        full += p;
    size_t pos = full.find(stops[0]);
    std::string nonstream = full.substr(0, pos);

    StreamSim s;
    for (auto& p : pieces)
        s.feed(p, msl, stops);
    s.finish();

    EXPECT_EQ(s.emitted, nonstream);
    EXPECT_EQ(s.emitted, "answer: 42");
}
