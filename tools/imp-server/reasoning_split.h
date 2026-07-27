#pragma once

// Pure (no-engine, no-HTTP) streaming reasoning/content demux for the DeepSeek
// "<think>…</think>" format. This is the single source of truth used by both
// streaming handlers (handlers_chat_stream.cpp, OpenAI; handlers_messages.cpp,
// Anthropic) — previously each carried its own copy of this state machine.
//
// Why it exists as a unit: the non-streaming path (utils.cpp extract_reasoning)
// splits at the LAST </think>, so a model that re-deliberates after closing its
// first think block keeps that second pass in reasoning_content. The streaming
// copies flipped REASONING->CONTENT at the FIRST </think> and could only detect
// a *second* <think> via a single-token-id compare — which never fires for
// Qwen3.6, whose <think>/</think> ship as multi-BPE added_tokens (special=False;
// see src/runtime/request.h:84-90). The result was reasoning leaking into
// `content` on the streaming path only
// (BUGREPORT-qwen36-reasoning-leaks-into-content.md).
//
// This unit closes that gap: in the CONTENT phase it detects a re-opened
// <think> by TEXT scan (mirroring the SCAN/REASONING phases), holds a small
// overlap so a marker split across SSE pieces is still caught, and reclassifies
// a stray </think> back into reasoning. What it cannot do online is un-stream
// content already flushed before a far-later </think> arrives — that is a
// fundamental streaming-vs-offline limit (shared with vLLM's qwen3 parser);
// only marker-less reasoning prose with no further </think> can still leak.

#include <cstddef>
#include <string>

#include "stream_pipeline.h"  // imp::stream::utf8_complete_len

namespace imp::server {

// Reconcile a heuristic "thinking is on" default against what the chat
// template ACTUALLY rendered into the prompt tail. The render is ground truth
// for whether the model's generation begins inside an open <think> block:
//   open block   (<think> present, no matching </think>) -> thinking ON
//   closed block (<think> AND </think> present)          -> thinking OFF
//   no <think>   (neither)                               -> keep `current`
// The closed-block downgrade is #934: templates such as Qwen3.5-4B default
// enable_thinking to a pre-closed empty block `<think>\n\n</think>\n\n` (the
// model answers directly), but the upstream heuristic only sees the template
// *mention* thinking (its Jinja names `enable_thinking`) and turns it ON.
// Starting the reasoning splitter in REASONING then traps the whole answer in
// reasoning_content with empty user-visible content. The upgrade direction was
// already handled inline; this makes the decision symmetric and testable. An
// explicit caller `enable_thinking` request is left untouched (a closed-block
// template cannot honor an explicit `true` anyway, and we do not silently flip
// an explicit choice — that is the caller's to make).
inline bool reconcile_thinking_with_prompt_tail(bool current, bool explicit_set,
                                                bool tail_has_think, bool tail_has_close) {
    if (tail_has_think && !tail_has_close)
        return true;  // open prefix: model is mid-reasoning
    if (tail_has_think && tail_has_close && !explicit_set)
        return false;  // pre-closed block: model answers directly (#934)
    return current;    // no <think> at all, or explicit request — leave as-is
}

enum class ThinkPhase { SCAN, REASONING, CONTENT };

class StreamReasoningSplitter {
public:
    // The reasoning/content text produced by one feed()/finish() call. Both
    // fields may be non-empty (a single piece can finish reasoning AND open
    // content). reasoning_tokens is the count to add to the handler's
    // n_reasoning_tokens for this step.
    struct Result {
        std::string reasoning;
        std::string content;
        int reasoning_tokens = 0;
    };

    // start: REASONING when the prompt injected a <think> opener
    // (enable_thinking), SCAN when the model decides whether to think, CONTENT
    // when no reasoning extraction applies (the splitter is then a pass-through).
    // think_start_id / think_end_id are the special-token ids when the markers
    // are single tokens (-1 if the model emits them as multi-BPE text — the
    // text-scan paths still catch those).
    StreamReasoningSplitter(ThinkPhase start, int think_start_id, int think_end_id,
                            int scan_limit = 8, int max_reentries = 1)
        : phase_(start),
          think_start_id_(think_start_id),
          think_end_id_(think_end_id),
          scan_limit_(scan_limit),
          max_reentries_(max_reentries),
          content_started_(start == ThinkPhase::CONTENT) {}

    ThinkPhase phase() const { return phase_; }

    // Text SCAN is currently holding while it waits to learn whether this is
    // reasoning. Callers that can prove it is not — a tool-call opener is proof,
    // since a call is never reasoning — use this to inspect and then release it.
    const std::string& held() const { return scan_buf_; }

    // Give up the hold: everything buffered is content, and the phase moves on.
    // Without this the hold that stops a chain of thought leaking would also
    // swallow a tool call whole, and its argument deltas would never stream.
    Result flush_scan() {
        Result r;
        if (phase_ != ThinkPhase::SCAN)
            return r;
        r.content = scan_buf_;
        scan_buf_.clear();
        phase_ = ThinkPhase::CONTENT;
        content_started_ = true;
        return r;
    }

    // Feed one decoded token piece (with its token id). Returns the
    // reasoning/content split for this step.
    Result feed(std::string piece, int token) {
        Result r;
        std::string work = std::move(piece);
        bool token_live = true;  // token-id checks apply only to the real token

        for (;;) {
            if (phase_ == ThinkPhase::SCAN) {
                if (token_live && think_start_id_ >= 0 && token == think_start_id_) {
                    phase_ = ThinkPhase::REASONING;
                    r.reasoning_tokens++;
                    return r;
                }
                // A CLOSER with no opener. The chat template rendered the
                // `<think>` into the PROMPT (a pre-closed block on a suppressed
                // -thinking request, say) and the model reasoned anyway, so the
                // output carries only `</think>` — scanning for an opener can
                // never succeed. Everything held so far was reasoning. The
                // offline path reaches the same conclusion via split_last_think;
                // streaming only gets one shot at it, which is what the scan
                // buffer is holding output for.
                if (token_live && think_end_id_ >= 0 && token == think_end_id_) {
                    r.reasoning += scan_buf_;
                    r.reasoning_tokens += scan_count_ + 1;
                    scan_buf_.clear();
                    phase_ = ThinkPhase::CONTENT;
                    return r;
                }
                scan_buf_ += work;
                scan_count_++;
                auto p = scan_buf_.find("<think>");
                if (p != std::string::npos) {
                    phase_ = ThinkPhase::REASONING;
                    r.reasoning_tokens += scan_count_;
                    work = scan_buf_.substr(p + 7);
                    scan_buf_.clear();
                    token_live = false;
                    continue;  // process the post-<think> remainder as reasoning
                }
                // Same, for models that spell the closer as plain text.
                auto pc = scan_buf_.find("</think>");
                if (pc != std::string::npos) {
                    r.reasoning += scan_buf_.substr(0, pc);
                    r.reasoning_tokens += scan_count_;
                    work = scan_buf_.substr(pc + 8);
                    scan_buf_.clear();
                    phase_ = ThinkPhase::CONTENT;
                    token_live = false;
                    continue;  // the remainder after the closer is content
                }
                if (scan_count_ == 1 && work.empty()) {
                    phase_ = ThinkPhase::REASONING;
                    r.reasoning_tokens++;
                    return r;
                }
                if (scan_count_ >= scan_limit_) {
                    phase_ = ThinkPhase::CONTENT;
                    work = scan_buf_;
                    scan_buf_.clear();
                    token_live = false;
                    continue;  // model never thought — flush scan buffer as content
                }
                return r;  // keep scanning
            }

            if (phase_ == ThinkPhase::REASONING) {
                if (token_live)
                    r.reasoning_tokens++;
                if (token_live && think_end_id_ >= 0 && token == think_end_id_) {
                    r.reasoning += rbuf_;
                    rbuf_.clear();
                    phase_ = ThinkPhase::CONTENT;
                    return r;
                }
                if (token_live && think_start_id_ >= 0 && token == think_start_id_)
                    return r;  // duplicate <think> while already reasoning
                rbuf_ += work;
                work.clear();
                // Strip <think> openers that arrive as multi-token text.
                for (size_t tp; (tp = rbuf_.find("<think>")) != std::string::npos;)
                    rbuf_.erase(tp, 7);
                auto end_pos = rbuf_.find("</think>");
                if (end_pos != std::string::npos) {
                    r.reasoning += rbuf_.substr(0, end_pos);
                    std::string after = rbuf_.substr(end_pos + 8);
                    rbuf_.clear();
                    phase_ = ThinkPhase::CONTENT;
                    auto ns = after.find_first_not_of("\n\r\t ");
                    if (ns == std::string::npos)
                        return r;
                    work = after.substr(ns);
                    content_started_ = true;  // leading whitespace already dropped
                    token_live = false;
                    continue;  // process the post-</think> remainder as content
                }
                emit_with_overlap(rbuf_, r.reasoning);  // hold 7B for a split marker
                return r;
            }

            // CONTENT
            if (!content_started_) {
                auto ns = work.find_first_not_of("\n\r\t ");
                if (ns == std::string::npos)
                    return r;
                work = work.substr(ns);
                content_started_ = true;
            }
            cbuf_ += work;
            work.clear();
            bool reenter = false;
            for (;;) {
                size_t tp = (reentries_ < max_reentries_) ? cbuf_.find("<think>")
                                                          : std::string::npos;
                size_t cp = cbuf_.find("</think>");
                if (tp != std::string::npos && (cp == std::string::npos || tp < cp)) {
                    // Model re-opened thinking: text before it is content, the
                    // rest re-enters REASONING.
                    r.content += cbuf_.substr(0, tp);
                    reentries_++;
                    work = cbuf_.substr(tp + 7);
                    cbuf_.clear();
                    phase_ = ThinkPhase::REASONING;
                    token_live = false;
                    reenter = true;
                    break;
                }
                if (cp != std::string::npos) {
                    // Stray </think>: the still-buffered text before it was
                    // reasoning, not content — reclassify it.
                    r.reasoning += cbuf_.substr(0, cp);
                    cbuf_.erase(0, cp + 8);
                    continue;
                }
                break;
            }
            if (reenter)
                continue;  // re-process `work` as reasoning
            // Hold back ONLY a trailing partial <think>/</think> (so a marker
            // split across pieces is still caught next call) or an incomplete
            // trailing UTF-8 codepoint. Everything else streams immediately: a
            // fixed overlap here would desync the handler's tool-call/stop
            // machinery, which reorders the content stream (the held tail gets
            // appended after a tool-call delta instead of in place).
            size_t hold = pending_tag_prefix(cbuf_);
            size_t utf8_tail = cbuf_.size() - imp::stream::utf8_complete_len(cbuf_);
            if (utf8_tail > hold)
                hold = utf8_tail;
            if (cbuf_.size() > hold) {
                r.content += cbuf_.substr(0, cbuf_.size() - hold);
                cbuf_.erase(0, cbuf_.size() - hold);
            }
            return r;
        }
    }

    // Flush any buffered tail at generation end. MUST be called once the stream
    // finishes, or the held overlap (last bytes of the answer) is lost.
    Result finish() {
        Result r;
        if (phase_ == ThinkPhase::REASONING) {
            r.reasoning += rbuf_;
            rbuf_.clear();
        } else if (phase_ == ThinkPhase::CONTENT) {
            for (size_t p; (p = cbuf_.find("<think>")) != std::string::npos;)
                cbuf_.erase(p, 7);
            for (size_t p; (p = cbuf_.find("</think>")) != std::string::npos;)
                cbuf_.erase(p, 8);
            r.content += cbuf_;
            cbuf_.clear();
        } else {  // SCAN: model never reached a <think> — undecided text is content
            r.content += scan_buf_;
            scan_buf_.clear();
        }
        return r;
    }

private:
    static constexpr size_t kOverlap = 7;  // longest partial "</think>"/"<think>"

    // Length of the longest suffix of `buf` that is a proper prefix of "<think>"
    // or "</think>" — the bytes that might still grow into a marker and so must
    // be held back in the CONTENT stream. Returns 0 for content with no pending
    // partial marker (the common case: plain text, JSON tool calls) so it streams
    // with no added latency.
    static size_t pending_tag_prefix(const std::string& buf) {
        static const char* const needles[] = {"<think>", "</think>"};
        size_t best = 0;
        for (const char* nd : needles) {
            size_t nl = std::char_traits<char>::length(nd);
            size_t maxk = nl - 1 < buf.size() ? nl - 1 : buf.size();  // proper prefix
            for (size_t k = maxk; k > best; --k) {
                if (buf.compare(buf.size() - k, k, nd, k) == 0) {
                    best = k;
                    break;
                }
            }
        }
        return best;
    }

    // Emit the complete-UTF-8 prefix of `buf` except a trailing kOverlap bytes
    // (kept so a marker split across pieces is still detectable next call),
    // backing the cut up to a codepoint boundary. Appends emitted text to `out`.
    static void emit_with_overlap(std::string& buf, std::string& out) {
        size_t complete = imp::stream::utf8_complete_len(buf);
        if (complete <= kOverlap)
            return;
        size_t emit_end = complete - kOverlap;
        while (emit_end > 0 &&
               (static_cast<unsigned char>(buf[emit_end]) & 0xC0) == 0x80)
            --emit_end;
        if (emit_end == 0)
            return;
        out += buf.substr(0, emit_end);
        buf.erase(0, emit_end);
    }

    ThinkPhase phase_;
    int think_start_id_;
    int think_end_id_;
    int scan_limit_;
    int max_reentries_;
    bool content_started_;
    int scan_count_ = 0;
    int reentries_ = 0;
    std::string scan_buf_;  // SCAN: undecided leading text
    std::string rbuf_;      // REASONING: reasoning buffer (overlap-held)
    std::string cbuf_;      // CONTENT: content buffer (overlap-held)
};

}  // namespace imp::server
