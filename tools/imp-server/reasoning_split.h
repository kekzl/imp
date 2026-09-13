#pragma once

// Pure streaming reasoning/content demux for DeepSeek "<think>...</think>", the single source of
// truth for both OpenAI and Anthropic streaming (each used to carry its own copy).
// Non-streaming splits at the LAST </think>; the old streaming code flipped at the FIRST and
// detected re-opening only via token-id compare, which never fires for Qwen3.6 (multi-BPE
// <think>), leaking reasoning into content on streaming only. This unit re-detects by text scan
// with overlap across SSE pieces instead; it cannot un-stream content already flushed before a
// far-later </think> (shared limit with vLLM's qwen3 parser).

#include <cstddef>
#include <string>

#include "stream_pipeline.h"  // imp::stream::utf8_complete_len

namespace imp::server {

// Reconcile the heuristic "thinking on" default against what the template actually rendered:
// open <think> (no closing) -> ON; closed block (<think>+</think>) -> OFF (#934, e.g. Qwen3.5-4B
// pre-closes despite mentioning enable_thinking); neither -> keep current. Explicit caller
// enable_thinking is left untouched either way.
// structured_output_excludes_thinking: true when the reply is constrained end-to-end (json_mode,
// tools, json_schema - #1431 added the last), leaving no room for a reasoning preamble.
inline bool structured_output_excludes_thinking(bool json_mode, bool has_tools, bool has_json_schema,
                                                bool has_regex, bool has_grammar) {
    return json_mode || has_tools || has_json_schema || has_regex || has_grammar;
}

// should_stamp_thinking_off: NOT stamping is not the same as stamping false - an unstamped
// template falls back to its own default, which can be OPEN <think> (Qwen3.8). Any reason not
// to want thinking must reach the template, not just the server's own flags.
inline bool should_stamp_thinking_off(bool is_think_model, bool enable_thinking, bool budget_disabled,
                                      bool want_thinking) {
    return is_think_model && !enable_thinking && (budget_disabled || !want_thinking);
}

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
    // Result of one feed()/finish() call: both reasoning and content may be non-empty (a piece can
    // finish reasoning and open content). reasoning_tokens is the count to add for this step.
    struct Result {
        std::string reasoning;
        std::string content;
        int reasoning_tokens = 0;
    };

    // StreamReasoningSplitter start phase: REASONING (prompt injected <think>), SCAN (model decides),
    // CONTENT (pass-through). think_start_id/think_end_id are -1 when the model emits multi-BPE
    // markers as text (the text-scan paths still catch those).
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

    // set_release_on_plain_text: off by default (agent path) holds SCAN for the full window to keep
    // an unmarked chain-of-thought from streaming as the answer; a plain chat request with thinking
    // off paid that whole 8-token hold (~85ms, Qwen3.8-27B) for no benefit, so this can release early.
    void set_release_on_plain_text(bool on) { release_on_plain_text_ = on; }

    // could_open_marker: true while `buf` (leading whitespace ignored) is empty or a prefix of
    // "<think>"/"</think>" (still undecided); false once a real word proves the answer started.
    static bool could_open_marker(const std::string& buf) {
        size_t ns = buf.find_first_not_of("\n\r\t ");
        if (ns == std::string::npos)
            return true;
        const size_t n = buf.size() - ns;
        auto prefix_of = [&](const char* marker, size_t len) {
            return n <= len && buf.compare(ns, n, marker, n) == 0;
        };
        return prefix_of("<think>", 7) || prefix_of("</think>", 8);
    }

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
                // A </think> with no opener: the template rendered <think> into the PROMPT (e.g. a
                // suppressed-
                // thinking request) and the model reasoned anyway, so scanning for an opener never succeeds -
                // everything held so far is reasoning (offline reaches the same via split_last_think).
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
                if ((release_on_plain_text_ && !could_open_marker(scan_buf_)) || scan_count_ >= scan_limit_) {
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
                // Holds back only a trailing partial marker or incomplete codepoint (not a fixed overlap):
                // the
                // fixed 7-byte overlap this replaced cost the first reasoning delta 1-2 tokens (20-27ms
                // measured,
                // Qwen3.8-27B-NVFP4).
                {
                    size_t hold = pending_tag_prefix(rbuf_);
                    size_t utf8_tail = rbuf_.size() - imp::stream::utf8_complete_len(rbuf_);
                    if (utf8_tail > hold)
                        hold = utf8_tail;
                    if (rbuf_.size() > hold) {
                        r.reasoning += rbuf_.substr(0, rbuf_.size() - hold);
                        rbuf_.erase(0, rbuf_.size() - hold);
                    }
                }
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
            // Holds back only a trailing partial <think>/</think> marker or incomplete UTF-8 codepoint;
            // everything else streams immediately - a fixed overlap here would desync the handler's
            // tool-call/stop machinery (which reorders content around tool-call deltas).
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

    // pending_tag_prefix: length of the longest trailing suffix of `buf` that could still grow into
    // "<think>"/"</think>" - must be held back. Returns 0 for plain text/JSON (no added latency).
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
    bool release_on_plain_text_ = false;
    int scan_count_ = 0;
    int reentries_ = 0;
    std::string scan_buf_;  // SCAN: undecided leading text
    std::string rbuf_;      // REASONING: reasoning buffer (overlap-held)
    std::string cbuf_;      // CONTENT: content buffer (overlap-held)
};

}  // namespace imp::server
