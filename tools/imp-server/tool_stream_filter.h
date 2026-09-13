#pragma once

// Streaming tool-call demux shared by OpenAI and Anthropic SSE paths: pure text-level state
// machine (feed decoded pieces in, get content + tool calls out, in order). Dialects: ChatML
// <tool_call>{json}</tool_call> (Qwen3.6 XML fallback), Llama3 <function=NAME>{json}</function>,
// Gemma-4 <|tool_call>call:NAME{...}<tool_call|>.
// JSON-layout calls stream incrementally (CALL_BEGIN once name+args start are known, then
// CALL_ARGS_DELTA per raw byte, CALL_END at close) instead of buffering the whole body (which
// cost 20-60s of zero SSE bytes on a large code edit); non-JSON layouts still buffer (their wire
// format needs the complete body to transform to JSON). Unparseable buffered bodies restore to
// content verbatim; once CALL_BEGIN fires that option is gone, so it waits for a parsed name AND
// a '{'/'[' value first.

#include "tool_call.h"
#include "stream_pipeline.h"  // imp::stream::utf8_complete_len (#1554)

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace imp::server {

class StreamToolCallFilter {
public:
    explicit StreamToolCallFilter(imp::ChatTemplateFamily family) : family_(family) {}

    // One step of output, in stream order. tc.id is NOT assigned — the caller
    // owns the id counter (assign it on CALL / CALL_BEGIN).
    struct Segment {
        enum class Kind {
            TEXT,             // plain content (text)
            CALL,             // complete buffered call (call) — non-JSON layouts
            CALL_BEGIN,       // streamed call opened (call.name set, arguments empty)
            CALL_ARGS_DELTA,  // raw argument bytes for the open call (text)
            CALL_END,         // streamed call closed (call complete; arguments =
                              //   concatenation of all deltas)
        };
        Kind kind = Kind::TEXT;
        std::string text;
        ParsedToolCall call;
    };
    using Result = std::vector<Segment>;

    Result feed(std::string piece) {
        Result r;
        for (;;) {
            switch (phase_) {
                case Phase::CONTENT: {
                    // Trim the cosmetic whitespace right after a completed
                    // tool call. Stateful (not just on the same-feed `after`
                    // remainder) so the output is chunking-invariant.
                    if (trim_ws_) {
                        size_t ws = piece.find_first_not_of("\n\r\t ");
                        if (ws == std::string::npos) {
                            piece.clear();
                            return r;  // all whitespace — keep skipping
                        }
                        piece.erase(0, ws);
                        trim_ws_ = false;
                    }
                    size_t lt = piece.find('<');
                    if (lt == std::string::npos) {
                        append_text(r, std::move(piece));
                        return r;
                    }
                    append_text(r, piece.substr(0, lt));
                    tag_buf_ = piece.substr(lt);
                    piece.clear();
                    phase_ = Phase::TAG;
                    break;  // scan immediately (the tag may already be complete)
                }
                case Phase::TAG: {
                    tag_buf_ += piece;
                    piece.clear();
                    auto scan = scan_tool_tag(tag_buf_, family_);
                    if (scan.kind == ToolTagScan::Kind::PARTIAL)
                        return r;  // could still become a tag — keep holding
                    if (scan.kind == ToolTagScan::Kind::NONE) {
                        // Provably not a tool tag anywhere in the buffer:
                        // release verbatim WITHOUT re-scanning (re-scanning the
                        // leading '<' would hold this content forever).
                        append_text(r, std::move(tag_buf_));
                        tag_buf_.clear();
                        phase_ = Phase::CONTENT;
                        return r;
                    }
                    append_text(r, tag_buf_.substr(0, scan.content_len));
                    open_text_ = tag_buf_.substr(scan.content_len, scan.body_start - scan.content_len);
                    body_buf_ = tag_buf_.substr(scan.body_start);
                    close_tag_ = scan.close_tag;
                    fn_name_ = std::move(scan.fn_name);
                    gemma_body_ = scan.gemma_body;
                    tag_buf_.clear();
                    reset_streaming_();
                    phase_ = Phase::BODY;
                    break;  // the close marker may already be in body_buf_
                }
                case Phase::BODY: {
                    body_buf_ += piece;
                    piece.clear();

                    // Incremental path: try to open + stream the arguments of
                    // a JSON-layout call while the body is still arriving.
                    if (!gemma_body_ && stream_state_ != StreamState::REJECTED)
                        pump_streaming_(r);

                    size_t close_pos = body_buf_.find(close_tag_);
                    if (close_pos == std::string::npos)
                        return r;  // still collecting the body

                    if (stream_state_ == StreamState::ARGS ||
                        stream_state_ == StreamState::TAIL) {
                        // Streamed call: close it out (any confirmed bytes up
                        // to the close marker were already emitted by
                        // pump_streaming_'s close-tag-aware limit).
                        finish_streaming_(r, close_pos);
                    } else {
                        // Buffered call (non-JSON layout, or the JSON value
                        // never materialized): parse the complete body.
                        std::string body = body_buf_.substr(0, close_pos);
                        auto bs = body.find_first_not_of("\n\r\t ");
                        auto be = body.find_last_not_of("\n\r\t ");
                        body = (bs == std::string::npos) ? std::string()
                                                         : body.substr(bs, be - bs + 1);

                        Segment seg;
                        if (parse_stream_tool_body(body, gemma_body_, fn_name_, seg.call)) {
                            seg.kind = Segment::Kind::CALL;
                            r.push_back(std::move(seg));
                        } else {
                            // Neither JSON nor a known fallback layout — restore
                            // the raw text (markers included) instead of dropping.
                            append_text(r, open_text_ +
                                               body_buf_.substr(0, close_pos + close_tag_.size()));
                        }
                    }

                    piece = body_buf_.substr(close_pos + close_tag_.size());
                    body_buf_.clear();
                    open_text_.clear();
                    fn_name_.clear();
                    gemma_body_ = false;
                    reset_streaming_();
                    phase_ = Phase::CONTENT;
                    trim_ws_ = true;  // drop cosmetic ws after the close marker
                    if (piece.empty())
                        return r;
                    break;  // re-scan `after` — it may contain another call
                }
            }
        }
    }

    // finish(): returns held raw bytes for a stream that ended mid-tag/mid-body. Once CALL_BEGIN was
    // emitted (arguments already streamed) nothing is restorable - the caller must close the open
    // call instead (see call_open()).
    std::string finish() const {
        if (call_open())
            return std::string();
        return tag_buf_ + open_text_ + body_buf_;
    }

    // True while text is being held as a potential/partial tool call.
    bool mid_tool() const { return phase_ != Phase::CONTENT; }

    // True when a streamed call is open (CALL_BEGIN emitted, CALL_END not
    // yet). On end-of-stream the caller must close the call/block itself;
    // streamed_arguments() is everything emitted so far.
    bool call_open() const {
        return phase_ == Phase::BODY &&
               (stream_state_ == StreamState::ARGS || stream_state_ == StreamState::TAIL);
    }
    const std::string& streamed_arguments() const { return streamed_args_; }

private:
    enum class Phase { CONTENT, TAG, BODY };

    // StreamState within BODY: PROBE (locating name + args-value start, nothing emitted), ARGS
    // (CALL_BEGIN emitted, streaming raw arg bytes, JSON-nesting tracked), TAIL (value closed,
    // swallowing the cosmetic remainder), REJECTED (not a streamable JSON layout, buffer as before).
    enum class StreamState { PROBE, ARGS, TAIL, REJECTED };

    static void append_text(Result& r, std::string text) {
        if (text.empty())
            return;
        if (!r.empty() && r.back().kind == Segment::Kind::TEXT) {
            r.back().text += text;
            return;
        }
        Segment seg;
        seg.text = std::move(text);
        r.push_back(std::move(seg));
    }

    void reset_streaming_() {
        stream_state_ = StreamState::PROBE;
        args_emitted_ = 0;
        args_end_ = std::string::npos;
        depth_ = 0;
        in_string_ = false;
        escaped_ = false;
        streamed_name_.clear();
        streamed_args_.clear();
    }

    // pump_streaming_: emits CALL_BEGIN once name+args-value start are known, then CALL_ARGS_DELTA
    // per confirmed byte. Never streams at/past a potential close-marker occurrence (withholds a
    // straddling prefix), so output matches the buffered path when the marker appears mid-body.
    void pump_streaming_(Result& r) {
        if (stream_state_ == StreamState::PROBE) {
            size_t args_start = 0;
            if (!try_open_streaming_(args_start))
                return;
            Segment seg;
            seg.kind = Segment::Kind::CALL_BEGIN;
            seg.call.name = streamed_name_;
            seg.call.valid = true;
            r.push_back(std::move(seg));
            stream_state_ = StreamState::ARGS;
            args_emitted_ = args_start;
        }
        if (stream_state_ == StreamState::ARGS) {
            // Confirmed-safe limit: stop at the close marker if present;
            // otherwise withhold a tail that could be the start of one.
            size_t limit = body_buf_.size();
            size_t cp = body_buf_.find(close_tag_, args_emitted_);
            if (cp != std::string::npos) {
                limit = cp;
            } else if (!close_tag_.empty()) {
                size_t hold = close_tag_.size() - 1;
                limit = (limit > args_emitted_ + hold) ? (limit - hold) : args_emitted_;
            }
            size_t i = args_emitted_;
            for (; i < limit && args_end_ == std::string::npos; ++i) {
                char c = body_buf_[i];
                if (in_string_) {
                    if (escaped_)
                        escaped_ = false;
                    else if (c == '\\')
                        escaped_ = true;
                    else if (c == '"')
                        in_string_ = false;
                    continue;
                }
                if (c == '"')
                    in_string_ = true;
                else if (c == '{' || c == '[')
                    depth_++;
                else if (c == '}' || c == ']') {
                    depth_--;
                    if (depth_ == 0)
                        args_end_ = i + 1;  // one past the value end
                }
            }
            size_t upto = (args_end_ != std::string::npos) ? args_end_ : i;
            if (upto > args_emitted_) {
                std::string piece = body_buf_.substr(args_emitted_, upto - args_emitted_);
                // #1554: the close-tag pullback (limit -= close_tag_.size()-1 bytes) can land mid-codepoint,
                // each half JSON-encoded separately into U+FFFD (measured: 10 replacement chars in one arg
                // string, Qwen3-8B-Q8_0 forced tool_choice). Holds the incomplete UTF-8 tail for the next
                // feed
                // (per-token path has done this since #1310) - except when args_end_ is already known, where
                // a
                // tail is genuinely ill-formed, not split.
                if (args_end_ == std::string::npos) {
                    const size_t complete = imp::stream::utf8_complete_len(piece);
                    // Same 3-byte bound as Utf8Stitch::feed: a split codepoint
                    // is at most 3 bytes short, and holding a longer tail would
                    // stall the stream on genuinely invalid input.
                    if (complete < piece.size() && piece.size() - complete <= 3)
                        piece.resize(complete);
                }
                if (!piece.empty()) {
                    Segment seg;
                    seg.kind = Segment::Kind::CALL_ARGS_DELTA;
                    args_emitted_ += piece.size();
                    streamed_args_ += piece;
                    seg.text = std::move(piece);
                    r.push_back(std::move(seg));
                }
            }
            if (args_end_ != std::string::npos)
                stream_state_ = StreamState::TAIL;
        }
        // TAIL: nothing to emit — the remainder up to the close tag is the
        // cosmetic JSON-wrapper close (ChatML) / whitespace.
    }

    // try_open_streaming_: decides whether the body is a streamable JSON layout and locates the
    // arguments value (Llama3: whole body; ChatML: "arguments" field). Sets REJECTED when the layout
    // is provably not streamable, so the buffered path handles it at close as before.
    bool try_open_streaming_(size_t& args_start) {
        size_t first = body_buf_.find_first_not_of("\n\r\t ");
        if (first == std::string::npos)
            return false;  // still whitespace — keep probing
        if (!fn_name_.empty()) {
            // Llama3: body must be the arguments object itself.
            if (body_buf_[first] != '{' && body_buf_[first] != '[') {
                stream_state_ = StreamState::REJECTED;
                return false;
            }
            streamed_name_ = fn_name_;
            args_start = first;
            return true;
        }
        // ChatML JSON: expect {"name": "...", ..., "arguments": <value>. The
        // Qwen3.6 XML fallback starts with '<' — reject to the buffered path.
        if (body_buf_[first] != '{') {
            stream_state_ = StreamState::REJECTED;
            return false;
        }
        size_t pos = first + 1;
        std::string name;
        if (!scan_json_key_string_(pos, "name", name))
            return false;  // need more bytes (or REJECTED was set)
        size_t apos = body_buf_.find("\"arguments\"", pos);
        if (apos == std::string::npos) {
            // The key may still arrive; but a long stretch past the name
            // without it means this is not the expected layout.
            if (body_buf_.size() > pos + 256)
                stream_state_ = StreamState::REJECTED;
            return false;
        }
        size_t vpos = body_buf_.find_first_not_of("\n\r\t :", apos + 11);
        if (vpos == std::string::npos)
            return false;  // value not arrived yet
        if (body_buf_[vpos] != '{' && body_buf_[vpos] != '[') {
            // String-encoded or scalar arguments — not streamable as raw
            // bytes (the client expects the decoded JSON object text).
            stream_state_ = StreamState::REJECTED;
            return false;
        }
        streamed_name_ = std::move(name);
        args_start = vpos;
        return true;
    }

    // scan_json_key_string_: scans `"key": "value"` at/after `pos`. Returns false when more bytes
    // are needed (not yet a mismatch); sets REJECTED only on a definite mismatch.
    bool scan_json_key_string_(size_t& pos, const char* key, std::string& out) {
        size_t p = body_buf_.find_first_not_of("\n\r\t ", pos);
        if (p == std::string::npos)
            return false;
        std::string want = std::string("\"") + key + "\"";
        size_t avail = body_buf_.size() - p;
        size_t cmp_len = std::min(want.size(), avail);
        if (body_buf_.compare(p, cmp_len, want, 0, cmp_len) != 0) {
            stream_state_ = StreamState::REJECTED;
            return false;
        }
        if (avail < want.size())
            return false;  // prefix matches so far — need more bytes
        p = body_buf_.find_first_not_of("\n\r\t :", p + want.size());
        if (p == std::string::npos)
            return false;
        if (body_buf_[p] != '"') {
            stream_state_ = StreamState::REJECTED;
            return false;
        }
        std::string val;
        size_t q = p + 1;
        bool esc = false;
        for (; q < body_buf_.size(); ++q) {
            char c = body_buf_[q];
            if (esc) {
                val += c;  // good enough for a function name (no \uXXXX needed)
                esc = false;
            } else if (c == '\\') {
                esc = true;
            } else if (c == '"') {
                out = std::move(val);
                pos = q + 1;
                return true;
            } else {
                val += c;
            }
        }
        return false;  // string not terminated yet
    }

    // finish_streaming_: close marker found while a call is open -> CALL_END with the accumulated
    // arguments. If the JSON value never closed before the marker, closes with what was already
    // streamed (those deltas are on the wire regardless).
    void finish_streaming_(Result& r, size_t /*close_pos*/) {
        Segment seg;
        seg.kind = Segment::Kind::CALL_END;
        seg.call.name = streamed_name_;
        seg.call.arguments = streamed_args_.empty() ? std::string("{}") : streamed_args_;
        seg.call.valid = true;
        r.push_back(std::move(seg));
    }

    imp::ChatTemplateFamily family_;
    Phase phase_ = Phase::CONTENT;
    std::string tag_buf_;    // TAG: bytes from the triggering '<' onward
    std::string open_text_;  // BODY: the open marker text (for raw restore)
    std::string body_buf_;   // BODY: accumulated body (+ trailing bytes)
    std::string close_tag_;  // BODY: expected close marker
    std::string fn_name_;    // BODY, Llama3: name from the open tag
    bool gemma_body_ = false;
    bool trim_ws_ = false;   // CONTENT: skip leading ws (right after a call)

    // Incremental-argument streaming state (BODY sub-state).
    StreamState stream_state_ = StreamState::PROBE;
    size_t args_emitted_ = 0;              // body_buf_ index streamed so far
    size_t args_end_ = std::string::npos;  // one past the value end, once closed
    int depth_ = 0;
    bool in_string_ = false;
    bool escaped_ = false;
    std::string streamed_name_;
    std::string streamed_args_;
};

}  // namespace imp::server
