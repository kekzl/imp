#pragma once

// Streaming tool-call demux shared by the OpenAI (handlers_chat_stream.cpp)
// and Anthropic (handlers_messages.cpp) SSE paths. Pure text-level state
// machine (same pattern as reasoning_split.h): feed decoded token pieces in,
// get back the user-visible content plus any completed tool calls, in order.
//
// Dialects (see scan_tool_tag / parse_stream_tool_body in tool_call.h):
//   ChatML  <tool_call>{json}</tool_call>          (JSON, Qwen3.6 XML fallback)
//   Llama3  <function=NAME>{json}</function>
//   Gemma-4 <|tool_call>call:NAME{...}<tool_call|> (plus the ChatML fallback)
//
// Unparseable bodies are restored to the content stream verbatim instead of
// being silently dropped; text that is provably not a tag is released without
// re-scanning (the previous in-handler machines could hold content forever
// once a bare '<' appeared in prose).

#include "tool_call.h"

#include <string>
#include <utility>
#include <vector>

namespace imp::server {

class StreamToolCallFilter {
public:
    explicit StreamToolCallFilter(imp::ChatTemplateFamily family) : family_(family) {}

    // One step of output, in stream order: zero or more segments, each either
    // plain content text or a completed tool call (tc.id is NOT assigned —
    // the caller owns the id counter).
    struct Segment {
        bool is_call = false;
        std::string text;    // content segment (is_call == false)
        ParsedToolCall call; // completed call (is_call == true)
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
                    phase_ = Phase::BODY;
                    break;  // the close marker may already be in body_buf_
                }
                case Phase::BODY: {
                    body_buf_ += piece;
                    piece.clear();
                    size_t close_pos = body_buf_.find(close_tag_);
                    if (close_pos == std::string::npos)
                        return r;  // still collecting the body

                    std::string body = body_buf_.substr(0, close_pos);
                    auto bs = body.find_first_not_of("\n\r\t ");
                    auto be = body.find_last_not_of("\n\r\t ");
                    body = (bs == std::string::npos) ? std::string()
                                                     : body.substr(bs, be - bs + 1);

                    Segment seg;
                    if (parse_stream_tool_body(body, gemma_body_, fn_name_, seg.call)) {
                        seg.is_call = true;
                        r.push_back(std::move(seg));
                    } else {
                        // Neither JSON nor a known fallback layout — restore
                        // the raw text (markers included) instead of dropping.
                        append_text(r, open_text_ +
                                           body_buf_.substr(0, close_pos + close_tag_.size()));
                    }

                    piece = body_buf_.substr(close_pos + close_tag_.size());
                    body_buf_.clear();
                    open_text_.clear();
                    fn_name_.clear();
                    gemma_body_ = false;
                    phase_ = Phase::CONTENT;
                    trim_ws_ = true;  // drop cosmetic ws after the close marker
                    if (piece.empty())
                        return r;
                    break;  // re-scan `after` — it may contain another call
                }
            }
        }
    }

    // Stream ended mid-tag/mid-body: the held raw bytes (incomplete tool call).
    std::string finish() const { return tag_buf_ + open_text_ + body_buf_; }

    // True while text is being held as a potential/partial tool call.
    bool mid_tool() const { return phase_ != Phase::CONTENT; }

private:
    enum class Phase { CONTENT, TAG, BODY };

    static void append_text(Result& r, std::string text) {
        if (text.empty())
            return;
        if (!r.empty() && !r.back().is_call) {
            r.back().text += text;
            return;
        }
        Segment seg;
        seg.text = std::move(text);
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
};

}  // namespace imp::server
