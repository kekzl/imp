// Shared per-token SSE streaming loop (see stream_driver.h). The dialect
// adapters live in handlers_chat_stream.cpp (OpenAI chat),
// handlers_messages.cpp (Anthropic) and handlers_responses.cpp (Responses).

#include "stream_driver.h"

#include "utils.h"
#include "tool_stream_filter.h"
#include "stream_pipeline.h"
#include "reasoning_split.h"

#include "runtime/request.h"

#include <chrono>
#include <cstdio>
#include <cstring>

bool run_stream_loop_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                      const std::shared_ptr<ServerRequest>& server_req, StreamDialect& d,
                      StreamLoopResult& out) {
    const auto& stop_sequences = ctx.params.stop_sequences;
    // Derive max_stop_len from the FINAL stop list, not ctx.params.max_stop_len:
    // the snapshot phase may inject server-side stops ("\nHuman" turn guard for
    // think models) AFTER request parsing computed max_stop_len. A stale 0 made
    // the partial-match holdback `size - max_stop_len + 1` flush one byte PAST
    // pending_text's end — emitting the std::string NUL terminator into every
    // SSE content delta ("4\0") and disabling cross-token stop matching.
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());
    const bool enable_thinking = ctx.snap.enable_thinking;
    const bool has_tools = ctx.params.has_tools;
    const auto tpl_family = ctx.snap.tpl_family;
    const float think_budget = ctx.params.think_budget;
    const auto snap_tok = ctx.snap.tok;
    const bool snap_have_template = ctx.snap.have_template;
    const auto& snap_stop_token_ids = ctx.snap.stop_token_ids;
    const auto t_start = ctx.t_start;

    const char* finish = nullptr;

    // Buffer for incomplete UTF-8 sequences across token boundaries.
    std::string utf8_buf;
    // Buffered output for stop-sequence matching: text is held back until it
    // provably does not contain (a prefix of) a stop match.
    std::string pending_text;
    bool text_stop_matched = false;

    // Streaming tool-call demux (tool_stream_filter.h) — pure state machine.
    // Detects ChatML/Llama3/Gemma-4 open markers, holds back potential-tag
    // text, parses completed bodies (JSON, Qwen3.6 XML fallback, Gemma
    // call:NAME{...}) and restores unparseable ones to the content stream.
    imp::server::StreamToolCallFilter tool_filter(tpl_family);
    // parallel_tool_calls=false: a second streamed call was opened by the
    // filter but is being suppressed (skip its deltas/END too).
    bool stream_call_suppressed = false;

    // Reasoning/content demux (DeepSeek <think>) — shared state machine in
    // reasoning_split.h. enable_thinking also covers text-level thinkers
    // (Nemotron: template-injected "<think>" as plain text, no special token).
    const bool use_reasoning = (state.default_args.reasoning_format == "deepseek" &&
                                (ctx.snap.is_think_model || enable_thinking));
    const bool think_active = use_reasoning || enable_thinking;
    imp::server::ThinkPhase think_start_phase;
    if (enable_thinking)
        think_start_phase = imp::server::ThinkPhase::REASONING;  // <think> in prefill
    else if (use_reasoning && think_budget > 0.0f)
        think_start_phase = imp::server::ThinkPhase::SCAN;  // model decides
    else
        think_start_phase = imp::server::ThinkPhase::CONTENT;  // no extraction
    imp::server::StreamReasoningSplitter think_split(think_start_phase, ctx.snap.think_start_id,
                                                     ctx.snap.think_end_id);

    // Rejoins characters the tokenizer split across two tokens, before any
    // consumer sees the piece — the think splitter and tool filter match on raw
    // bytes, so half a character must never reach them either.
    Utf8Stitch utf8_stitch;

    // Gemma-4 channel filter state: when we see <|channel> or <channel|>,
    // skip tokens until the next newline (the channel header).
    bool channel_header_active = false;

    // gpt-oss Harmony streaming filter. The model emits
    //   <|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…
    // Route analysis/commentary channels to the reasoning sink and the final
    // channel to content, stripping the control markers (which arrive as atomic
    // special-token pieces) and the <|start|>role plumbing. hm_buf holds the
    // current channel's bytes so a token that splits a multibyte char is not
    // emitted mid-codepoint (#760).
    const bool harmony = (tpl_family == imp::ChatTemplateFamily::HARMONY);
    std::string hm_channel, hm_name, hm_buf;
    bool hm_in_msg = false, hm_reading_name = false;
    auto hm_flush = [&](bool force) -> bool {
        size_t complete = force ? hm_buf.size() : utf8_complete_len(hm_buf);
        if (complete == 0)
            return true;
        std::string chunk = hm_buf.substr(0, complete);
        hm_buf.erase(0, complete);
        if (hm_channel == "analysis" || hm_channel == "commentary")
            return d.harmony_reasoning_on ? d.emit_reasoning(chunk) : true;
        return d.emit_text(chunk);
    };

    // Flush confirmed holdback text up to a byte position.
    auto flush_text = [&](size_t up_to) -> bool {
        up_to = std::min(up_to, pending_text.size());  // never read past the buffer
        if (up_to == 0)
            return true;
        bool ok = d.emit_text(pending_text.substr(0, up_to));
        pending_text.erase(0, up_to);
        return ok;
    };

    // Flush held content buffers before emitting a tool call (or directly
    // emitted text), so stream order is preserved. A complete stop match
    // cannot be pending here: the normal emission path checks after every
    // append with the same holdback decision.
    auto flush_buffered_content = [&]() -> bool {
        if (stop_sequences.empty()) {
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                std::string chunk = utf8_buf.substr(0, complete);
                utf8_buf.erase(0, complete);
                if (!d.emit_text(chunk))
                    return false;
            }
        } else if (!pending_text.empty()) {
            auto hd = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(hd.flush_len))
                return false;
        }
        return true;
    };

    auto request_start = std::chrono::steady_clock::now();
    auto last_keepalive = request_start;
    for (;;) {
        // Terminate as soon as a finish reason has been recorded. The is_last
        // token sets `finish` and then falls through to the per-token
        // post-processing below, where a think/reasoning/channel `continue`
        // can skip the trailing `if (finish) break`. Re-checking here means the
        // stream always ends (and the terminal SSE frame is emitted) even when
        // the final token is swallowed by one of those paths (#755/#757).
        if (finish)
            break;

        // Check client disconnect.
        if (!sink.is_writable()) {
            server_req->cancel();
            state.metrics.requests_cancelled++;
            finish = "cancelled";
            break;
        }

        // Check request timeout.
        if (state.request_timeout > 0) {
            auto elapsed = std::chrono::steady_clock::now() - request_start;
            if (elapsed > std::chrono::seconds(state.request_timeout)) {
                server_req->cancel();
                finish = "length";
                break;
            }
        }

        // Read next token from the batching engine (with timeout).
        TokenEvent evt{};
        if (!server_req->pop_token(evt)) {
            // No token ready yet (long prefill / queued behind other work).
            // Emit a dialect keepalive every ~10s so reverse proxies and SDK
            // idle-timeouts don't kill the connection. A failed keepalive
            // write means the client is gone — cancel like a disconnect.
            auto now = std::chrono::steady_clock::now();
            if (now - last_keepalive > std::chrono::seconds(10)) {
                last_keepalive = now;
                if (!d.keepalive()) {
                    server_req->cancel();
                    state.metrics.requests_cancelled++;
                    finish = "cancelled";
                    break;
                }
            }
            continue;  // timeout — loop back to check disconnect/timeout
        }

        if (evt.token_id < 0) {
            // Finish event with no token.
            finish = evt.finish_reason ? evt.finish_reason : "stop";
            break;
        }

        int32_t token = evt.token_id;

        // Silently drop structural stop tokens that slipped through. The
        // engine's think-block implicit-close (Engine::should_stop) passes ONE
        // EOS-like token through to recover from empty thinking. That token
        // must not appear as user-visible content (would render as
        // "<|im_end|>" / "<|endoftext|>" in chat).
        if (!evt.is_last) {
            bool is_structural_stop = (token == snap_tok->eos_id());
            if (!is_structural_stop && snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids) {
                    if (token == stop_id) {
                        is_structural_stop = true;
                        break;
                    }
                }
            }
            if (is_structural_stop)
                continue;
        }

        // Check stop conditions (EOS/stop tokens already detected by engine).
        if (evt.is_last) {
            // The engine marked this as the last token. Don't emit EOS/stop
            // tokens — they're structural, not content.
            if (token == snap_tok->eos_id()) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            bool is_stop = false;
            if (snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids) {
                    if (token == stop_id) {
                        is_stop = true;
                        break;
                    }
                }
            }
            if (is_stop) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            // Not a stop token — emit it, then finish after this iteration.
            finish = evt.finish_reason ? evt.finish_reason : "length";
        }

        out.n_output_tokens++;
        if (out.n_output_tokens == 1) {
            auto t_first = std::chrono::high_resolution_clock::now();
            out.ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
        }
        // A token can end mid-character; hold the partial bytes until the next
        // one completes them, or the delta ships half a character as U+FFFD.
        std::string piece = utf8_stitch.feed(snap_tok->decode_token(token));

        // gpt-oss Harmony channel routing. Markers arrive as atomic
        // special-token pieces.
        if (harmony) {
            if (piece == "<|channel|>" || piece == "<|message|>" || piece == "<|end|>" ||
                piece == "<|return|>" || piece == "<|start|>") {
                if (hm_in_msg && !hm_flush(/*force=*/true))
                    return false;
                if (piece == "<|channel|>") {
                    hm_reading_name = true;
                    hm_in_msg = false;
                    hm_name.clear();
                } else if (piece == "<|message|>") {
                    size_t s = hm_name.find_first_not_of("\n\r\t ");
                    size_t e = hm_name.find_last_not_of("\n\r\t ");
                    hm_channel = (s == std::string::npos) ? std::string() : hm_name.substr(s, e - s + 1);
                    hm_reading_name = false;
                    hm_in_msg = true;
                } else {  // <|end|> / <|return|> / <|start|>: close the block
                    hm_in_msg = false;
                    hm_reading_name = false;
                    hm_channel.clear();
                }
                continue;
            }
            if (hm_reading_name) {  // channel name between <|channel|> and <|message|>
                hm_name += piece;
                continue;
            }
            if (!hm_in_msg)  // role text / inter-block plumbing
                continue;
            hm_buf += piece;
            if (!hm_flush(/*force=*/false))
                return false;
            continue;
        }

        // Gemma-4 channel filter: strip "<|channel>NAME\n" structural headers
        // from the content stream. `<channel|>` is the channel-switch marker —
        // strip the token but do NOT enter the scan-until-newline mode,
        // because Q5_K_M sometimes emits the final answer directly after it
        // with no newline (observed: "<|channel>thought\n<channel|>5 + 3 = 8").
        if (ctx.snap.channel_open_id >= 0) {
            if (channel_header_active) {
                if (token == ctx.snap.channel_newline_id || (!piece.empty() && piece.back() == '\n')) {
                    channel_header_active = false;
                }
                continue;
            }
            if (token == ctx.snap.channel_open_id) {
                channel_header_active = true;
                continue;
            }
            if (token == ctx.snap.channel_close_id) {
                // Drop just the marker; the next token is body.
                continue;
            }
        }

        // Reasoning/content demux (DeepSeek <think>): routes reasoning to the
        // dialect's reasoning sink and hands back the user-visible content for
        // this step (empty when the whole piece was reasoning or is being held
        // for boundary detection).
        if (think_active) {
            auto rs = think_split.feed(std::move(piece), token);
            out.n_reasoning_tokens += rs.reasoning_tokens;
            if (!rs.reasoning.empty() && !d.emit_reasoning(rs.reasoning))
                return false;
            if (rs.content.empty())
                continue;
            piece = std::move(rs.content);
        }

        // Streaming tool-call demux (only when tools are present): the filter
        // returns the user-visible content and any completed tool calls, in
        // stream order. Content before/between calls is emitted directly;
        // trailing content after the last call falls through to the normal
        // emission path below.
        if (has_tools) {
            auto segs = tool_filter.feed(std::move(piece));
            piece.clear();
            using SegKind = imp::server::StreamToolCallFilter::Segment::Kind;
            for (size_t si = 0; si < segs.size(); ++si) {
                auto& seg = segs[si];
                if (seg.kind == SegKind::TEXT) {
                    if (si + 1 == segs.size()) {
                        piece = std::move(seg.text);  // trailing content
                    } else {
                        if (!flush_buffered_content())
                            return false;
                        if (!d.emit_text(seg.text))
                            return false;
                    }
                    continue;
                }
                // parallel_tool_calls=false: stream at most one tool call.
                // (For a streamed call the gate fires at CALL_BEGIN, so the
                // later deltas/END of a suppressed call are skipped too.)
                if (!ctx.params.parallel_tool_calls &&
                    ((seg.kind == SegKind::CALL && !out.tool_calls.empty()) ||
                     (seg.kind == SegKind::CALL_BEGIN && !out.tool_calls.empty()) ||
                     (seg.kind != SegKind::CALL && stream_call_suppressed))) {
                    if (seg.kind == SegKind::CALL_BEGIN)
                        stream_call_suppressed = true;
                    if (seg.kind == SegKind::CALL_END)
                        stream_call_suppressed = false;
                    continue;
                }

                if (seg.kind == SegKind::CALL_BEGIN) {
                    // Streamed call opens: the dialect emits its open frame
                    // now; the argument bytes follow as CALL_ARGS_DELTA
                    // segments while the model is still generating them
                    // (previously the whole body was buffered until the close
                    // tag — 20-60 s of zero SSE bytes on a big code edit).
                    ParsedToolCall tc = std::move(seg.call);
                    tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                    if (!flush_buffered_content())
                        return false;
                    out.tool_calls.push_back(std::move(tc));
                    out.tool_calls_emitted = true;
                    if (!d.on_call_begin(out.tool_calls.back()))
                        return false;
                    continue;
                }
                if (seg.kind == SegKind::CALL_ARGS_DELTA) {
                    if (!d.on_call_args_delta(seg.text))
                        return false;
                    continue;
                }
                if (seg.kind == SegKind::CALL_END) {
                    // Deltas already on the wire — record the full arguments
                    // for bookkeeping; the dialect closes its frame.
                    if (!out.tool_calls.empty())
                        out.tool_calls.back().arguments = std::move(seg.call.arguments);
                    if (!d.on_call_end(out.tool_calls.empty() ? nullptr : &out.tool_calls.back()))
                        return false;
                    continue;
                }

                // SegKind::CALL — buffered call (non-JSON layouts): the
                // dialect emits the whole call (open + arguments + close).
                ParsedToolCall tc = std::move(seg.call);
                tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                if (!flush_buffered_content())
                    return false;
                out.tool_calls.push_back(std::move(tc));
                out.tool_calls_emitted = true;
                if (!d.on_call_buffered(out.tool_calls.back()))
                    return false;
            }
            if (piece.empty())
                continue;
        }

        // Normal content emission (no tool tag detected).
        if (stop_sequences.empty()) {
            // No stop sequences: stream directly (with UTF-8 buffering).
            utf8_buf += piece;
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                std::string chunk = utf8_buf.substr(0, complete);
                utf8_buf.erase(0, complete);
                if (!d.emit_content_token(chunk))
                    return false;
            }
        } else {
            // Buffer text and check for stop matches via the pure holdback
            // pipeline (stream_pipeline.h). It returns the safe-to-emit prefix
            // and whether a complete stop sequence is present.
            pending_text += piece;
            auto hd = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(hd.flush_len))
                return false;
            if (hd.complete_match) {
                text_stop_matched = true;
                finish = "stop";
                break;
            }
        }

        // Break after processing the last non-EOS token from batching engine.
        if (finish)
            break;
    }

    // Harmony: flush the final channel's tail (the final block usually ends at
    // EOS/<|return|> with no trailing <|end|>). The other buffers below stay
    // empty for harmony, so they're no-ops.
    if (harmony && !hm_buf.empty())
        hm_flush(/*force=*/true);

    // Flush the splitter's held tail at stream end: buffered reasoning -> the
    // reasoning sink, any held/undecided content -> the content flush below.
    if (think_active) {
        auto rs = think_split.finish();
        if (!rs.reasoning.empty())
            d.emit_reasoning(rs.reasoning);
        if (!rs.content.empty())
            utf8_buf += rs.content;
    }

    // The model exhausted max_tokens while still reasoning and never produced
    // content (finish == "length" — a model that naturally hit EOS during
    // thinking already has its reasoning delivered). The chat dialect emits a
    // user-visible notice on this flag.
    out.reasoning_truncated = think_active && think_split.phase() == imp::server::ThinkPhase::REASONING &&
                              utf8_buf.empty() && pending_text.empty() && finish &&
                              std::strcmp(finish, "length") == 0;

    // Handle incomplete tool call at end (max_tokens hit while in tag/body):
    // release the held raw text as content, finish_reason stays "length".
    if (has_tools && tool_filter.mid_tool() && !out.tool_calls_emitted) {
        std::string leftover = tool_filter.finish();
        if (!leftover.empty())
            utf8_buf += leftover;
    }
    // A STREAMED call cut off mid-arguments: its open frame + deltas are
    // already on the wire (nothing restorable) — record what was streamed for
    // bookkeeping; the client sees finish_reason=length.
    if (has_tools && tool_filter.call_open() && !out.tool_calls.empty() &&
        out.tool_calls.back().arguments.empty()) {
        out.tool_calls.back().arguments = tool_filter.streamed_arguments();
    }

    // Flush any remaining buffers (skip after a text-level stop match or when
    // tool calls were emitted).
    if (!utf8_buf.empty() && !text_stop_matched && !out.tool_calls_emitted)
        d.emit_text(utf8_buf);
    if (!pending_text.empty() && !text_stop_matched && !out.tool_calls_emitted)
        d.emit_text(pending_text);

    if (!finish)
        finish = out.tool_calls_emitted ? "tool_calls" : "length";
    else if (out.tool_calls_emitted && std::strcmp(finish, "stop") == 0)
        finish = "tool_calls";
    out.finish = finish;
    return true;
}

void finish_stream_accounting_(ServerState& state, ChatRequestContext& ctx,
                               const std::shared_ptr<imp::Request>& active_req, const StreamLoopResult& out,
                               const std::string& req_id, const char* label) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - ctx.t_start).count();
    int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;
    fprintf(stderr, "[%s] %s%d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms, cached=%d)\n",
            req_id.c_str(), label, n_prompt_tokens, out.n_output_tokens, ms, out.ttft_ms, cached);
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += n_prompt_tokens;
    state.metrics.tokens_completion_total += out.n_output_tokens;
    state.metrics.tokens_cached_total += cached;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.last_ttft_ms = static_cast<int64_t>(out.ttft_ms);
    state.metrics.request_duration.observe(ms / 1000.0);
    if (out.n_output_tokens > 0)
        state.metrics.ttft.observe(out.ttft_ms / 1000.0);
    // Mean inter-token latency: post-first-token decode time spread over the
    // remaining tokens. Streaming-only (non-stream has no per-token cadence).
    if (out.n_output_tokens > 1)
        state.metrics.inter_token.observe((ms - out.ttft_ms) / 1000.0 / (out.n_output_tokens - 1));

    // Streaming response content is not accumulated across SSE chunks, so the
    // JSONL `response` field stays null. The request body, token counts,
    // finish reason, and latency still reflect everything the client did.
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, req_id, ctx.log_endpoint, ctx.log_client_ip,
                      ctx.log_raw_body, ms, n_prompt_tokens, out.n_output_tokens, out.finish, json());
}
