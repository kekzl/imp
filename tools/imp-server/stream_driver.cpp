// Shared per-token SSE streaming loop (see stream_driver.h). The dialect
// adapters live in handlers_chat_stream.cpp (OpenAI chat),
// handlers_messages.cpp (Anthropic) and handlers_responses.cpp (Responses).

#include "stream_driver.h"

#include "utils.h"
#include "core/logging.h"
#include "tool_stream_filter.h"
#include "stream_pipeline.h"
#include "reasoning_split.h"

#include "runtime/engine.h"
#include "runtime/request.h"

#include <chrono>
#include <cstdio>
#include <cstring>

bool run_stream_loop_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                      const std::shared_ptr<ServerRequest>& server_req, StreamDialect& d,
                      StreamLoopResult& out) {
    const auto& stop_sequences = ctx.params.stop_sequences;
    // max_stop_len must come from the FINAL stop list, not ctx.params.max_stop_len: the snapshot
    // phase can inject server-side stops (e.g. the "\nHuman" guard) after parsing computed it. A
    // stale 0 flushed one byte past pending_text's end, leaking the NUL terminator into SSE deltas.
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
    auto t_prev_token = t_start;  // for the per-token ITL observation (#1577)

    const char* finish = nullptr;

    // Buffer for incomplete UTF-8 sequences across token boundaries.
    std::string utf8_buf;
    // Buffered output for stop-sequence matching: text is held back until it
    // provably does not contain (a prefix of) a stop match.
    std::string pending_text;
    bool text_stop_matched = false;

    // Which token produced which bytes of the two holdback buffers (#1588).
    // See stream_pipeline.h for why the live counter cannot answer that.
    imp::stream::TokenSpans pending_spans;
    imp::stream::TokenSpans utf8_spans;

    // Streaming tool-call demux (tool_stream_filter.h): pure state machine detecting ChatML/Llama3/
    // Gemma-4 open markers, parsing completed bodies, and restoring unparseable ones to content.
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
    // agent_scan_limit: when tools suppress the thinking default, the template renders a pre-closed
    // think block, but a model that reasons anyway emits only the CLOSER - the SCAN opener it's
    // waiting for never arrives, so at the default budget the whole CoT streamed as the answer
    // (found via Claude Code against imp-server). Holding longer here is safe: an agent client is
    // waiting for a tool call anyway. Default 256; server.agent_scan_limit tunes it (AUDIT_arch_2026 E-4).
    const int agent_scan_limit = (state.ctx && state.ctx->engine)
                                     ? state.ctx->engine->runtime_config().server.agent_scan_limit
                                     : 256;
    const int scan_limit = has_tools ? std::max(agent_scan_limit, 1) : 8;
    imp::server::StreamReasoningSplitter think_split(think_start_phase, ctx.snap.think_start_id,
                                                     ctx.snap.think_end_id, scan_limit);
    // Without tools the SCAN hold (8 tokens) buys almost nothing but every thinking-off answer paid
    // it in full before its first delta - measured TTFT 99-126ms client-side vs 17-43ms server-side
    // (Qwen3.8-27B-NVFP4). Released as soon as the first word proves the answer started.
    think_split.set_release_on_plain_text(!has_tools);

    // Rejoins characters the tokenizer split across two tokens, before any
    // consumer sees the piece — the think splitter and tool filter match on raw
    // bytes, so half a character must never reach them either.
    Utf8Stitch utf8_stitch;

    // Gemma-4 channel filter state: when we see <|channel> or <channel|>,
    // skip tokens until the next newline (the channel header).
    bool channel_header_active = false;

    // gpt-oss Harmony streaming filter: routes analysis/commentary channels to the reasoning sink,
    // final to content, stripping control markers and <|start|>role plumbing. hm_buf holds the
    // current channel's bytes so a split multibyte char isn't emitted mid-codepoint (#760); every
    // content byte passes through one of these two sinks, so content_emitted cannot drift from the wire.
    auto send_text = [&](const std::string& chunk) -> bool {
        if (!chunk.empty())
            out.content_emitted = true;
        return d.emit_text(chunk);
    };
    auto send_content_token = [&](const std::string& chunk, int token_index) -> bool {
        if (!chunk.empty())
            out.content_emitted = true;
        return d.emit_content_token(chunk, token_index);
    };

    const bool harmony = (tpl_family == imp::ChatTemplateFamily::HARMONY);
    std::string hm_channel, hm_name, hm_buf, hm_recipient, hm_args;
    bool hm_in_msg = false, hm_reading_name = false, hm_call_open = false;
    auto hm_flush = [&](bool force) -> bool {
        size_t complete = force ? hm_buf.size() : utf8_complete_len(hm_buf);
        if (complete == 0)
            return true;
        std::string chunk = hm_buf.substr(0, complete);
        hm_buf.erase(0, complete);
        // A channel addressed to `functions.NAME` is a tool call, not text (#1716): routing by channel
        // name alone let "commentary to=functions.X" match neither the reasoning nor content branch,
        // dropping the arguments from the stream.
        if (hm_call_open) {
            hm_args += chunk;
            return d.on_call_args_delta(chunk);
        }
        if (hm_channel == "analysis" || hm_channel == "commentary")
            return d.harmony_reasoning_on ? d.emit_reasoning(chunk) : true;
        return send_text(chunk);
    };
    // Close an open call: record the arguments and let the dialect close its
    // frame, exactly as the tag path does at CALL_END.
    auto hm_close_call = [&]() -> bool {
        if (!hm_call_open)
            return true;
        hm_call_open = false;
        if (!out.tool_calls.empty())
            out.tool_calls.back().arguments = hm_args;
        hm_args.clear();
        return d.on_call_end(out.tool_calls.empty() ? nullptr : &out.tool_calls.back());
    };

    // flush_text: emits confirmed holdback text up to a byte position, one emission per token so the
    // dialect can attach that token's logprob (#1588). `up_to` (from the stop matcher) can land
    // mid-token; the remainder then ships as one chunk with index -1 rather than a guessed index.
    auto flush_text = [&](size_t up_to) -> bool {
        up_to = std::min(up_to, pending_text.size());  // never read past the buffer
        if (up_to == 0)
            return true;
        for (const auto& e : pending_spans.flush(up_to)) {
            if (!send_content_token(pending_text.substr(e.offset, e.length), e.token_index))
                return false;
        }
        pending_text.erase(0, up_to);
        return true;
    };

    // flush_buffered_content: flushes held content before a tool call (or direct text) to preserve
    // stream order. A complete stop match cannot be pending here - the normal path already checked
    // after every append.
    auto flush_buffered_content = [&]() -> bool {
        if (stop_sequences.empty()) {
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                std::string chunk = utf8_buf.substr(0, complete);
                utf8_buf.erase(0, complete);
                if (!send_text(chunk))
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
        // Re-checks `finish` here (not just at the loop's trailing break): is_last sets it, but a
        // think/reasoning/channel `continue` earlier in the per-token path can skip that trailing break,
        // leaving the terminal SSE frame ([DONE]) unsent (#755/#757).
        if (finish)
            break;

        // Check client disconnect.
        if (!sink.is_writable()) {
            server_req->cancel();
            state.metrics.requests_cancelled++;
            state.metrics.observe_unadmitted_queue_wait(server_req->t_submit,
                                                        server_req->queue_ms.load(std::memory_order_relaxed));
            finish = "cancelled";
            break;
        }

        // A server --request-timeout reports finish="length" on the wire (no OpenAI enum member for "the
        // server gave up", #1590) but is now COUNTED (#1640) - previously invisible, a timing-out server
        // under load looked identical to clients asking for short answers.
        if (state.request_timeout > 0) {
            auto elapsed = std::chrono::steady_clock::now() - request_start;
            if (elapsed > std::chrono::seconds(state.request_timeout)) {
                server_req->cancel();
                state.metrics.requests_timed_out++;
                state.metrics.observe_unadmitted_queue_wait(server_req->t_submit,
                                                            server_req->queue_ms.load(
                                                                std::memory_order_relaxed));
                IMP_LOG_WARN(
                    "request ended at --request-timeout (%d s); the client sees "
                    "finish_reason=length, which is indistinguishable from a spent "
                    "token budget - see imp_requests_timed_out_total",
                    state.request_timeout);
                finish = "length";
                out.error_type = "timeout_error";
                out.error_message = "request exceeded the server's --request-timeout of " +
                                    std::to_string(state.request_timeout) + " s";
                break;
            }
        }

        // Read next token from the batching engine (with timeout).
        TokenEvent evt{};
        if (!server_req->pop_token(evt)) {
            // Emits a dialect keepalive every ~10s while no token is ready (long prefill / queued) so
            // reverse proxies and SDK idle-timeouts don't kill the connection. A failed write means the
            // client is gone - cancel like a disconnect.
            auto now = std::chrono::steady_clock::now();
            if (now - last_keepalive > std::chrono::seconds(10)) {
                last_keepalive = now;
                if (!d.keepalive()) {
                    server_req->cancel();
                    state.metrics.requests_cancelled++;
                    state.metrics.observe_unadmitted_queue_wait(server_req->t_submit,
                                                                server_req->queue_ms.load(
                                                                    std::memory_order_relaxed));
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

        // Drops structural stop tokens that slip through: Engine::should_stop's think-block implicit-
        // close passes one EOS-like token through to recover from empty thinking, which must not render
        // as "<|im_end|>"/"<|endoftext|>" in user-visible content.
        bool is_structural_stop = (token == snap_tok->eos_id());
        if (!is_structural_stop && snap_have_template) {
            for (int32_t stop_id : snap_stop_token_ids) {
                if (token == stop_id) {
                    is_structural_stop = true;
                    break;
                }
            }
        }
        // ignore_eos (vLLM semantics): EOS and stop tokens count as output
        // tokens and carry no text; only max_tokens ends the request.
        if (ctx.params.ignore_eos && is_structural_stop) {
            out.n_output_tokens++;
            if (evt.is_last) {
                finish = evt.finish_reason ? evt.finish_reason : "length";
                break;
            }
            continue;
        }
        if (!evt.is_last && is_structural_stop)
            continue;

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
        {
            // One ITL observation per token, taken here rather than as a
            // per-request mean after the fact (#1577): a mean cannot show the
            // variance, and variance is the whole reason to keep a histogram.
            auto t_tok = std::chrono::high_resolution_clock::now();
            if (out.n_output_tokens == 1) {
                out.ttft_ms = std::chrono::duration<double, std::milli>(t_tok - t_start).count();
                // Queue time is known once the worker has admitted the request,
                // which is guaranteed by the time a token comes back (#1580).
                const double q = server_req->queue_ms.load(std::memory_order_relaxed);
                ctx.trace.queue_ms = q;  // the span's `queue` child
                if (q >= 0.0)
                    state.metrics.record_queue_wait(ctx.log_endpoint, q / 1000.0);
            } else {
                state.metrics.record_inter_token(ctx.log_endpoint,
                                                 std::chrono::duration<double>(t_tok - t_prev_token).count());
            }
            t_prev_token = t_tok;
        }
        // A token can end mid-character; hold the partial bytes until the next
        // one completes them, or the delta ships half a character as U+FFFD.
        std::string piece = utf8_stitch.feed(snap_tok->decode_token(token));

        // gpt-oss Harmony channel routing. Markers arrive as atomic
        // special-token pieces.
        if (harmony) {
            if (piece == "<|channel|>" || piece == "<|message|>" || piece == "<|end|>" ||
                piece == "<|return|>" || piece == "<|start|>" || piece == "<|call|>") {
                if (hm_in_msg && !hm_flush(/*force=*/true))
                    return false;
                if (piece == "<|channel|>") {
                    hm_reading_name = true;
                    hm_in_msg = false;
                    hm_name.clear();
                } else if (piece == "<|message|>") {
                    // Harmony header shape: `NAME [to=RECIPIENT] [<|constrain|>FMT]`. Must split on
                    // whitespace to
                    // separate the channel name from the recipient - taking the whole string as the name is
                    // how a
                    // tool call used to disappear.
                    size_t s = hm_name.find_first_not_of("\n\r\t ");
                    hm_recipient.clear();
                    if (s == std::string::npos) {
                        hm_channel.clear();
                    } else {
                        size_t sp = hm_name.find_first_of("\n\r\t ", s);
                        hm_channel = hm_name.substr(s, sp == std::string::npos ? std::string::npos : sp - s);
                        const size_t to = hm_name.find("to=functions.", s);
                        if (to != std::string::npos) {
                            const size_t b = to + 13;
                            size_t e2 = hm_name.find_first_of("\n\r\t <", b);
                            hm_recipient = hm_name.substr(b, e2 == std::string::npos ? std::string::npos
                                                                                     : e2 - b);
                        }
                    }
                    hm_reading_name = false;
                    hm_in_msg = true;
                    if (!hm_recipient.empty()) {
                        ParsedToolCall tc;
                        tc.name = hm_recipient;
                        tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                        if (!flush_buffered_content())
                            return false;
                        out.tool_calls.push_back(std::move(tc));
                        out.tool_calls_emitted = true;
                        hm_call_open = true;
                        hm_args.clear();
                        if (!d.on_call_begin(out.tool_calls.back()))
                            return false;
                    }
                } else {  // <|end|> / <|return|> / <|start|> / <|call|>: close
                    if (!hm_close_call())
                        return false;
                    hm_in_msg = false;
                    hm_reading_name = false;
                    hm_channel.clear();
                    hm_recipient.clear();
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

        // Gemma-4 channel filter strips "<|channel>NAME\n" headers; `<channel|>` is a channel-switch
        // marker - strip the token but do NOT scan-until-newline after it, since some quants (Q5_K_M)
        // emit the answer directly with no following newline.
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

        // Reasoning/content demux (DeepSeek <think>): routes reasoning to the dialect's reasoning sink,
        // returns the user-visible content for this step (empty when the whole piece was reasoning or
        // still held for boundary detection).
        if (think_active) {
            auto rs = think_split.feed(std::move(piece), token);
            out.n_reasoning_tokens += rs.reasoning_tokens;
            if (!rs.reasoning.empty() && !d.emit_reasoning(rs.reasoning))
                return false;
            if (rs.content.empty()) {
                // A tool-call opener while SCAN is still holding output settles the question (a call is never
                // reasoning): release the hold immediately, or the whole call would buffer and its argument
                // deltas would never stream.
                if (has_tools && think_split.phase() == imp::server::ThinkPhase::SCAN &&
                    scan_tool_tag(think_split.held(), tpl_family).kind == ToolTagScan::Kind::OPEN) {
                    rs = think_split.flush_scan();
                }
                if (rs.content.empty())
                    continue;
            }
            piece = std::move(rs.content);
        }

        // With tools present, the streaming tool-call filter returns user-visible content and completed
        // calls in stream order; content after the last call falls through to the normal emission path.
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
                        if (!send_text(seg.text))
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
                    // A streamed call opens its dialect frame immediately; argument bytes follow as
                    // CALL_ARGS_DELTA segments while still generating - previously the whole body buffered
                    // until the
                    // close tag, producing 20-60s of zero SSE bytes on a large code-edit call.
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
            utf8_spans.append(piece.size(), out.n_output_tokens - 1);
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                for (const auto& e : utf8_spans.flush(complete)) {
                    if (!send_content_token(utf8_buf.substr(e.offset, e.length), e.token_index))
                        return false;
                }
                utf8_buf.erase(0, complete);
            }
        } else {
            // Buffer text and check for stop matches via the pure holdback
            // pipeline (stream_pipeline.h). It returns the safe-to-emit prefix
            // and whether a complete stop sequence is present.
            pending_text += piece;
            pending_spans.append(piece.size(), out.n_output_tokens - 1);
            auto hd = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(hd.flush_len))
                return false;
            if (hd.complete_match) {
                text_stop_matched = true;
                if (hd.matched_index >= 0 && static_cast<size_t>(hd.matched_index) < stop_sequences.size())
                    out.stop_sequence = stop_sequences[hd.matched_index];
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
        if (!rs.content.empty()) {
            // No token span recorded here on purpose: the splitter reorders bytes between its reasoning and
            // content sinks, so which token produced a held byte isn't recoverable. These bytes ship with
            // token_index -1 (no logprob) rather than a guessed one.
            utf8_buf += rs.content;
        }
    }

    // reasoning_truncated: max_tokens exhausted while still reasoning, no content produced
    // (finish=="length" only - a model that hit EOS naturally mid-think already delivered its
    // reasoning). The chat dialect surfaces a notice on this flag.
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
    if (!utf8_buf.empty() && !text_stop_matched && !out.tool_calls_emitted) {
        for (const auto& e : utf8_spans.flush(utf8_buf.size()))
            send_content_token(utf8_buf.substr(e.offset, e.length), e.token_index);
    }
    if (!pending_text.empty() && !text_stop_matched && !out.tool_calls_emitted)
        flush_text(pending_text.size());

    if (!finish)
        finish = out.tool_calls_emitted ? "tool_calls" : "length";
    else if (out.tool_calls_emitted && std::strcmp(finish, "stop") == 0)
        finish = "tool_calls";

    // The engine's "capacity" finish (four emission sites in batching_engine.cpp) becomes an `error`
    // SSE event here, since a stream can't answer 503 like non-streaming does and no dialect defines
    // "capacity" as a stop_reason (#1552, #1553).
    if (std::strcmp(finish, "capacity") == 0 && !out.error_type) {
        out.error_type = "overloaded_error";
        out.error_message =
            "the KV pool cannot hold this request; it was admitted and then dropped. "
            "Shorten the prompt or retry when the server is less loaded.";
    }
    // "internal_error" is the worker's finish for a request cancelled from
    // inside a failed step (host throw or device fault, AUDIT_arch_2026 D-1).
    // Non-streaming answers 500; the stream says it in the same `error` event.
    if (std::strcmp(finish, "internal_error") == 0 && !out.error_type) {
        out.error_type = "internal_error";
        out.error_message =
            "the engine step failed and this request was cancelled. GET /health reports "
            "whether the engine is faulted (code engine_faulted: restart the process).";
    }
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
    IMP_LOG_INFO("[%s] %s%d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms, cached=%d)", req_id.c_str(),
                 label, n_prompt_tokens, out.n_output_tokens, ms, out.ttft_ms, cached);
    state.metrics.record_completion(ctx.log_endpoint, ms, out.n_output_tokens > 0 ? out.ttft_ms : -1.0,
                                    n_prompt_tokens, out.n_output_tokens, cached);

    // Inter-token latency is observed per token inside the loop (#1577).

    // Streaming response content is not accumulated across SSE chunks, so the
    // JSONL `response` field stays null. The request body, token counts,
    // finish reason, and latency still reflect everything the client did.
        ctx.trace.model = ctx.snap.model_name;
    ctx.trace.stream = true;
    ctx.trace.ttft_ms = out.n_output_tokens > 0 ? out.ttft_ms : -1.0;
    ctx.trace.cached_tokens = cached;
    if (active_req)
        ctx.trace.set_spec(active_req->spec_drafted, active_req->spec_accepted, active_req->spec_verifies);
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, req_id, ctx.log_endpoint, ctx.log_client_ip,
                      ctx.log_raw_body, ms, n_prompt_tokens, out.n_output_tokens, out.finish, json(),
                      ctx.log_client_request_id, &ctx.trace);
}
