// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// Anthropic /v1/messages: the handle_messages endpoint (bottom of file) plus
// its streaming machinery — the AnthropicSSE event writer and
// run_anthropic_stream_ (drives the real token loop, emits native Anthropic
// SSE events).

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "tool_stream_filter.h"
#include "anthropic.h"
#include "stream_pipeline.h"
#include "reasoning_split.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"
#include "model/hf_hub.h"
#include "runtime/config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

// ===========================================================================
// Anthropic /v1/messages — native SSE streaming
// ===========================================================================
//
// Non-streaming reuses the OpenAI code path (Anthropic→OpenAI body in,
// OpenAI→Anthropic response out). Streaming drives the same real per-token
// batching-engine loop the OpenAI streaming path uses (pop_token), but emits
// native Anthropic SSE events incrementally so TTFT == real first-token
// latency rather than full-generation latency.
// ---------------------------------------------------------------------------

namespace {

// Anthropic SSE event writer. Emits "event: <name>\ndata: <json>\n\n".
struct AnthropicSSE {
    httplib::DataSink& sink;
    bool emit(const char* event_name, const json& payload) const {
        std::string buf = "event: ";
        buf += event_name;
        buf += "\ndata: ";
        buf += dump_safe(payload);
        buf += "\n\n";
        return sink.write(buf.data(), buf.size());
    }
};

// Tracks which content block (if any) is currently open in the stream so we
// can close it before opening one of a different kind. Anthropic requires a
// content_block_start before deltas and a content_block_stop after.
enum class AnthBlock { NONE, THINKING, TEXT, TOOL_USE };

}  // anonymous namespace

// Drives the real token loop and emits native Anthropic SSE events. Mirrors
// the token-handling of run_chat_stream_ (reasoning extraction, channel
// filter, tool-call tag state machine) but maps it onto Anthropic blocks:
//   reasoning -> thinking block (thinking_delta)
//   content   -> text block (text_delta)
//   tool call -> tool_use block (input_json_delta, chunked)
bool run_anthropic_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                                  const std::shared_ptr<ServerRequest>& server_req,
                                  const std::string& anth_model, const std::string& msg_id) {
    AnthropicSSE out{sink};

    const auto& stop_sequences = ctx.params.stop_sequences;
    // Derive from the FINAL stop list (server-injected stops update it late) —
    // see the matching comment in the chat streaming path.
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());
    bool enable_thinking = ctx.snap.enable_thinking;
    bool has_tools = ctx.params.has_tools;
    auto tpl_family = ctx.snap.tpl_family;
    float think_budget = ctx.params.think_budget;
    auto snap_tok = ctx.snap.tok;
    bool snap_have_template = ctx.snap.have_template;
    bool snap_is_think_model = ctx.snap.is_think_model;
    int snap_think_start_id = ctx.snap.think_start_id;
    int snap_think_end_id = ctx.snap.think_end_id;
    int snap_channel_open_id = ctx.snap.channel_open_id;
    int snap_channel_close_id = ctx.snap.channel_close_id;
    int snap_channel_newline_id = ctx.snap.channel_newline_id;
    const auto& snap_stop_token_ids = ctx.snap.stop_token_ids;
    auto active_req = server_req->request;
    auto t_start = ctx.t_start;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;

    // ---- message_start ----------------------------------------------------
    {
        json msg = {
            {"id", msg_id},
            {"type", "message"},
            {"role", "assistant"},
            {"content", json::array()},
            {"model", anth_model},
            {"stop_reason", nullptr},
            {"stop_sequence", nullptr},
            {"usage", {{"input_tokens", n_prompt_tokens}, {"output_tokens", 0}}},
        };
        if (!out.emit("message_start", json{{"type", "message_start"}, {"message", std::move(msg)}}))
            return false;
    }

    // ---- ping (initial keepalive) -----------------------------------------
    // Anthropic streams emit periodic `ping` events; sending one immediately
    // signals liveness before the first token (TTFT can be >1s under load /
    // long prefills), and the loop below re-pings during idle gaps so clients
    // and intermediary proxies don't time out the connection.
    if (!out.emit("ping", json{{"type", "ping"}}))
        return false;
    auto last_ping = std::chrono::steady_clock::now();

    int block_index = -1;
    AnthBlock open_block = AnthBlock::NONE;

    auto stop_block = [&]() -> bool {
        if (open_block == AnthBlock::NONE)
            return true;
        bool ok = out.emit("content_block_stop",
                           json{{"type", "content_block_stop"}, {"index", block_index}});
        open_block = AnthBlock::NONE;
        return ok;
    };
    auto start_text_block = [&]() -> bool {
        if (open_block == AnthBlock::TEXT)
            return true;
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::TEXT;
        return out.emit("content_block_start",
                        json{{"type", "content_block_start"},
                             {"index", block_index},
                             {"content_block", {{"type", "text"}, {"text", ""}}}});
    };
    auto start_thinking_block = [&]() -> bool {
        if (open_block == AnthBlock::THINKING)
            return true;
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::THINKING;
        return out.emit("content_block_start",
                        json{{"type", "content_block_start"},
                             {"index", block_index},
                             {"content_block", {{"type", "thinking"}, {"thinking", ""}}}});
    };
    auto emit_text = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_text_block())
            return false;
        return out.emit("content_block_delta",
                        json{{"type", "content_block_delta"},
                             {"index", block_index},
                             {"delta", {{"type", "text_delta"}, {"text", text}}}});
    };
    auto emit_thinking = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_thinking_block())
            return false;
        return out.emit("content_block_delta",
                        json{{"type", "content_block_delta"},
                             {"index", block_index},
                             {"delta", {{"type", "thinking_delta"}, {"thinking", text}}}});
    };

    // gpt-oss Harmony streaming filter (analysis/commentary -> thinking block,
    // final -> text block); markers arrive as atomic special-token pieces. See
    // the matching filter in run_chat_stream_ (#760).
    const bool harmony = (ctx.snap.tpl_family == imp::ChatTemplateFamily::HARMONY);
    std::string hm_channel, hm_name, hm_buf;
    bool hm_in_msg = false, hm_reading_name = false;
    auto hm_flush = [&](bool force) -> bool {
        size_t complete = force ? hm_buf.size() : utf8_complete_len(hm_buf);
        if (complete == 0)
            return true;
        std::string chunk = hm_buf.substr(0, complete);
        hm_buf.erase(0, complete);
        if (hm_channel == "analysis" || hm_channel == "commentary")
            return emit_thinking(chunk);
        return emit_text(chunk);
    };
    // Open a tool_use block (content_block_start). Arguments follow as
    // input_json_delta events — incrementally for streamed (JSON-layout)
    // calls, chunked-after-the-fact for buffered ones.
    auto open_tool_use_block = [&](const ParsedToolCall& tc) -> bool {
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::TOOL_USE;
        namespace anth = imp_server::anthropic;
        return out.emit("content_block_start",
                        json{{"type", "content_block_start"},
                             {"index", block_index},
                             {"content_block",
                              {{"type", "tool_use"},
                               {"id", anth::tool_call_id_to_anthropic(tc.id)},
                               {"name", tc.name},
                               {"input", json::object()}}}});
    };
    auto emit_tool_args_delta = [&](const std::string& partial) -> bool {
        return out.emit("content_block_delta",
                        json{{"type", "content_block_delta"},
                             {"index", block_index},
                             {"delta",
                              {{"type", "input_json_delta"}, {"partial_json", partial}}}});
    };
    // Buffered call path: open block + chunked arg deltas + close.
    auto emit_tool_use = [&](const ParsedToolCall& tc) -> bool {
        if (!open_tool_use_block(tc))
            return false;
        const std::string& args = tc.arguments;
        constexpr size_t kChunk = 48;
        for (size_t off = 0; off < args.size(); off += kChunk) {
            size_t n = std::min(kChunk, args.size() - off);
            if (!emit_tool_args_delta(args.substr(off, n)))
                return false;
        }
        return stop_block();
    };

    int n_output_tokens = 0;
    const char* finish = nullptr;
    double ttft_ms = 0.0;

    std::string utf8_buf;        // confirmed-UTF8 content buffer
    std::string pending_text;    // stop-sequence holdback
    bool text_stop_matched = false;

    // Streaming tool-call demux — same shared state machine as
    // run_chat_stream_ (tool_stream_filter.h): ChatML/Llama3/Gemma-4 markers,
    // JSON + Qwen3.6-XML + Gemma body parsing, raw-restore on failure.
    imp::server::StreamToolCallFilter tool_filter(tpl_family);
    std::vector<ParsedToolCall> stream_tool_calls;
    bool tool_calls_emitted = false;
    // parallel_tool_calls=false: a second streamed call was opened by the
    // filter but is being suppressed (skip its deltas/END too).
    bool stream_call_suppressed = false;

    // Reasoning extraction (DeepSeek <think>). enable_thinking also covers
    // text-level thinkers (Nemotron) — see the chat streaming path.
    // Shared reasoning/content demux (reasoning_split.h) — identical to the
    // OpenAI streaming path; emit_thinking is the Anthropic reasoning sink.
    const bool use_reasoning = (state.default_args.reasoning_format == "deepseek" &&
                                (snap_is_think_model || enable_thinking));
    const bool think_active = use_reasoning || enable_thinking;
    imp::server::ThinkPhase think_start_phase;
    if (enable_thinking)
        think_start_phase = imp::server::ThinkPhase::REASONING;
    else if (use_reasoning && think_budget > 0.0f)
        think_start_phase = imp::server::ThinkPhase::SCAN;
    else
        think_start_phase = imp::server::ThinkPhase::CONTENT;
    imp::server::StreamReasoningSplitter think_split(think_start_phase, snap_think_start_id,
                                                     snap_think_end_id);
    bool channel_header_active = false;

    auto flush_text = [&](size_t up_to) -> bool {
        if (up_to == 0)
            return true;
        bool ok = emit_text(pending_text.substr(0, up_to));
        pending_text.erase(0, up_to);
        return ok;
    };

    // Flush held content buffers before a tool_use block (or directly emitted
    // text) so block order matches the model's output order. A complete stop
    // match cannot be pending here — the normal emission path checks after
    // every append with the same holdback decision.
    auto flush_buffered_content = [&]() -> bool {
        if (stop_sequences.empty()) {
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                if (!emit_text(utf8_buf.substr(0, complete)))
                    return false;
                utf8_buf.erase(0, complete);
            }
        } else if (!pending_text.empty()) {
            auto d = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(d.flush_len))
                return false;
        }
        return true;
    };

    auto request_start = std::chrono::steady_clock::now();
    for (;;) {
        // #755: when a thinking model exhausts its token budget the final
        // (is_last) token lands inside the REASONING phase, which `continue`s
        // and skips the trailing `if (finish) break`. The loop would then spin
        // on pop_token forever — message_delta/message_stop never emitted, so
        // the Anthropic client (SDK or curl -N) hangs indefinitely. Breaking
        // here guarantees the terminal events are always sent.
        if (finish)
            break;

        if (!sink.is_writable()) {
            server_req->cancel();
            state.metrics.requests_cancelled++;
            finish = "cancelled";
            break;
        }
        if (state.request_timeout > 0) {
            auto elapsed = std::chrono::steady_clock::now() - request_start;
            if (elapsed > std::chrono::seconds(state.request_timeout)) {
                server_req->cancel();
                finish = "length";
                break;
            }
        }

        TokenEvent evt{};
        if (!server_req->pop_token(evt)) {
            // No token ready yet (prefill / generation gap). Re-ping at most
            // every ~10s so idle streams stay alive without spamming.
            auto now = std::chrono::steady_clock::now();
            if (now - last_ping > std::chrono::seconds(10)) {
                last_ping = now;
                if (!out.emit("ping", json{{"type", "ping"}}))
                    break;
            }
            continue;
        }

        if (evt.token_id < 0) {
            finish = evt.finish_reason ? evt.finish_reason : "stop";
            break;
        }
        int32_t token = evt.token_id;

        if (!evt.is_last) {
            bool is_structural_stop = (token == snap_tok->eos_id());
            if (!is_structural_stop && snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids)
                    if (token == stop_id) {
                        is_structural_stop = true;
                        break;
                    }
            }
            if (is_structural_stop)
                continue;
        }
        if (evt.is_last) {
            if (token == snap_tok->eos_id()) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            bool is_stop = false;
            if (snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids)
                    if (token == stop_id) {
                        is_stop = true;
                        break;
                    }
            }
            if (is_stop) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            finish = evt.finish_reason ? evt.finish_reason : "length";
        }

        n_output_tokens++;
        if (n_output_tokens == 1)
            ttft_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t_start)
                          .count();
        std::string piece = snap_tok->decode_token(token);

        // gpt-oss Harmony channel routing (analysis/commentary -> thinking,
        // final -> text). Markers arrive as atomic special-token pieces.
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
                } else {
                    hm_in_msg = false;
                    hm_reading_name = false;
                    hm_channel.clear();
                }
                continue;
            }
            if (hm_reading_name) {
                hm_name += piece;
                continue;
            }
            if (!hm_in_msg)
                continue;
            hm_buf += piece;
            if (!hm_flush(/*force=*/false))
                return false;
            continue;
        }

        // Gemma-4 channel filter.
        if (snap_channel_open_id >= 0) {
            if (channel_header_active) {
                if (token == snap_channel_newline_id || (!piece.empty() && piece.back() == '\n'))
                    channel_header_active = false;
                continue;
            }
            if (token == snap_channel_open_id) {
                channel_header_active = true;
                continue;
            }
            if (token == snap_channel_close_id)
                continue;
        }

        // Reasoning extraction.
        // Reasoning/content demux (reasoning_split.h) — shared with the OpenAI
        // streaming path. Reasoning goes to the Anthropic thinking sink.
        if (think_active) {
            auto rs = think_split.feed(std::move(piece), token);
            if (!rs.reasoning.empty() && !emit_thinking(rs.reasoning))
                return false;
            if (rs.content.empty())
                continue;
            piece = std::move(rs.content);
        }

        // Streaming tool-call demux (see the matching block in
        // run_chat_stream_): completed calls become tool_use blocks; content
        // before/between calls is emitted directly; trailing content after
        // the last call falls through to the normal emission path below.
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
                        if (!emit_text(seg.text))
                            return false;
                    }
                    continue;
                }
                // parallel_tool_calls=false: stream at most one tool call.
                // (For a streamed call the gate fires at CALL_BEGIN, so the
                // later deltas/END of a suppressed call are skipped too.)
                if (!ctx.params.parallel_tool_calls &&
                    ((seg.kind == SegKind::CALL && !stream_tool_calls.empty()) ||
                     (seg.kind == SegKind::CALL_BEGIN && !stream_tool_calls.empty()) ||
                     (seg.kind != SegKind::CALL && stream_call_suppressed))) {
                    if (seg.kind == SegKind::CALL_BEGIN)
                        stream_call_suppressed = true;
                    if (seg.kind == SegKind::CALL_END)
                        stream_call_suppressed = false;
                    continue;
                }
                if (seg.kind == SegKind::CALL_BEGIN) {
                    // Streamed call: open the tool_use block now; the
                    // argument bytes follow as input_json_delta events while
                    // the model is still generating them.
                    ParsedToolCall tc = std::move(seg.call);
                    tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                    if (!flush_buffered_content())
                        return false;
                    if (!open_tool_use_block(tc))
                        return false;
                    stream_tool_calls.push_back(std::move(tc));
                    tool_calls_emitted = true;
                    continue;
                }
                if (seg.kind == SegKind::CALL_ARGS_DELTA) {
                    if (!emit_tool_args_delta(seg.text))
                        return false;
                    continue;
                }
                if (seg.kind == SegKind::CALL_END) {
                    if (!stream_tool_calls.empty()) {
                        stream_tool_calls.back().arguments = std::move(seg.call.arguments);
                        validate_tool_call(stream_tool_calls.back(), ctx.params.tools);
                        if (!stream_tool_calls.back().valid) {
                            fprintf(stderr, "[%s] tool-call arg validation failed: %s: %s\n",
                                    msg_id.c_str(), stream_tool_calls.back().name.c_str(),
                                    stream_tool_calls.back().error.c_str());
                        }
                    }
                    if (!stop_block())
                        return false;
                    continue;
                }
                // SegKind::CALL — buffered call (non-JSON layouts).
                ParsedToolCall tc = std::move(seg.call);
                tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                validate_tool_call(tc, ctx.params.tools);
                if (!tc.valid) {
                    fprintf(stderr, "[%s] tool-call arg validation failed: %s: %s\n",
                            msg_id.c_str(), tc.name.c_str(), tc.error.c_str());
                }
                if (!flush_buffered_content())
                    return false;
                if (!emit_tool_use(tc))
                    return false;
                stream_tool_calls.push_back(std::move(tc));
                tool_calls_emitted = true;
            }
            if (piece.empty())
                continue;
        }

        // Normal content emission.
        if (stop_sequences.empty()) {
            utf8_buf += piece;
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                if (!emit_text(utf8_buf.substr(0, complete)))
                    return false;
                utf8_buf.erase(0, complete);
            }
        } else {
            pending_text += piece;
            auto d = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(d.flush_len))
                return false;
            if (d.complete_match) {
                text_stop_matched = true;
                finish = "stop";
                break;
            }
        }

        if (finish)
            break;
    }

    // Flush trailing buffers.
    // Harmony: flush the final channel's tail (ends at EOS/<|return|> with no
    // trailing <|end|>); the other buffers below stay empty for harmony.
    if (harmony && !hm_buf.empty())
        hm_flush(/*force=*/true);

    // Flush the splitter's held tail: buffered reasoning -> thinking sink,
    // held/undecided content -> the content flush below.
    if (think_active) {
        auto rs = think_split.finish();
        if (!rs.reasoning.empty())
            emit_thinking(rs.reasoning);
        if (!rs.content.empty())
            utf8_buf += rs.content;
    }
    if (has_tools && tool_filter.mid_tool() && !tool_calls_emitted) {
        std::string leftover = tool_filter.finish();
        if (!leftover.empty())
            utf8_buf += leftover;
    }
    if (!utf8_buf.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(utf8_buf);
    if (!pending_text.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(pending_text);

    // Close any block still open.
    stop_block();

    if (!finish)
        finish = tool_calls_emitted ? "tool_calls" : "length";
    else if (tool_calls_emitted && strcmp(finish, "stop") == 0)
        finish = "tool_calls";

    // Map finish_reason -> Anthropic stop_reason.
    std::string stop_reason;
    if (strcmp(finish, "stop") == 0)
        stop_reason = "end_turn";
    else if (strcmp(finish, "length") == 0)
        stop_reason = "max_tokens";
    else if (strcmp(finish, "tool_calls") == 0)
        stop_reason = "tool_use";
    else if (strcmp(finish, "cancelled") == 0)
        stop_reason = "end_turn";
    else
        stop_reason = finish;

    // ---- message_delta + message_stop ------------------------------------
    // Cache accounting is only known after prefill ran, so it rides on the
    // final usage update instead of message_start.
    json delta_usage = {{"output_tokens", n_output_tokens}};
    {
        int cached_now = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
        int creation = cache_creation_tokens_(active_req, n_prompt_tokens);
        if (cached_now > 0 || creation > 0) {
            delta_usage["input_tokens"] = n_prompt_tokens - cached_now;
            delta_usage["cache_read_input_tokens"] = cached_now;
            delta_usage["cache_creation_input_tokens"] = creation;
        }
    }
    out.emit("message_delta",
             json{{"type", "message_delta"},
                  {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
                  {"usage", std::move(delta_usage)}});
    out.emit("message_stop", json{{"type", "message_stop"}});
    sink.done();

    // Metrics + log.
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += n_prompt_tokens;
    state.metrics.tokens_completion_total += n_output_tokens;
    state.metrics.tokens_cached_total += cached;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.last_ttft_ms = static_cast<int64_t>(ttft_ms);
    state.metrics.request_duration.observe(ms / 1000.0);
    if (n_output_tokens > 0)
        state.metrics.ttft.observe(ttft_ms / 1000.0);
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, msg_id, ctx.log_endpoint, ctx.log_client_ip,
                      ctx.log_raw_body, ms, n_prompt_tokens, n_output_tokens, finish, json());
    fprintf(stderr, "[%s] messages stream: %d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms)\n",
            msg_id.c_str(), n_prompt_tokens, n_output_tokens, ms, ttft_ms);
    return true;
}

// ===========================================================================
// Anthropic /v1/messages endpoint (moved here from handlers_chat.cpp to keep
// that TU under the file-size gate; co-located with run_anthropic_stream_).
// Non-streaming reuses the OpenAI path via a shim; streaming drives the real
// per-token loop above.
// ===========================================================================
static void handle_messages_impl(const httplib::Request& req, httplib::Response& res, ServerState& state);

void handle_messages(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // Any exception escaping the impl — notably from the inner
    // handle_chat_completions shim on the non-streaming path — must return the
    // Anthropic error envelope ({"type":"error",...}), not the OpenAI-shaped one
    // the global exception handler emits, which strict Anthropic SDK clients
    // fail to parse (#891). res is untouched by the shim (it writes a separate
    // shim_res) when the throw propagates, so it is safe to rewrite here.
    try {
        handle_messages_impl(req, res, state);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"type", "error"}, {"error", {{"type", "server_error"}, {"message", e.what()}}}};
        res.set_content(dump_safe(err), "application/json");
    } catch (...) {
        res.status = 500;
        json err = {{"type", "error"}, {"error", {{"type", "server_error"}, {"message", "internal error"}}}};
        res.set_content(dump_safe(err), "application/json");
    }
}

static void handle_messages_impl(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    namespace anth = imp_server::anthropic;

    // Capture original Anthropic request data for opt-in JSONL logging.
    const auto t_log_start = std::chrono::system_clock::now();
    const std::string log_endpoint = req.path;
    std::string log_client_ip = req.get_header_value("X-Forwarded-For");
    if (log_client_ip.empty())
        log_client_ip = req.remote_addr;
    const std::string log_raw_body = req.body;

    json anth_body;
    try {
        anth_body = json::parse(req.body);
    } catch (const std::exception& e) {
        res.status = 400;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "invalid_request_error"},
                      {"message", std::string("Invalid JSON: ") + e.what()}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    if (!anth_body.is_object()) {
        res.status = 400;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "invalid_request_error"}, {"message", "Request body must be a JSON object"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Anthropic requires max_tokens — if it's missing, supply a sane default
    // matching the server's chat-completions default (handled downstream).
    std::string anth_model = anth_body.value("model", "");
    const bool want_stream = anth_body.value("stream", false);

    // Transform -> OpenAI body.
    json oai_body;
    try {
        oai_body = anth::anthropic_to_openai_body(anth_body);
    } catch (const std::exception& e) {
        res.status = 400;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "invalid_request_error"},
                      {"message", std::string("Failed to transform Anthropic body: ") + e.what()}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // ---- Real streaming path -------------------------------------------
    // For stream=true we drive the same per-token batching-engine loop the
    // OpenAI streaming path uses and emit native Anthropic SSE events as
    // tokens arrive — TTFT is real first-token latency, not full-gen latency.
    if (want_stream) {
        // Build the chat request context from the transformed OpenAI body.
        httplib::Request shim_req = req;
        json shim_body = oai_body;
        shim_body["stream"] = true;
        shim_req.body = dump_safe(shim_body);
        shim_req.headers.erase("Content-Length");
        shim_req.headers.erase("content-length");

        ChatRequestContext ctx;
        g_in_anthropic_shim = true;  // suppress inner request-log (we log here)
        bool ok = parse_chat_request_params(shim_req, res, state, ctx) &&
                  snapshot_state_and_tokenize_(res, state, ctx);
        g_in_anthropic_shim = false;
        if (!ok) {
            // parse/snapshot set an OpenAI-shaped error on res; re-wrap as
            // an Anthropic error envelope.
            json parsed;
            try {
                parsed = json::parse(res.body);
            } catch (...) {
                parsed = {{"error", {{"message", res.body}, {"type", "invalid_request_error"}}}};
            }
            json out = {{"type", "error"},
                        {"error", parsed.value("error",
                                               json{{"type", "invalid_request_error"}, {"message", "bad request"}})}};
            res.set_content(dump_safe(out), "application/json");
            return;
        }

        // Restore Anthropic logging context (parse_chat_request_params set
        // these from the shim request; we log the outer Anthropic request).
        ctx.log_skip = false;
        ctx.log_endpoint = log_endpoint;
        ctx.log_client_ip = log_client_ip;
        ctx.log_raw_body = log_raw_body;
        ctx.t_log_start = t_log_start;

        // Vision is per-request now (req->image) and streams like any request.
        auto imp_req = std::make_shared<imp::Request>();
        imp_req->image = ctx.snap.vision_image;  // per-request vision (null for text)
        imp_req->input_tokens = ctx.snap.tokens;
        imp_req->max_tokens = ctx.params.max_tokens;
        imp_req->temperature = ctx.params.temperature;
        imp_req->top_p = ctx.params.top_p;
        imp_req->top_k = ctx.params.top_k;
        imp_req->seed = ctx.params.seed;
        imp_req->pin_kv_prefix = ctx.params.cache_prompt;
        imp_req->spec_ngram_override = ctx.params.spec_ngram_override;
        // This is the streaming /v1/messages path — stay on per-step decode so
        // SSE is real per-token rather than one burst at generation end (#754).
        imp_req->stream = true;
        imp_req->min_p = ctx.params.min_p;
        imp_req->typical_p = ctx.params.typical_p;
        imp_req->repetition_penalty = ctx.params.repetition_penalty;
        imp_req->frequency_penalty = ctx.params.frequency_penalty;
        imp_req->presence_penalty = ctx.params.presence_penalty;
        imp_req->repeat_last_n = ctx.params.repeat_last_n;
        imp_req->dry_multiplier = ctx.params.dry_multiplier;
        imp_req->dry_base = ctx.params.dry_base;
        imp_req->dry_allowed_length = ctx.params.dry_allowed_length;
        imp_req->dry_penalty_last_n = ctx.params.dry_penalty_last_n;
        imp_req->mirostat = ctx.params.mirostat;
        imp_req->mirostat_tau = ctx.params.mirostat_tau;
        imp_req->mirostat_eta = ctx.params.mirostat_eta;
        imp_req->logprobs = ctx.params.req_logprobs;
        imp_req->top_logprobs = ctx.params.top_logprobs;
        imp_req->json_mode = ctx.params.json_mode;
        imp_req->json_schema = ctx.params.json_schema_str;
        imp_req->has_tools = ctx.params.has_tools;
        imp_req->tpl_family = ctx.snap.tpl_family;
        imp_req->logit_bias = ctx.params.logit_bias;
        imp_req->think_budget = ctx.params.think_budget;
        imp_req->status = imp::RequestStatus::PENDING;

        auto server_req = std::make_shared<ServerRequest>();
        server_req->request = imp_req;
        {
            std::lock_guard<std::timed_mutex> lock(state.mtx);
            if (!state.batching || !state.batching->is_running()) {
                res.status = 503;
                json err = {{"type", "error"},
                            {"error",
                             {{"type", "server_error"}, {"message", "Inference engine not ready. Please retry."}}}};
                res.set_content(dump_safe(err), "application/json");
                return;
            }
            state.batching->submit(server_req);
        }

        std::string msg_id = anth::make_message_id(static_cast<uint64_t>(state.next_id.fetch_add(1)));
        ctx.t_start = std::chrono::high_resolution_clock::now();

        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [stream_ctx = std::move(ctx), &state, server_req, anth_model, msg_id](
                size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
                return run_anthropic_stream_(sink, stream_ctx, state, server_req, anth_model, msg_id);
            });
        return;
    }

    // ---- Non-streaming path: reuse the OpenAI handler via a shim --------
    // httplib::Request is a plain struct, safe to copy. Force stream=false on
    // the inner OpenAI call — we re-serialize the response as Anthropic JSON.
    httplib::Request shim_req = req;
    json shim_body = oai_body;
    shim_body["stream"] = false;
    shim_req.body = dump_safe(shim_body);
    shim_req.headers.erase("Content-Length");
    shim_req.headers.erase("content-length");

    httplib::Response shim_res;
    g_in_anthropic_shim = true;
    handle_chat_completions(shim_req, shim_res, state);
    g_in_anthropic_shim = false;

    // Propagate error envelopes (transform them to Anthropic error shape).
    // httplib::Response defaults status to -1 and auto-promotes to 200 only
    // at send time; any other non-200 code set by handle_chat_completions is
    // a real error we should forward.
    const bool is_error = shim_res.status >= 400;
    if (is_error) {
        res.status = shim_res.status;
        json parsed;
        try {
            parsed = json::parse(shim_res.body);
        } catch (...) {
            parsed = {{"error", {{"message", shim_res.body}, {"type", "server_error"}}}};
        }
        json out = {{"type", "error"},
                    {"error", parsed.value("error", json{{"type", "server_error"}, {"message", "unknown"}})}};
        res.set_content(dump_safe(out), "application/json");
        return;
    }

    json oai_response;
    try {
        oai_response = json::parse(shim_res.body);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "server_error"},
                      {"message", std::string("Upstream returned non-JSON: ") + e.what()}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    json anth_response = anth::openai_to_anthropic_response(oai_response, anth_model);

    // JSONL log — built from Anthropic shapes so /v1/messages clients see
    // exactly what they sent and what they got back.
    {
        auto t_end = std::chrono::system_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_log_start).count();
        int prompt_t = oai_response.value("usage", json::object()).value("prompt_tokens", 0);
        int completion_t = oai_response.value("usage", json::object()).value("completion_tokens", 0);
        std::string stop_reason = anth_response.value("stop_reason", "");
        std::string req_id = anth_response.value("id", make_completion_id(state));
        log_request_jsonl(state, /*skip=*/false, t_log_start, req_id, log_endpoint, log_client_ip,
                          log_raw_body, ms, prompt_t, completion_t,
                          stop_reason.empty() ? nullptr : stop_reason.c_str(), anth_response);
    }

    // Non-streaming requests are fully assembled above (the want_stream path
    // returned earlier with a native incremental SSE stream).
    res.status = 200;
    res.set_content(dump_safe(anth_response), "application/json");
}
