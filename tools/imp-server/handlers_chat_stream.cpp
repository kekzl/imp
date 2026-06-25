// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// Streaming chat-completion machinery: the SSE chunked-provider setup
// (stream_chat_response_) and the per-token streaming loop body
// (run_chat_stream_). Used by handlers_chat.cpp and handlers_messages.cpp.

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
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

// Set up SSE chunked content provider for streaming chat completion.
// Captures state and ctx by reference for the chunked-provider lambda. ctx
// must outlive the SSE response (httplib invokes the chunked provider after
// this function returns; ctx is a stack-local in handle_chat_completions
// which keeps the request frame alive until the response is fully sent).
bool run_chat_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                             const std::shared_ptr<ServerRequest>& server_req);

void stream_chat_response_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx,
                                  const std::shared_ptr<ServerRequest>& server_req) {
    // SSE streaming response
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    ctx.comp_id = ctx.req_id;
    ctx.created = unix_timestamp();

    res.set_chunked_content_provider(
        "text/event-stream",
        [stream_ctx = ctx, &state, server_req](size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
            return run_chat_stream_(sink, stream_ctx, state, server_req);
        });
}

// Streaming chat response loop body. Extracted from the
// res.set_chunked_content_provider() lambda in stream_chat_response_ so
// the 760-LOC body is no longer a god-function nested four levels deep.
// httplib calls the lambda repeatedly until it returns false; the lambda
// just dispatches to this function. ctx is captured by value into the
// lambda (so it survives stream_chat_response_'s return); state is
// captured by reference (lives in the long-lived ServerState).
bool run_chat_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                             const std::shared_ptr<ServerRequest>& server_req) {
    // Local aliases so the body reads unchanged from its previous
    // capture-list-based form. These were 30+ individual lambda captures
    // before this refactor.
    const std::string& comp_id          = ctx.comp_id;
    int64_t            created          = ctx.created;
    int                n_prompt_tokens  = ctx.snap.n_prompt_tokens;
    auto               t_start          = ctx.t_start;
    const auto&        stop_sequences   = ctx.params.stop_sequences;
    // Derive max_stop_len from the FINAL stop list, not ctx.params.max_stop_len:
    // the snapshot phase may inject server-side stops ("\nHuman" turn guard for
    // think models) AFTER request parsing computed max_stop_len. A stale 0 made
    // the partial-match holdback `size - max_stop_len + 1` flush one byte PAST
    // pending_text's end — emitting the std::string NUL terminator into every
    // SSE content delta ("4\0") and disabling cross-token stop matching.
    size_t             max_stop_len     = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());
    int                req_logprobs     = ctx.params.req_logprobs;
    bool               include_usage    = ctx.params.include_usage;
    bool               enable_thinking  = ctx.snap.enable_thinking;
    bool               has_tools        = ctx.params.has_tools;
    auto               tpl_family       = ctx.snap.tpl_family;
    float              think_budget     = ctx.params.think_budget;
    auto               snap_tok         = ctx.snap.tok;
    bool               snap_have_template = ctx.snap.have_template;
    const std::string& snap_model_name  = ctx.snap.model_name;
    bool               snap_is_think_model      = ctx.snap.is_think_model;
    int                snap_think_start_id      = ctx.snap.think_start_id;
    int                snap_think_end_id        = ctx.snap.think_end_id;
    int                snap_channel_open_id     = ctx.snap.channel_open_id;
    int                snap_channel_close_id    = ctx.snap.channel_close_id;
    int                snap_channel_newline_id  = ctx.snap.channel_newline_id;
    const auto&        snap_stop_token_ids      = ctx.snap.stop_token_ids;
    bool               log_skip         = ctx.log_skip;
    auto               t_log_start      = ctx.t_log_start;
    const std::string& log_endpoint     = ctx.log_endpoint;
    const std::string& log_client_ip    = ctx.log_client_ip;
    const std::string& log_raw_body     = ctx.log_raw_body;

    // Active request ref for logprobs access
    auto active_req = server_req->request;

    // Pre-build SSE envelope templates for fast content/reasoning emission
    SSEChunkWriter sse_writer(comp_id, created, snap_model_name);

    // Send initial chunk with role
    json role_delta = {{"role", "assistant"}};
    std::string chunk = sse_chunk(comp_id, created, snap_model_name, role_delta, nullptr);
    sink.write(chunk.data(), chunk.size());

    int n_output_tokens = 0;
    const char* finish = nullptr;
    double ttft_ms = 0.0;  // Time to first token

    // Buffer for incomplete UTF-8 sequences across token boundaries
    std::string utf8_buf;

    // Buffered output for stop sequence matching in streaming mode.
    // We hold back text until we're sure it doesn't contain a stop match.
    std::string pending_text;
    bool text_stop_matched = false;

    // Tool call detection state machine for streaming
    enum class ToolPhase { CONTENT, TAG_SCANNING, TOOL_CALL_BODY };
    ToolPhase tool_phase = ToolPhase::CONTENT;
    std::string tool_tag_buf;    // buffer for partial tag match
    std::string tool_body_buf;   // buffer for tool call body
    std::string tool_close_tag;  // expected closing tag
    std::string tool_fn_name;    // Llama3: extracted function name from open tag
    std::vector<ParsedToolCall> stream_tool_calls;
    bool tool_calls_emitted = false;
    // The full accumulated output (only used when has_tools, for fallback)
    std::string full_output;

    // Reasoning content extraction (DeepSeek format). enable_thinking also
    // covers text-level thinkers (Nemotron: template-injected "<think>" as
    // plain text, no special token — is_think_model is false but the output
    // starts mid-reasoning and exits via the literal "</think>").
    // Reasoning/content demux is shared with the Anthropic path via the pure
    // StreamReasoningSplitter (reasoning_split.h) — single source of truth for
    // the streaming first-vs-last </think> handling
    // (BUGREPORT-qwen36-reasoning-leaks-into-content).
    const bool use_reasoning = (state.default_args.reasoning_format == "deepseek" &&
                                (snap_is_think_model || enable_thinking));
    const bool think_active = use_reasoning || enable_thinking;
    imp::server::ThinkPhase think_start_phase;
    if (enable_thinking)
        think_start_phase = imp::server::ThinkPhase::REASONING;  // <think> in prefill
    else if (use_reasoning && think_budget > 0.0f)
        think_start_phase = imp::server::ThinkPhase::SCAN;  // model decides
    else
        think_start_phase = imp::server::ThinkPhase::CONTENT;  // no extraction
    imp::server::StreamReasoningSplitter think_split(think_start_phase, snap_think_start_id,
                                                     snap_think_end_id);
    int n_reasoning_tokens = 0;

    // Gemma-4 channel filter state: when we see <|channel> or <channel|>,
    // skip tokens until the next newline (the channel header).
    bool channel_header_active = false;

    // Helper: emit reasoning_content SSE chunk
    auto emit_reasoning = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        return sse_writer.write_reasoning(text, sink);
    };

    // gpt-oss Harmony streaming filter. The model emits
    //   <|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…
    // Route analysis/commentary channels to reasoning_content and the final
    // channel to content, stripping the control markers (which arrive as atomic
    // special-token pieces) and the <|start|>role plumbing. hm_buf holds the
    // current channel's bytes so a token that splits a multibyte char is not
    // emitted mid-codepoint (#760).
    const bool harmony = (tpl_family == imp::ChatTemplateFamily::HARMONY);
    // Harmony reasoning is its own mechanism (not the deepseek <think> path), so
    // it's gated on reasoning_format alone — emit reasoning_content unless the
    // caller explicitly asked for none.
    const bool hm_reasoning_on = (state.default_args.reasoning_format != "none");
    std::string hm_channel, hm_name, hm_buf;
    bool hm_in_msg = false, hm_reading_name = false;
    auto hm_flush = [&](bool force) -> bool {
        size_t complete = force ? hm_buf.size() : utf8_complete_len(hm_buf);
        if (complete == 0)
            return true;
        std::string chunk = hm_buf.substr(0, complete);
        hm_buf.erase(0, complete);
        if (hm_channel == "analysis" || hm_channel == "commentary")
            return hm_reasoning_on ? emit_reasoning(chunk) : true;
        return sse_writer.write_content(chunk.data(), chunk.size(), sink);
    };

    // Helper: flush confirmed text up to a byte position
    auto flush_text = [&](size_t up_to) {
        up_to = std::min(up_to, pending_text.size());  // never read past the buffer
        if (up_to == 0)
            return true;
        bool ok = sse_writer.write_content(pending_text.data(), up_to, sink);
        pending_text.erase(0, up_to);
        return ok;
    };

    auto request_start = std::chrono::steady_clock::now();
    for (;;) {
        // Terminate as soon as a finish reason has been recorded. The is_last
        // token sets `finish` and then falls through to the per-token
        // post-processing below, where a think/reasoning/channel `continue`
        // can skip the trailing `if (finish) break`. Re-checking here means the
        // stream always ends (and the terminal SSE frame is emitted) even when
        // the final token is swallowed by one of those paths (#755/#757).
        if (finish)
            break;

        // Check client disconnect
        if (!sink.is_writable()) {
            server_req->cancel();
            state.metrics.requests_cancelled++;
            finish = "cancelled";
            break;
        }

        // Check request timeout
        if (state.request_timeout > 0) {
            auto elapsed = std::chrono::steady_clock::now() - request_start;
            if (elapsed > std::chrono::seconds(state.request_timeout)) {
                server_req->cancel();
                finish = "length";
                break;
            }
        }

        // Read next token from the batching engine (with timeout)
        TokenEvent evt{};
        if (!server_req->pop_token(evt)) {
            continue;  // timeout — loop back to check disconnect/timeout
        }

        if (evt.token_id < 0) {
            // Finish event with no token
            finish = evt.finish_reason ? evt.finish_reason : "stop";
            break;
        }

        int32_t token = evt.token_id;

        // Silently drop structural stop tokens that slipped through.
        // The engine's think-block implicit-close (Engine::should_stop)
        // passes ONE EOS-like token through to recover from empty
        // thinking. That token must not appear as user-visible content
        // (would render as "<|im_end|>" / "<|endoftext|>" in chat).
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

        // Check stop conditions (EOS/stop tokens already detected by engine)
        if (evt.is_last) {
            // The engine marked this as the last token.
            // Don't emit EOS/stop tokens — they're structural, not content.
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
            // Not a stop token — emit it, then finish after this iteration
            finish = evt.finish_reason ? evt.finish_reason : "length";
        }

        n_output_tokens++;
        if (n_output_tokens == 1) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
        }
        std::string piece = snap_tok->decode_token(token);

        // gpt-oss Harmony channel routing (analysis/commentary -> reasoning,
        // final -> content). Markers arrive as atomic special-token pieces.
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

        // Gemma-4 channel filter: strip "<|channel>NAME\n" structural
        // headers from the content stream. `<channel|>` is the
        // channel-switch marker — strip the token but do NOT enter
        // the scan-until-newline mode, because Q5_K_M sometimes
        // emits the final answer directly after it with no newline
        // (observed: "<|channel>thought\n<channel|>5 + 3 = 8").
        if (snap_channel_open_id >= 0) {
            if (channel_header_active) {
                if (token == snap_channel_newline_id ||
                    (!piece.empty() && piece.back() == '\n')) {
                    channel_header_active = false;
                }
                continue;
            }
            if (token == snap_channel_open_id) {
                channel_header_active = true;
                continue;
            }
            if (token == snap_channel_close_id) {
                // Drop just the marker; the next token is body.
                continue;
            }
        }

        // Reasoning/content demux (DeepSeek <think>) — shared, fixed state
        // machine in reasoning_split.h. Routes reasoning to reasoning_content
        // and hands back the user-visible content for this step (empty when the
        // whole piece was reasoning or is being held for boundary detection).
        if (think_active) {
            auto rs = think_split.feed(std::move(piece), token);
            n_reasoning_tokens += rs.reasoning_tokens;
            if (!rs.reasoning.empty() && !emit_reasoning(rs.reasoning))
                return false;
            if (rs.content.empty())
                continue;
            piece = std::move(rs.content);
        }

        // CONTENT phase — with tool call tag detection
        if (has_tools)
            full_output += piece;

        // Tool call state machine (only active when tools are present)
        if (has_tools && tool_phase == ToolPhase::TOOL_CALL_BODY) {
            tool_body_buf += piece;
            // Check for close tag
            auto close_pos = tool_body_buf.find(tool_close_tag);
            if (close_pos != std::string::npos) {
                std::string body = tool_body_buf.substr(0, close_pos);
                auto bs = body.find_first_not_of("\n\r\t ");
                auto be = body.find_last_not_of("\n\r\t ");
                if (bs != std::string::npos && be != std::string::npos)
                    body = body.substr(bs, be - bs + 1);

                // Parse and emit tool call
                try {
                    json j = json::parse(body);
                    ParsedToolCall tc;
                    tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                    if (tpl_family == imp::ChatTemplateFamily::LLAMA3) {
                        tc.name = tool_fn_name;
                        tc.arguments = dump_safe(j);
                    } else {
                        tc.name = j.value("name", "");
                        if (j.contains("arguments")) {
                            tc.arguments = dump_safe(j["arguments"]);
                        } else {
                            json args = j;
                            args.erase("name");
                            tc.arguments = dump_safe(args);
                        }
                    }
                    // parallel_tool_calls=false: stream at most one tool call.
                    if (!tc.name.empty() &&
                        (ctx.params.parallel_tool_calls || stream_tool_calls.empty())) {
                        int idx = static_cast<int>(stream_tool_calls.size());
                        // Emit name chunk
                        json name_delta = {
                            {"tool_calls",
                             json::array(
                                 {{{"index", idx},
                                   {"id", tc.id},
                                   {"type", "function"},
                                   {"function", {{"name", tc.name}, {"arguments", ""}}}}})}};
                        std::string sse = sse_chunk(comp_id, created, snap_model_name, name_delta,
                                                    nullptr);
                        sink.write(sse.data(), sse.size());

                        // Emit arguments incrementally as partial-JSON deltas
                        // (Task 6) so OpenAI streaming clients see the tool
                        // arguments grow rather than land in one block.
                        constexpr size_t kArgChunk = 48;
                        const std::string& full_args = tc.arguments;
                        if (full_args.empty()) {
                            json args_delta = {
                                {"tool_calls",
                                 json::array({{{"index", idx},
                                               {"function", {{"arguments", ""}}}}})}};
                            sse = sse_chunk(comp_id, created, snap_model_name, args_delta, nullptr);
                            sink.write(sse.data(), sse.size());
                        }
                        for (size_t aoff = 0; aoff < full_args.size(); aoff += kArgChunk) {
                            size_t an = std::min(kArgChunk, full_args.size() - aoff);
                            json args_delta = {
                                {"tool_calls",
                                 json::array(
                                     {{{"index", idx},
                                       {"function", {{"arguments", full_args.substr(aoff, an)}}}}})}};
                            sse = sse_chunk(comp_id, created, snap_model_name, args_delta, nullptr);
                            sink.write(sse.data(), sse.size());
                        }

                        stream_tool_calls.push_back(std::move(tc));
                        tool_calls_emitted = true;
                    }
                } catch (...) {
                    // Malformed JSON — skip
                }

                // Check for more content after close tag
                std::string after = tool_body_buf.substr(close_pos + tool_close_tag.size());
                tool_body_buf.clear();
                tool_phase = ToolPhase::CONTENT;
                // If there's remaining text, it might contain more tool calls
                if (!after.empty()) {
                    auto ws = after.find_first_not_of("\n\r\t ");
                    if (ws != std::string::npos) {
                        piece = after.substr(ws);
                        // Fall through to CONTENT handling below
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;  // Still collecting body
            }
        }

        if (has_tools && tool_phase == ToolPhase::TAG_SCANNING) {
            tool_tag_buf += piece;
            // ChatML: check for <tool_call>
            if (tpl_family != imp::ChatTemplateFamily::LLAMA3) {
                if (tool_tag_buf.size() >= 11) {  // len("<tool_call>")
                    if (tool_tag_buf.find("<tool_call>") != std::string::npos) {
                        auto pos = tool_tag_buf.find("<tool_call>");
                        // Flush content before the tag
                        std::string before = tool_tag_buf.substr(0, pos);
                        if (!before.empty()) {
                            json cd = {{"content", before}};
                            std::string sse = sse_chunk(comp_id, created, snap_model_name, cd,
                                                        nullptr);
                            sink.write(sse.data(), sse.size());
                        }
                        tool_body_buf = tool_tag_buf.substr(pos + 11);
                        tool_close_tag = "</tool_call>";
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::TOOL_CALL_BODY;
                        continue;
                    }
                    // Check if it's definitely not a tool_call tag
                    if (tool_tag_buf.find("<tool_call") == std::string::npos &&
                        tool_tag_buf.find("<tool_c") == std::string::npos &&
                        tool_tag_buf.find("<tool_") == std::string::npos &&
                        tool_tag_buf.find("<tool") == std::string::npos &&
                        tool_tag_buf.find("<too") == std::string::npos &&
                        tool_tag_buf.find("<to") == std::string::npos &&
                        tool_tag_buf.find("<t") == std::string::npos) {
                        // Not a tool tag — flush as content
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                        // Fall through to content emission
                    } else {
                        continue;  // Still scanning
                    }
                } else {
                    // Check partial match
                    const char* tc_tag = "<tool_call>";
                    bool could_match = true;
                    for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 11; ci++) {
                        if (tool_tag_buf[ci] != tc_tag[ci]) {
                            could_match = false;
                            break;
                        }
                    }
                    if (!could_match) {
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                    } else {
                        continue;  // Still matching prefix
                    }
                }
            } else {
                // Llama3: check for <function=
                if (tool_tag_buf.size() >= 10) {  // len("<function=")
                    auto fn_pos = tool_tag_buf.find("<function=");
                    if (fn_pos != std::string::npos) {
                        auto gt = tool_tag_buf.find('>', fn_pos + 10);
                        if (gt != std::string::npos) {
                            std::string before = tool_tag_buf.substr(0, fn_pos);
                            if (!before.empty()) {
                                json cd = {{"content", before}};
                                std::string sse = sse_chunk(comp_id, created, snap_model_name, cd,
                                                            nullptr);
                                sink.write(sse.data(), sse.size());
                            }
                            tool_fn_name = tool_tag_buf.substr(fn_pos + 10, gt - (fn_pos + 10));
                            tool_body_buf = tool_tag_buf.substr(gt + 1);
                            tool_close_tag = "</function>";
                            tool_tag_buf.clear();
                            tool_phase = ToolPhase::TOOL_CALL_BODY;
                            continue;
                        } else {
                            continue;  // Still scanning for >
                        }
                    }
                    // Check prefix match
                    const char* fn_tag = "<function=";
                    bool could_match = true;
                    for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 10; ci++) {
                        if (tool_tag_buf[ci] != fn_tag[ci]) {
                            could_match = false;
                            break;
                        }
                    }
                    if (!could_match) {
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                    } else {
                        continue;
                    }
                } else {
                    const char* fn_tag = "<function=";
                    bool could_match = true;
                    for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 10; ci++) {
                        if (tool_tag_buf[ci] != fn_tag[ci]) {
                            could_match = false;
                            break;
                        }
                    }
                    if (!could_match) {
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                    } else {
                        continue;
                    }
                }
            }
        }

        // In CONTENT phase, check for start of tool call tag
        if (has_tools && tool_phase == ToolPhase::CONTENT) {
            // Look for < that might start a tool call tag
            size_t lt_pos = piece.find('<');
            if (lt_pos != std::string::npos) {
                // Emit everything before the <
                if (lt_pos > 0) {
                    std::string before = piece.substr(0, lt_pos);
                    if (stop_sequences.empty()) {
                        utf8_buf += before;
                    } else {
                        pending_text += before;
                    }
                }
                // Start tag scanning with the < and everything after
                tool_tag_buf = piece.substr(lt_pos);
                tool_phase = ToolPhase::TAG_SCANNING;
                // Flush any buffered content before entering tag scan
                if (stop_sequences.empty() && !utf8_buf.empty()) {
                    size_t complete = utf8_complete_len(utf8_buf);
                    if (complete > 0) {
                        if (!sse_writer.write_content(utf8_buf.data(), complete, sink))
                            return false;
                        utf8_buf.erase(0, complete);
                    }
                } else if (!stop_sequences.empty()) {
                    auto d = imp::stream::holdback_decision(pending_text, max_stop_len,
                                                            stop_sequences);
                    if (!flush_text(d.flush_len))
                        return false;
                    if (d.complete_match) {
                        text_stop_matched = true;
                        finish = "stop";
                        break;
                    }
                }
                continue;
            }
        }

        // Normal content emission (no tool tag detected)
        if (stop_sequences.empty()) {
            // No stop sequences: stream directly (with UTF-8 buffering)
            utf8_buf += piece;
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                if (req_logprobs && active_req) {
                    // Logprobs path: fall back to sse_chunk (rare)
                    std::string to_emit = utf8_buf.substr(0, complete);
                    utf8_buf.erase(0, complete);
                    json content_delta = {{"content", to_emit}};
                    json lp_chunk = nullptr;
                    size_t lp_idx = n_output_tokens - 1;
                    if (lp_idx < active_req->output_logprobs.size()) {
                        const auto& lp = active_req->output_logprobs[lp_idx];
                        json top_arr = json::array();
                        for (const auto& t : lp.top) {
                            top_arr.push_back({{"token", safe_token_json(t.text)},
                                               {"logprob", t.logprob},
                                               {"bytes", token_bytes_json(t.text)}});
                        }
                        lp_chunk = {
                            {"content", json::array({{{"token", safe_token_json(lp.text)},
                                                      {"logprob", lp.logprob},
                                                      {"bytes", token_bytes_json(lp.text)},
                                                      {"top_logprobs", top_arr}}})}};
                    }
                    std::string chunk = sse_chunk(comp_id, created, snap_model_name,
                                                  content_delta, nullptr, lp_chunk);
                    if (!sink.write(chunk.data(), chunk.size()))
                        return false;
                } else {
                    // Fast path: pre-formatted template
                    if (!sse_writer.write_content(utf8_buf.data(), complete, sink))
                        return false;
                    utf8_buf.erase(0, complete);
                }
            }
        } else {
            // Buffer text and check for stop matches via the pure holdback
            // pipeline (stream_pipeline.h). It returns the safe-to-emit prefix
            // and whether a complete stop sequence is present.
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

        // Break after processing the last non-EOS token from batching engine
        if (finish)
            break;
    }

    // Flush scan buffer if we never left SCAN phase (model didn't think)
    // Harmony: flush the final channel's tail (the final block usually ends at
    // EOS/<|return|> with no trailing <|end|>). The other buffers below stay
    // empty for harmony, so they're no-ops.
    if (harmony && !hm_buf.empty())
        hm_flush(/*force=*/true);

    // Flush the splitter's held tail at stream end: buffered reasoning ->
    // reasoning_content, any held/undecided content -> the content flush below.
    if (think_active) {
        auto rs = think_split.finish();
        if (!rs.reasoning.empty())
            emit_reasoning(rs.reasoning);
        if (!rs.content.empty())
            utf8_buf += rs.content;
    }

    // If the model exhausted tokens while still reasoning and never
    // produced content, emit a notice so the user sees something
    // instead of a blank response. Only fire this when max_tokens
    // was actually the cause (finish == "length") — a model that
    // naturally hit EOS during thinking will already have its
    // reasoning_content delivered, and the notice would be
    // misleading ("increase max_tokens" doesn't help when the model
    // chose to stop).
    if (think_active && think_split.phase() == imp::server::ThinkPhase::REASONING &&
        utf8_buf.empty() && pending_text.empty() && finish &&
        std::strcmp(finish, "length") == 0) {
        std::string notice = "[Reasoning truncated — increase max_tokens for a complete answer]";
        sse_writer.write_content(notice, sink);
    }

    // Handle incomplete tool call at end (max_tokens hit while in tag)
    if (tool_phase != ToolPhase::CONTENT && !tool_calls_emitted) {
        // Partial tool call — emit as content, finish_reason stays "length"
        std::string leftover;
        if (!tool_tag_buf.empty())
            leftover += tool_tag_buf;
        if (!tool_body_buf.empty())
            leftover += tool_body_buf;
        if (!leftover.empty()) {
            utf8_buf += leftover;
        }
    }

    // Flush any remaining UTF-8 buffer (only if no tool calls were emitted)
    if (!utf8_buf.empty() && !text_stop_matched && !tool_calls_emitted) {
        sse_writer.write_content(utf8_buf, sink);
    }

    // Flush any remaining buffered text (skip if text-level stop was matched)
    if (!pending_text.empty() && !text_stop_matched && !tool_calls_emitted) {
        sse_writer.write_content(pending_text, sink);
    }

    if (!finish) {
        finish = tool_calls_emitted ? "tool_calls" : "length";
    } else if (tool_calls_emitted && strcmp(finish, "stop") == 0) {
        finish = "tool_calls";
    }

    // Send final chunk with finish_reason
    json empty_delta = json::object();
    std::string final_chunk = sse_chunk(comp_id, created, snap_model_name, empty_delta, finish);
    sink.write(final_chunk.data(), final_chunk.size());

    // Send usage chunk if requested
    if (include_usage) {
        json usage = {{"prompt_tokens", n_prompt_tokens},
                      {"completion_tokens", n_output_tokens},
                      {"total_tokens", n_prompt_tokens + n_output_tokens}};
        // Report prefix cache hit (OpenAI-compatible prompt_tokens_details)
        if (active_req && (active_req->cached_tokens > 0 || active_req->pin_kv_prefix)) {
            json details = {{"cached_tokens", active_req->cached_tokens}};
            int creation = cache_creation_tokens_(active_req, n_prompt_tokens);
            if (creation > 0)
                details["cache_creation_tokens"] = creation;
            usage["prompt_tokens_details"] = std::move(details);
        }
        if (n_reasoning_tokens > 0) {
            usage["completion_tokens_details"] = {{"reasoning_tokens", n_reasoning_tokens}};
        }
        json usage_obj = {{"id", comp_id},
                          {"object", "chat.completion.chunk"},
                          {"created", created},
                          {"model", snap_model_name},
                          {"choices", json::array()},
                          {"usage", usage}};
        std::string usage_chunk = "data: " + dump_safe(usage_obj) + "\n\n";
        sink.write(usage_chunk.data(), usage_chunk.size());
    }

    // Send [DONE]
    std::string done = "data: [DONE]\n\n";
    sink.write(done.data(), done.size());
    sink.done();

    // Log request with TTFT and cache hit info
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
    fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms, cached=%d)\n",
            comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms, ttft_ms, cached);
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += n_prompt_tokens;
    state.metrics.tokens_completion_total += n_output_tokens;
    state.metrics.tokens_cached_total += cached;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.last_ttft_ms = static_cast<int64_t>(ttft_ms);
    state.metrics.request_duration.observe(ms / 1000.0);
    if (n_output_tokens > 0)
        state.metrics.ttft.observe(ttft_ms / 1000.0);
    // Mean inter-token latency: post-first-token decode time spread over the
    // remaining tokens. Streaming-only (non-stream has no per-token cadence).
    if (n_output_tokens > 1)
        state.metrics.inter_token.observe((ms - ttft_ms) / 1000.0 / (n_output_tokens - 1));

    // Streaming response content is not accumulated across SSE
    // chunks, so the JSONL `response` field stays null. The
    // request body, token counts, finish reason, and latency
    // still reflect everything the client did.
    log_request_jsonl(state, log_skip, t_log_start, comp_id, log_endpoint, log_client_ip,
                      log_raw_body, ms, n_prompt_tokens, n_output_tokens, finish, json());

    return true;
}
