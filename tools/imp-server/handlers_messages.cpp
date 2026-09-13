// Anthropic /v1/messages: handle_messages endpoint plus its streaming machinery (AnthropicSSE
// event writer, dialect adapter for the shared token loop) emitting native Anthropic SSE events.

#include "runtime/engine.h"
#include "handlers.h"
#include "handlers_internal.h"
#include "stream_driver.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"

#include "runtime/request.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

// Non-streaming reuses the OpenAI path (Anthropic->OpenAI body in, OpenAI->Anthropic response
// out). Streaming drives the same per-token batching-engine loop as OpenAI streaming
// (run_stream_loop_) but emits native Anthropic SSE events so TTFT is real first-token latency.

namespace {

// Anthropic SSE event writer. Emits "event: <name>\ndata: <json>\n\n".
struct AnthropicSSE {
    httplib::DataSink& sink;
    std::string hot_buf;  // reused by emit_delta, never by emit

    bool emit(const char* event_name, const json& payload) const {
        std::string buf = "event: ";
        buf += event_name;
        buf += "\ndata: ";
        buf += dump_safe(payload);
        buf += "\n\n";
        return sink.write(buf.data(), buf.size());
    }

    // #1657: builds the SSE frame once per block and only escapes the token per call - dumping a
    // json object per token violates the hot-path rule (utils.h:105-106) that
    // /v1/chat/completions already avoids via SSEChunkWriter.
    bool emit_delta(const std::string& prefix, const std::string& suffix, const std::string& text) {
        hot_buf.clear();
        hot_buf += prefix;
        json_escape_into(hot_buf, text.data(), text.size());
        hot_buf += suffix;
        return sink.write(hot_buf.data(), hot_buf.size());
    }
};

// The constant half of a content_block_delta frame, built once per block.
// `field` is "text" for a text_delta and "thinking" for a thinking_delta.
inline std::string anth_delta_prefix(int block_index, const char* type, const char* field) {
    std::string p = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":";
    p += std::to_string(block_index);
    p += ",\"delta\":{\"type\":\"";
    p += type;
    p += "\",\"";
    p += field;
    p += "\":\"";
    return p;
}

// Tracks which content block (if any) is currently open in the stream so we
// can close it before opening one of a different kind. Anthropic requires a
// content_block_start before deltas and a content_block_stop after.
enum class AnthBlock { NONE, THINKING, TEXT, TOOL_USE };

}  // anonymous namespace

// Anthropic dialect adapter: reasoning -> thinking_delta, content -> text_delta, tool call ->
// tool_use block (input_json_delta, chunked).
bool run_anthropic_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                           const std::shared_ptr<ServerRequest>& server_req, const std::string& anth_model,
                           const std::string& msg_id, bool omit_thinking) {
    namespace anth = imp_server::anthropic;
    AnthropicSSE out{sink};
    auto active_req = server_req->request;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;

    // ---- message_start ----------------------------------------------------
    {
        // Cache accounting (#1006): cached_tokens read at ADMISSION, not submit. A prior version polled
        // for PENDING->PREFILLING first, claiming no TTFT cost; measured cost was real (median 118.5 ms
        // vs 11.4 ms, 8 streams, #1558). message_delta re-reports the final, corrective count.
        const int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
        const int creation = active_req ? cache_creation_tokens_(active_req, n_prompt_tokens) : 0;
        json usage = {{"input_tokens", n_prompt_tokens - cached},
                      {"output_tokens", 0},
                      {"cache_read_input_tokens", cached},
                      {"cache_creation_input_tokens", creation}};
        json msg = {
            {"id", msg_id},
            {"type", "message"},
            {"role", "assistant"},
            {"content", json::array()},
            {"model", anth_model},
            {"stop_reason", nullptr},
            {"stop_sequence", nullptr},
            {"usage", std::move(usage)},
        };
        if (!out.emit("message_start", json{{"type", "message_start"}, {"message", std::move(msg)}}))
            return false;
    }

    // Sends an Anthropic `ping` event immediately (signals liveness before TTFT, which can exceed 1s
    // under load) and periodically during idle gaps so clients/proxies don't time out.
    if (!out.emit("ping", json{{"type", "ping"}}))
        return false;

    int block_index = -1;
    AnthBlock open_block = AnthBlock::NONE;
    // Rebuilt whenever a block opens; constant for every token inside it.
    std::string text_delta_prefix, thinking_delta_prefix;
    static const std::string kDeltaSuffix = "\"}}\n\n";

    // The thinking text as it goes out, so the block can be signed at its close
    // (#1555): Anthropic emits signature_delta immediately before
    // content_block_stop on a thinking block, and its SDKs round-trip the pair.
    std::string thinking_so_far;

    auto stop_block = [&]() -> bool {
        if (open_block == AnthBlock::NONE)
            return true;
        if (open_block == AnthBlock::THINKING && !thinking_so_far.empty()) {
            if (!out.emit("content_block_delta",
                          json{{"type", "content_block_delta"},
                               {"index", block_index},
                               {"delta",
                                {{"type", "signature_delta"},
                                 {"signature", anth::thinking_signature(thinking_so_far)}}}}))
                return false;
        }
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
        text_delta_prefix = anth_delta_prefix(block_index, "text_delta", "text");
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
        thinking_delta_prefix = anth_delta_prefix(block_index, "thinking_delta", "thinking");
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
        return out.emit_delta(text_delta_prefix, kDeltaSuffix, text);
    };
    auto emit_thinking = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        // thinking.display "omitted": the model still reasons, the client asked
        // not to be shown it. Dropping the deltas is the whole of it - no
        // block is opened, so no start/stop pair goes out either (#1560).
        if (omit_thinking)
            return true;
        if (!start_thinking_block())
            return false;
        thinking_so_far += text;
        return out.emit_delta(thinking_delta_prefix, kDeltaSuffix, text);
    };
    // Open a tool_use block (content_block_start). Arguments follow as
    // input_json_delta events — incrementally for streamed (JSON-layout)
    // calls, chunked-after-the-fact for buffered ones.
    auto open_tool_use_block = [&](const ParsedToolCall& tc) -> bool {
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::TOOL_USE;
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

    StreamLoopResult res;
    StreamDialect dialect;
    dialect.emit_text = emit_text;
    dialect.emit_reasoning = emit_thinking;
    dialect.emit_content_token = [&](const std::string& t, int) { return emit_text(t); };
    dialect.keepalive = [&]() -> bool { return out.emit("ping", json{{"type", "ping"}}); };
    dialect.on_call_begin = [&](const ParsedToolCall& tc) -> bool {
        // Streamed call: open the tool_use block now; the argument bytes
        // follow as input_json_delta events while the model is still
        // generating them.
        return open_tool_use_block(tc);
    };
    dialect.on_call_args_delta = emit_tool_args_delta;
    dialect.on_call_end = [&](ParsedToolCall* tc) -> bool {
        if (tc) {
            validate_tool_call(*tc, ctx.params.tools);
            if (!tc->valid) {
                IMP_LOG_INFO("[%s] tool-call arg validation failed: %s: %s", msg_id.c_str(), tc->name.c_str(),
                             tc->error.c_str());
            }
        }
        return stop_block();
    };
    dialect.on_call_buffered = [&](ParsedToolCall& tc) -> bool {
        // Buffered call (non-JSON layouts): open block + chunked arg deltas +
        // close.
        validate_tool_call(tc, ctx.params.tools);
        if (!tc.valid) {
            IMP_LOG_INFO("[%s] tool-call arg validation failed: %s: %s", msg_id.c_str(), tc.name.c_str(),
                         tc.error.c_str());
        }
        if (!open_tool_use_block(tc))
            return false;
        const std::string& args = tc.arguments;
        constexpr size_t kChunk = 48;
        // #1554: slices tool-argument text at most kChunk bytes AND on a codepoint boundary - a fixed
        // byte slice cut multi-byte UTF-8 in half, each half becoming U+FFFD in dump_safe.
        for (size_t off = 0; off < args.size();) {
            const size_t n = utf8_chunk_len(args, off, kChunk);
            if (!emit_tool_args_delta(args.substr(off, n)))
                return false;
            off += n;
        }
        return stop_block();
    };

    if (!run_stream_loop_(sink, ctx, state, server_req, dialect, res))
        return false;

    // Close any block still open.
    stop_block();

    // Map finish_reason -> Anthropic stop_reason, through the same function the
    // non-streaming builder uses. This copy passed "capacity" through verbatim
    // (#1552) and could not report a stop-sequence match (#1550).
    const std::string stop_reason = anth::anthropic_stop_reason(res.finish, !res.stop_sequence.empty());

    // #1553: a fault that ends the stream emits an `error` event, not a fake completed turn - a
    // server timeout used to arrive as stop_reason "max_tokens" (indistinguishable from budget
    // exhaustion) and admission refusal as "capacity" (not a real Anthropic stop_reason).
    if (res.error_type) {
        out.emit("error", json{{"type", "error"},
                               {"error", {{"type", res.error_type}, {"message", res.error_message}}}});
        sink.done();
        finish_stream_accounting_(state, ctx, active_req, res, msg_id, "messages stream: ");
        return true;
    }

    // ---- message_delta + message_stop ------------------------------------
    // Cache accounting is only known after prefill ran, so it rides on the
    // final usage update instead of message_start.
    json delta_usage = {{"output_tokens", res.n_output_tokens}};
    if (res.n_reasoning_tokens > 0)
        delta_usage["output_tokens_details"] = {{"reasoning_tokens", res.n_reasoning_tokens}};
    {
        int cached_now = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
        int creation = cache_creation_tokens_(active_req, n_prompt_tokens);
        if (cached_now > 0 || creation > 0) {
            delta_usage["input_tokens"] = n_prompt_tokens - cached_now;
            delta_usage["cache_read_input_tokens"] = cached_now;
            delta_usage["cache_creation_input_tokens"] = creation;
        }
    }
    // The exhaustion detail rides in the same delta as stop_reason, which keeps
    // its Anthropic enum (client compatibility). Same condition and same value
    // as the OpenAI dialect's `imp_finish_detail` on the final chunk.
    json delta = {{"stop_reason", stop_reason},
                  {"stop_sequence", res.stop_sequence.empty() ? json(nullptr) : json(res.stop_sequence)}};
    attach_reasoning_finish_detail(delta, res.tool_calls_emitted, !res.content_emitted,
                                   res.n_reasoning_tokens > 0);
    if (delta.contains("imp_finish_detail"))
        state.metrics.requests_reasoning_exhausted++;
    out.emit("message_delta",
             json{{"type", "message_delta"}, {"delta", std::move(delta)}, {"usage", std::move(delta_usage)}});
    out.emit("message_stop", json{{"type", "message_stop"}});
    sink.done();

    finish_stream_accounting_(state, ctx, active_req, res, msg_id, "messages stream: ");
    return true;
}

// Anthropic /v1/messages endpoint (moved here from handlers_chat.cpp for the file-size gate;
// co-located with run_anthropic_stream_). Non-streaming shims to the OpenAI path; streaming
// drives the real per-token loop above.
static void handle_messages_impl(const httplib::Request& req, httplib::Response& res, ServerState& state,
                                 const std::string& request_id);

void handle_messages(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // One id for this request, on every answer this endpoint gives - success or
    // error (#1561). It is what a client quotes in a bug report and what ties
    // the response to its line in the JSONL log.
    const std::string request_id = make_request_id(state);
    res.set_header("request-id", request_id);

    // anthropic-version/anthropic-beta were read by nothing (#1562). imp accepts both rather than
    // 400ing like upstream (a client that works here and fails there is the lesser harm) but echoes
    // both back and warns once per unknown beta value, so a false-accept isn't silent.
    {
        const std::string version = req.get_header_value("anthropic-version");
        if (!version.empty())
            res.set_header("anthropic-version", version);
        const std::string beta = req.get_header_value("anthropic-beta");
        if (!beta.empty()) {
            res.set_header("anthropic-beta", beta);
            static std::mutex warned_mu;
            static std::set<std::string> warned;
            bool first = false;
            {
                std::lock_guard<std::mutex> lk(warned_mu);
                first = warned.insert(beta).second;
            }
            if (first)
                IMP_LOG_WARN(
                    "anthropic-beta: %s - imp implements no beta surfaces, so this request is "
                    "served as if the flag were absent. Upstream would refuse an unknown beta.",
                    sanitize_for_echo(beta, 96).c_str());
        }
    }

    // Any exception escaping this handler (notably from the inner handle_chat_completions shim) must
    // return the Anthropic error envelope, not the OpenAI-shaped one from the global handler, which
    // strict Anthropic SDK clients fail to parse (#891).
    try {
        handle_messages_impl(req, res, state, request_id);
    } catch (const std::exception& e) {
        // api_error, not server_error: the latter is not one of Anthropic's
        // error types, so an SDK switching on it lands in its default branch
        // (#1556).
        send_anthropic_error(res, 500, "api_error", e.what(), request_id);
    } catch (...) {
        send_anthropic_error(res, 500, "api_error", "internal error", request_id);
    }
}

static void handle_messages_impl(const httplib::Request& req, httplib::Response& res, ServerState& state,
                                 const std::string& request_id) {
    namespace anth = imp_server::anthropic;

    // Capture original Anthropic request data for opt-in JSONL logging.
    const auto t_log_start = std::chrono::system_clock::now();
    const std::string log_endpoint = req.path;
    // Same key the rate limiter uses: an untrusted X-Forwarded-For in the
    // request log is a forged identity in the audit trail (#1614).
    std::string log_client_ip = state.rate_limit_key(req.remote_addr,
                                                     req.get_header_value("X-Forwarded-For"));
    const std::string log_client_request_id =
        sanitize_for_echo(req.get_header_value("X-Request-Id"), 128);
    const std::string log_raw_body = req.body;

    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return;

    json anth_body;
    try {
        anth_body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_anthropic_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what(),
                             request_id);
        return;
    }

    if (!anth_body.is_object()) {
        send_anthropic_error(res, 400, "invalid_request_error", "Request body must be a JSON object",
                             request_id);
        return;
    }

    // Anthropic requires max_tokens — if it's missing, supply a sane default
    // matching the server's chat-completions default (handled downstream).
    std::string anth_model = anth_body.value("model", "");
    const bool want_stream = anth_body.value("stream", false);

    // Before the transform, not after: the transform is what makes an
    // unreadable block invisible, so checking the OpenAI body downstream finds
    // a clean request. Refusing beats answering from the blocks that survived.
    if (std::string why; anthropic_unreadable_block(anth_body, why)) {
        send_anthropic_error(res, 400, "invalid_request_error", why, request_id);
        return;
    }

    // Transform -> OpenAI body.
    json oai_body;
    try {
        oai_body = anth::anthropic_to_openai_body(anth_body);
    } catch (const std::exception& e) {
        send_anthropic_error(res, 400, "invalid_request_error",
                             std::string("Failed to transform Anthropic body: ") + e.what(), request_id);
        return;
    }

    // stream=true drives the same per-token batching-engine loop as OpenAI streaming, emitting
    // native Anthropic SSE events - TTFT is real first-token latency, not full-generation latency.
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
        ctx.log_client_request_id = log_client_request_id;
        ctx.log_raw_body = log_raw_body;
        ctx.t_log_start = t_log_start;

        // This is the streaming /v1/messages path — stay on per-step decode so
        // SSE is real per-token rather than one burst at generation end (#754).
        auto imp_req = build_imp_request_(ctx, ctx.snap.tokens, /*completion_idx=*/0,
                                          /*stream=*/true);

        auto server_req = std::make_shared<ServerRequest>();
        server_req->request = imp_req;
        {
            std::lock_guard<std::timed_mutex> lock(state.mtx);
            if (!state.batching || !state.batching->is_running()) {
                send_anthropic_error(res, 503, "overloaded_error",
                                     "Inference engine not ready. Please retry.", request_id);
                return;
            }
            state.batching->submit(server_req);
        }

        std::string msg_id = anth::make_message_id(static_cast<uint64_t>(state.next_id.fetch_add(1)));
        ctx.t_start = std::chrono::high_resolution_clock::now();
        const bool omit_thinking = anth::thinking_display_omitted(anth_body);

        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream",
                                         [stream_ctx = std::move(ctx), &state, server_req, anth_model, msg_id,
                                          omit_thinking](size_t /*offset*/,
                                                         httplib::DataSink& sink) mutable -> bool {
                                             return run_anthropic_stream_(sink, stream_ctx, state, server_req,
                                                                          anth_model, msg_id, omit_thinking);
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

    // httplib::Response defaults status to -1 (auto-promotes to 200 only at send time): any other
    // non-200 status set by the inner handle_chat_completions shim is a real error to forward.
    const bool is_error = shim_res.status >= 400;
    if (is_error) {
        // #1556: the OpenAI shim's error `type` was forwarded verbatim inside the Anthropic envelope,
        // leaking non-Anthropic types (capacity_error, server_error) to SDK clients. Translated here;
        // param/code are kept (additive fields).
        json inner;
        try {
            inner = json::parse(shim_res.body).value("error", json::object());
        } catch (...) {
            inner = json::object();
        }
        const std::string msg = inner.value("message", shim_res.body.empty() ? "unknown" : shim_res.body);
        const std::string oai_type = inner.value("type", "");
        json e = {{"type", anthropic_error_type_for(oai_type, shim_res.status)}, {"message", msg}};
        if (inner.contains("code") && !inner["code"].is_null())
            e["code"] = inner["code"];
        if (inner.contains("param") && !inner["param"].is_null())
            e["param"] = inner["param"];
        json out = {{"type", "error"}, {"error", std::move(e)}, {"request_id", request_id}};
        res.status = shim_res.status;
        res.set_content(dump_safe(out), "application/json");
        return;
    }

    json oai_response;
    try {
        oai_response = json::parse(shim_res.body);
    } catch (const std::exception& e) {
        send_anthropic_error(res, 500, "api_error", std::string("Upstream returned non-JSON: ") + e.what(),
                             request_id);
        return;
    }

    // The shim's OpenAI body cannot say which stop sequence ended the
    // generation; the handler that matched it left the answer beside the body
    // (#1550).
    json anth_response = anth::openai_to_anthropic_response(oai_response, anth_model, g_shim_stop_sequence,
                                                            anth::thinking_display_omitted(anth_body));

    // JSONL log — built from Anthropic shapes so /v1/messages clients see
    // exactly what they sent and what they got back.
    {
        auto t_end = std::chrono::system_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_log_start).count();
        int prompt_t = oai_response.value("usage", json::object()).value("prompt_tokens", 0);
        int completion_t = oai_response.value("usage", json::object()).value("completion_tokens", 0);
        std::string stop_reason = anth_response.value("stop_reason", "");
        std::string req_id = anth_response.value("id", make_completion_id(state));
        // Trace join (the shim's inner header lands on the discarded
        // shim_res): client id when sent, this dialect's message id otherwise.
        res.set_header("X-Request-Id",
                       log_client_request_id.empty() ? req_id : log_client_request_id);
        RequestSpan trace;
        trace.traceparent = req.get_header_value("traceparent");
        trace.model = anth_response.value("model", std::string());
        trace.cached_tokens =
            anth_response.value("usage", json::object()).value("cache_read_input_tokens", 0);
        {
            const json u = anth_response.value("usage", json::object());
            if (u.contains("imp_spec_verify_steps"))
                trace.set_spec(u.value("imp_spec_drafted", 0LL), u.value("imp_spec_accepted", 0LL),
                               u.value("imp_spec_verify_steps", 0));
        }
        log_request_jsonl(state, /*skip=*/false, t_log_start, req_id, log_endpoint, log_client_ip,
                          log_raw_body, ms, prompt_t, completion_t,
                          stop_reason.empty() ? nullptr : stop_reason.c_str(), anth_response,
                          log_client_request_id, &trace);
    }

    // Non-streaming requests are fully assembled above (the want_stream path
    // returned earlier with a native incremental SSE stream).
    res.status = 200;
    res.set_content(dump_safe(anth_response), "application/json");
}
