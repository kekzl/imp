// Anthropic /v1/messages: the handle_messages endpoint (bottom of file) plus
// its streaming machinery — the AnthropicSSE event writer and the Anthropic
// dialect adapter for the shared token loop (stream_driver.h), which emits
// native Anthropic SSE events.

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

// ===========================================================================
// Anthropic /v1/messages — native SSE streaming
// ===========================================================================
//
// Non-streaming reuses the OpenAI code path (Anthropic→OpenAI body in,
// OpenAI→Anthropic response out). Streaming drives the same real per-token
// batching-engine loop the OpenAI streaming path uses (run_stream_loop_), but
// emits native Anthropic SSE events incrementally so TTFT == real first-token
// latency rather than full-generation latency.
// ---------------------------------------------------------------------------

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

    // #1657: the per-token path. emit() above builds a nested json object and
    // dump()s it for EVERY token, which is exactly what the shared writer's own
    // header forbids on the hot path (utils.h:167-168) and what
    // /v1/chat/completions has avoided since it got SSEChunkWriter. The frame
    // around a delta is constant for the whole block, so it is built once at
    // block start and the token only gets escaped between the two halves.
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

// Anthropic dialect adapter: maps the shared token loop onto Anthropic blocks:
//   reasoning -> thinking block (thinking_delta)
//   content   -> text block (text_delta)
//   tool call -> tool_use block (input_json_delta, chunked)
bool run_anthropic_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                           const std::shared_ptr<ServerRequest>& server_req, const std::string& anth_model,
                           const std::string& msg_id, bool omit_thinking) {
    namespace anth = imp_server::anthropic;
    AnthropicSSE out{sink};
    auto active_req = server_req->request;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;

    // ---- message_start ----------------------------------------------------
    {
        // Cache accounting (#1006): harnesses read cache_read/creation from
        // message_start to display live hit rates, and cached_tokens is set at
        // ADMISSION rather than at submit. This used to wait for the
        // PENDING->PREFILLING transition first - 50 x 2 ms - on the claim that
        // it cost no measurable TTFT. It cost exactly what it looks like, and
        // inverted against its own justification: the poll exits on the first
        // iteration when the queue is empty and runs the full 100 ms when the
        // request is queued, which is when TTFT matters. Measured on
        // Qwen3-4B-Instruct-2507-Q8_0 with 8 concurrent streams, time to
        // message_start: median 118.5 ms with the poll (max 121.0), 11.4 ms
        // without (max 12.8) (#1558). The final message_delta already re-reports the accounting
        // and stays the corrective source, which is what makes the wait
        // buy presentation accuracy rather than correctness.
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

    // ---- ping (initial keepalive) -----------------------------------------
    // Anthropic streams emit periodic `ping` events; sending one immediately
    // signals liveness before the first token (TTFT can be >1s under load /
    // long prefills), and the shared loop re-pings during idle gaps so clients
    // and intermediary proxies don't time out the connection.
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
        // #1554: the slice is at most kChunk bytes AND ends on a codepoint
        // boundary. A fixed byte slice cut multi-byte characters in half and
        // each half became U+FFFD in dump_safe, so a tool argument with a
        // German city name or an emoji reached the client corrupted.
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

    // A fault that ends the stream is an `error` event, not a completed turn
    // (#1553). The status line is long gone by here, so the event is the only
    // way to say the answer is not the model's: a server-side timeout used to
    // arrive as stop_reason "max_tokens", indistinguishable from the model
    // reaching its budget, and an admission refusal as "capacity", which is not
    // an Anthropic stop_reason at all.
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
                  {"delta",
                   {{"stop_reason", stop_reason},
                    {"stop_sequence", res.stop_sequence.empty() ? json(nullptr) : json(res.stop_sequence)}}},
                  {"usage", std::move(delta_usage)}});
    out.emit("message_stop", json{{"type", "message_stop"}});
    sink.done();

    finish_stream_accounting_(state, ctx, active_req, res, msg_id, "messages stream: ");
    return true;
}

// ===========================================================================
// Anthropic /v1/messages endpoint (moved here from handlers_chat.cpp to keep
// that TU under the file-size gate; co-located with run_anthropic_stream_).
// Non-streaming reuses the OpenAI path via a shim; streaming drives the real
// per-token loop above.
// ===========================================================================
static void handle_messages_impl(const httplib::Request& req, httplib::Response& res, ServerState& state,
                                 const std::string& request_id);

void handle_messages(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // One id for this request, on every answer this endpoint gives - success or
    // error (#1561). It is what a client quotes in a bug report and what ties
    // the response to its line in the JSONL log.
    const std::string request_id = make_request_id(state);
    res.set_header("request-id", request_id);

    // anthropic-version and anthropic-beta were read by nothing (#1562).
    // Upstream, a missing version is a 400 and an unknown beta is refused; imp
    // deliberately does neither, because a client that works here and fails
    // there is the lesser harm compared to 400-ing every request that omits a
    // header this server does not need. What it must not do is stay silent: a
    // beta header is a request for behaviour imp does not implement, and
    // answering 200 makes that a false accept the client cannot see. Both are
    // echoed back so the client can tell it was read, and an unknown beta warns
    // once per value.
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

    // Any exception escaping the impl — notably from the inner
    // handle_chat_completions shim on the non-streaming path — must return the
    // Anthropic error envelope ({"type":"error",...}), not the OpenAI-shaped one
    // the global exception handler emits, which strict Anthropic SDK clients
    // fail to parse (#891). res is untouched by the shim (it writes a separate
    // shim_res) when the throw propagates, so it is safe to rewrite here.
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

    // Transform -> OpenAI body.
    json oai_body;
    try {
        oai_body = anth::anthropic_to_openai_body(anth_body);
    } catch (const std::exception& e) {
        send_anthropic_error(res, 400, "invalid_request_error",
                             std::string("Failed to transform Anthropic body: ") + e.what(), request_id);
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

    // Propagate error envelopes (transform them to Anthropic error shape).
    // httplib::Response defaults status to -1 and auto-promotes to 200 only
    // at send time; any other non-200 code set by handle_chat_completions is
    // a real error we should forward.
    const bool is_error = shim_res.status >= 400;
    if (is_error) {
        // The shim answers in the OpenAI dialect, and its `type` was forwarded
        // verbatim inside the Anthropic envelope - so `capacity_error` and
        // `server_error`, neither of which Anthropic defines, reached SDK
        // clients (#1556). Translate; keep param/code, which are additive.
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
        log_request_jsonl(state, /*skip=*/false, t_log_start, req_id, log_endpoint, log_client_ip,
                          log_raw_body, ms, prompt_t, completion_t,
                          stop_reason.empty() ? nullptr : stop_reason.c_str(), anth_response,
                          log_client_request_id);
    }

    // Non-streaming requests are fully assembled above (the want_stream path
    // returned earlier with a native incremental SSE stream).
    res.status = 200;
    res.set_content(dump_safe(anth_response), "application/json");
}
