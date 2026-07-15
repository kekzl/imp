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
#include <string>
#include <thread>

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

// Anthropic dialect adapter: maps the shared token loop onto Anthropic blocks:
//   reasoning -> thinking block (thinking_delta)
//   content   -> text block (text_delta)
//   tool call -> tool_use block (input_json_delta, chunked)
bool run_anthropic_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                                  const std::shared_ptr<ServerRequest>& server_req,
                                  const std::string& anth_model, const std::string& msg_id) {
    AnthropicSSE out{sink};
    auto active_req = server_req->request;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;

    // ---- message_start ----------------------------------------------------
    {
        // Cache accounting (#1006): harnesses read cache_read/creation from
        // message_start to display live hit rates. cached_tokens is set at
        // ADMISSION (before prefill compute), which happens within a scheduler
        // step of submit — a short bounded poll for the PENDING→PREFILLING
        // transition makes the values authoritative here without measurable
        // TTFT cost. On timeout the fields ride at their initial values and
        // the final message_delta (below) stays the corrective source.
        if (active_req) {
            for (int i = 0; i < 50 && active_req->status == imp::RequestStatus::PENDING; i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
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

    StreamLoopResult res;
    StreamDialect dialect;
    dialect.emit_text = emit_text;
    dialect.emit_reasoning = emit_thinking;
    dialect.emit_content_token = emit_text;
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
                fprintf(stderr, "[%s] tool-call arg validation failed: %s: %s\n",
                        msg_id.c_str(), tc->name.c_str(), tc->error.c_str());
            }
        }
        return stop_block();
    };
    dialect.on_call_buffered = [&](ParsedToolCall& tc) -> bool {
        // Buffered call (non-JSON layouts): open block + chunked arg deltas +
        // close.
        validate_tool_call(tc, ctx.params.tools);
        if (!tc.valid) {
            fprintf(stderr, "[%s] tool-call arg validation failed: %s: %s\n",
                    msg_id.c_str(), tc.name.c_str(), tc.error.c_str());
        }
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

    if (!run_stream_loop_(sink, ctx, state, server_req, dialect, res))
        return false;

    // Close any block still open.
    stop_block();

    // Map finish_reason -> Anthropic stop_reason.
    std::string stop_reason;
    if (strcmp(res.finish, "stop") == 0)
        stop_reason = "end_turn";
    else if (strcmp(res.finish, "length") == 0)
        stop_reason = "max_tokens";
    else if (strcmp(res.finish, "tool_calls") == 0)
        stop_reason = "tool_use";
    else if (strcmp(res.finish, "cancelled") == 0)
        stop_reason = "end_turn";
    else
        stop_reason = res.finish;

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
                  {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
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

        // This is the streaming /v1/messages path — stay on per-step decode so
        // SSE is real per-token rather than one burst at generation end (#754).
        auto imp_req = build_imp_request_(ctx, ctx.snap.tokens, /*completion_idx=*/0,
                                          /*stream=*/true);

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
