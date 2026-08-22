// OpenAI Responses API (/v1/responses) — the dialect the OpenAI Agents SDK
// and Codex CLI speak by default. Non-streaming reuses the chat-completions
// code path via the transform shim (responses.h, same pattern as the
// Anthropic adapter); streaming drives the shared token loop
// (stream_driver.h) and emits native Responses events
// (response.created … response.output_text.delta …
// response.function_call_arguments.delta … response.completed) so TTFT is
// real first-token latency and tool-call arguments stream incrementally
// (tool_stream_filter.h CALL_BEGIN/ARGS_DELTA segments).

#include "handlers.h"
#include "handlers_internal.h"
#include "stream_driver.h"
#include "utils.h"
#include "tool_call.h"
#include "responses.h"

#include "runtime/request.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>

namespace rsp = imp_server::responses;

namespace {

// Responses SSE event writer: "event: <type>\ndata: {...}\n\n" with the
// monotonically increasing sequence_number every event carries.
struct ResponsesSSE {
    httplib::DataSink& sink;
    uint64_t seq = 0;
    std::string hot_buf;  // reused by emit_delta, never by emit

    bool emit(const char* type, json payload) {
        payload["type"] = type;
        payload["sequence_number"] = seq++;
        std::string buf = "event: ";
        buf += type;
        buf += "\ndata: ";
        buf += dump_safe(payload);
        buf += "\n\n";
        return sink.write(buf.data(), buf.size());
    }

    // #1657: the per-token path. emit() builds a json object and dump()s it for
    // every token; /v1/chat/completions has not done that since SSEChunkWriter.
    // Everything but the sequence number and the text is constant for the whole
    // item, so `prefix` is built once when the item opens and ends at
    // `"sequence_number":`; `mid` reopens the object for the delta value.
    bool emit_delta(const std::string& prefix, const char* mid, const std::string& text) {
        hot_buf.clear();
        hot_buf += prefix;
        hot_buf += std::to_string(seq++);
        hot_buf += mid;
        json_escape_into(hot_buf, text.data(), text.size());
        hot_buf += "\"}\n\n";
        return sink.write(hot_buf.data(), hot_buf.size());
    }
};

// The constant half of a delta frame, up to and including `"sequence_number":`.
inline std::string rsp_delta_prefix(const char* event, const std::string& item_id, int output_index) {
    std::string p = "event: ";
    p += event;
    p += "\ndata: {\"type\":\"";
    p += event;
    p += "\",\"item_id\":\"";
    json_escape_into(p, item_id.data(), item_id.size());
    p += "\",\"output_index\":";
    p += std::to_string(output_index);
    p += ",\"content_index\":0,\"sequence_number\":";
    return p;
}
inline constexpr const char* kRspDeltaMid = ",\"delta\":\"";

// Which output item is currently open in the stream.
enum class RspItem { NONE, REASONING, MESSAGE, FUNCTION_CALL };

// Skeleton `response` object used by response.created/in_progress/completed.
json response_skeleton(const std::string& response_id, const std::string& model,
                       const char* status) {
    return {{"id", response_id},        {"object", "response"},
            {"created_at", static_cast<int64_t>(time(nullptr))},
            {"model", model},           {"status", status},
            {"error", nullptr},         {"incomplete_details", nullptr},
            {"output", json::array()},  {"parallel_tool_calls", true},
            {"tool_choice", "auto"},    {"tools", json::array()}};
}

}  // anonymous namespace

// Responses dialect adapter: maps the shared token loop onto Responses output
// items:
//   reasoning -> reasoning item (reasoning_summary_text.delta)
//   content   -> message item (output_text.delta)
//   tool call -> function_call item (function_call_arguments.delta,
//                incremental for JSON layouts)
static bool run_responses_stream_(httplib::DataSink& sink, ChatRequestContext& ctx,
                                  ServerState& state,
                                  const std::shared_ptr<ServerRequest>& server_req,
                                  const std::string& req_model,
                                  const std::string& response_id) {
    ResponsesSSE out{sink};
    auto active_req = server_req->request;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;

    const std::string model_name = req_model.empty() ? ctx.snap.model_name : req_model;

    // ---- response.created / response.in_progress ---------------------------
    if (!out.emit("response.created",
                  json{{"response", response_skeleton(response_id, model_name, "in_progress")}}))
        return false;
    if (!out.emit("response.in_progress",
                  json{{"response", response_skeleton(response_id, model_name, "in_progress")}}))
        return false;

    int output_index = -1;
    // Rebuilt when the message item opens; constant for every token in it.
    std::string text_delta_prefix_;
    RspItem open_item = RspItem::NONE;
    uint64_t item_counter = 0;
    std::string cur_item_id;

    // Loop result — declared before the item lambdas so stop_item can read the
    // current tool call (the driver appends to lres.tool_calls live).
    StreamLoopResult lres;

    // Accumulators for the final `response` object.
    std::string acc_text, acc_reasoning;
    json final_output = json::array();

    auto stop_item = [&]() -> bool {
        if (open_item == RspItem::NONE)
            return true;
        bool ok = true;
        if (open_item == RspItem::MESSAGE) {
            ok = out.emit("response.output_text.done",
                          json{{"item_id", cur_item_id},
                               {"output_index", output_index},
                               {"content_index", 0},
                               {"text", acc_text}}) &&
                 out.emit("response.content_part.done",
                          json{{"item_id", cur_item_id},
                               {"output_index", output_index},
                               {"content_index", 0},
                               {"part", {{"type", "output_text"},
                                         {"text", acc_text},
                                         {"annotations", json::array()}}}});
            json item = {{"type", "message"},
                         {"id", cur_item_id},
                         {"status", "completed"},
                         {"role", "assistant"},
                         {"content", json::array({{{"type", "output_text"},
                                                   {"text", acc_text},
                                                   {"annotations", json::array()}}})}};
            ok = ok && out.emit("response.output_item.done",
                                json{{"output_index", output_index}, {"item", item}});
            final_output.push_back(std::move(item));
        } else if (open_item == RspItem::REASONING) {
            ok = out.emit("response.reasoning_summary_text.done",
                          json{{"item_id", cur_item_id},
                               {"output_index", output_index},
                               {"summary_index", 0},
                               {"text", acc_reasoning}});
            json item = {{"type", "reasoning"},
                         {"id", cur_item_id},
                         {"summary", json::array({{{"type", "summary_text"},
                                                   {"text", acc_reasoning}}})}};
            ok = ok && out.emit("response.output_item.done",
                                json{{"output_index", output_index}, {"item", item}});
            final_output.push_back(std::move(item));
        } else if (open_item == RspItem::FUNCTION_CALL) {
            const auto& tc = lres.tool_calls.back();
            ok = out.emit("response.function_call_arguments.done",
                          json{{"item_id", cur_item_id},
                               {"output_index", output_index},
                               {"arguments", tc.arguments}});
            json item = {{"type", "function_call"},
                         {"id", cur_item_id},
                         {"call_id", tc.id},
                         {"name", tc.name},
                         {"arguments", tc.arguments},
                         {"status", "completed"}};
            ok = ok && out.emit("response.output_item.done",
                                json{{"output_index", output_index}, {"item", item}});
            final_output.push_back(std::move(item));
        }
        open_item = RspItem::NONE;
        return ok;
    };
    auto start_message_item = [&]() -> bool {
        if (open_item == RspItem::MESSAGE)
            return true;
        if (!stop_item())
            return false;
        ++output_index;
        open_item = RspItem::MESSAGE;
        acc_text.clear();
        cur_item_id = rsp::make_item_id("msg", item_counter++);
        text_delta_prefix_ = rsp_delta_prefix("response.output_text.delta", cur_item_id, output_index);
        json item = {{"type", "message"},
                     {"id", cur_item_id},
                     {"status", "in_progress"},
                     {"role", "assistant"},
                     {"content", json::array()}};
        return out.emit("response.output_item.added",
                        json{{"output_index", output_index}, {"item", item}}) &&
               out.emit("response.content_part.added",
                        json{{"item_id", cur_item_id},
                             {"output_index", output_index},
                             {"content_index", 0},
                             {"part", {{"type", "output_text"},
                                       {"text", ""},
                                       {"annotations", json::array()}}}});
    };
    auto start_reasoning_item = [&]() -> bool {
        if (open_item == RspItem::REASONING)
            return true;
        if (!stop_item())
            return false;
        ++output_index;
        open_item = RspItem::REASONING;
        acc_reasoning.clear();
        cur_item_id = rsp::make_item_id("rs", item_counter++);
        json item = {{"type", "reasoning"}, {"id", cur_item_id}, {"summary", json::array()}};
        return out.emit("response.output_item.added",
                        json{{"output_index", output_index}, {"item", item}}) &&
               out.emit("response.reasoning_summary_part.added",
                        json{{"item_id", cur_item_id},
                             {"output_index", output_index},
                             {"summary_index", 0},
                             {"part", {{"type", "summary_text"}, {"text", ""}}}});
    };
    auto emit_text = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_message_item())
            return false;
        acc_text += text;
        // #1657: same per-token json object and dump() the Anthropic dialect
        // had. The frame is constant for the item, so it is built when the item
        // opens and the token is only escaped between the halves.
        return out.emit_delta(text_delta_prefix_, kRspDeltaMid, text);
    };
    auto emit_thinking = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_reasoning_item())
            return false;
        acc_reasoning += text;
        return out.emit("response.reasoning_summary_text.delta",
                        json{{"item_id", cur_item_id},
                             {"output_index", output_index},
                             {"summary_index", 0},
                             {"delta", text}});
    };
    // Open a function_call item; arguments follow as incremental deltas.
    auto open_function_call_item = [&](const ParsedToolCall& tc) -> bool {
        if (!stop_item())
            return false;
        ++output_index;
        open_item = RspItem::FUNCTION_CALL;
        cur_item_id = rsp::make_item_id("fc", item_counter++);
        json item = {{"type", "function_call"},
                     {"id", cur_item_id},
                     {"call_id", tc.id},
                     {"name", tc.name},
                     {"arguments", ""},
                     {"status", "in_progress"}};
        return out.emit("response.output_item.added",
                        json{{"output_index", output_index}, {"item", item}});
    };
    auto emit_args_delta = [&](const std::string& partial) -> bool {
        return out.emit("response.function_call_arguments.delta",
                        json{{"item_id", cur_item_id},
                             {"output_index", output_index},
                             {"delta", partial}});
    };

    StreamDialect dialect;
    dialect.emit_text = emit_text;
    dialect.emit_reasoning = emit_thinking;
    dialect.emit_content_token = [&](const std::string& t, int) { return emit_text(t); };
    dialect.keepalive = [&]() -> bool {
        // SSE comment lines are spec-compliant and ignored by SSE parsers —
        // same cadence as the chat stream (#941).
        static constexpr char kKeepalive[] = ": keepalive\n\n";
        return sink.write(kKeepalive, sizeof(kKeepalive) - 1);
    };
    dialect.on_call_begin = [&](const ParsedToolCall& tc) -> bool {
        return open_function_call_item(tc);
    };
    dialect.on_call_args_delta = emit_args_delta;
    dialect.on_call_end = [&](ParsedToolCall*) -> bool { return stop_item(); };
    dialect.on_call_buffered = [&](ParsedToolCall& tc) -> bool {
        // Buffered call (non-JSON layouts): open the item, emit the whole
        // arguments as one delta, close (stop_item reads the call the driver
        // just recorded in lres.tool_calls).
        if (!open_function_call_item(tc))
            return false;
        if (!tc.arguments.empty() && !emit_args_delta(tc.arguments))
            return false;
        return stop_item();
    };

    if (!run_stream_loop_(sink, ctx, state, server_req, dialect, lres))
        return false;

    stop_item();

    // ---- response.completed / response.incomplete --------------------------
    const bool incomplete =
        (strcmp(lres.finish, "length") == 0 || strcmp(lres.finish, "cancelled") == 0);
    json response = response_skeleton(response_id, model_name,
                                      incomplete ? "incomplete" : "completed");
    response["output"] = std::move(final_output);
    if (incomplete)
        response["incomplete_details"] = {{"reason", "max_output_tokens"}};
    int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
    json in_details = {{"cached_tokens", cached}};
    // Context lost to StreamingLLM eviction — see prompt_tokens_details_().
    if (active_req && active_req->evicted_kv_tokens > 0)
        in_details["evicted_tokens"] = active_req->evicted_kv_tokens;
    response["usage"] = {{"input_tokens", n_prompt_tokens},
                         {"output_tokens", lres.n_output_tokens},
                         {"total_tokens", n_prompt_tokens + lres.n_output_tokens},
                         {"input_tokens_details", std::move(in_details)},
                         {"output_tokens_details", {{"reasoning_tokens", lres.n_reasoning_tokens}}}};
    out.emit(incomplete ? "response.incomplete" : "response.completed",
             json{{"response", std::move(response)}});
    sink.done();

    finish_stream_accounting_(state, ctx, active_req, lres, response_id, "responses stream: ");
    return true;
}

// POST /v1/responses
void handle_responses(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    const auto t_log_start = std::chrono::system_clock::now();
    const std::string log_endpoint = req.path;
    std::string log_client_ip = req.get_header_value("X-Forwarded-For");
    if (log_client_ip.empty())
        log_client_ip = req.remote_addr;
    const std::string log_raw_body = req.body;

    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return;

    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }
    if (!body.is_object()) {
        send_json_error(res, 400, "invalid_request_error", "Request body must be a JSON object");
        return;
    }

    const std::string req_model = body.value("model", "");
    const bool want_stream = body.value("stream", false);

    json oai_body;
    try {
        oai_body = rsp::responses_to_openai_body(body);
    } catch (const std::exception& e) {
        send_json_error(res, 400, "invalid_request_error", e.what());
        return;
    }

    const std::string response_id =
        rsp::make_response_id(static_cast<uint64_t>(state.next_id.fetch_add(1)));

    // ---- Streaming: native Responses SSE ------------------------------------
    if (want_stream) {
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
        if (!ok)
            return;  // parse/snapshot set an OpenAI-shaped error (same envelope)

        ctx.log_skip = false;
        ctx.log_endpoint = log_endpoint;
        ctx.log_client_ip = log_client_ip;
        ctx.log_raw_body = log_raw_body;
        ctx.t_log_start = t_log_start;

        // Streaming stays on per-step decode for real per-token SSE (#754).
        auto imp_req = build_imp_request_(ctx, ctx.snap.tokens, /*completion_idx=*/0,
                                          /*stream=*/true);

        auto server_req = std::make_shared<ServerRequest>();
        server_req->request = imp_req;
        {
            std::lock_guard<std::timed_mutex> lock(state.mtx);
            if (!state.batching || !state.batching->is_running()) {
                send_json_error(res, 503, "server_error", "Inference engine not ready. Please retry.");
                return;
            }
            state.batching->submit(server_req);
        }

        ctx.t_start = std::chrono::high_resolution_clock::now();
        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [stream_ctx = std::move(ctx), &state, server_req, req_model, response_id](
                size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
                return run_responses_stream_(sink, stream_ctx, state, server_req, req_model,
                                             response_id);
            });
        return;
    }

    // ---- Non-streaming: reuse the OpenAI handler via a shim -----------------
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

    if (shim_res.status >= 400) {
        // Same error envelope — forward as-is.
        res.status = shim_res.status;
        res.set_content(shim_res.body, "application/json");
        return;
    }

    json oai_response;
    try {
        oai_response = json::parse(shim_res.body);
    } catch (const std::exception& e) {
        send_json_error(res, 500, "server_error", std::string("Upstream returned non-JSON: ") + e.what());
        return;
    }

    json response = rsp::openai_to_responses_response(oai_response, req_model, response_id);

    {
        auto t_end = std::chrono::system_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_log_start).count();
        int prompt_t = oai_response.value("usage", json::object()).value("prompt_tokens", 0);
        int completion_t =
            oai_response.value("usage", json::object()).value("completion_tokens", 0);
        std::string status = response.value("status", "");
        log_request_jsonl(state, /*skip=*/false, t_log_start, response_id, log_endpoint,
                          log_client_ip, log_raw_body, ms, prompt_t, completion_t,
                          status.empty() ? nullptr : status.c_str(), response);
    }

    res.status = 200;
    res.set_content(dump_safe(response), "application/json");
}
