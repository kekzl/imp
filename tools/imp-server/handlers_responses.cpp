// OpenAI Responses API (/v1/responses) — the dialect the OpenAI Agents SDK
// and Codex CLI speak by default. Non-streaming reuses the chat-completions
// code path via the transform shim (responses.h, same pattern as the
// Anthropic adapter); streaming drives the same real per-token batching-
// engine loop the other SSE paths use and emits native Responses events
// (response.created … response.output_text.delta …
// response.function_call_arguments.delta … response.completed) so TTFT is
// real first-token latency and tool-call arguments stream incrementally
// (tool_stream_filter.h CALL_BEGIN/ARGS_DELTA segments).

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "tool_stream_filter.h"
#include "responses.h"
#include "stream_pipeline.h"
#include "reasoning_split.h"

#include "api/imp_internal.h"
#include "runtime/request.h"
#include "runtime/config.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <functional>
#include <vector>

namespace rsp = imp_server::responses;

namespace {

// Responses SSE event writer: "event: <type>\ndata: {...}\n\n" with the
// monotonically increasing sequence_number every event carries.
struct ResponsesSSE {
    httplib::DataSink& sink;
    uint64_t seq = 0;
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
};

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

// Drives the real token loop and emits native Responses SSE events. Mirrors
// run_anthropic_stream_ (handlers_messages.cpp) with the Anthropic block
// model replaced by Responses output items:
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

    const auto& stop_sequences = ctx.params.stop_sequences;
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

    const std::string model_name = req_model.empty() ? ctx.snap.model_name : req_model;

    // ---- response.created / response.in_progress ---------------------------
    if (!out.emit("response.created",
                  json{{"response", response_skeleton(response_id, model_name, "in_progress")}}))
        return false;
    if (!out.emit("response.in_progress",
                  json{{"response", response_skeleton(response_id, model_name, "in_progress")}}))
        return false;

    int output_index = -1;
    RspItem open_item = RspItem::NONE;
    uint64_t item_counter = 0;
    std::string cur_item_id;

    // Accumulators for the final `response` object.
    std::string acc_text, acc_reasoning;
    std::vector<ParsedToolCall> stream_tool_calls;
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
            const auto& tc = stream_tool_calls.back();
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
        return out.emit("response.output_text.delta", json{{"item_id", cur_item_id},
                                                           {"output_index", output_index},
                                                           {"content_index", 0},
                                                           {"delta", text}});
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

    // gpt-oss Harmony streaming filter (analysis/commentary -> reasoning item,
    // final -> message item). Markers arrive as atomic special-token pieces.
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

    int n_output_tokens = 0;
    const char* finish = nullptr;

    std::string utf8_buf;      // confirmed-UTF8 content buffer
    std::string pending_text;  // stop-sequence holdback
    bool text_stop_matched = false;

    imp::server::StreamToolCallFilter tool_filter(tpl_family);
    bool tool_calls_emitted = false;
    // parallel_tool_calls=false: a second streamed call was opened by the
    // filter but is being suppressed (skip its deltas/END too).
    bool stream_call_suppressed = false;

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
        if (!server_req->pop_token(evt))
            continue;

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
        std::string piece = snap_tok->decode_token(token);

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
                    hm_channel =
                        (s == std::string::npos) ? std::string() : hm_name.substr(s, e - s + 1);
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

        // Reasoning/content demux (reasoning_split.h).
        if (think_active) {
            auto rs2 = think_split.feed(std::move(piece), token);
            if (!rs2.reasoning.empty() && !emit_thinking(rs2.reasoning))
                return false;
            if (rs2.content.empty())
                continue;
            piece = std::move(rs2.content);
        }

        // Streaming tool-call demux — incremental argument deltas for JSON
        // layouts (see the matching block in run_chat_stream_).
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
                    ParsedToolCall tc = std::move(seg.call);
                    tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                    if (!flush_buffered_content())
                        return false;
                    if (!open_function_call_item(tc))
                        return false;
                    stream_tool_calls.push_back(std::move(tc));
                    tool_calls_emitted = true;
                    continue;
                }
                if (seg.kind == SegKind::CALL_ARGS_DELTA) {
                    if (!emit_args_delta(seg.text))
                        return false;
                    continue;
                }
                if (seg.kind == SegKind::CALL_END) {
                    if (!stream_tool_calls.empty())
                        stream_tool_calls.back().arguments = std::move(seg.call.arguments);
                    if (!stop_item())
                        return false;
                    continue;
                }
                // SegKind::CALL — buffered call (non-JSON layouts): open the
                // item, emit the whole arguments as one delta, close.
                ParsedToolCall tc = std::move(seg.call);
                tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                if (!flush_buffered_content())
                    return false;
                if (!open_function_call_item(tc))
                    return false;
                stream_tool_calls.push_back(std::move(tc));
                if (!tc.arguments.empty() && !emit_args_delta(tc.arguments))
                    return false;
                if (!stop_item())
                    return false;
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
    }

    // ---- Stream-end flushing (mirrors the Anthropic path) ------------------
    if (think_active) {
        auto rs2 = think_split.finish();
        if (!rs2.reasoning.empty())
            emit_thinking(rs2.reasoning);
        if (!rs2.content.empty())
            utf8_buf += rs2.content;
    }
    if (has_tools && tool_filter.mid_tool() && !tool_calls_emitted) {
        std::string leftover = tool_filter.finish();
        if (!leftover.empty())
            utf8_buf += leftover;
    }
    if (has_tools && tool_filter.call_open() && !stream_tool_calls.empty() &&
        stream_tool_calls.back().arguments.empty())
        stream_tool_calls.back().arguments = tool_filter.streamed_arguments();
    if (!utf8_buf.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(utf8_buf);
    if (!pending_text.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(pending_text);

    stop_item();

    if (!finish)
        finish = tool_calls_emitted ? "tool_calls" : "length";

    // ---- response.completed / response.incomplete --------------------------
    const bool incomplete = (strcmp(finish, "length") == 0 || strcmp(finish, "cancelled") == 0);
    json response = response_skeleton(response_id, model_name,
                                      incomplete ? "incomplete" : "completed");
    response["output"] = std::move(final_output);
    if (incomplete)
        response["incomplete_details"] = {{"reason", "max_output_tokens"}};
    {
        int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
        response["usage"] = {{"input_tokens", n_prompt_tokens},
                             {"output_tokens", n_output_tokens},
                             {"total_tokens", n_prompt_tokens + n_output_tokens},
                             {"input_tokens_details", {{"cached_tokens", cached}}},
                             {"output_tokens_details", {{"reasoning_tokens", 0}}}};
    }
    out.emit(incomplete ? "response.incomplete" : "response.completed",
             json{{"response", std::move(response)}});

    // JSONL request log (outer Responses shapes).
    if (!ctx.log_skip) {
        auto t_end = std::chrono::system_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t_end - ctx.t_log_start).count();
        log_request_jsonl(state, /*skip=*/false, ctx.t_log_start, response_id, ctx.log_endpoint,
                          ctx.log_client_ip, ctx.log_raw_body, ms, n_prompt_tokens,
                          n_output_tokens, finish, json());
    }

    sink.done();
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

        auto imp_req = std::make_shared<imp::Request>();
        imp_req->image = ctx.snap.vision_image;
        imp_req->input_tokens = ctx.snap.tokens;
        imp_req->max_tokens = ctx.params.max_tokens;
        imp_req->temperature = ctx.params.temperature;
        imp_req->top_p = ctx.params.top_p;
        imp_req->top_k = ctx.params.top_k;
        imp_req->seed = ctx.params.seed;
        imp_req->pin_kv_prefix = ctx.params.cache_prompt;
        imp_req->spec_ngram_override = ctx.params.spec_ngram_override;
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
