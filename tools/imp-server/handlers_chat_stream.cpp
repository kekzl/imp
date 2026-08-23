// OpenAI chat-completions streaming: the SSE chunked-provider setup
// (stream_chat_response_) and the chat dialect adapter for the shared token
// loop (stream_driver.h) — pre-built envelope templates, per-token logprobs,
// tool_calls deltas, usage chunk, [DONE].

#include "handlers.h"
#include "handlers_internal.h"
#include "stream_driver.h"
#include "utils.h"
#include "core/logging.h"
#include "tool_call.h"

#include "runtime/request.h"

#include <cstdio>
#include <cstring>
#include <string>

// Set up SSE chunked content provider for streaming chat completion.
// Captures state and ctx by reference for the chunked-provider lambda. ctx
// must outlive the SSE response (httplib invokes the chunked provider after
// this function returns; ctx is a stack-local in handle_chat_completions
// which keeps the request frame alive until the response is fully sent).
// run_chat_stream_ itself is declared in handlers_internal.h.
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

// Chat dialect adapter: emits OpenAI chat.completion.chunk SSE frames around
// the shared token loop (run_stream_loop_). httplib calls the chunked-provider
// lambda repeatedly until it returns false; the lambda dispatches here. ctx is
// captured by value into the lambda (so it survives stream_chat_response_'s
// return); state is captured by reference (lives in the long-lived
// ServerState).
bool run_chat_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                             const std::shared_ptr<ServerRequest>& server_req) {
    const std::string& comp_id = ctx.comp_id;
    int64_t created = ctx.created;
    const std::string& snap_model_name = ctx.snap.model_name;
    bool req_logprobs = ctx.params.req_logprobs;

    // Active request ref for logprobs access
    auto active_req = server_req->request;

    // Pre-build SSE envelope templates for fast content/reasoning emission
    SSEChunkWriter sse_writer(comp_id, created, snap_model_name);

    // Send initial chunk with role
    json role_delta = {{"role", "assistant"}};
    std::string chunk = sse_chunk(comp_id, created, snap_model_name, role_delta, nullptr);
    sink.write(chunk.data(), chunk.size());

    StreamLoopResult out;
    StreamDialect dialect;
    // Harmony reasoning is its own mechanism (not the deepseek <think> path), so
    // it's gated on reasoning_format alone — emit reasoning_content unless the
    // caller explicitly asked for none.
    dialect.harmony_reasoning_on = (state.default_args.reasoning_format != "none");
    dialect.emit_text = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        return sse_writer.write_content(text, sink);
    };
    dialect.emit_reasoning = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        return sse_writer.write_reasoning(text, sink);
    };
    dialect.emit_content_token = [&](const std::string& text, int token_index) -> bool {
        if (req_logprobs && active_req) {
            // Logprobs path: fall back to sse_chunk (rare)
            json content_delta = {{"content", text}};
            json lp_chunk = nullptr;
            size_t lp_idx = static_cast<size_t>(token_index);
            if (token_index >= 0 && lp_idx < active_req->output_logprobs.size()) {
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
            std::string sse = sse_chunk(comp_id, created, snap_model_name, content_delta,
                                        nullptr, lp_chunk);
            return sink.write(sse.data(), sse.size());
        }
        // Fast path: pre-formatted template
        return sse_writer.write_content(text, sink);
    };
    dialect.keepalive = [&]() -> bool {
        // SSE comment lines are spec-compliant and ignored by SSE parsers.
        static constexpr char kKeepalive[] = ": keepalive\n\n";
        return sink.write(kKeepalive, sizeof(kKeepalive) - 1);
    };
    dialect.on_call_begin = [&](const ParsedToolCall& tc) -> bool {
        int idx = static_cast<int>(out.tool_calls.size()) - 1;
        json name_delta = {
            {"tool_calls",
             json::array({{{"index", idx},
                           {"id", tc.id},
                           {"type", "function"},
                           {"function", {{"name", tc.name}, {"arguments", ""}}}}})}};
        std::string sse = sse_chunk(comp_id, created, snap_model_name, name_delta, nullptr);
        sink.write(sse.data(), sse.size());
        return true;
    };
    dialect.on_call_args_delta = [&](const std::string& partial) -> bool {
        int idx = static_cast<int>(out.tool_calls.size()) - 1;
        json args_delta = {
            {"tool_calls",
             json::array({{{"index", idx}, {"function", {{"arguments", partial}}}}})}};
        std::string sse = sse_chunk(comp_id, created, snap_model_name, args_delta, nullptr);
        sink.write(sse.data(), sse.size());
        return true;
    };
    dialect.on_call_end = [&](ParsedToolCall*) -> bool {
        // Deltas already on the wire; arguments recorded by the driver.
        return true;
    };
    dialect.on_call_buffered = [&](ParsedToolCall& tc) -> bool {
        // Buffered call (non-JSON layouts): emit the name chunk + arguments in
        // bounded deltas.
        int idx = static_cast<int>(out.tool_calls.size()) - 1;
        json name_delta = {
            {"tool_calls",
             json::array({{{"index", idx},
                           {"id", tc.id},
                           {"type", "function"},
                           {"function", {{"name", tc.name}, {"arguments", ""}}}}})}};
        std::string sse = sse_chunk(comp_id, created, snap_model_name, name_delta, nullptr);
        sink.write(sse.data(), sse.size());

        constexpr size_t kArgChunk = 48;
        const std::string& full_args = tc.arguments;
        if (full_args.empty()) {
            json args_delta = {
                {"tool_calls",
                 json::array({{{"index", idx}, {"function", {{"arguments", ""}}}}})}};
            sse = sse_chunk(comp_id, created, snap_model_name, args_delta, nullptr);
            sink.write(sse.data(), sse.size());
        }
        // #1554: codepoint-aligned slices, same reason as the Anthropic
        // dialect. The /v1/responses dialect never chunked and was immune.
        for (size_t aoff = 0; aoff < full_args.size();) {
            const size_t an = utf8_chunk_len(full_args, aoff, kArgChunk);
            json args_delta = {
                {"tool_calls",
                 json::array({{{"index", idx},
                               {"function", {{"arguments", full_args.substr(aoff, an)}}}}})}};
            sse = sse_chunk(comp_id, created, snap_model_name, args_delta, nullptr);
            sink.write(sse.data(), sse.size());
            aoff += an;
        }
        return true;
    };

    if (!run_stream_loop_(sink, ctx, state, server_req, dialect, out))
        return false;

    // The model exhausted its budget while still reasoning and never produced
    // content. This used to write a server-authored English sentence into
    // delta.content, which made the streaming and non-streaming answers to the
    // identical request differ, and put text into `content` that no token
    // produced (#1593). It is the same shape as the reasoning leak this file's
    // invariant already forbids, so it goes where the non-streaming path
    // already puts it: the server log. finish_reason is "length" either way,
    // which is the machine-readable half a client can act on.
    if (out.reasoning_truncated) {
        IMP_LOG_WARN(
            "empty content: the answer never started because the token budget went to "
            "reasoning (streaming, finish_reason=length). Raise max_tokens — a thinking "
            "model needs room to answer AFTER it thinks.");
    }

    // Send final chunk with finish_reason
    json empty_delta = json::object();
    std::string final_chunk = sse_chunk(comp_id, created, snap_model_name, empty_delta,
                                        openai_finish_reason(out.finish));
    sink.write(final_chunk.data(), final_chunk.size());

    // Send usage chunk if requested
    if (ctx.params.include_usage) {
        int n_prompt_tokens = ctx.snap.n_prompt_tokens;
        json usage = {{"prompt_tokens", n_prompt_tokens},
                      {"completion_tokens", out.n_output_tokens},
                      {"total_tokens", n_prompt_tokens + out.n_output_tokens}};
        // Report prefix cache hit (OpenAI-compatible prompt_tokens_details)
        // Also carries `evicted_tokens` (StreamingLLM dropped context mid-run).
        if (json details = prompt_tokens_details_(active_req, n_prompt_tokens); !details.is_null())
            usage["prompt_tokens_details"] = std::move(details);
        if (out.n_reasoning_tokens > 0) {
            usage["completion_tokens_details"] = {{"reasoning_tokens", out.n_reasoning_tokens}};
        }
        // Predicted Outputs accounting (mirrors the non-streaming path).
        if (active_req && !active_req->prediction_tokens.empty()) {
            usage["completion_tokens_details"]["accepted_prediction_tokens"] =
                active_req->pred_accepted;
            usage["completion_tokens_details"]["rejected_prediction_tokens"] =
                active_req->pred_rejected;
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

    finish_stream_accounting_(state, ctx, active_req, out, comp_id, "");
    return true;
}
