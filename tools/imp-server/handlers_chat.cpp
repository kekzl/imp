// AUTO-SPLIT from handlers.cpp. OpenAI chat endpoint: handle_chat_completions
// (/v1/chat/completions), Anthropic handle_count_tokens. /v1/completions: handlers_completions.cpp.
// handle_messages lives in handlers_messages.cpp; stream machinery in handlers_chat_core/stream.cpp.

#include "runtime/engine.h"
#include "handlers.h"
#include "handlers_internal.h"
#include "request_field_types.h"
#include "utils.h"
#include "completion_prompt.h"
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

void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    ChatRequestContext ctx;
    if (!parse_chat_request_params(req, res, state, ctx))
        return;
    if (!snapshot_state_and_tokenize_(res, state, ctx))
        return;
    // req.path keeps /v1/messages through the shim, so the refusal is already in its dialect.
    ctx.queued_lease = admit_queued_tokens(req.path, res, state.queued_tokens, ctx.snap.n_prompt_tokens,
                                           state.max_queued_tokens);
    if (!ctx.queued_lease)
        return;

    // Save input tokens for potential reuse with n > 1
    std::vector<int32_t> saved_tokens = ctx.snap.tokens;

    // Create first request
    auto imp_req = build_imp_request_(ctx, saved_tokens, /*completion_idx=*/0, ctx.params.stream);

    // Create a ServerRequest wrapper and submit to the batching engine
    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = imp_req;
    server_req->queued_lease = ctx.queued_lease;

    // Vision requests are now per-request (req->image, encoded by the worker on
    // admission) and flow through the normal batching path below — no blocking
    // C-API fallback, no engine pause.

    // Submit to batching engine for continuous batching
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (!state.batching || !state.batching->is_running()) {
            res.status = 503;
            json err = {
                {"error",
                 {{"message", "Inference engine not ready. Please retry."}, {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }
        state.batching->submit(server_req);
    }

    std::string comp_id = ctx.req_id;
    int64_t created = unix_timestamp();

    if (ctx.params.stream) {
        stream_chat_response_(res, state, ctx, server_req);
    } else {
        nonstream_chat_response_(res, state, ctx, imp_req, server_req, saved_tokens, comp_id, created);
    }
}


// POST /v1/messages/count_tokens: runs the real request chain (convert, parse, snapshot,
// tokenize) without submitting to the engine; returns {"input_tokens": N}. Used by Claude Code
// for context tracking/auto-compaction.
void send_capacity_error_(httplib::Response& res, const ServerState& state, bool recurrent) {
    // floored: the pool fell back to its rescue floor (a few hundred tokens) - "shorten the prompt"
    // is not actionable there, it's a startup fault. Name which situation this is.
    const bool floored = state.ctx && state.ctx->engine && state.ctx->engine->kv_pool_floored();
    const char* msg =
        recurrent ? "No recurrent-state slot could be committed for this request: the card has no VRAM left "
                    "above the allocator headroom (a lazily loaded vision tower or another tenant took it). "
                    "Retry after other requests finish, or give the server more VRAM; the prompt length is "
                    "not the cause."
        : floored ? "The KV pool fell back to its rescue floor at startup, so it holds only "
                    "a few hundred tokens. This lasts as long as the process and retrying "
                    "will not help: restart the server on a free card. GET /health reports "
                    "code kv_pool_floored and the exact capacity."
                  : "Request does not fit the KV cache: the prompt needs more blocks than "
                    "the pool can hold. Shorten the prompt, lower --max-seq-len, or give "
                    "the server more VRAM (see the engine log for the exact block counts).";
    send_json_error(res, 503, "capacity_error", msg, /*param=*/nullptr,
                    recurrent ? "recurrent_state_unavailable" : floored ? "kv_pool_floored" : "context_length_exceeded");
}

void handle_count_tokens(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    namespace anth = imp_server::anthropic;

    auto send_anthropic_error = [&](int status, const char* type, const std::string& message) {
        res.status = status;
        json err = {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
        res.set_content(dump_safe(err), "application/json");
    };

    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return;

    json anth_body;
    try {
        anth_body = json::parse(req.body);
        drop_null_fields(anth_body);
    } catch (const std::exception& e) {
        send_anthropic_error(400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }
    if (!anth_body.is_object()) {
        send_anthropic_error(400, "invalid_request_error", "Request body must be a JSON object");
        return;
    }

    json oai_body;
    try {
        oai_body = anth::anthropic_to_openai_body(anth_body);
    } catch (const std::exception& e) {
        const std::string field = wrong_field_type_message(req.body);
        send_anthropic_error(400, "invalid_request_error",
                             field.empty() ? std::string("Failed to transform Anthropic body: ") + e.what() : field);
        return;
    }

    // Reuse the chat parsing + tokenize chain via a shim request; the inner
    // handlers write OpenAI-shaped errors into shim_res, re-wrapped below.
    httplib::Request shim_req = req;
    shim_req.body = dump_safe(oai_body);
    shim_req.headers.erase("Content-Length");
    shim_req.headers.erase("content-length");

    ChatRequestContext ctx;
    httplib::Response shim_res;
    g_in_anthropic_shim = true;  // suppress inner request-log entries
    bool ok = parse_chat_request_params(shim_req, shim_res, state, ctx) &&
              snapshot_state_and_tokenize_(shim_res, state, ctx);
    g_in_anthropic_shim = false;

    if (!ok) {
        // Tokenization itself may have succeeded with only a post-tokenize
        // limit check failing (context window / --max-input-tokens). Counting
        // is exactly what such callers need — report the count anyway.
        if (ctx.snap.n_prompt_tokens > 0) {
            res.status = 200;
            res.set_content(dump_safe(json{{"input_tokens", ctx.snap.n_prompt_tokens}}),
                            "application/json");
            return;
        }
        res.status = shim_res.status >= 400 ? shim_res.status : 400;
        json parsed;
        try {
            parsed = json::parse(shim_res.body);
        } catch (...) {
            parsed = {{"error", {{"message", shim_res.body}, {"type", "invalid_request_error"}}}};
        }
        json out = {{"type", "error"},
                    {"error", parsed.value("error", json{{"type", "invalid_request_error"},
                                                         {"message", "bad request"}})}};
        res.set_content(dump_safe(out), "application/json");
        return;
    }

    res.status = 200;
    res.set_content(dump_safe(json{{"input_tokens", ctx.snap.n_prompt_tokens}}), "application/json");
}
