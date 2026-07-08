// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// OpenAI chat endpoints: handle_chat_completions (/v1/chat/completions),
// handle_completions (/v1/completions), plus the Anthropic
// handle_count_tokens (/v1/messages/count_tokens). The handle_messages
// (/v1/messages) endpoint lives in handlers_messages.cpp next to its streaming
// loop. Streaming/parse machinery lives in handlers_chat_core.cpp /
// handlers_chat_stream.cpp / handlers_messages.cpp.

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"

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

    // Save input tokens for potential reuse with n > 1
    std::vector<int32_t> saved_tokens = ctx.snap.tokens;

    // Helper to create an imp::Request with the given completion index
    auto make_imp_request = [&](int completion_idx) {
        auto req = std::make_shared<imp::Request>();
        req->image = ctx.snap.vision_image;  // per-request vision (null for text)
        req->input_tokens = saved_tokens;
        req->max_tokens = ctx.params.max_tokens;
        req->temperature = ctx.params.temperature;
        req->top_p = ctx.params.top_p;
        req->top_k = ctx.params.top_k;
        req->seed = (ctx.params.seed != -1) ? ctx.params.seed + completion_idx : -1;
        req->pin_kv_prefix = ctx.params.cache_prompt;
        req->spec_ngram_override = ctx.params.spec_ngram_override;
        req->prediction_tokens = ctx.snap.prediction_tokens;
        req->min_p = ctx.params.min_p;
        req->typical_p = ctx.params.typical_p;
        req->repetition_penalty = ctx.params.repetition_penalty;
        req->frequency_penalty = ctx.params.frequency_penalty;
        req->presence_penalty = ctx.params.presence_penalty;
        req->repeat_last_n = ctx.params.repeat_last_n;
        req->dry_multiplier = ctx.params.dry_multiplier;
        req->dry_base = ctx.params.dry_base;
        req->dry_allowed_length = ctx.params.dry_allowed_length;
        req->dry_penalty_last_n = ctx.params.dry_penalty_last_n;
        req->mirostat = ctx.params.mirostat;
        req->mirostat_tau = ctx.params.mirostat_tau;
        req->mirostat_eta = ctx.params.mirostat_eta;
        req->logprobs = ctx.params.req_logprobs;
        req->top_logprobs = ctx.params.top_logprobs;
        req->json_mode = ctx.params.json_mode;
        req->json_schema = ctx.params.json_schema_str;
        req->has_tools = ctx.params.has_tools;
        req->tpl_family = ctx.snap.tpl_family;
        req->logit_bias = ctx.params.logit_bias;
        req->think_budget = ctx.params.think_budget;
        // Generation starts INSIDE the think block when the prompt carries the
        // <think> prefix (template-injected or server-appended). Without this
        // the engine's think-budget enforcement never sees an opener in the
        // output, counts zero reasoning tokens, and lets the model think until
        // max_tokens (content empty, finish=length).
        req->started_in_think = ctx.snap.enable_thinking;
        req->in_think_block = ctx.snap.enable_thinking;
        // Stream requests stay on per-step decode for real per-token SSE (#754).
        req->stream = ctx.params.stream;
        req->status = imp::RequestStatus::PENDING;
        return req;
    };

    // Create first request
    auto imp_req = make_imp_request(0);

    // Create a ServerRequest wrapper and submit to the batching engine
    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = imp_req;

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

void handle_completions(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    // Validate sampling parameters
    if (!validate_sampling_params(body, res))
        return;

    // /v1/completions does not implement multi-choice generation. Reject n>1
    // explicitly instead of validating n in [1,4] and then silently returning a
    // single choice (only the chat endpoint honors n, via n_completions).
    if (body.value("n", 1) > 1) {
        send_json_error(res, 400, "invalid_request_error",
                        "n>1 is not supported on /v1/completions; request one completion per call");
        return;
    }

    // Extract prompt
    std::string prompt = body.value("prompt", "");
    if (prompt.empty()) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "\"prompt\" is required and must not be empty"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Extract parameters
    float temperature = body.value("temperature", 0.7f);
    float top_p = body.value("top_p", 0.95f);
    int top_k = body.value("top_k", 40);
    int max_tokens = body.value("max_tokens", state.default_max_tokens);
    int seed = body.value("seed", -1);
    bool stream = body.value("stream", false);
    bool echo = body.value("echo", false);
    float min_p = body.value("min_p", 0.0f);
    float typical_p = body.value("typical_p", 1.0f);
    float repetition_penalty = body.value("repetition_penalty", 1.05f);
    float frequency_penalty = body.value("frequency_penalty", 0.0f);
    float presence_penalty = body.value("presence_penalty", 0.0f);
    int repeat_last_n = body.value("repeat_last_n", 0);
    float dry_multiplier = body.value("dry_multiplier", 0.0f);
    float dry_base = body.value("dry_base", 1.75f);
    int dry_allowed_length = body.value("dry_allowed_length", 2);
    int dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    int mirostat = body.value("mirostat", 0);
    float mirostat_tau = body.value("mirostat_tau", 5.0f);
    float mirostat_eta = body.value("mirostat_eta", 0.1f);

    // Completions API types `logprobs` as an integer (top-N count); Chat uses a
    // bool `logprobs` + int `top_logprobs`. Accept both so a spec-compliant
    // Completions client sending `logprobs: 5` isn't 400'd on a json type error.
    bool req_logprobs = false;
    int top_logprobs = body.value("top_logprobs", 0);
    if (body.contains("logprobs") && !body["logprobs"].is_null()) {
        const auto& lp = body["logprobs"];
        if (lp.is_boolean()) {
            req_logprobs = lp.get<bool>();
        } else if (lp.is_number_integer()) {
            int n = lp.get<int>();
            if (n > 0) {
                req_logprobs = true;
                top_logprobs = std::max(top_logprobs, n);
            }
        } else {
            send_json_error(res, 400, "invalid_request_error",
                            "\"logprobs\" must be an integer (Completions) or boolean");
            return;
        }
    }
    if (top_logprobs < 0)
        top_logprobs = 0;
    if (top_logprobs > 20)
        top_logprobs = 20;

    // Parse stop sequences (same 16-entry cap as the chat parser).
    std::vector<std::string> stop_sequences;
    if (parse_stop_field(body, 16, stop_sequences)) {
        fprintf(stderr, "warning: request sent %zu stop sequences; keeping the first 16\n",
                body["stop"].size());
    }
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());

    // Parse logit_bias: map of token_id (string) -> bias (float)
    std::vector<std::pair<int32_t, float>> logit_bias;
    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
        for (auto& [key, val] : body["logit_bias"].items()) {
            try {
                int32_t token_id = std::stoi(key);
                float bias = val.get<float>();
                logit_bias.emplace_back(token_id, bias);
            } catch (...) {
                // Skip invalid entries
            }
        }
    }

    // Parse stream_options for include_usage
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Log request received
    std::string req_id = make_completion_id(state);
    fprintf(stderr, "[%s] completions: prompt_len=%zu stream=%s max_tokens=%d temp=%.2f\n", req_id.c_str(),
            prompt.size(), stream ? "true" : "false", max_tokens, temperature);

    // Validate model field (required per OpenAI spec)
    std::string requested_model = body.value("model", "");
    if (requested_model.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"model\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Snapshot state fields under lock for thread-safe access
    imp::Tokenizer* snap_tok;
    std::string snap_model_name;
    bool snap_is_think_model;
    int32_t snap_channel_open_id;
    int snap_max_seq_len;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (!ensure_model_loaded(state, requested_model, res))
            return;
        snap_tok = state.tok;
        snap_model_name = state.model_name;
        snap_is_think_model = state.is_think_model;
        snap_channel_open_id = state.channel_open_id;
        snap_max_seq_len = state.max_seq_len;
    }

    // Tokenize raw prompt (no chat template)
    std::vector<int32_t> tokens = snap_tok->encode(prompt);
    int n_prompt_tokens = static_cast<int>(tokens.size());

    // Server-side input-token limit (--max-input-tokens). Reject pre-prefill.
    if (state.max_input_tokens > 0 && n_prompt_tokens > state.max_input_tokens) {
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds max input tokens (" + std::to_string(n_prompt_tokens) +
                                        " > " + std::to_string(state.max_input_tokens) + ")"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    if (n_prompt_tokens >= snap_max_seq_len) {
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds context window (" + std::to_string(n_prompt_tokens) +
                                        " tokens >= " + std::to_string(snap_max_seq_len) + " max)"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    int remaining = snap_max_seq_len - n_prompt_tokens;
    if (max_tokens > remaining)
        max_tokens = remaining;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Create an imp::Request and submit to batching engine (/v1/completions is
    // text-only — no vision).
    auto imp_req = std::make_shared<imp::Request>();
    imp_req->input_tokens = std::move(tokens);
    imp_req->max_tokens = max_tokens;
    imp_req->temperature = temperature;
    imp_req->top_p = top_p;
    imp_req->top_k = top_k;
    imp_req->seed = seed;
    imp_req->min_p = min_p;
    imp_req->typical_p = typical_p;
    imp_req->repetition_penalty = repetition_penalty;
    imp_req->frequency_penalty = frequency_penalty;
    imp_req->presence_penalty = presence_penalty;
    imp_req->repeat_last_n = repeat_last_n;
    imp_req->dry_multiplier = dry_multiplier;
    imp_req->dry_base = dry_base;
    imp_req->dry_allowed_length = dry_allowed_length;
    imp_req->dry_penalty_last_n = dry_penalty_last_n;
    imp_req->mirostat = mirostat;
    imp_req->mirostat_tau = mirostat_tau;
    imp_req->mirostat_eta = mirostat_eta;
    imp_req->logprobs = req_logprobs;
    imp_req->top_logprobs = top_logprobs;
    imp_req->logit_bias = std::move(logit_bias);
    imp_req->think_budget = body.value("think_budget", state.default_think_budget);
    imp_req->pin_kv_prefix = body.value("cache_prompt", false);
    if (body.contains("speculative") && body["speculative"].is_boolean())
        imp_req->spec_ngram_override = body["speculative"].get<bool>() ? 1 : 0;
    // Predicted Outputs (string-content form) on the completions route: the
    // prediction only seeds the n-gram draft corpus, output is unchanged.
    if (body.contains("prediction") && body["prediction"].is_object()) {
        const auto& pred = body["prediction"];
        if (pred.value("type", "content") == "content" && pred.contains("content") &&
            pred["content"].is_string()) {
            imp_req->prediction_tokens = snap_tok->encode(pred["content"].get<std::string>());
            if (snap_max_seq_len > 0 &&
                imp_req->prediction_tokens.size() > static_cast<size_t>(snap_max_seq_len))
                imp_req->prediction_tokens.resize(snap_max_seq_len);
        }
    }
    // Stream requests stay on per-step decode for real per-token SSE (#754).
    imp_req->stream = stream;
    imp_req->status = imp::RequestStatus::PENDING;

    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = imp_req;

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

    std::string comp_id = req_id;
    int64_t created = unix_timestamp();

    if (stream) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, server_req, comp_id, created, n_prompt_tokens, t_start, stop_sequences,
             max_stop_len, echo, prompt, include_usage, snap_tok, snap_model_name,
             snap_is_think_model](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                int n_output_tokens = 0;
                const char* finish = nullptr;

                // Echo prompt as first chunk if requested
                if (echo && !prompt.empty()) {
                    std::string chunk = sse_completion_chunk(comp_id, created, snap_model_name, prompt,
                                                             nullptr);
                    sink.write(chunk.data(), chunk.size());
                }

                std::string utf8_buf;
                std::string pending_text;
                bool text_stop_matched = false;

                // Strip <think> blocks for completions (no reasoning_content field).
                // think_confirmed starts FALSE so a raw /v1/completions prompt
                // (no chat template → no injected <think>) streams incrementally
                // instead of buffering every token into think_buf waiting for a
                // </think> that never comes (#760: completions stream arrived as
                // one frame). It flips true only if a real <think> opener shows
                // up in the first kThinkScanLimit tokens, so genuine think blocks
                // are still stripped.
                bool think_strip = (snap_is_think_model && state.default_args.reasoning_format != "none");
                bool think_confirmed = false;
                std::string think_buf;
                int think_tokens = 0;
                const int kThinkScanLimit = 8;

                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0)
                        return true;
                    std::string to_send = pending_text.substr(0, up_to);
                    pending_text = pending_text.substr(up_to);
                    std::string sse = sse_completion_chunk(comp_id, created, snap_model_name, to_send,
                                                           nullptr);
                    return sink.write(sse.data(), sse.size());
                };

                auto request_start_c = std::chrono::steady_clock::now();
                auto last_keepalive_c = request_start_c;
                for (;;) {
                    // #757: the is_last token sets `finish` then falls through
                    // to think-stripping, which `continue`s on every swallowed
                    // token — bypassing the trailing `if (finish) break`. For a
                    // think-capable model whose final token lands inside the
                    // think buffer the loop would otherwise spin on pop_token
                    // until the client gives up (0 bytes, never terminates).
                    // Break here so the buffers flush and [DONE] is sent.
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
                        auto elapsed = std::chrono::steady_clock::now() - request_start_c;
                        if (elapsed > std::chrono::seconds(state.request_timeout)) {
                            server_req->cancel();
                            finish = "length";
                            break;
                        }
                    }

                    TokenEvent evt{};
                    if (!server_req->pop_token(evt)) {
                        // SSE comment keepalive while waiting (long prefill /
                        // queueing) — ignored by SSE parsers, keeps proxies
                        // and SDK idle-timeouts from killing the connection.
                        auto now = std::chrono::steady_clock::now();
                        if (now - last_keepalive_c > std::chrono::seconds(10)) {
                            last_keepalive_c = now;
                            static constexpr char kKeepalive[] = ": keepalive\n\n";
                            sink.write(kKeepalive, sizeof(kKeepalive) - 1);
                        }
                        continue;
                    }

                    if (evt.token_id < 0) {
                        finish = evt.finish_reason ? evt.finish_reason : "stop";
                        break;
                    }

                    int32_t token = evt.token_id;

                    if (evt.is_last) {
                        if (token == snap_tok->eos_id()) {
                            finish = evt.finish_reason ? evt.finish_reason : "stop";
                            break;
                        }
                        finish = evt.finish_reason ? evt.finish_reason : "length";
                    }

                    n_output_tokens++;
                    std::string piece = snap_tok->decode_token(token);

                    // Strip <think>...</think> block for text completions
                    if (think_strip) {
                        think_buf += piece;
                        think_tokens++;

                        if (!think_confirmed) {
                            if (think_buf.find("<think>") != std::string::npos)
                                think_confirmed = true;
                            else if (think_tokens == 1 && piece.empty())
                                think_confirmed = true;
                        }

                        auto end_pos = think_buf.find("</think>");
                        if (end_pos != std::string::npos) {
                            think_strip = false;
                            std::string after = think_buf.substr(end_pos + 8);
                            think_buf.clear();
                            auto start = after.find_first_not_of("\n\r\t ");
                            piece = (start != std::string::npos) ? after.substr(start) : "";
                            if (piece.empty())
                                continue;
                        } else if (think_confirmed) {
                            continue;
                        } else if (think_tokens < kThinkScanLimit) {
                            continue;
                        } else {
                            think_strip = false;
                            piece = think_buf;
                            think_buf.clear();
                        }
                    }

                    if (stop_sequences.empty()) {
                        utf8_buf += piece;
                        size_t complete = utf8_complete_len(utf8_buf);
                        if (complete > 0) {
                            std::string to_emit = utf8_buf.substr(0, complete);
                            utf8_buf = utf8_buf.substr(complete);
                            std::string chunk = sse_completion_chunk(comp_id, created, snap_model_name,
                                                                     to_emit, nullptr);
                            if (!sink.write(chunk.data(), chunk.size()))
                                return false;
                        }
                    } else {
                        pending_text += piece;
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

                    if (finish)
                        break;
                }

                // Flush think buffer: strip think blocks and emit remaining content
                if (!think_buf.empty()) {
                    strip_think_block(think_buf);
                    if (!think_buf.empty()) {
                        utf8_buf += think_buf;
                    }
                    think_buf.clear();
                }

                // Flush remaining buffers
                if (!utf8_buf.empty() && !text_stop_matched) {
                    std::string sse = sse_completion_chunk(comp_id, created, snap_model_name, utf8_buf,
                                                           nullptr);
                    sink.write(sse.data(), sse.size());
                }
                if (!pending_text.empty() && !text_stop_matched) {
                    std::string sse = sse_completion_chunk(comp_id, created, snap_model_name, pending_text,
                                                           nullptr);
                    sink.write(sse.data(), sse.size());
                }

                if (!finish)
                    finish = "length";

                // Final chunk with finish_reason
                std::string final_chunk = sse_completion_chunk(comp_id, created, snap_model_name, "", finish);
                sink.write(final_chunk.data(), final_chunk.size());

                // Usage chunk if requested
                if (include_usage) {
                    json usage_obj = {{"id", comp_id},
                                      {"object", "text_completion"},
                                      {"created", created},
                                      {"model", snap_model_name},
                                      {"choices", json::array()},
                                      {"usage",
                                       {{"prompt_tokens", n_prompt_tokens},
                                        {"completion_tokens", n_output_tokens},
                                        {"total_tokens", n_prompt_tokens + n_output_tokens}}}};
                    std::string usage_chunk = "data: " + dump_safe(usage_obj) + "\n\n";
                    sink.write(usage_chunk.data(), usage_chunk.size());
                }

                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();

                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n", comp_id.c_str(),
                        n_prompt_tokens, n_output_tokens, ms);
                state.metrics.requests_total++;
                state.metrics.tokens_prompt_total += n_prompt_tokens;
                state.metrics.tokens_completion_total += n_output_tokens;
                state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

                return true;
            });
    } else {
        // Non-streaming
        auto active_req = server_req->request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;

        auto ns_comp_start = std::chrono::steady_clock::now();
        for (;;) {
            if (state.request_timeout > 0) {
                auto elapsed = std::chrono::steady_clock::now() - ns_comp_start;
                if (elapsed > std::chrono::seconds(state.request_timeout)) {
                    server_req->cancel();
                    finish = "length";
                    break;
                }
            }

            TokenEvent evt{};
            if (!server_req->pop_token(evt)) {
                continue;
            }

            if (evt.token_id < 0) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }

            int32_t token = evt.token_id;

            if (evt.is_last) {
                if (token == snap_tok->eos_id()) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                finish = evt.finish_reason ? evt.finish_reason : "length";
            }

            output_ids.push_back(token);

            if (!stop_sequences.empty()) {
                output_text += snap_tok->decode_token(token);
                bool stop_found = false;
                for (const auto& stop : stop_sequences) {
                    auto pos = output_text.find(stop);
                    if (pos != std::string::npos) {
                        output_text = output_text.substr(0, pos);
                        stop_found = true;
                        break;
                    }
                }
                if (stop_found) {
                    finish = "stop";
                    break;
                }
            }

            if (finish)
                break;
        }

        if (!finish)
            finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string text = !stop_sequences.empty() ? output_text : snap_tok->decode(output_ids);

        // Strip <think>...</think> for text completions (no reasoning_content field)
        if (snap_is_think_model && state.default_args.reasoning_format != "none") {
            strip_think_block(text);
        }
        if (snap_channel_open_id >= 0) {
            strip_channel_headers(text);
        }

        // Prepend prompt if echo requested
        if (echo)
            text = prompt + text;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n", comp_id.c_str(), n_prompt_tokens,
                n_output_tokens, ms);
        state.metrics.requests_total++;
        state.metrics.tokens_prompt_total += n_prompt_tokens;
        state.metrics.tokens_completion_total += n_output_tokens;
        state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

        // Build logprobs if requested
        json logprobs_obj = nullptr;
        if (req_logprobs && active_req) {
            const auto& lp_data = active_req->output_logprobs;
            json content_logprobs = json::array();
            for (size_t idx = 0; idx < lp_data.size() && idx < output_ids.size(); idx++) {
                const auto& lp = lp_data[idx];
                json top_arr = json::array();
                for (const auto& t : lp.top) {
                    top_arr.push_back({{"token", safe_token_json(t.text)},
                                       {"logprob", t.logprob},
                                       {"bytes", token_bytes_json(t.text)}});
                }
                content_logprobs.push_back({{"token", safe_token_json(lp.text)},
                                            {"logprob", lp.logprob},
                                            {"bytes", token_bytes_json(lp.text)},
                                            {"top_logprobs", top_arr}});
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        json choice = {{"index", 0}, {"text", text}, {"finish_reason", finish}};
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        json response = {{"id", comp_id},
                         {"object", "text_completion"},
                         {"created", created},
                         {"model", snap_model_name},
                         {"choices", json::array({choice})},
                         {"usage",
                          {{"prompt_tokens", n_prompt_tokens},
                           {"completion_tokens", n_output_tokens},
                           {"total_tokens", n_prompt_tokens + n_output_tokens}}}};

        res.set_content(dump_safe(response), "application/json");
    }
}

// POST /v1/messages/count_tokens — Anthropic token-counting endpoint. Claude
// Code calls it for context tracking / auto-compaction. Runs the exact chain a
// real request would take (anthropic_to_openai_body -> common param parse ->
// state snapshot + chat-template tokenize, including tool defs and the think
// prefix) WITHOUT submitting to the engine, and returns {"input_tokens": N}.
void handle_count_tokens(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    namespace anth = imp_server::anthropic;

    auto send_anthropic_error = [&](int status, const char* type, const std::string& message) {
        res.status = status;
        json err = {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
        res.set_content(dump_safe(err), "application/json");
    };

    json anth_body;
    try {
        anth_body = json::parse(req.body);
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
        send_anthropic_error(400, "invalid_request_error",
                             std::string("Failed to transform Anthropic body: ") + e.what());
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
