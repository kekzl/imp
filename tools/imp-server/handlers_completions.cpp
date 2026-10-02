// Text completion endpoints: handle_completions (/v1/completions) and handle_infill (POST /infill,
// fill-in-the-middle, #2201). Moved out of handlers_chat.cpp (file-size gate, 800 code LOC).

#include "runtime/engine.h"
#include "handlers.h"
#include "handlers_internal.h"
#include "request_field_types.h"
#include "utils.h"
#include "completion_prompt.h"
#include "prompt_logprobs.h"
#include "fim_request.h"
#include "completions_params.h"
#include "stream_pipeline.h"
#include "reasoning_split.h"

#include "runtime/request.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

// CompletionCtx mirrors ChatRequestContext (handlers_internal.h) so /v1/completions parse/stream/
// response are three phases instead of one 593-LOC body (#1905 file-size gate).
struct CompletionCtx {
    const std::string& prompt;
    const std::vector<std::string>& stop_sequences;
    const std::string& comp_id;
    const std::string& snap_model_name;
    imp::Tokenizer* snap_tok;
    std::chrono::high_resolution_clock::time_point t_start;
    int64_t created;
    int n_prompt_tokens;
    int remaining;
    int32_t snap_channel_open_id;
    size_t max_stop_len;
    bool echo;
    bool include_usage;
    bool req_logprobs;
    bool snap_is_think_model;
    bool ignore_eos;
    bool infill;                        // POST /infill: llama.cpp `content` mirrors the text (#2201)
    PromptLogprobsRequest plp;
    std::function<bool()> client_gone;  // req.is_connection_closed
};

// SSE path. The body is the move-verbatim contents of the old `if (stream)`
// branch; the aliases below keep it byte-identical rather than re-spelling
// every local as `c.x`, which would have rewritten lines nobody is changing.
void stream_completion_response_(httplib::Response& res, ServerState& state, const CompletionCtx& c,
                                 const std::shared_ptr<ServerRequest>& server_req) {
    const std::string& prompt = c.prompt;
    const std::vector<std::string>& stop_sequences = c.stop_sequences;
    const std::string& comp_id = c.comp_id;
    const std::string& snap_model_name = c.snap_model_name;
    imp::Tokenizer* snap_tok = c.snap_tok;
    const auto t_start = c.t_start;
    const int64_t created = c.created;
    const int n_prompt_tokens = c.n_prompt_tokens;
    const size_t max_stop_len = c.max_stop_len;
    const bool echo = c.echo;
    const bool include_usage = c.include_usage;
    const bool req_logprobs = c.req_logprobs;
    const bool snap_is_think_model = c.snap_is_think_model;
    const bool ignore_eos = c.ignore_eos;
    const bool infill = c.infill;
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_chunked_content_provider(
        "text/event-stream",
        [&state, server_req, comp_id, created, n_prompt_tokens, t_start, stop_sequences, max_stop_len,
         ignore_eos, echo, prompt, include_usage, snap_tok, snap_model_name, snap_is_think_model,
         req_logprobs, infill](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            int n_output_tokens = 0;
            const char* finish = nullptr;

            // Echo prompt as first chunk if requested
            if (echo && !prompt.empty()) {
                std::string chunk = sse_completion_chunk(comp_id, created, snap_model_name, prompt, nullptr,
                                                         nullptr, infill);
                sink.write(chunk.data(), chunk.size());
            }

            std::string utf8_buf;
            std::string pending_text;
            bool text_stop_matched = false;

            // think_strip starts false for /v1/completions (no chat template -> no injected <think>) so a
            // raw prompt streams incrementally instead of buffering for a </think> that never comes (#760).
            // Flips true only if a real <think> opener appears within kThinkScanLimit tokens.
            bool think_strip = (snap_is_think_model && state.default_args.reasoning_format != "none");
            bool think_confirmed = false;
            std::string think_buf;
            int think_tokens = 0;
            const int kThinkScanLimit = 8;

            // #1589: emit one SSE chunk per token when logprobs are requested - a chunk carrying two tokens
            // has nowhere to put two offsets.
            imp::stream::TokenSpans pending_spans;
            imp::stream::TokenSpans utf8_spans;
            // think_buf must track spans the same way pending_spans does: a completion shorter than the
            // 8-token think-scan window never leaves the buffer mid-loop, so attribution must flush at end.
            imp::stream::TokenSpans think_spans;
            std::vector<imp::stream::TokenSpans::Emit> carried_spans;
            size_t completion_offset = 0;  // byte offset of the next token in the completion

            auto emit_completion_piece = [&](const std::string& piece_text, int token_index) {
                json lp_obj = nullptr;
                if (req_logprobs && token_index >= 0) {
                    const auto& lps = server_req->request->output_logprobs;
                    if (static_cast<size_t>(token_index) < lps.size()) {
                        lp_obj = completions_logprobs_json_one(lps[static_cast<size_t>(token_index)],
                                                               completion_offset);
                    }
                }
                completion_offset += piece_text.size();
                std::string sse = sse_completion_chunk(comp_id, created, snap_model_name, piece_text, nullptr,
                                                       lp_obj, infill);
                return sink.write(sse.data(), sse.size());
            };

            auto flush_text = [&](size_t up_to) {
                if (up_to == 0)
                    return true;
                for (const auto& e : pending_spans.flush(up_to)) {
                    if (!emit_completion_piece(pending_text.substr(e.offset, e.length), e.token_index))
                        return false;
                }
                pending_text = pending_text.substr(up_to);
                return true;
            };

            auto request_start_c = std::chrono::steady_clock::now();
            auto last_keepalive_c = request_start_c;
            double ttft_ms = -1.0;
            auto t_prev_token = t_start;  // last delivered token (ITL)
            for (;;) {
                // #757: is_last sets `finish` then falls into think-stripping's `continue`, bypassing the
                // trailing `if (finish) break` - a think-capable model whose last token lands in the think
                // buffer spun on pop_token forever (0 bytes, never terminates). Break here to flush and send
                // [DONE].
                if (finish)
                    break;

                // Check client disconnect
                if (!sink.is_writable()) {
                    server_req->cancel();
                    state.metrics.requests_cancelled++;
                    state.metrics.observe_unadmitted_queue_wait(server_req->t_submit,
                                                                server_req->queue_ms.load(
                                                                    std::memory_order_relaxed));
                    finish = "cancelled";
                    break;
                }

                // Check request timeout
                if (state.request_timeout > 0) {
                    auto elapsed = std::chrono::steady_clock::now() - request_start_c;
                    if (elapsed > std::chrono::seconds(state.request_timeout)) {
                        server_req->cancel();
                        state.metrics.requests_timed_out++;
                        state.metrics.observe_unadmitted_queue_wait(server_req->t_submit,
                                                                    server_req->queue_ms.load(
                                                                        std::memory_order_relaxed));
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

                // First token closes TTFT/queue-wait histograms; every later token is an ITL sample. This
                // loop
                // fed none of the four before, so all /v1/completions traffic left them empty.
                {
                    const auto t_tok = std::chrono::high_resolution_clock::now();
                    if (ttft_ms < 0.0) {
                        ttft_ms = std::chrono::duration<double, std::milli>(t_tok - t_start).count();
                        const double q = server_req->queue_ms.load(std::memory_order_relaxed);
                        if (q >= 0.0)
                            state.metrics.record_queue_wait("/v1/completions", q / 1000.0);
                    } else {
                        state.metrics.record_inter_token(
                            "/v1/completions", std::chrono::duration<double>(t_tok - t_prev_token).count());
                    }
                    t_prev_token = t_tok;
                }

                if (evt.is_last) {
                    if (token == snap_tok->eos_id() && !ignore_eos) {
                        finish = evt.finish_reason ? evt.finish_reason : "stop";
                        break;
                    }
                    finish = evt.finish_reason ? evt.finish_reason : "length";
                }
                if (ignore_eos && token == snap_tok->eos_id()) {
                    n_output_tokens++;  // counted, no text
                    if (evt.is_last)
                        break;
                    continue;
                }

                n_output_tokens++;
                std::string piece = snap_tok->decode_token(token);

                // Strip <think>...</think> block for text completions
                if (think_strip) {
                    think_buf += piece;
                    think_spans.append(piece.size(), n_output_tokens - 1);
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
                        // Everything before </think> is dropped, so its
                        // attribution goes with it; what follows belongs to
                        // the token that closed the block.
                        think_spans.clear();
                        auto start = after.find_first_not_of("\n\r\t ");
                        piece = (start != std::string::npos) ? after.substr(start) : "";
                        if (piece.empty())
                            continue;
                    } else if (think_confirmed) {
                        continue;
                    } else if (think_tokens < kThinkScanLimit &&
                               imp::server::StreamReasoningSplitter::could_open_marker(think_buf)) {
                        // Hold only while the text could still be the
                        // start of a marker; a first word releases it
                        // (the same trade as the chat stream's SCAN).
                        continue;
                    } else {
                        think_strip = false;
                        piece = think_buf;
                        think_buf.clear();
                        carried_spans = think_spans.flush(piece.size());
                        think_spans.clear();
                    }
                }

                if (stop_sequences.empty()) {
                    utf8_buf += piece;
                    if (carried_spans.empty()) {
                        utf8_spans.append(piece.size(), n_output_tokens - 1);
                    } else {
                        for (const auto& e : carried_spans)
                            utf8_spans.append(e.length, e.token_index);
                        carried_spans.clear();
                    }
                    size_t complete = utf8_complete_len(utf8_buf);
                    if (complete > 0) {
                        for (const auto& e : utf8_spans.flush(complete)) {
                            if (!emit_completion_piece(utf8_buf.substr(e.offset, e.length), e.token_index))
                                return false;
                        }
                        utf8_buf = utf8_buf.substr(complete);
                    }
                } else {
                    pending_text += piece;
                    if (carried_spans.empty()) {
                        pending_spans.append(piece.size(), n_output_tokens - 1);
                    } else {
                        for (const auto& e : carried_spans)
                            pending_spans.append(e.length, e.token_index);
                        carried_spans.clear();
                    }
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

            // Flush think buffer: strip think blocks and emit remaining content
            if (!think_buf.empty()) {
                const size_t before = think_buf.size();
                strip_think_block(think_buf);
                if (!think_buf.empty()) {
                    // Carry attribution across only when the strip changed nothing: once bytes are removed
                    // the
                    // recorded offsets no longer match the string, and a wrong index is worse than none.
                    if (think_buf.size() == before) {
                        for (const auto& e : think_spans.flush(before))
                            utf8_spans.append(e.length, e.token_index);
                    } else {
                        utf8_spans.append(think_buf.size(), -1);
                    }
                    utf8_buf += think_buf;
                }
                think_spans.clear();
                think_buf.clear();
            }

            // Flush remaining buffers
            if (!utf8_buf.empty() && !text_stop_matched) {
                for (const auto& e : utf8_spans.flush(utf8_buf.size()))
                    emit_completion_piece(utf8_buf.substr(e.offset, e.length), e.token_index);
            }
            if (!pending_text.empty() && !text_stop_matched)
                flush_text(pending_text.size());

            if (!finish)
                finish = "length";

            // Final chunk with finish_reason
            std::string final_chunk = sse_completion_chunk(comp_id, created, snap_model_name, "",
                                                           openai_finish_reason(finish), nullptr, infill);
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
                if (json details = prompt_tokens_details_(server_req->request, n_prompt_tokens);
                    !details.is_null())
                    usage_obj["usage"]["prompt_tokens_details"] = std::move(details);
                add_spec_usage_(usage_obj["usage"], server_req->request);
                std::string usage_chunk = "data: " + dump_safe(usage_obj) + "\n\n";
                sink.write(usage_chunk.data(), usage_chunk.size());
            }

            std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();

            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            // Same shape as the chat driver's line: ttft and the admission wait
            // are what a TTFT attribution reads from the log.
            IMP_LOG_INFO("[%s] %d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms, queue=%.1f ms)",
                         comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms, ttft_ms,
                         server_req->queue_ms.load(std::memory_order_relaxed));
            state.metrics.record_completion("/v1/completions", ms, ttft_ms, n_prompt_tokens, n_output_tokens);

            return true;
        });
}

// Blocking path: collect the whole completion, then one JSON response.
void nonstream_completion_response_(httplib::Response& res, ServerState& state, const CompletionCtx& c,
                                    const std::shared_ptr<ServerRequest>& server_req) {
    const std::string& prompt = c.prompt;
    const std::vector<std::string>& stop_sequences = c.stop_sequences;
    const std::string& comp_id = c.comp_id;
    const std::string& snap_model_name = c.snap_model_name;
    imp::Tokenizer* snap_tok = c.snap_tok;
    const auto t_start = c.t_start;
    const int64_t created = c.created;
    const int n_prompt_tokens = c.n_prompt_tokens;
    const int32_t snap_channel_open_id = c.snap_channel_open_id;
    const bool echo = c.echo;
    const bool req_logprobs = c.req_logprobs;
    const bool snap_is_think_model = c.snap_is_think_model;
    const bool ignore_eos = c.ignore_eos;
    int n_eos_counted = 0;
    // Non-streaming
    auto active_req = server_req->request;
    std::vector<int32_t> output_ids;
    const char* finish = nullptr;
    std::string output_text;

    auto ns_comp_start = std::chrono::steady_clock::now();
    double ttft_ms = -1.0;
    auto t_prev_token = t_start;  // last delivered token (ITL)
    for (;;) {
        finish = nonstream_should_stop_(state, *server_req, ns_comp_start, c.client_gone);
        if (finish)
            break;

        TokenEvent evt{};
        if (!server_req->pop_token(evt)) {
            continue;
        }

        if (evt.token_id < 0) {
            finish = evt.finish_reason ? evt.finish_reason : "stop";
            break;
        }

        int32_t token = evt.token_id;

        {
            const auto t_tok = std::chrono::high_resolution_clock::now();
            if (ttft_ms < 0.0) {
                ttft_ms = std::chrono::duration<double, std::milli>(t_tok - t_start).count();
                const double q = server_req->queue_ms.load(std::memory_order_relaxed);
                if (q >= 0.0)
                    state.metrics.record_queue_wait("/v1/completions", q / 1000.0);
            } else {
                state.metrics.record_inter_token("/v1/completions",
                                                 std::chrono::duration<double>(t_tok - t_prev_token).count());
            }
            t_prev_token = t_tok;
        }

        if (evt.is_last) {
            if (token == snap_tok->eos_id() && !ignore_eos) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            finish = evt.finish_reason ? evt.finish_reason : "length";
        }
        if (ignore_eos && token == snap_tok->eos_id()) {
            n_eos_counted++;  // counted in usage, kept out of the text
            if (evt.is_last)
                break;
            continue;
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

    int n_output_tokens = static_cast<int>(output_ids.size()) + n_eos_counted;
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
    IMP_LOG_INFO("[%s] %d prompt + %d completion tokens, %.1f ms", comp_id.c_str(), n_prompt_tokens,
                 n_output_tokens, ms);
    state.metrics.record_completion("/v1/completions", ms, ttft_ms, n_prompt_tokens, n_output_tokens);

    // #1589: build the Completions logprobs shape here (not the Chat shape) - an OpenAI SDK reading
    // `.logprobs.tokens` on /v1/completions cannot see the Chat object.
    json logprobs_obj = nullptr;
    if (req_logprobs && active_req) {
        logprobs_obj = completions_logprobs_json(active_req->output_logprobs, output_ids.size(), text);
    }

    json choice = {{"index", 0}, {"text", text}, {"finish_reason", openai_finish_reason(finish)}};
    if (!logprobs_obj.is_null()) {
        choice["logprobs"] = logprobs_obj;
    }
    const auto token_text = [snap_tok](int32_t id) { return snap_tok->decode_token(id); };
    if (active_req &&
        !attach_prompt_logprobs(choice, c.plp, *active_req, token_text, output_ids.size(), text)) {
        send_json_error(res, 500, "server_error",
                        "prompt logprobs incomplete: not every prompt token was scored");
        return;
    }

    json response = {{"id", comp_id},
                     {"object", "text_completion"},
                     {"created", created},
                     {"model", snap_model_name},
                     {"system_fingerprint", system_fingerprint(snap_model_name)},
                     {"choices", json::array({choice})},
                     {"usage",
                      {{"prompt_tokens", n_prompt_tokens},
                       {"completion_tokens", n_output_tokens},
                       {"total_tokens", n_prompt_tokens + n_output_tokens}}}};
    if (json details = prompt_tokens_details_(active_req, n_prompt_tokens); !details.is_null())
        response["usage"]["prompt_tokens_details"] = std::move(details);
    add_spec_usage_(response["usage"], active_req);
    if (c.infill) {
        response["content"] = text;
        response["stop"] = true;
    }

    res.set_content(dump_safe(response), "application/json");
}

// /v1/completions and POST /infill (#2201) share one path; `infill` switches the prompt fields.
void completions_impl_(const httplib::Request& req, httplib::Response& res, ServerState& state, bool infill) {
    CompletionRequestParams p;
    if (!parse_completions_request_params(req, res, state, infill, p))
        return;
    std::string& prompt = p.prompt;
    const std::vector<int32_t>& prompt_ids = p.prompt_ids;
    FimRequest& fim = p.fim;
    std::vector<std::string>& stop_sequences = p.stop_sequences;
    size_t max_stop_len = p.max_stop_len;
    int max_tokens = p.max_tokens;
    std::string requested_model = p.requested_model;
    const std::string& req_id = p.req_id;

    // Snapshot state fields under lock for thread-safe access
    imp::Tokenizer* snap_tok;
    std::string snap_model_name;
    bool snap_is_think_model;
    int32_t snap_channel_open_id;
    int snap_max_seq_len;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (requested_model.empty())
            requested_model = state.model_name;
        if (!ensure_model_loaded(state, requested_model, res))
            return;
        snap_tok = state.tok;
        snap_model_name = state.model_name;
        snap_is_think_model = state.is_think_model;
        snap_channel_open_id = state.channel_open_id;
        snap_max_seq_len = state.max_seq_len;
    }
    if (const std::string err = logit_bias_vocab_error(p.logit_bias, snap_tok->vocab_size()); !err.empty()) {
        send_json_error(res, 400, "invalid_request_error", err, "logit_bias");
        return;
    }

    // The byte bound first: the merge walk over the prompt is the cost the
    // token check below arrives too late for (AUDIT_arch_2026 F2-9).
    if (!prompt_within_input_budget(res, fim.active ? fim.bytes() : prompt.size(), state.max_input_tokens,
                                    "prompt"))
        return;

    // Tokenize raw prompt (no chat template); token ids go in as given, echo reads their text.
    for (const int32_t id : prompt_ids) {
        if (id >= snap_tok->vocab_size()) {
            send_json_error(res, 400, "invalid_request_error",
                            "prompt token id " + std::to_string(id) + " is outside the vocabulary (" +
                                std::to_string(snap_tok->vocab_size()) + ")",
                            "prompt");
            return;
        }
    }
    if (!prompt_ids.empty())
        prompt = snap_tok->decode(prompt_ids);
    std::vector<int32_t> tokens;
    if (fim.active) {
        // FIM prompt from the tokenizer's FIM tokens; their texts end the generation.
        if (const std::string err = build_fim_request_tokens(fim, *snap_tok, tokens, stop_sequences);
            !err.empty()) {
            send_json_error(res, 400, "invalid_request_error", err, infill ? nullptr : "suffix",
                            "fim_not_supported");
            return;
        }
        for (const auto& s : stop_sequences)
            max_stop_len = std::max(max_stop_len, s.size());
    } else {
        tokens = prompt_ids.empty() ? snap_tok->encode(prompt) : prompt_ids;
    }
    // A raw text prompt gets the BOS the tokenizer asks for (add_bos_token; llama.cpp and vLLM do the
    // same); the chat path gets it from its template. Without it Gemma-4 answered "The capital of
    // France is" with " is is is ...". Token-id prompts are taken as given.
    if (!fim.active && prompt_ids.empty() && snap_tok->add_bos() && snap_tok->bos_id() >= 0 &&
        (tokens.empty() || tokens.front() != snap_tok->bos_id()))
        tokens.insert(tokens.begin(), snap_tok->bos_id());
    int n_prompt_tokens = static_cast<int>(tokens.size());

    // Server-side input-token limit (--max-input-tokens). Reject pre-prefill.
    if (state.max_input_tokens > 0 && n_prompt_tokens > state.max_input_tokens) {
        send_json_error(res, 400, "invalid_request_error",
                        "Prompt exceeds max input tokens (" + std::to_string(n_prompt_tokens) + " > " +
                            std::to_string(state.max_input_tokens) + ")",
                        "prompt", "context_length_exceeded");
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
    imp_req->trace_id = req_id;  // #1582: join the engine's log lines to this request
    imp_req->input_tokens = std::move(tokens);
    imp_req->max_tokens = max_tokens;
    p.apply_to(*imp_req);
    imp_req->seed = p.seed;
    imp_req->logprobs = p.req_logprobs;
    imp_req->top_logprobs = p.top_logprobs;
    imp_req->prompt_logprobs = p.plp.engine_top_n;
    // With ignore_eos the engine keeps sampling past EOS; every EOS it emits
    // counts as an output token (vLLM semantics) but carries no text.
    const bool ignore_eos = imp_req->ignore_eos;
    imp_req->logit_bias = std::move(p.logit_bias);
    apply_spec_contract_(*imp_req, p.spec_override, p.spec_mtp_k,
                         state.armed_mtp_k.load(std::memory_order_relaxed),
                         state.mtp_head_present.load(std::memory_order_relaxed),
                         state.mtp_head_loaded.load(std::memory_order_relaxed));
    // Predicted Outputs (string-content form): seeds the n-gram draft corpus only, output unchanged.
    if (p.has_prediction) {
        imp_req->prediction_tokens = snap_tok->encode(p.prediction_text);
        if (snap_max_seq_len > 0 && imp_req->prediction_tokens.size() > static_cast<size_t>(snap_max_seq_len))
            imp_req->prediction_tokens.resize(snap_max_seq_len);
    }
    // Stream requests stay on per-step decode for real per-token SSE (#754).
    imp_req->stream = p.stream;
    imp_req->status = imp::RequestStatus::PENDING;

    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = std::move(imp_req);

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

    const CompletionCtx cctx{prompt,
                             stop_sequences,
                             comp_id,
                             snap_model_name,
                             snap_tok,
                             t_start,
                             created,
                             n_prompt_tokens,
                             remaining,
                             snap_channel_open_id,
                             max_stop_len,
                             p.echo,
                             p.include_usage,
                             p.req_logprobs,
                             snap_is_think_model,
                             ignore_eos,
                             infill,
                             p.plp,
                             req.is_connection_closed};
    if (p.stream) {
        stream_completion_response_(res, state, cctx, server_req);
    } else {
        nonstream_completion_response_(res, state, cctx, server_req);
    }
}

}  // namespace

void handle_completions(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    completions_impl_(req, res, state, /*infill=*/false);
}

void handle_infill(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    completions_impl_(req, res, state, /*infill=*/true);
}
