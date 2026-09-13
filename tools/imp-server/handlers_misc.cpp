// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// Misc endpoints: handle_tokenize, handle_detokenize, handle_metrics, and
// handle_embeddings (with its fp16->fp32 helper).

#include "runtime/engine.h"
#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"

#include "api/imp_internal.h"
#include "core/fp_bits.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
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

void handle_tokenize(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return;
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    // `content` is the llama.cpp spelling, `prompt` the vLLM one; a client written
    // against either gets the exact count instead of a 400 and a length estimate (#1980).
    std::string content = body.value("content", "");
    if (content.empty())
        content = body.value("prompt", "");
    if (content.empty()) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "\"content\" (or its alias \"prompt\") is required"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Snapshot model pointer (+ context size) under lock
    ImpModel snap_model;
    int snap_max_seq_len = 0;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        snap_model = state.model;
        snap_max_seq_len = state.max_seq_len;
    }
    if (!snap_model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"}, {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // --max-input-tokens applies here as on every prompt-taking route
    // (AUDIT_arch_2026 F2-9): the byte bound refuses before the merge walk,
    // the token check after it.
    if (!prompt_within_input_budget(res, content.size(), state.max_input_tokens, "content"))
        return;

    // Size the token buffer from the model's context (min 256k): agentic
    // prompts run 10k-200k tokens; the previous fixed 32768 silently failed
    // above that.
    const int tok_cap = std::max(snap_max_seq_len, 262144);
    std::vector<int32_t> tokens(tok_cap);
    int n_tokens = 0;
    ImpError err = imp_tokenize(snap_model, content.c_str(), tokens.data(), &n_tokens, tok_cap);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error",
                       {{"message", std::string("Tokenize failed: ") + imp_error_string(err)},
                        {"type", "server_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }
    if (state.max_input_tokens > 0 && n_tokens > state.max_input_tokens) {
        send_json_error(res, 400, "invalid_request_error",
                        "content exceeds --max-input-tokens (" + std::to_string(n_tokens) + " > " +
                            std::to_string(state.max_input_tokens) + ")",
                        "content", "context_length_exceeded");
        return;
    }

    tokens.resize(n_tokens);
    json response = {{"tokens", tokens}};
    res.set_content(dump_safe(response), "application/json");
}

void handle_detokenize(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return;
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    if (!body.contains("tokens") || !body["tokens"].is_array()) {
        res.status = 400;
        json err = {
            {"error", {{"message", "\"tokens\" array is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Snapshot model pointer under lock
    ImpModel snap_model;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        snap_model = state.model;
    }
    if (!snap_model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"}, {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Bound the array before allocating buf = tokens.size()*32 + 256: a ~100 MiB
    // JSON array of small ints would otherwise force a multi-GB host allocation.
    if (body["tokens"].size() > 1000000) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "tokens array exceeds maximum of 1000000 entries"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }
    // The operator's prompt bound holds on the way out as well (F2-9).
    if (state.max_input_tokens > 0 && static_cast<int>(body["tokens"].size()) > state.max_input_tokens) {
        send_json_error(res, 400, "invalid_request_error",
                        "tokens array exceeds --max-input-tokens (" + std::to_string(body["tokens"].size()) +
                            " > " + std::to_string(state.max_input_tokens) + ")",
                        "tokens", "context_length_exceeded");
        return;
    }

    std::vector<int32_t> tokens = body["tokens"].get<std::vector<int32_t>>();
    std::vector<char> buf(tokens.size() * 32 + 256);
    ImpError err = imp_detokenize(snap_model, tokens.data(), static_cast<int>(tokens.size()), buf.data(),
                                  buf.size());
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error",
                       {{"message", std::string("Detokenize failed: ") + imp_error_string(err)},
                        {"type", "server_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    json response = {{"content", std::string(buf.data())}};
    res.set_content(dump_safe(response), "application/json");
}

void handle_metrics(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    auto& m = state.metrics;
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                                   m.start_time)
                      .count();

    std::string out;
    out.reserve(1024);
    out += "# HELP imp_uptime_seconds Server uptime in seconds\n";
    out += "# TYPE imp_uptime_seconds gauge\n";
    out += "imp_uptime_seconds " + std::to_string(uptime) + "\n";
    out += "# HELP imp_requests_total Total inference requests\n";
    out += "# TYPE imp_requests_total counter\n";
    out += "imp_requests_total " + std::to_string(m.requests_total.load()) + "\n";
    out += "# HELP imp_requests_failed_total Failed inference requests\n";
    out += "# TYPE imp_requests_failed_total counter\n";
    out += "imp_requests_failed_total " + std::to_string(m.requests_failed.load()) + "\n";
    // OTLP span export (server.otlp_endpoint); all zero while it is off.
    out += "# HELP imp_otlp_spans_exported_total Spans accepted by the OTLP collector\n";
    out += "# TYPE imp_otlp_spans_exported_total counter\n";
    out += "imp_otlp_spans_exported_total " + std::to_string(state.tracer.exported_spans()) + "\n";
    out += "# HELP imp_otlp_export_failures_total OTLP batches the collector did not accept (spans dropped)\n";
    out += "# TYPE imp_otlp_export_failures_total counter\n";
    out += "imp_otlp_export_failures_total " + std::to_string(state.tracer.failed_batches()) + "\n";
    out += "# HELP imp_otlp_unsampled_requests_total Requests whose traceparent cleared the sampled flag (not exported)\n";
    out += "# TYPE imp_otlp_unsampled_requests_total counter\n";
    out += "imp_otlp_unsampled_requests_total " + std::to_string(state.tracer.unsampled_requests()) + "\n";

    out += "# HELP imp_constrained_eager_fallback_total Constrained requests that requested "
           "logprobs and fell back from the ConstrainedPipeline to eager decode\n";
    out += "# TYPE imp_constrained_eager_fallback_total counter\n";
    out += "imp_constrained_eager_fallback_total " +
           std::to_string(m.constrained_eager_fallback.load()) + "\n";
    out += "# HELP imp_tokens_prompt_total Total prompt tokens processed\n";
    out += "# TYPE imp_tokens_prompt_total counter\n";
    out += "imp_tokens_prompt_total " + std::to_string(m.tokens_prompt_total.load()) + "\n";
    out += "# HELP imp_tokens_completion_total Total completion tokens generated\n";
    out += "# TYPE imp_tokens_completion_total counter\n";
    out += "imp_tokens_completion_total " + std::to_string(m.tokens_completion_total.load()) + "\n";
    out += "# HELP imp_last_request_duration_ms Duration of last request in milliseconds\n";
    out += "# TYPE imp_last_request_duration_ms gauge\n";
    out += "imp_last_request_duration_ms " + std::to_string(m.last_request_duration_ms.load()) + "\n";
    out += "# HELP imp_model_loads_total Total model loads\n";
    out += "# TYPE imp_model_loads_total counter\n";
    out += "imp_model_loads_total " + std::to_string(m.model_loads_total.load()) + "\n";
    out += "# HELP imp_tokens_cached_total Total prompt tokens served from prefix cache\n";
    out += "# TYPE imp_tokens_cached_total counter\n";
    out += "imp_tokens_cached_total " + std::to_string(m.tokens_cached_total.load()) + "\n";
    out += "# HELP imp_requests_cancelled_total Requests cancelled by client disconnect\n";
    out += "# TYPE imp_requests_cancelled_total counter\n";
    out += "imp_requests_cancelled_total " + std::to_string(m.requests_cancelled.load()) + "\n";
    out += "# HELP imp_requests_timed_out_total Requests the server ended at --request-timeout\n";
    out += "# TYPE imp_requests_timed_out_total counter\n";
    out += "imp_requests_timed_out_total " + std::to_string(m.requests_timed_out.load()) + "\n";
    out +=
        "# HELP imp_requests_reasoning_exhausted_total Requests whose answer never started: "
        "the token budget went to reasoning\n";
    out += "# TYPE imp_requests_reasoning_exhausted_total counter\n";
    out += "imp_requests_reasoning_exhausted_total " + std::to_string(m.requests_reasoning_exhausted.load()) +
           "\n";
    // Read from the engine at scrape time rather than mirrored into
    // ServerMetrics: the decision is the engine's, and a second copy is a
    // second thing to keep in sync (#1641).
    {
        uint64_t kv_rejects = 0, kv_growths = 0, kv_auto_stream = 0, prefix_evictions = 0;
        if (state.ctx && state.ctx->engine) {
            kv_rejects = state.ctx->engine->kv_pressure_rejections();
            kv_growths = state.ctx->engine->kv_pool_growths();
            kv_auto_stream = state.ctx->engine->streaming_kv_auto_enables();
            prefix_evictions = state.ctx->engine->prefix_cache_evictions();
        }
        out +=
            "# HELP imp_kv_pressure_rejections_total Requests cancelled because the KV pool "
            "could not give them blocks\n";
        out += "# TYPE imp_kv_pressure_rejections_total counter\n";
        out += "imp_kv_pressure_rejections_total " + std::to_string(kv_rejects) + "\n";
        uint64_t mixed_steps = 0;
        if (state.ctx && state.ctx->engine)
            mixed_steps = state.ctx->engine->mixed_decode_steps();
        out +=
            "# HELP imp_mixed_decode_steps_total Engine steps whose decoders rode the ragged prefill "
            "forward (runtime.prefill_mixed_decode)\n";
        out += "# TYPE imp_mixed_decode_steps_total counter\n";
        out += "imp_mixed_decode_steps_total " + std::to_string(mixed_steps) + "\n";
        out += "# HELP imp_kv_pool_growths_total Times the growable KV pool committed more memory\n";
        out += "# TYPE imp_kv_pool_growths_total counter\n";
        out += "imp_kv_pool_growths_total " + std::to_string(kv_growths) + "\n";
        // Exposes StreamingLLM auto-enable (demotes CUDA graphs one-way) and prefix-cache block reclaim
        // as rate metrics - previously log lines only. usage.evicted_tokens (id 573) is the per-request size.
        out +=
            "# HELP imp_streaming_kv_auto_enables_total Times the KV pool ran >90% full and "
            "StreamingLLM eviction was auto-enabled (CUDA graphs demoted one-way)\n";
        out += "# TYPE imp_streaming_kv_auto_enables_total counter\n";
        out += "imp_streaming_kv_auto_enables_total " + std::to_string(kv_auto_stream) + "\n";
        out +=
            "# HELP imp_prefix_cache_evictions_total Cached prefix blocks reclaimed for new "
            "allocations\n";
        out += "# TYPE imp_prefix_cache_evictions_total counter\n";
        out += "imp_prefix_cache_evictions_total " + std::to_string(prefix_evictions) + "\n";
    }
    out += "# HELP imp_last_ttft_ms Time to first token of last request in milliseconds\n";
    out += "# TYPE imp_last_ttft_ms gauge\n";
    out += "imp_last_ttft_ms " + std::to_string(m.last_ttft_ms.load()) + "\n";

    append_memory_metrics(out, state);

    // Emits a Prometheus histogram (cumulative _bucket{le=...}, _sum, _count; seconds). `labels` is
    // the label set without the trailing `le` (empty for the unlabelled totals).
    auto emit_histogram_body = [&out](const char* name, const std::string& labels,
                                      const LatencyHistogram& h) {
        for (int i = 0; i < LatencyHistogram::kNumBuckets; ++i) {
            char le[32];
            std::snprintf(le, sizeof(le), "%g", h.bounds[i]);
            out += name;
            out += "_bucket{";
            out += labels;
            out += "le=\"";
            out += le;
            out += "\"} ";
            out += std::to_string(h.buckets[i].load());
            out += "\n";
        }
        out += name;
        out += "_bucket{";
        out += labels;
        out += "le=\"+Inf\"} ";
        out += std::to_string(h.count.load());
        out += "\n";
        char sum[48];
        std::snprintf(sum, sizeof(sum), "%g", h.sum_us.load() / 1e6);
        const std::string tail = labels.empty() ? std::string(" ")
                                                : "{" + labels.substr(0, labels.size() - 1) + "} ";
        out += name;
        out += "_sum";
        out += tail;
        out += sum;
        out += "\n";
        out += name;
        out += "_count";
        out += tail;
        out += std::to_string(h.count.load());
        out += "\n";
    };
    auto emit_header = [&out](const char* name, const char* help, const char* type) {
        out += "# HELP ";
        out += name;
        out += ' ';
        out += help;
        out += "\n# TYPE ";
        out += name;
        out += ' ';
        out += type;
        out += "\n";
    };
    auto emit_histogram = [&](const char* name, const char* help, const LatencyHistogram& h) {
        emit_header(name, help, "histogram");
        emit_histogram_body(name, "", h);
    };
    emit_histogram("imp_request_duration_seconds", "Request end-to-end latency in seconds",
                   m.request_duration);
    emit_histogram("imp_ttft_seconds", "Time to first token in seconds", m.ttft);
    emit_histogram("imp_inter_token_seconds", "Inter-token latency in seconds, one observation per token",
                   m.inter_token);
    emit_histogram(
        "imp_queue_time_seconds",
        "Seconds from submit to the scheduler's first batch (the wait behind max_batch_size and KV)",
        m.queue_time);
    // The same series per endpoint (AUDIT_arch_2026 E-7), every endpoint
    // emitted even at zero so a panel can be built before traffic arrives.
    emit_header("imp_endpoint_requests_total", "Completed generation requests per endpoint", "counter");
    for (int e = 0; e < ServerMetrics::kEndpointCount; ++e) {
        out += "imp_endpoint_requests_total{endpoint=\"";
        out += ServerMetrics::endpoint_name(e);
        out += "\"} ";
        out += std::to_string(m.series(static_cast<ServerMetrics::Endpoint>(e)).requests_total.load());
        out += "\n";
    }
    struct EndpointHist {
        const char* name;
        const char* help;
        LatencyHistogram ServerMetrics::EndpointSeries::*member;
    };
    const EndpointHist endpoint_hists[] = {
        {"imp_endpoint_request_duration_seconds", "Request end-to-end latency in seconds, per endpoint",
         &ServerMetrics::EndpointSeries::request_duration},
        {"imp_endpoint_ttft_seconds", "Time to first token in seconds, per endpoint",
         &ServerMetrics::EndpointSeries::ttft},
        {"imp_endpoint_inter_token_seconds", "Inter-token latency in seconds, per endpoint",
         &ServerMetrics::EndpointSeries::inter_token},
        {"imp_endpoint_queue_time_seconds",
         "Seconds from submit to the scheduler's first batch, per endpoint",
         &ServerMetrics::EndpointSeries::queue_time},
    };
    for (const auto& eh : endpoint_hists) {
        emit_header(eh.name, eh.help, "histogram");
        for (int e = 0; e < ServerMetrics::kEndpointCount; ++e) {
            const std::string labels = std::string("endpoint=\"") + ServerMetrics::endpoint_name(e) + "\",";
            emit_histogram_body(eh.name, labels,
                                m.series(static_cast<ServerMetrics::Endpoint>(e)).*eh.member);
        }
    }
    out += "# HELP imp_requests_rejected_total Requests refused with a 4xx\n";
    out += "# TYPE imp_requests_rejected_total counter\n";
    out += "imp_requests_rejected_total " + std::to_string(m.requests_rejected.load()) + "\n";
    // Decode batch size exposed as a counter pair so a dashboard can rate() both and divide for the
    // average over any window, without the server maintaining one itself.
    {
        int64_t steps = 0, rows = 0, bmax = 0, blast = 0;
        std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
        if (lock.owns_lock() && state.batching) {
            steps = state.batching->decode_steps.load();
            rows = state.batching->decode_rows.load();
            bmax = state.batching->decode_batch_max.load();
            blast = state.batching->decode_batch_last.load();
        }
        out += "# HELP imp_decode_batch_steps_total Decode steps executed\n";
        out += "# TYPE imp_decode_batch_steps_total counter\n";
        out += "imp_decode_batch_steps_total " + std::to_string(steps) + "\n";
        out += "# HELP imp_decode_batch_rows_total Sequences decoded, summed over steps\n";
        out += "# TYPE imp_decode_batch_rows_total counter\n";
        out += "imp_decode_batch_rows_total " + std::to_string(rows) + "\n";
        out += "# HELP imp_decode_batch_max Largest decode batch seen since start\n";
        out += "# TYPE imp_decode_batch_max gauge\n";
        out += "imp_decode_batch_max " + std::to_string(bmax) + "\n";
        out +=
            "# HELP imp_decode_batch_last_rows Sequences in the most recent decode step, 0 while "
            "idle\n";
        out += "# TYPE imp_decode_batch_last_rows gauge\n";
        out += "imp_decode_batch_last_rows " + std::to_string(blast) + "\n";
    }
    out += "# HELP imp_model_loaded Whether a model is currently loaded\n";
    out += "# TYPE imp_model_loaded gauge\n";
    bool loaded = false;
    std::string model_label;
    int queue = -1;
    long long waiting = -1, running = -1;
    {
        // Bounded lock so a scrape can't hang behind a long /v1/embeddings
        // holder (#889); fall back to the lock-free status snapshot.
        std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
        if (lock.owns_lock()) {
            loaded = state.model_loaded();
            model_label = loaded ? state.model_name : std::string();
            if (state.batching) {
                queue = state.batching->queue_depth();
                waiting = state.batching->queue_waiting.load(std::memory_order_relaxed);
                running = state.batching->queue_running.load(std::memory_order_relaxed);
            }
        } else {
            const auto snap = state.model_status_snapshot();  // queue stays -1 (unknown)
            loaded = snap.loaded;
            model_label = loaded ? snap.model_name : std::string();
        }
    }
    // Labelled with the model (E-7): a swapped model's numbers and its
    // predecessor's were one series before. Quotes and backslashes in a
    // name are escaped the way the exposition format wants.
    std::string model_escaped;
    for (char c : model_label) {
        if (c == '"' || c == '\\')
            model_escaped += '\\';
        model_escaped += (c == '\n') ? ' ' : c;
    }
    out += "imp_model_loaded{model=\"" + model_escaped + "\"} " + std::string(loaded ? "1" : "0") + "\n";
    out += "# HELP imp_queue_depth Current number of active and pending requests\n";
    out += "# TYPE imp_queue_depth gauge\n";
    out += "imp_queue_depth " + std::to_string(queue) + "\n";
    // The split the depth hides (AUDIT_arch_2026 C-5): a request behind
    // max_batch_size or KV admission is waiting, one in a batch is running.
    out += "# HELP imp_queue_waiting Requests submitted and not yet in a prefill or decode batch\n";
    out += "# TYPE imp_queue_waiting gauge\n";
    out += "imp_queue_waiting " + std::to_string(waiting) + "\n";
    out += "# HELP imp_queue_running Requests in the current prefill or decode batch\n";
    out += "# TYPE imp_queue_running gauge\n";
    out += "imp_queue_running " + std::to_string(running) + "\n";

    res.set_content(out, "text/plain; version=0.0.4; charset=utf-8");
}

void handle_embeddings(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return;
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    // Collect inputs: "input" can be a string or array of strings
    std::vector<std::string> inputs;
    if (body.contains("input")) {
        if (body["input"].is_string()) {
            inputs.push_back(body["input"].get<std::string>());
        } else if (body["input"].is_array()) {
            // One request, one rate-limit unit, N forward passes (#1616).
            if (state.max_batch_items > 0 && static_cast<int>(body["input"].size()) > state.max_batch_items) {
                res.status = 400;
                json err = {{"error",
                             {{"message", "\"input\" has " + std::to_string(body["input"].size()) +
                                              " entries, above the server limit of " +
                                              std::to_string(state.max_batch_items) + " (--max-batch-items)"},
                              {"type", "invalid_request_error"}}}};
                res.set_content(dump_safe(err), "application/json");
                return;
            }
            for (const auto& item : body["input"]) {
                if (item.is_string()) {
                    inputs.push_back(item.get<std::string>());
                } else {
                    res.status = 400;
                    json err = {
                        {"error",
                         {{"message", "Each input must be a string"}, {"type", "invalid_request_error"}}}};
                    res.set_content(dump_safe(err), "application/json");
                    return;
                }
            }
        } else {
            res.status = 400;
            json err = {{"error",
                         {{"message", "\"input\" must be a string or array of strings"},
                          {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }
    } else {
        res.status = 400;
        json err = {{"error", {{"message", "\"input\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    if (inputs.empty()) {
        res.status = 400;
        json err = {
            {"error", {{"message", "\"input\" must not be empty"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // encoding_format: OpenAI supports "float" (JSON array) or "base64" (default in the OpenAI
    // Python SDK, little-endian float32 bytes). Anything else is rejected (400), not silently floats.
    std::string encoding_format = body.value("encoding_format", std::string("float"));
    if (encoding_format != "float" && encoding_format != "base64") {
        send_json_error(res, 400, "invalid_request_error",
                        "Unsupported encoding_format '" + encoding_format +
                            "' (expected 'float' or 'base64')");
        return;
    }
    // Matryoshka dimension truncation is not supported. Accept the field only
    // when it matches the model's native width (checked against d_model once a
    // model is confirmed loaded, below) — never silently ignore it.
    bool has_dimensions = body.contains("dimensions") && !body["dimensions"].is_null();
    int requested_dims = 0;
    if (has_dimensions) {
        if (!body["dimensions"].is_number_integer()) {
            send_json_error(res, 400, "invalid_request_error", "\"dimensions\" must be an integer");
            return;
        }
        requested_dims = body["dimensions"].get<int>();
    }

    // Acquire inference lock and pause batching engine for exclusive access
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(1));
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error",
                     {{"message", "Server is busy processing another request. Please retry."},
                      {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Validates/auto-loads the requested model exactly like chat/completions (ensure_model_loaded,
    // 404 on unknown). Absent "model" field is lenient: uses the loaded model.
    std::string requested_model = body.value("model", std::string());
    if (requested_model.empty())
        requested_model = state.model_name;
    if (!ensure_model_loaded(state, requested_model, res))
        return;

    // Decoder models with a running batching worker take the SCHEDULED path (#1005): embeddings
    // queue as prefill-only requests, batching alongside concurrent decodes. Encoder models
    // (bidirectional forward) and worker-less setups keep the legacy exclusive pause path.
    const bool is_encoder =
        state.ctx && state.ctx->engine && state.ctx->engine->is_encoder_model();
    if (!is_encoder && state.batching && state.batching->is_running()) {
        state.metrics.requests_total++;
        state.metrics.series(ServerMetrics::kEmbeddings).requests_total++;
        auto t0b = std::chrono::steady_clock::now();
        const int d_model = imp_model_d_model(state.model);
        if (has_dimensions && requested_dims != d_model) {
            send_json_error(res, 400, "invalid_request_error",
                            "This server does not support Matryoshka dimension truncation; "
                            "\"dimensions\" must equal the model width (" +
                                std::to_string(d_model) + ")");
            return;
        }
        auto embedding_field_b = [&](const std::vector<float>& v) -> json {
            if (encoding_format == "base64")
                return base64_encode(reinterpret_cast<const uint8_t*>(v.data()),
                                     v.size() * sizeof(float));
            return json(v);
        };

        // Tokenize + submit every input while the lock is held (submission
        // order == queue order), then release and await completions.
        std::vector<std::shared_ptr<ServerRequest>> submitted;
        int total_prompt_tokens = 0;
        const int tok_cap = std::max(state.max_seq_len, 262144);
        for (const auto& text : inputs) {
            std::vector<int32_t> tokens(tok_cap);
            int n_tokens = 0;
            ImpError terr = imp_tokenize(state.model, text.c_str(), tokens.data(), &n_tokens, tok_cap);
            if (terr != IMP_SUCCESS) {
                send_json_error(res, 500, "server_error",
                                std::string("Tokenize failed: ") + imp_error_string(terr));
                return;
            }
            if (n_tokens == 0) {
                send_json_error(res, 400, "invalid_request_error", "Input tokenizes to zero tokens");
                return;
            }
            if (state.max_input_tokens > 0 && n_tokens > state.max_input_tokens) {
                send_json_error(res, 400, "invalid_request_error",
                                "Input exceeds --max-input-tokens (" + std::to_string(n_tokens) + " > " +
                                    std::to_string(state.max_input_tokens) + ")",
                                "input", "context_length_exceeded");
                return;
            }
            // Chunked prefill pools per chunk (#1005), so the only hard bound
            // is the engine context.
            if (state.max_seq_len > 0 && n_tokens > state.max_seq_len) {
                send_json_error(res, 400, "invalid_request_error",
                                "Input exceeds the model context (" + std::to_string(n_tokens) +
                                    " tokens > " + std::to_string(state.max_seq_len) + " max)");
                return;
            }
            tokens.resize(n_tokens);
            total_prompt_tokens += n_tokens;

            auto req = std::make_shared<imp::Request>();
            req->input_tokens = std::move(tokens);
            req->max_tokens = 1;
            req->temperature = 0.0f;
            req->stream = false;
            req->embedding_request = true;
            req->status = imp::RequestStatus::PENDING;
            auto sr = std::make_shared<ServerRequest>();
            sr->request = req;
            state.batching->submit(sr);
            submitted.push_back(std::move(sr));
        }
        lock.unlock();

        json data = json::array();
        for (size_t i = 0; i < submitted.size(); i++) {
            auto& sr = submitted[i];
            bool finished = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
            while (!finished) {
                std::unique_lock<std::mutex> ql(sr->token_mutex);
                if (!sr->token_cv.wait_until(ql, deadline,
                                             [&] { return !sr->token_queue.empty(); })) {
                    sr->cancelled = true;
                    send_json_error(res, 503, "server_error", "Embedding request timed out");
                    return;
                }
                while (!sr->token_queue.empty()) {
                    auto ev = sr->token_queue.front();
                    sr->token_queue.pop_front();
                    if (ev.is_last)
                        finished = true;
                }
            }
            auto& ereq = sr->request;
            if (ereq->status != imp::RequestStatus::FINISHED || ereq->embedding_out.empty()) {
                send_json_error(res, 500, "server_error",
                                "Embedding forward failed (request cancelled or empty result)");
                return;
            }
            std::vector<float> embedding = ereq->embedding_out;  // mean-pooled engine-side
            float norm_sq = 0.0f;
            for (float v : embedding)
                norm_sq += v * v;
            const float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
            for (float& v : embedding)
                v *= inv_norm;
            data.push_back({{"object", "embedding"},
                            {"embedding", embedding_field_b(embedding)},
                            {"index", i}});
        }

        auto t1b = std::chrono::steady_clock::now();
        state.metrics.last_request_duration_ms.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(t1b - t0b).count());
        state.metrics.tokens_prompt_total += total_prompt_tokens;
        json response = {{"object", "list"},
                         {"data", data},
                         {"model", body.value("model", state.model_name)},
                         {"usage",
                          {{"prompt_tokens", total_prompt_tokens},
                           {"total_tokens", total_prompt_tokens}}}};
        res.set_content(dump_safe(response), "application/json");
        return;
    }

    // Pauses (not stops) the batching engine for exclusive C-API access during imp_prefill: stop()
    // would cancel concurrent chat (the "0 completion tokens" wedge); pause() lets in-flight finish.
    // state.mtx blocks new admission meanwhile; resume() unparks on scope exit.
    bool had_batching = (state.batching && state.batching->is_running());
    if (had_batching) {
        if (!state.batching->pause()) {
            res.status = 503;
            json err = {{"error",
                         {{"message", "Server busy draining in-flight requests. Please retry."},
                          {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }
    }
    auto restart_batching = [&] {
        if (had_batching && state.batching)
            state.batching->resume();
    };
    // Use a simple scope guard
    struct ScopeGuard {
        std::function<void()> fn;
        ~ScopeGuard() { fn(); }
    } batching_guard{restart_batching};

    state.metrics.requests_total++;
    state.metrics.series(ServerMetrics::kEmbeddings).requests_total++;
    auto t0 = std::chrono::steady_clock::now();

    // Get model dimensions
    int d_model = imp_model_d_model(state.model);
    int total_prompt_tokens = 0;

    if (has_dimensions && requested_dims != d_model) {
        send_json_error(res, 400, "invalid_request_error",
                        "This server does not support Matryoshka dimension truncation; "
                        "\"dimensions\" must equal the model width (" + std::to_string(d_model) + ")");
        return;
    }

    // Serialize one embedding per the requested encoding_format: a JSON float
    // array ("float") or base64 of the little-endian float32 bytes ("base64").
    auto embedding_field = [&](const std::vector<float>& v) -> json {
        if (encoding_format == "base64") {
            return base64_encode(reinterpret_cast<const uint8_t*>(v.data()), v.size() * sizeof(float));
        }
        return json(v);
    };

    json data = json::array();

    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        const auto& text = inputs[input_idx];

        // Tokenize (buffer sized from the model context, min 256k — the
        // embed_cap check below rejects over-long inputs AFTER counting).
        const int tok_cap = std::max(state.max_seq_len, 262144);
        std::vector<int32_t> tokens(tok_cap);
        int n_tokens = 0;
        ImpError err = imp_tokenize(state.model, text.c_str(), tokens.data(), &n_tokens, tok_cap);
        if (err != IMP_SUCCESS) {
            res.status = 500;
            json error = {{"error",
                           {{"message", std::string("Tokenize failed: ") + imp_error_string(err)},
                            {"type", "server_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return;
        }
        tokens.resize(n_tokens);

        if (n_tokens == 0) {
            res.status = 400;
            json error = {
                {"error",
                 {{"message", "Input tokenizes to zero tokens"}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return;
        }

        // Enforce the operator's --max-input-tokens per-request cap, like chat
        // and completions do — bounding embedding prefill cost regardless of
        // the (much larger) engine context / encoder capacity.
        if (state.max_input_tokens > 0 && n_tokens > state.max_input_tokens) {
            send_json_error(res, 400, "invalid_request_error",
                            "Input exceeds --max-input-tokens (" + std::to_string(n_tokens) + " > " +
                                std::to_string(state.max_input_tokens) + ")");
            return;
        }

        // Encoder-only embedder (#836, nomic-bert): the dedicated bidirectional
        // forward pools + L2-normalizes on device. Frame with [CLS]/[SEP]
        // (BERT convention; ids come from the GGUF bos/eos metadata).
        if (state.ctx && state.ctx->engine && state.ctx->engine->is_encoder_model()) {
            auto* engine = state.ctx->engine.get();
            std::vector<int32_t> framed;
            framed.reserve(n_tokens + 2);
            const int32_t cls = imp_model_bos_token(state.model);
            const int32_t sep = (engine->model() && engine->model()->tokenizer())
                                    ? engine->model()->tokenizer()->eos_id()
                                    : -1;
            if (cls >= 0 && (n_tokens == 0 || tokens[0] != cls))
                framed.push_back(cls);
            framed.insert(framed.end(), tokens.begin(), tokens.begin() + n_tokens);
            if (sep >= 0 && framed.back() != sep)
                framed.push_back(sep);
            std::vector<float> emb;
            if (!engine->encoder_embed(framed, emb)) {
                nlohmann::json error = {
                    {"error",
                     {{"message", "encoder forward failed (input too long for the encoder "
                                  "workspace?)"},
                      {"type", "server_error"}}}};
                res.status = 500;
                res.set_content(dump_safe(error), "application/json");
                return;
            }
            data.push_back(
                {{"object", "embedding"}, {"embedding", embedding_field(emb)}, {"index", input_idx}});
            total_prompt_tokens += static_cast<int>(framed.size());
            continue;
        }

        // Rejects over-long embeddings input against TWO bounds: state.max_seq_len (engine context) and
        // the executor's single-pass hidden capacity max_tokens() - embeddings mean-pools every token,
        // needing the whole input in one prefill pass; exceeding it slices out of bounds and aborts the
        // process (Tensor::slice IMP_CHECK). max_tokens can be far below max_seq_len.
        int embed_cap = state.max_seq_len;
        if (state.ctx && state.ctx->engine && state.ctx->engine->executor()) {
            int hid = state.ctx->engine->executor()->max_tokens();
            if (hid > 0 && (embed_cap <= 0 || hid < embed_cap))
                embed_cap = hid;
        }
        if (embed_cap > 0 && n_tokens > embed_cap) {
            send_json_error(res, 400, "invalid_request_error",
                            "Input exceeds the embedding context (" + std::to_string(n_tokens) +
                                " tokens > " + std::to_string(embed_cap) + " max)");
            return;
        }

        total_prompt_tokens += n_tokens;

        // Run prefill (forward pass without generation)
        err = imp_prefill(state.ctx, tokens.data(), n_tokens);
        if (err != IMP_SUCCESS) {
            imp_context_reset(state.ctx);
            res.status = 500;
            json error = {{"error",
                           {{"message", std::string("Prefill failed: ") + imp_error_string(err)},
                            {"type", "server_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return;
        }

        // Extract hidden states from the executor
        // hidden_ is [n_tokens, d_model] FP16 on GPU after forward_logits()
        auto* engine = state.ctx->engine.get();
        auto* executor = engine->executor();
        imp::Tensor hidden_view = executor->view_hidden(n_tokens);

        // Copy FP16 hidden states from GPU to host as uint16_t
        size_t n_elements = static_cast<size_t>(n_tokens) * d_model;
        std::vector<uint16_t> h_hidden(n_elements);
        cudaError_t cuda_err = cudaMemcpy(h_hidden.data(), hidden_view.data, n_elements * sizeof(uint16_t),
                                          cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            imp_context_reset(state.ctx);
            res.status = 500;
            json error = {{"error",
                           {{"message", std::string("CUDA memcpy failed: ") + cudaGetErrorString(cuda_err)},
                            {"type", "server_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return;
        }

        // Mean-pool across tokens: average all token hidden states
        std::vector<float> embedding(d_model, 0.0f);
        for (int t = 0; t < n_tokens; ++t) {
            for (int d = 0; d < d_model; ++d) {
                embedding[d] += imp::half_to_float(h_hidden[t * d_model + d]);
            }
        }
        float inv_n = 1.0f / static_cast<float>(n_tokens);
        for (int d = 0; d < d_model; ++d) {
            embedding[d] *= inv_n;
        }

        // L2 normalize
        float norm_sq = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            norm_sq += embedding[d] * embedding[d];
        }
        float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
        for (int d = 0; d < d_model; ++d) {
            embedding[d] *= inv_norm;
        }

        data.push_back(
            {{"object", "embedding"}, {"embedding", embedding_field(embedding)}, {"index", input_idx}});

        // Reset context for next input
        imp_context_reset(state.ctx);
    }

    auto t1 = std::chrono::steady_clock::now();
    int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    state.metrics.last_request_duration_ms.store(duration_ms);
    state.metrics.tokens_prompt_total += total_prompt_tokens;

    // batching_guard restarts the batching engine automatically on scope exit

    json response = {{"object", "list"},
                     {"data", data},
                     {"model", body.value("model", state.model_name)},
                     {"usage",
                      {{"prompt_tokens", total_prompt_tokens}, {"total_tokens", total_prompt_tokens}}}};
    res.set_content(dump_safe(response), "application/json");
}
