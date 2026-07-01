// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// Misc endpoints: handle_tokenize, handle_detokenize, handle_metrics, and
// handle_embeddings (with its fp16->fp32 helper).

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

void handle_tokenize(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    std::string content = body.value("content", "");
    if (content.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"content\" is required"}, {"type", "invalid_request_error"}}}};
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

    tokens.resize(n_tokens);
    json response = {{"tokens", tokens}};
    res.set_content(dump_safe(response), "application/json");
}

void handle_detokenize(const httplib::Request& req, httplib::Response& res, ServerState& state) {
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
    out += "# HELP imp_last_ttft_ms Time to first token of last request in milliseconds\n";
    out += "# TYPE imp_last_ttft_ms gauge\n";
    out += "imp_last_ttft_ms " + std::to_string(m.last_ttft_ms.load()) + "\n";

    // Latency histograms (Prometheus histogram: cumulative _bucket{le=...},
    // plus _sum and _count). Buckets are in seconds.
    auto emit_histogram = [&out](const char* name, const char* help, const LatencyHistogram& h) {
        out += "# HELP ";
        out += name;
        out += ' ';
        out += help;
        out += "\n# TYPE ";
        out += name;
        out += " histogram\n";
        for (int i = 0; i < LatencyHistogram::kNumBuckets; ++i) {
            char le[32];
            std::snprintf(le, sizeof(le), "%g", LatencyHistogram::kBounds[i]);
            out += name;
            out += "_bucket{le=\"";
            out += le;
            out += "\"} ";
            out += std::to_string(h.buckets[i].load());
            out += "\n";
        }
        out += name;
        out += "_bucket{le=\"+Inf\"} ";
        out += std::to_string(h.count.load());
        out += "\n";
        char sum[48];
        std::snprintf(sum, sizeof(sum), "%g", h.sum_us.load() / 1e6);
        out += name;
        out += "_sum ";
        out += sum;
        out += "\n";
        out += name;
        out += "_count ";
        out += std::to_string(h.count.load());
        out += "\n";
    };
    emit_histogram("imp_request_duration_seconds", "Request end-to-end latency in seconds",
                   m.request_duration);
    emit_histogram("imp_ttft_seconds", "Time to first token in seconds", m.ttft);
    emit_histogram("imp_inter_token_seconds", "Mean inter-token latency (ITL) per request in seconds",
                   m.inter_token);
    out += "# HELP imp_model_loaded Whether a model is currently loaded\n";
    out += "# TYPE imp_model_loaded gauge\n";
    bool loaded;
    int queue = 0;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        loaded = state.model_loaded();
        if (state.batching)
            queue = state.batching->queue_depth();
    }
    out += "imp_model_loaded " + std::string(loaded ? "1" : "0") + "\n";
    out += "# HELP imp_queue_depth Current number of active and pending requests\n";
    out += "# TYPE imp_queue_depth gauge\n";
    out += "imp_queue_depth " + std::to_string(queue) + "\n";

    res.set_content(out, "text/plain; version=0.0.4; charset=utf-8");
}

// Convert IEEE 754 FP16 (uint16_t) to FP32 on host
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // Subnormal: normalize
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

void handle_embeddings(const httplib::Request& req, httplib::Response& res, ServerState& state) {
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

    if (!state.model_loaded()) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"}, {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Pause the batching engine for exclusive C-API access (imp_prefill drives
    // engine->step() directly, which must not race the worker). pause() lets
    // in-flight generations FINISH before parking the worker — stop() would
    // cancel them, so any chat running concurrently with this embeddings call
    // returned an empty `finish_reason:"cancelled"` completion (the "0
    // completion tokens" wedge). We hold state.mtx, so no new chat is admitted
    // while paused; resume() unparks on scope exit.
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
    auto t0 = std::chrono::steady_clock::now();

    // Get model dimensions
    int d_model = imp_model_d_model(state.model);
    int total_prompt_tokens = 0;

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

        // Reject over-long inputs before prefill. Two independent bounds:
        //   1. the engine's allocated context (state.max_seq_len), and
        //   2. the executor's single-pass hidden capacity (max_tokens()): the
        //      embeddings path mean-pools EVERY token's hidden state, which only
        //      works when the whole input is prefilled in one pass. A longer
        //      input is chunked and hidden_ keeps only the last chunk, so
        //      view_hidden(n) would slice [0,n) out of a [max_tokens,*] buffer
        //      and abort the whole server (Tensor::slice IMP_CHECK). max_tokens
        //      can be far below max_seq_len (e.g. 4096 vs a 32768 context).
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
                embedding[d] += fp16_to_fp32(h_hidden[t * d_model + d]);
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

        data.push_back({{"object", "embedding"}, {"embedding", embedding}, {"index", input_idx}});

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
