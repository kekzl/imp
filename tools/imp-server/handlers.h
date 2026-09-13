#pragma once
#include "tracing.h"

#include <map>

#include "args.h"
#include "batching_engine.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "runtime/config.h"

#include <imp/imp.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include "rate_limit.h"

#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

// /health, /metrics, /v1/models take state.mtx with this bounded timeout, never unbounded: a
// long /v1/embeddings call holds the lock for its whole computation (#889).
inline constexpr std::chrono::milliseconds kObservabilityLockTimeout{250};

// Per-request JSONL logger. Opt-in via --log-requests <path>; appends one
// line per chat/completions or messages call with the raw client body, basic
// metadata, and (for non-streaming) the assistant response. Thread-safe.
struct RequestLogger {
    std::ofstream file;
    std::mutex mtx;
    bool enabled = false;

    bool open(const std::string& path) {
        if (path.empty())
            return true;
        file.open(path, std::ios::app);
        if (!file.is_open()) {
            fprintf(stderr, "warning: --log-requests: failed to open %s\n", path.c_str());
            return false;
        }
        enabled = true;
        return true;
    }

    void log(const json& record) {
        if (!enabled)
            return;
        std::lock_guard<std::mutex> lock(mtx);
        // replace: the record echoes raw client bodies / decoded model text,
        // which can contain ill-formed UTF-8 — a plain dump() would throw
        // (json::type_error.316) and take down the request mid-log.
        file << record.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
        file.flush();
    }
};

// Prometheus-style cumulative-bucket histogram, lock-free (each observation bumps buckets+sum+count).
// Ladder is per-instance (#1577): ITL is single-digit ms, request duration/TTFT is sub-second-to-minutes.
struct LatencyHistogram {
    static constexpr int kNumBuckets = 11;
    // Sub-second-to-minutes: request duration, TTFT.
    static constexpr double kSecondsBounds[kNumBuckets] = {0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                                                           0.5,   1.0,  2.5,   5.0,  10.0};
    // Milliseconds: inter-token latency. 400 tok/s is 2.5 ms, so the ladder
    // has to resolve either side of that; the top end covers a stalled or
    // heavily batched step.
    static constexpr double kItlBounds[kNumBuckets] = {0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075,
                                                       0.01,   0.025, 0.05,  0.1,   0.5};

    const double* bounds = kSecondsBounds;  // set once at construction
    std::atomic<int64_t> buckets[kNumBuckets] = {};
    std::atomic<int64_t> count{0};
    // Sum of observed seconds, stored as micros to keep an integer atomic.
    std::atomic<int64_t> sum_us{0};

    LatencyHistogram() = default;
    explicit LatencyHistogram(const double* ladder) : bounds(ladder) {}

    void observe(double seconds) {
        if (seconds < 0)
            seconds = 0;
        for (int i = 0; i < kNumBuckets; ++i) {
            if (seconds <= bounds[i])
                buckets[i].fetch_add(1, std::memory_order_relaxed);
        }
        count.fetch_add(1, std::memory_order_relaxed);
        sum_us.fetch_add(static_cast<int64_t>(seconds * 1e6), std::memory_order_relaxed);
    }
};

// Server-wide metrics (atomics for lock-free reads from /metrics endpoint)
struct ServerMetrics {
    // Per-endpoint request/latency series (AUDIT_arch_2026 E-7): imp_endpoint_*{endpoint="..."}
    // so a dashboard can separate error rate per dialect. Unlabelled totals kept for existing panels.
    enum Endpoint { kChat = 0, kCompletions, kMessages, kResponses, kEmbeddings, kRerank, kEndpointCount };
    static const char* endpoint_name(int e) {
        static const char* const names[kEndpointCount] = {"chat_completions", "completions", "messages",
                                                          "responses",        "embeddings",  "rerank"};
        return (e >= 0 && e < kEndpointCount) ? names[e] : "other";
    }
    // -1 for a path that is not a generation route (count_tokens, tokenize).
    static int endpoint_index(const std::string& path) {
        if (path == "/v1/chat/completions")
            return kChat;
        if (path == "/v1/completions")
            return kCompletions;
        if (path == "/v1/messages")
            return kMessages;
        if (path == "/v1/responses")
            return kResponses;
        if (path == "/v1/embeddings")
            return kEmbeddings;
        if (path == "/v1/rerank" || path == "/rerank")
            return kRerank;
        return -1;
    }
    struct EndpointSeries {
        std::atomic<int64_t> requests_total{0};
        LatencyHistogram request_duration;
        LatencyHistogram ttft;
        LatencyHistogram queue_time;
        LatencyHistogram inter_token{LatencyHistogram::kItlBounds};
    };
    EndpointSeries endpoint_series[kEndpointCount];
    EndpointSeries& series(Endpoint e) { return endpoint_series[e]; }
    // nullptr for a path that is not a generation route.
    EndpointSeries* series_for(const std::string& path) {
        const int i = endpoint_index(path);
        return i < 0 ? nullptr : &endpoint_series[i];
    }
    // One completed generation: the totals and the endpoint's own series.
    // ttft_ms < 0 means no token was produced. The six handler sites used to
    // spell the same nine lines each.
    void record_completion(const std::string& endpoint_path, double ms, double ttft_ms, int prompt_tokens,
                           int completion_tokens, int cached = 0) {
        requests_total++;
        tokens_prompt_total += prompt_tokens;
        tokens_completion_total += completion_tokens;
        tokens_cached_total += cached;
        last_request_duration_ms = static_cast<int64_t>(ms);
        request_duration.observe(ms / 1000.0);
        auto* es = series_for(endpoint_path);
        if (es) {
            es->requests_total++;
            es->request_duration.observe(ms / 1000.0);
        }
        if (ttft_ms >= 0.0) {
            last_ttft_ms = static_cast<int64_t>(ttft_ms);
            ttft.observe(ttft_ms / 1000.0);
            if (es)
                es->ttft.observe(ttft_ms / 1000.0);
        }
    }
    void record_queue_wait(const std::string& endpoint_path, double seconds) {
        queue_time.observe(seconds);
        if (auto* es = series_for(endpoint_path))
            es->queue_time.observe(seconds);
    }
    void record_inter_token(const std::string& endpoint_path, double seconds) {
        inter_token.observe(seconds);
        if (auto* es = series_for(endpoint_path))
            es->inter_token.observe(seconds);
    }

    std::atomic<int64_t> requests_total{0};
    std::atomic<int64_t> requests_failed{0};
    std::atomic<int64_t> tokens_prompt_total{0};
    std::atomic<int64_t> tokens_completion_total{0};
    std::atomic<int64_t> tokens_cached_total{0};  // Prefix cache hits
    std::atomic<int64_t> requests_cancelled{0};   // Client-disconnect cancellations
    // requests_timed_out: server-initiated --request-timeout, distinct from requests_cancelled
    // (client gone). Without this counter a timeout was invisible (finish_reason "length" either way, #1640).
    std::atomic<int64_t> requests_timed_out{0};
    // requests_reasoning_exhausted: empty content beside a non-empty reasoning channel (budget spent
    // thinking). Same blind-spot class as requests_timed_out; per-request detail is imp_finish_detail.
    std::atomic<int64_t> requests_reasoning_exhausted{0};
    // constrained_eager_fallback: json_schema/json_mode/enforced-tools requests that also ask for
    // logprobs drop the ConstrainedPipeline fast path for eager decode (~102 vs ~235 tok/s, 8B ref, #1006).
    std::atomic<int64_t> constrained_eager_fallback{0};
    std::atomic<int64_t> last_request_duration_ms{0};
    std::atomic<int64_t> last_ttft_ms{0};  // Time to first token (ms)
    std::atomic<int64_t> model_loads_total{0};
    LatencyHistogram request_duration;  // end-to-end request latency
    LatencyHistogram ttft;              // time to first token
    // inter_token: per-TOKEN latency on a millisecond ladder, not a per-request mean on the
    // request-duration ladder (#1577), which cannot answer "how does it vary".
    LatencyHistogram inter_token{LatencyHistogram::kItlBounds};
    // Time from admission to the first decode step, i.e. how long a request
    // waited behind others. Nothing measured queueing before (#1580).
    LatencyHistogram queue_time;
    // Queue wait for a request given up on (client gone, or --request-timeout) before admission:
    // otherwise wait is only closed at first token, so a timed-out request reads as 0 in
    // imp_queue_time_seconds.
    void observe_unadmitted_queue_wait(std::chrono::steady_clock::time_point t_submit, double queue_ms) {
        if (queue_ms >= 0.0 || t_submit == std::chrono::steady_clock::time_point{})
            return;
        queue_time.observe(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_submit).count());
    }
    // requests_rejected counts 4xx refusals separately from requests_failed (5xx only, #1579):
    // "the server broke" and "the server refused" are different alerts.
    std::atomic<int64_t> requests_rejected{0};
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

struct ServerState {
    ImpModel model = nullptr;
    // OTLP span exporter (server.otlp_endpoint); one span set per request.
    Tracer tracer;
    ImpContext ctx = nullptr;
    // LoRA adapters loaded at startup (--lora NAME=PATH): name -> C-API id. Per-request "lora" field
    // selects; empty/absent = base. Swap recaptures decode graphs; adapter is engine-global (single-user).
    std::map<std::string, int32_t> lora_ids;
    imp::Tokenizer* tok = nullptr;
    imp::ChatTemplate chat_tpl;
    bool have_template = false;
    std::string model_name;
    // Loaded once at startup (imp.conf + --set overrides). load_model_into_state
    // re-stashes this via set_pending_runtime_config() before each Engine
    // construction (server may swap models at runtime via /v1/models POST).
    imp::RuntimeConfig runtime_config;
    std::timed_mutex mtx;
    int default_max_tokens = 8192;
    int max_seq_len = 0;
    // resolved_max_batch_size: the CURRENT load's resolved batch size. Feeds speculative.mtp_k=auto
    // and /health's decline reason; both read the raw CLI flag before 2026-08-29 (see id 465).
    int resolved_max_batch_size = 0;
    // armed_mtp_k: MTP depth armed by the CURRENT load (Engine::mtp_spec_decode_k()), read lock-free
    // so /health and per-request validation avoid state.mtx (#888). Written once per load under the swap
    // lock.
    std::atomic<int> armed_mtp_k{0};        // Engine::mtp_spec_decode_k()
    std::atomic<bool> mtp_head_present{false};  // checkpoint ships MTP tensors
    std::atomic<bool> mtp_head_loaded{false};   // ... and this process uploaded them
    std::atomic<int> next_id{0};
    std::atomic<int> next_tool_call_id{0};
    ServerArgs default_args;
    std::string models_dir;       // directory to scan for available .gguf files
    std::string api_key;          // if non-empty, require Bearer token auth
    // --metrics-require-auth: gate /metrics behind api_key too (#1207). Default
    // off — the Prometheus scrape in monitoring/ is unauthenticated — but the
    // endpoint discloses model name, d_model and cumulative token counts.
    bool metrics_require_auth = false;
    bool is_think_model = false;  // model has <think> token (DeepSeek R1 etc.)
    int32_t think_start_id = -1;  // <think> token ID (-1 if not present)
    int32_t think_end_id = -1;    // </think> token ID (-1 if not present)
    // Gemma-4 emits reasoning/answer structure as "<|channel>NAME...<channel|>" (NAME: thought,
    // analysis, final, ...; closing tag often omitted on short answers). Routed out of user content
    // by the state-machine filter in handlers.cpp.
    int32_t channel_open_id = -1;       // <|channel>  (-1 if not a channel model)
    int32_t channel_close_id = -1;      // <channel|>
    int32_t channel_newline_id = -1;    // '\n' used to terminate a channel header
    float default_think_budget = 0.5f;  // fraction of max_tokens for reasoning (0=disabled, 0.5=50%)
    ServerMetrics metrics;

    // Continuous batching engine: runs inference in a background thread,
    // allowing multiple concurrent requests to be processed together.
    std::unique_ptr<BatchingEngine> batching;

    // suspended: true while /admin/suspend has torn down model+engine (VRAM freed, weights in host
    // snapshot); inference endpoints answer 503. Atomic so /health reads it lock-free; writes hold state.mtx.
    std::atomic<bool> suspended{false};
    // True while load_model_into_state() runs (startup load, auto-load, swap):
    // the old engine is gone and the new one is not up. GET /ready reads it
    // without the mutex, which the swap holds for the whole load.
    std::atomic<bool> swapping{false};
    std::string loaded_model_path;             // resolved path of the loaded model
    ImpWeightSnapshot weight_snapshot = nullptr;

    // Server limits
    int max_concurrent = 64;
    int request_timeout = 300;
    int max_input_tokens = 0;  // reject prompts longer than this many tokens (0=disabled)
    int max_n = 8;             // cap on `n` completions (0=unlimited)
    int max_batch_items = 512;  // cap on rerank documents / embeddings input (0=unlimited)
    int max_logit_bias = 1024;  // cap on logit_bias entries (0=unlimited)
    int max_images = 8;         // cap on image parts per request (0=unlimited)

    // Rate limiting lives in its own unit so the CPU lane can test it
    // (#1614); ServerState cannot be constructed there.
    RateLimiter rate_limiter;
    // Admitted inference handlers right now (E-2); entered in pre-routing,
    // left in post-routing.
    InflightGate inflight;

    std::string rate_limit_key(const std::string& remote_addr, const std::string& xff) const {
        return rate_limiter.key(remote_addr, xff);
    }
    bool check_rate_limit(const std::string& ip) { return rate_limiter.allow(ip); }

    // Per-request JSONL logger (opt-in via --log-requests).
    RequestLogger request_logger;

    bool model_loaded() const { return ctx != nullptr; }

    // obs_mtx guards a {loaded, model_name} snapshot for observability endpoints, held only for a
    // trivial copy (never across inference, #889). Published under state.mtx at every (un)load.
    std::mutex obs_mtx;
    bool obs_loaded = false;
    std::string obs_model_name;
    struct ObsStatus {
        bool loaded;
        std::string model_name;
    };
    void publish_model_status(bool loaded, const std::string& name) {
        std::lock_guard<std::mutex> lk(obs_mtx);
        obs_loaded = loaded;
        obs_model_name = name;
    }
    ObsStatus model_status_snapshot() {
        std::lock_guard<std::mutex> lk(obs_mtx);
        return {obs_loaded, obs_model_name};
    }

};

// Graceful shutdown
extern std::atomic<httplib::Server*> g_server;
// g_draining: set by the signal handler before the listener stops. Already-accepted requests get
// 503 instead of being served into teardown; main() then drains in-flight work
// (server.model_swap_drain_ms) before stopping the batching engine (AUDIT_arch_2026 E-6).
extern std::atomic<bool> g_draining;

void signal_handler(int sig);
std::string make_completion_id(ServerState& state);

// A `req_imp_...` id for one HTTP request: set as the `request-id` response
// header and echoed in Anthropic error bodies (#1561).
std::string make_request_id(ServerState& state);
int64_t unix_timestamp();

std::vector<std::pair<std::string, std::string>> scan_gguf_files(const std::string& dir);
std::string find_model_path(const ServerState& state, const std::string& name);

// resolve_max_batch_size: --max-batch > [runtime] max_batch_size (imp.conf) > 0 (auto). One
// function because multiple callers must agree (AUDIT_arch_2026 G-12 removed a dead per-load
// JSON overrides object: 11 keys parsed, no request body ever populated it).
int resolve_max_batch_size(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg);

ImpConfig build_config(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg,
                       const std::string& model_path = {});
std::string load_model_into_state(ServerState& state, const std::string& path);

void handle_health(const httplib::Request& req, httplib::Response& res, ServerState& state);
// Readiness (AUDIT_arch_2026 E-5): 200 only when an inference request would be
// taken right now; 503 with a stable `code` (no_model, suspended, swapping,
// draining) otherwise. /health stays liveness and answers 200 in all four.
void handle_ready(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_models(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_model_retrieve(const httplib::Request& req, httplib::Response& res, ServerState& state,
                           const std::string& model_id);
// Context-window auto-detect probes: /props (llama.cpp n_ctx), /info (TGI max_total_tokens /
// max_input_tokens), /v1/models (vLLM max_model_len + llama.cpp meta.n_ctx_train).
void handle_props(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_info(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
// Anthropic-compatible Messages API. Non-streaming requests are a thin shim
// over handle_chat_completions; streaming requests drive the real per-token
// batching-engine loop and emit native Anthropic SSE events incrementally.
void handle_messages(const httplib::Request& req, httplib::Response& res, ServerState& state);

// POST /v1/responses — OpenAI Responses API (Agents SDK / Codex dialect);
// reuses the chat-completions path via the transform shim (responses.h).
void handle_responses(const httplib::Request& req, httplib::Response& res, ServerState& state);
// Anthropic /v1/messages/count_tokens: same body transform + tokenize chain as
// handle_messages, but never submits to the engine; returns {"input_tokens":N}.
void handle_count_tokens(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_tokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_detokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_metrics(const httplib::Request& req, httplib::Response& res, ServerState& state);

// Appends the per-tier memory gauges (I7) to a /metrics body. Lives in
// metrics_memory.cpp.
void append_memory_metrics(std::string& out, ServerState& state);
void handle_embeddings(const httplib::Request& req, httplib::Response& res, ServerState& state);

// POST /v1/rerank (also /rerank) — Cohere/Jina/vLLM-compatible reranking.
// Scores each document against the query with a cross-encoder reranker, jointly
// in one forward. Requires a reranker model to be loaded; see handlers_rerank.cpp.
void handle_rerank(const httplib::Request& req, httplib::Response& res, ServerState& state);

// POST /admin/suspend: snapshot weights to host RAM, tear down model/engine, free VRAM.
// POST /admin/resume: reload with the snapshot armed (warm restore). Both idempotent; standard
// api-key auth applies.
void handle_suspend(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_resume(const httplib::Request& req, httplib::Response& res, ServerState& state);
