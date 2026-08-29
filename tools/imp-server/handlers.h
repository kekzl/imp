#pragma once

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

// Observability endpoints (/health, /metrics, /v1/models) grab state.mtx with
// this bounded timeout instead of blocking unbounded: a long /v1/embeddings
// call holds the lock for its whole computation, and an unbounded wait here
// would hang a liveness probe and get a healthy container killed (#889).
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

// Prometheus-style latency histogram (cumulative buckets, in seconds).
// Lock-free: each observation bumps the matching cumulative buckets + sum +
// count.
//
// The ladder is per-instance because one ladder does not fit every quantity
// (#1577). Request duration and TTFT are sub-second-to-minutes; inter-token
// latency is single-digit MILLISECONDS - at imp's own documented decode rates
// every ITL observation landed in the first bucket of the shared ladder
// (le=0.005), so histogram_quantile returned a function of the bucket bounds
// rather than of the data.
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
    std::atomic<int64_t> requests_total{0};
    std::atomic<int64_t> requests_failed{0};
    std::atomic<int64_t> tokens_prompt_total{0};
    std::atomic<int64_t> tokens_completion_total{0};
    std::atomic<int64_t> tokens_cached_total{0};  // Prefix cache hits
    std::atomic<int64_t> requests_cancelled{0};   // Client-disconnect cancellations
    // Requests the SERVER gave up on at --request-timeout. Distinct from
    // requests_cancelled, which is the client going away: this one is imp's
    // own decision and the operator's to tune (#1640). Without it a timeout
    // was invisible - the client saw finish_reason "length", the same value a
    // completed token budget produces, and no counter moved.
    std::atomic<int64_t> requests_timed_out{0};
    // Constrained requests (json_schema/json_mode/enforced tools) that ALSO
    // request logprobs: they silently leave the ConstrainedPipeline fast path
    // for eager decode (~102 vs ~235 tok/s on the 8B reference) — surfaced
    // here so the slowdown is diagnosable (#1006).
    std::atomic<int64_t> constrained_eager_fallback{0};
    std::atomic<int64_t> last_request_duration_ms{0};
    std::atomic<int64_t> last_ttft_ms{0};  // Time to first token (ms)
    std::atomic<int64_t> model_loads_total{0};
    LatencyHistogram request_duration;  // end-to-end request latency
    LatencyHistogram ttft;              // time to first token
    // Per-TOKEN inter-token latency, on a millisecond ladder. It used to
    // observe one per-request MEAN on the request-duration ladder, which
    // answers neither "how long between tokens" nor "how does that vary"
    // (#1577).
    LatencyHistogram inter_token{LatencyHistogram::kItlBounds};
    // Time from admission to the first decode step, i.e. how long a request
    // waited behind others. Nothing measured queueing before (#1580).
    LatencyHistogram queue_time;
    // (Decode batch size lives on the BatchingEngine, where the batch is
    // formed; /metrics reads it from there.)
    // 4xx refusals. requests_failed counts 5xx only, and every rejection this
    // server is designed to emit is a 4xx - so the error counter was blind to
    // the whole designed error surface (#1579). Kept as its own series rather
    // than folded in, because "the server broke" and "the server refused" are
    // different alerts.
    std::atomic<int64_t> requests_rejected{0};
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

struct ServerState {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    // LoRA adapters loaded at startup (--lora NAME=PATH): name -> C-API id.
    // Selected per request via the "lora" body field; empty/absent = base.
    // Swaps re-capture decode graphs — single-user semantics (imp's mission),
    // the active adapter is engine-global between requests.
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
    // The batch size the CURRENT load resolved to (flag > imp.conf > per-load
    // override > 0 = engine auto). Recorded because two decisions outside
    // build_config need it - whether speculative.mtp_k=auto engages, and what
    // /health reports as the reason it did not - and both read the raw CLI
    // flag before 2026-08-29, which made `runtime.max_batch_size=1` from
    // imp.conf a single-stream server that auto still declined.
    int resolved_max_batch_size = 0;
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
    // Gemma-4 emits its reasoning/answer structure as "<|channel>NAME\n...<channel|>\n..."
    // where NAME is one of {thought, analysis, final, ...}. The closing <channel|> is
    // often omitted on short answers. We route these headers out of the user-facing
    // content stream; see handlers.cpp for the state-machine filter.
    int32_t channel_open_id = -1;       // <|channel>  (-1 if not a channel model)
    int32_t channel_close_id = -1;      // <channel|>
    int32_t channel_newline_id = -1;    // '\n' used to terminate a channel header
    float default_think_budget = 0.5f;  // fraction of max_tokens for reasoning (0=disabled, 0.5=50%)
    ServerMetrics metrics;

    // Continuous batching engine: runs inference in a background thread,
    // allowing multiple concurrent requests to be processed together.
    std::unique_ptr<BatchingEngine> batching;

    // Suspend-to-RAM (/admin/suspend, /admin/resume): while suspended the
    // model/engine are torn down (VRAM freed), the weights live in the host
    // snapshot, and inference endpoints answer 503. Atomic so /health can
    // read it without state.mtx. All writes happen under state.mtx.
    std::atomic<bool> suspended{false};
    std::string loaded_model_path;             // resolved path of the loaded model
    ImpWeightSnapshot weight_snapshot = nullptr;

    // Server limits
    int max_concurrent = 64;
    int request_timeout = 300;
    int max_input_tokens = 0;  // reject prompts longer than this many tokens (0=disabled)
    int max_n = 8;             // cap on `n` completions (0=unlimited)
    int max_batch_items = 512;  // cap on rerank documents / embeddings input (0=unlimited)
    int max_logit_bias = 1024;  // cap on logit_bias entries (0=unlimited)

    // Rate limiting lives in its own unit so the CPU lane can test it
    // (#1614); ServerState cannot be constructed there.
    RateLimiter rate_limiter;

    std::string rate_limit_key(const std::string& remote_addr, const std::string& xff) const {
        return rate_limiter.key(remote_addr, xff);
    }
    bool check_rate_limit(const std::string& ip) { return rate_limiter.allow(ip); }

    // Per-request JSONL logger (opt-in via --log-requests).
    RequestLogger request_logger;

    bool model_loaded() const { return ctx != nullptr; }

    // Lock-free-ish snapshot of {loaded, model_name} for the observability
    // endpoints (/health, /metrics, /v1/models). Guarded by its own tiny mutex
    // that is only ever held for a trivial copy — never across inference — so
    // these endpoints can read model status without contending on `mtx`, which
    // a long /v1/embeddings call deliberately holds for its whole computation
    // (#889). Published under `mtx` at every (un)load; read on lock timeout.
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

void signal_handler(int sig);
std::string make_completion_id(ServerState& state);

// A `req_imp_...` id for one HTTP request: set as the `request-id` response
// header and echoed in Anthropic error bodies (#1561).
std::string make_request_id(ServerState& state);
int64_t unix_timestamp();

std::vector<std::pair<std::string, std::string>> scan_gguf_files(const std::string& dir);
std::string find_model_path(const ServerState& state, const std::string& name);

// Max batch size for one load: --max-batch > [runtime] max_batch_size from
// imp.conf > the per-load JSON override > 0 (engine auto-sizes). One place,
// because build_config is not the only caller that has to agree with it.
int resolve_max_batch_size(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg,
                           const nlohmann::json& overrides);

ImpConfig build_config(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg,
                       const std::string& model_path = {}, const json& overrides = json::object());
std::string load_model_into_state(ServerState& state, const std::string& path,
                                  const json& config_overrides = json::object());

void handle_health(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_models(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_model_retrieve(const httplib::Request& req, httplib::Response& res, ServerState& state,
                           const std::string& model_id);
// Context-window probes for OpenAI-compatible clients that auto-detect the max
// context length. /props follows llama.cpp (n_ctx), /info follows TGI
// (max_total_tokens / max_input_tokens); /v1/models carries vLLM's
// max_model_len + llama.cpp's meta.n_ctx_train on the model object.
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

// POST /admin/suspend — snapshot weights to host RAM, tear the model/engine
// down, free (approximately) all VRAM. POST /admin/resume — reload the same
// model with the snapshot armed (warm weight restore) and serve again.
// Both idempotent; auth via the standard pre-routing API-key check.
void handle_suspend(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_resume(const httplib::Request& req, httplib::Response& res, ServerState& state);
