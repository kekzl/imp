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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

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
        file << record.dump() << '\n';
        file.flush();
    }
};

// Prometheus-style latency histogram (cumulative buckets, in seconds).
// Lock-free: each observation bumps the matching cumulative buckets + sum +
// count. Bucket upper bounds are shared by request-duration and TTFT; both
// are sub-second-to-minutes scale so the same ladder fits.
struct LatencyHistogram {
    // le upper bounds in seconds; the implicit +Inf bucket is `count`.
    static constexpr int kNumBuckets = 11;
    static constexpr double kBounds[kNumBuckets] = {0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                                                    0.5,   1.0,  2.5,   5.0,  10.0};
    std::atomic<int64_t> buckets[kNumBuckets] = {};
    std::atomic<int64_t> count{0};
    // Sum of observed seconds, stored as micros to keep an integer atomic.
    std::atomic<int64_t> sum_us{0};

    void observe(double seconds) {
        if (seconds < 0)
            seconds = 0;
        for (int i = 0; i < kNumBuckets; ++i) {
            if (seconds <= kBounds[i])
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
    std::atomic<int64_t> last_request_duration_ms{0};
    std::atomic<int64_t> last_ttft_ms{0};  // Time to first token (ms)
    std::atomic<int64_t> model_loads_total{0};
    LatencyHistogram request_duration;  // end-to-end request latency
    LatencyHistogram ttft;              // time to first token
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
    std::atomic<int> next_id{0};
    std::atomic<int> next_tool_call_id{0};
    ServerArgs default_args;
    std::string models_dir;       // directory to scan for available .gguf files
    std::string api_key;          // if non-empty, require Bearer token auth
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

    // Server limits
    int max_concurrent = 64;
    int request_timeout = 300;
    int rate_limit = 0;        // requests per minute per IP (0=unlimited)
    int max_input_tokens = 0;  // reject prompts longer than this many tokens (0=disabled)

    // Rate limiter state: IP → list of request timestamps
    std::mutex rate_mutex;
    std::unordered_map<std::string, std::vector<std::chrono::steady_clock::time_point>> rate_tracker;

    // Per-request JSONL logger (opt-in via --log-requests).
    RequestLogger request_logger;

    bool model_loaded() const { return ctx != nullptr; }

    // Check rate limit for an IP. Returns true if allowed.
    bool check_rate_limit(const std::string& ip) {
        if (rate_limit <= 0)
            return true;
        std::lock_guard<std::mutex> lock(rate_mutex);
        auto now = std::chrono::steady_clock::now();
        auto cutoff = now - std::chrono::seconds(60);
        auto& stamps = rate_tracker[ip];
        // Remove old entries
        stamps.erase(std::remove_if(stamps.begin(), stamps.end(), [&](auto& t) { return t < cutoff; }),
                     stamps.end());
        if (static_cast<int>(stamps.size()) >= rate_limit)
            return false;
        stamps.push_back(now);
        return true;
    }
};

// Graceful shutdown
extern std::atomic<httplib::Server*> g_server;

void signal_handler(int sig);
std::string make_completion_id(ServerState& state);
int64_t unix_timestamp();

std::vector<std::pair<std::string, std::string>> scan_gguf_files(const std::string& dir);
std::string find_model_path(const ServerState& state, const std::string& name);

ImpConfig build_config(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg,
                       const std::string& model_path = {}, const json& overrides = json::object());
std::string load_model_into_state(ServerState& state, const std::string& path,
                                  const json& config_overrides = json::object());

void handle_health(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_models(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
// Anthropic-compatible Messages API. Non-streaming requests are a thin shim
// over handle_chat_completions; streaming requests drive the real per-token
// batching-engine loop and emit native Anthropic SSE events incrementally.
void handle_messages(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_tokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_detokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_metrics(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_embeddings(const httplib::Request& req, httplib::Response& res, ServerState& state);
