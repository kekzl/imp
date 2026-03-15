#pragma once

#include "args.h"
#include "batching_engine.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"

#include <imp/imp.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

// Server-wide metrics (atomics for lock-free reads from /metrics endpoint)
struct ServerMetrics {
    std::atomic<int64_t> requests_total{0};
    std::atomic<int64_t> requests_failed{0};
    std::atomic<int64_t> tokens_prompt_total{0};
    std::atomic<int64_t> tokens_completion_total{0};
    std::atomic<int64_t> tokens_cached_total{0};      // Prefix cache hits
    std::atomic<int64_t> last_request_duration_ms{0};
    std::atomic<int64_t> last_ttft_ms{0};              // Time to first token (ms)
    std::atomic<int64_t> model_loads_total{0};
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

struct ServerState {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    imp::Tokenizer* tok = nullptr;
    imp::ChatTemplate chat_tpl;
    bool have_template = false;
    std::string model_name;
    std::timed_mutex mtx;
    int default_max_tokens = 8192;
    int max_seq_len = 0;
    std::atomic<int> next_id{0};
    std::atomic<int> next_tool_call_id{0};
    ServerArgs default_args;
    std::string models_dir;  // directory to scan for available .gguf files
    std::string api_key;     // if non-empty, require Bearer token auth
    bool is_think_model = false;  // model has <think> token (DeepSeek R1 etc.)
    int32_t think_start_id = -1;  // <think> token ID (-1 if not present)
    int32_t think_end_id = -1;    // </think> token ID (-1 if not present)
    float default_think_budget = 0.5f;  // fraction of max_tokens for reasoning (0=disabled, 0.5=50%)
    ServerMetrics metrics;

    // Continuous batching engine: runs inference in a background thread,
    // allowing multiple concurrent requests to be processed together.
    std::unique_ptr<BatchingEngine> batching;

    // Server limits
    int max_concurrent = 64;
    int request_timeout = 300;
    int rate_limit = 0;  // requests per minute per IP (0=unlimited)

    // Rate limiter state: IP → list of request timestamps
    std::mutex rate_mutex;
    std::unordered_map<std::string, std::vector<std::chrono::steady_clock::time_point>> rate_tracker;

    bool model_loaded() const { return ctx != nullptr; }

    // Check rate limit for an IP. Returns true if allowed.
    bool check_rate_limit(const std::string& ip) {
        if (rate_limit <= 0) return true;
        std::lock_guard<std::mutex> lock(rate_mutex);
        auto now = std::chrono::steady_clock::now();
        auto cutoff = now - std::chrono::seconds(60);
        auto& stamps = rate_tracker[ip];
        // Remove old entries
        stamps.erase(std::remove_if(stamps.begin(), stamps.end(),
                     [&](auto& t) { return t < cutoff; }), stamps.end());
        if (static_cast<int>(stamps.size()) >= rate_limit) return false;
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

ImpConfig build_config(const ServerArgs& args, const std::string& model_path = {},
                       const json& overrides = json::object());
std::string load_model_into_state(ServerState& state, const std::string& path,
                                  const json& config_overrides = json::object());

void handle_health(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_models(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_completions(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_tokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_detokenize(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_load_model(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_unload_model(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_metrics(const httplib::Request& req, httplib::Response& res, ServerState& state);
void handle_embeddings(const httplib::Request& req, httplib::Response& res, ServerState& state);
