#include "args.h"
#include "handlers.h"
#include "runtime/presets.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

using json = nlohmann::json;

// Resolve HuggingFace cache directory:
//   $HUGGINGFACE_HUB_CACHE > $HF_HOME/hub > ~/.cache/huggingface/hub
static std::string resolve_hf_cache_dir() {
    if (const char* v = std::getenv("HUGGINGFACE_HUB_CACHE"))
        return v;
    if (const char* v = std::getenv("HF_HOME"))
        return std::string(v) + "/hub";
    if (const char* v = std::getenv("HOME"))
        return std::string(v) + "/.cache/huggingface/hub";
    return {};
}

int main(int argc, char** argv) {
    ServerArgs args = parse_server_args(argc, argv);

    // Load presets (TOML file or built-in fallback)
    imp::load_presets(args.presets_file);

    // Handle --preset list
    if (args.preset == "list") {
        imp::print_presets();
        return 0;
    }

    // Validate preset name early (allow "none" to disable auto-detection)
    if (!args.preset.empty() && args.preset != "none" &&
        !imp::find_preset(args.preset)) {
        fprintf(stderr, "Unknown preset: %s (use --preset list to see available presets)\n",
                args.preset.c_str());
        return 1;
    }

    printf("IMP Server %s\n", imp_version());

    ServerState state;
    state.default_max_tokens = args.max_tokens;
    state.default_think_budget = args.think_budget;
    state.default_args = args;

    // Set models directory (explicit flag → model parent → /models → HF cache)
    if (!args.models_dir.empty()) {
        state.models_dir = args.models_dir;
    } else if (!args.model_path.empty()) {
        auto parent = std::filesystem::path(args.model_path).parent_path().string();
        if (!parent.empty()) state.models_dir = parent;
    } else if (std::filesystem::is_directory("/models")) {
        state.models_dir = "/models";
    } else {
        std::string hf = resolve_hf_cache_dir();
        if (!hf.empty() && std::filesystem::is_directory(hf))
            state.models_dir = hf;
    }

    if (!state.models_dir.empty()) {
        printf("Models directory: %s\n", state.models_dir.c_str());
    }

    // Load model at startup if provided
    if (!args.model_path.empty()) {
        printf("Loading model: %s\n", args.model_path.c_str());
        std::string error = load_model_into_state(state, args.model_path);
        if (!error.empty()) {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
    } else {
        printf("No model specified — server will wait for POST /v1/models\n");
    }

    // Set up HTTP server
    httplib::Server svr;

    // Limit request body size to 100 MiB (prevents DoS via large base64 images)
    svr.set_payload_max_length(100 * 1024 * 1024);

    // Store API key and limits in state
    state.api_key = args.api_key;
    state.max_concurrent = args.max_concurrent;
    state.request_timeout = args.request_timeout;
    state.rate_limit = args.rate_limit;

    // CORS headers + API key auth on every response
    svr.set_pre_routing_handler([&state](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

        // Skip auth/limits for health checks and CORS preflight
        if (req.path == "/health" || req.path == "/metrics" || req.method == "OPTIONS")
            return httplib::Server::HandlerResponse::Unhandled;

        // Rate limiting (per-IP, inference endpoints only)
        if (state.rate_limit > 0 &&
            (req.path == "/v1/chat/completions" || req.path == "/v1/completions")) {
            std::string ip = req.get_header_value("X-Forwarded-For");
            if (ip.empty()) ip = req.remote_addr;
            if (!state.check_rate_limit(ip)) {
                res.status = 429;
                json err = {{"error", {{"message", "Rate limit exceeded"},
                                        {"type", "rate_limit_error"}}}};
                res.set_content(err.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        // Max concurrent requests
        if (state.max_concurrent > 0 && state.batching &&
            (req.path == "/v1/chat/completions" || req.path == "/v1/completions")) {
            if (state.batching->queue_depth() >= state.max_concurrent) {
                res.status = 429;
                json err = {{"error", {{"message", "Server overloaded, too many concurrent requests"},
                                        {"type", "rate_limit_error"}}}};
                res.set_content(err.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        // Enforce API key if configured
        if (!state.api_key.empty()) {
            std::string auth = req.get_header_value("Authorization");
            std::string expected = "Bearer " + state.api_key;
            if (auth != expected) {
                res.status = 401;
                json err = {{"error", {{"message", "Invalid API key"},
                                        {"type", "invalid_request_error"}}}};
                res.set_content(err.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    svr.Get("/health", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res, state);
    });

    svr.Get("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res, state);
    });

    svr.Post("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_load_model(req, res, state);
    });

    svr.Delete("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_unload_model(req, res, state);
    });

    svr.Post("/v1/chat/completions",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_chat_completions(req, res, state);
        });

    svr.Post("/v1/completions",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_completions(req, res, state);
        });

    svr.Post("/v1/embeddings",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_embeddings(req, res, state);
        });

    svr.Post("/tokenize",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_tokenize(req, res, state);
        });

    svr.Post("/detokenize",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_detokenize(req, res, state);
        });

    svr.Get("/metrics",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_metrics(req, res, state);
        });

    // Track failed requests via post-routing
    svr.set_post_routing_handler([&state](const httplib::Request&, httplib::Response& res) {
        if (res.status >= 500) state.metrics.requests_failed++;
    });

    // Graceful shutdown on SIGINT/SIGTERM
    g_server.store(&svr, std::memory_order_relaxed);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (!state.api_key.empty()) printf("API key: enabled\n");
    if (state.max_concurrent > 0) printf("Max concurrent: %d\n", state.max_concurrent);
    if (state.request_timeout > 0) printf("Request timeout: %ds\n", state.request_timeout);
    if (state.rate_limit > 0) printf("Rate limit: %d req/min per IP\n", state.rate_limit);

    printf("Server listening on http://%s:%d\n", args.host.c_str(), args.port);
    printf("Endpoints:\n");
    printf("  GET    /health\n");
    printf("  GET    /v1/models\n");
    printf("  POST   /v1/models          Load/swap model\n");
    printf("  DELETE /v1/models          Unload model\n");
    printf("  POST   /v1/chat/completions\n");
    printf("  POST   /v1/completions\n");
    printf("  POST   /v1/embeddings\n");
    printf("  POST   /tokenize\n");
    printf("  POST   /detokenize\n");
    printf("  GET    /metrics             Prometheus metrics\n");
    fflush(stdout);

    if (!svr.listen(args.host, args.port)) {
        // listen() returns false on stop() or bind failure
        if (!g_server.load(std::memory_order_relaxed)) {
            // Server was nulled by signal — clean shutdown
        } else {
            fprintf(stderr, "Failed to start server on %s:%d\n",
                    args.host.c_str(), args.port);
        }
    }

    g_server.store(nullptr, std::memory_order_relaxed);
    if (state.batching) {
        state.batching->stop();
        state.batching.reset();
    }
    imp_context_free(state.ctx);
    imp_model_free(state.model);
    return 0;
}
