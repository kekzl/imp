#include "args.h"
#include "handlers.h"
#include "utils.h"
#include "model/hf_hub.h"
#include "runtime/config.h"
#include "runtime/process_diag.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <csignal>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>

using json = nlohmann::json;

int main(int argc, char** argv) {
    ServerArgs args = parse_server_args(argc, argv);

    if (args.model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        return 1;
    }

    printf("IMP Server %s\n", imp_version());

    ServerState state;
    state.default_max_tokens = args.max_tokens;
    state.default_think_budget = args.think_budget;
    state.default_args = args;

    // Load imp.conf (if present) + apply --set overrides, then stash for
    // Engine::init to pick up (Phase 5 Track D follow-up: replaces the
    // RuntimeConfig::install() process-wide singleton). The server may load
    // a model at runtime (auto-load on first request when started without
    // --model); load_model_into_state re-stashes the same config snapshot
    // before each Engine construction.
    state.runtime_config = imp::RuntimeConfig::load(args.config_path, args.config_overrides);
    imp::process_diag_install(state.runtime_config);
    imp::set_pending_runtime_config(state.runtime_config);

    ImpModelFormat resolved_format = IMP_FORMAT_GGUF;
    std::string resolved_model = imp::resolve_model_auto(args.model_path, resolved_format, args.revision);
    if (resolved_model.empty()) {
        fprintf(stderr, "Failed to resolve model: %s\n", args.model_path.c_str());
        return 1;
    }
    if (resolved_model != args.model_path) {
        printf("Resolved model: %s -> %s (%s)\n", args.model_path.c_str(), resolved_model.c_str(),
               resolved_format == IMP_FORMAT_SAFETENSORS ? "SafeTensors" : "GGUF");
    }

    // Models directory: explicit --models-dir overrides, else the resolved model's parent.
    if (!args.models_dir.empty()) {
        state.models_dir = args.models_dir;
    } else {
        auto parent = std::filesystem::path(resolved_model).parent_path().string();
        if (!parent.empty())
            state.models_dir = parent;
    }
    if (!state.models_dir.empty()) {
        printf("Models directory: %s\n", state.models_dir.c_str());
    }

    printf("Loading model: %s\n", resolved_model.c_str());
    {
        std::string error = load_model_into_state(state, resolved_model);
        if (!error.empty()) {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
    }

    // --lora NAME=PATH: load PEFT adapters once; requests select by name.
    for (const auto& [name, path] : args.loras) {
        int32_t id = 0;
        if (imp_lora_load(state.ctx, path.c_str(), &id) != IMP_SUCCESS) {
            fprintf(stderr, "Failed to load LoRA adapter '%s' from %s\n", name.c_str(), path.c_str());
            return 1;
        }
        state.lora_ids[name] = id;
        printf("LoRA adapter loaded: %s (id=%d) from %s\n", name.c_str(), id, path.c_str());
    }

    // Set up HTTP server
    httplib::Server svr;

    // Limit request body size to 100 MiB (prevents DoS via large base64 images)
    svr.set_payload_max_length(static_cast<size_t>(100) * 1024 * 1024);

    // Store API key and limits in state
    state.api_key = args.api_key;
    state.max_concurrent = args.max_concurrent;
    state.request_timeout = args.request_timeout;
    state.rate_limit = args.rate_limit;

    // --max-input-tokens <n>: reject prompts whose tokenized length exceeds
    // <n> with HTTP 400 before prefill (0 = disabled).
    state.max_input_tokens = args.max_input_tokens;
    if (!args.log_requests_path.empty()) {
        if (state.request_logger.open(args.log_requests_path)) {
            printf("Request logging: appending JSONL to %s\n", args.log_requests_path.c_str());
        }
    }

    // CORS headers + API key auth on every response
    svr.set_pre_routing_handler([&state](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

        // Skip auth/limits for health checks and CORS preflight
        if (req.path == "/health" || req.path == "/metrics" || req.method == "OPTIONS")
            return httplib::Server::HandlerResponse::Unhandled;

        // Rate limiting (per-IP, inference endpoints only)
        if (state.rate_limit > 0 && (req.path == "/v1/chat/completions" || req.path == "/v1/completions")) {
            std::string ip = req.get_header_value("X-Forwarded-For");
            if (ip.empty())
                ip = req.remote_addr;
            if (!state.check_rate_limit(ip)) {
                res.status = 429;
                json err = {{"error", {{"message", "Rate limit exceeded"}, {"type", "rate_limit_error"}}}};
                res.set_content(err.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        // Max concurrent requests
        if (state.max_concurrent > 0 &&
            (req.path == "/v1/chat/completions" || req.path == "/v1/completions")) {
            int queue = 0;
            {
                std::lock_guard<std::timed_mutex> lock(state.mtx);
                if (state.batching)
                    queue = state.batching->queue_depth();
            }
            if (queue >= state.max_concurrent) {
                res.status = 429;
                json err = {{"error",
                             {{"message", "Server overloaded, too many concurrent requests"},
                              {"type", "rate_limit_error"}}}};
                res.set_content(err.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        // Enforce API key if configured. The constant-time compare lives in
        // bearer_token_matches() (utils.cpp) so it is unit-testable (test-core).
        if (!state.api_key.empty()) {
            std::string auth = req.get_header_value("Authorization");
            if (!bearer_token_matches(auth, state.api_key)) {
                res.status = 401;
                json err = {{"error", {{"message", "Invalid API key"}, {"type", "invalid_request_error"}}}};
                res.set_content(err.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });

    svr.Get("/health", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res, state);
    });

    svr.Get("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res, state);
    });

    svr.Post("/v1/chat/completions", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_chat_completions(req, res, state);
    });

    svr.Post("/v1/completions", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_completions(req, res, state);
    });

    // Anthropic-compatible Messages API. Supports both non-streaming and
    // native incremental SSE streaming (real per-token, not synthetic replay).
    svr.Post("/v1/messages", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res, state);
    });

    svr.Post("/v1/embeddings", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_embeddings(req, res, state);
    });

    svr.Post("/tokenize", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_tokenize(req, res, state);
    });

    svr.Post("/detokenize", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_detokenize(req, res, state);
    });

    svr.Get("/metrics", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res, state);
    });

    // Global safety net: any exception that escapes a handler must become a
    // JSON error envelope, never a bare "500 Internal Server Error". Malformed
    // or invalid-UTF-8 client input surfaces as a json::exception (parse_error,
    // or type_error.316 when an error message echoes the offending bytes) — map
    // those to 400. Everything else is a genuine internal failure → 500, but
    // still with a JSON body. dump_safe (inside send_json_error) guarantees the
    // envelope itself can't throw on bad bytes.
    svr.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            try {
                std::rethrow_exception(std::move(ep));
            } catch (const nlohmann::json::exception& e) {
                send_json_error(res, 400, "invalid_request_error", e.what());
            } catch (const std::exception& e) {
                send_json_error(res, 500, "server_error", e.what());
            } catch (...) {
                send_json_error(res, 500, "server_error", "unknown internal error");
            }
        });

    // Track failed requests via post-routing
    svr.set_post_routing_handler([&state](const httplib::Request&, httplib::Response& res) {
        if (res.status >= 500)
            state.metrics.requests_failed++;
    });

    // Graceful shutdown on SIGINT/SIGTERM
    g_server.store(&svr, std::memory_order_relaxed);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (!state.api_key.empty())
        printf("API key: enabled\n");
    if (state.max_concurrent > 0)
        printf("Max concurrent: %d\n", state.max_concurrent);
    if (state.request_timeout > 0)
        printf("Request timeout: %ds\n", state.request_timeout);
    if (state.rate_limit > 0)
        printf("Rate limit: %d req/min per IP\n", state.rate_limit);
    if (state.max_input_tokens > 0)
        printf("Max input tokens: %d\n", state.max_input_tokens);

    printf("Server listening on http://%s:%d\n", args.host.c_str(), args.port);
    printf("Endpoints:\n");
    printf("  GET    /health\n");
    printf("  GET    /v1/models\n");
    printf("  POST   /v1/chat/completions\n");
    printf("  POST   /v1/completions\n");
    printf("  POST   /v1/messages          Anthropic-compatible (streaming + non-streaming)\n");
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
            fprintf(stderr, "Failed to start server on %s:%d\n", args.host.c_str(), args.port);
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
