#include "args.h"
#include "handlers.h"
#include "utils.h"
#include "webui_asset.h"  // generated: IMP_WEBUI_HTML
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

// The routes that consume engine capacity and must go through --max-concurrent
// admission control. /v1/messages (Anthropic) and /v1/embeddings were once
// omitted from these checks, silently bypassing both guards (the non-stream
// /v1/messages path reaches inference by directly calling
// handle_chat_completions() without re-entering pre-routing).
static bool is_inference_endpoint(const std::string& path) {
    return path == "/v1/chat/completions" || path == "/v1/completions" || path == "/v1/responses" ||
           path == "/v1/messages" || path == "/v1/embeddings" || path == "/v1/rerank" ||
           path == "/rerank";
}

// What the RATE limit covers is a wider set, and the difference is the defect
// in #1615: --max-concurrent protects the engine, so it belongs on the routes
// that queue work, but --rate-limit is there to stop a client from hammering
// the process at all. Tokenisation walks the whole prompt through the BPE
// merge table on a server thread, and /admin/suspend flips global state; both
// were reachable at any rate. The exemptions below are deliberate and short.
static bool is_rate_limited_endpoint(const std::string& path) {
    if (path == "/health" || path == "/metrics")
        return false;
    return true;
}

int main(int argc, char** argv) {
    ServerArgs args = parse_server_args(argc, argv);

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
    std::vector<std::string> rejected_overrides;
    state.runtime_config = imp::RuntimeConfig::load(args.config_path, args.config_overrides, &rejected_overrides);
    if (!rejected_overrides.empty()) {
        // Serving with a configuration the operator did not ask for is worse
        // than refusing to start.
        for (const auto& bad : rejected_overrides)
            fprintf(stderr, "Error: --set %s\n", bad.c_str());
        fprintf(stderr, "See imp.conf.example for the key names.\n");
        return 1;
    }
    imp::process_diag_install(state.runtime_config);
    imp::set_pending_runtime_config(state.runtime_config);

    // --model is optional (it has always been documented that way in --help).
    // Without it the server starts model-less: the request-validation surface,
    // /health, /v1/models and /metrics all answer, and the first request that
    // names a model in --models-dir auto-loads it (ensure_model_loaded). A
    // request that cannot resolve a model gets 503 — never a silent success.
    // This is also what lets CI run the shipping binary on a GPU-less runner
    // instead of a Python stand-in (#1302).
    ImpModelFormat resolved_format = IMP_FORMAT_GGUF;
    std::string resolved_model;
    if (!args.model_path.empty()) {
        resolved_model = imp::resolve_model_auto(args.model_path, resolved_format, args.revision);
        if (resolved_model.empty()) {
            fprintf(stderr, "Failed to resolve model: %s\n", args.model_path.c_str());
            return 1;
        }
        if (resolved_model != args.model_path) {
            printf("Resolved model: %s -> %s (%s)\n", args.model_path.c_str(), resolved_model.c_str(),
                   resolved_format == IMP_FORMAT_SAFETENSORS ? "SafeTensors" : "GGUF");
        }
    }

    // Models directory: explicit --models-dir overrides, else the resolved model's parent.
    if (!args.models_dir.empty()) {
        state.models_dir = args.models_dir;
    } else if (!resolved_model.empty()) {
        auto parent = std::filesystem::path(resolved_model).parent_path().string();
        if (!parent.empty())
            state.models_dir = parent;
    }
    if (!state.models_dir.empty()) {
        printf("Models directory: %s\n", state.models_dir.c_str());
    }

    // Set up the HTTP server and BIND the listen socket now — before the (slow)
    // model load — so a port conflict fails in <1 s instead of after a full
    // model load (#760). Routes are registered once the model is ready;
    // listen_after_bind() below starts accepting connections then.
    httplib::Server svr;
    if (svr.bind_to_port(args.host, args.port) == 0) {
        fprintf(stderr, "Failed to start server on %s:%d: port already in use\n", args.host.c_str(),
                args.port);
        return 1;
    }

    if (resolved_model.empty()) {
        if (state.models_dir.empty()) {
            printf(
                "No model: started model-less and no --models-dir to auto-load from — "
                "inference endpoints answer 503.\n");
        } else {
            printf(
                "No model: started model-less — the first request naming a model in %s "
                "loads it.\n",
                state.models_dir.c_str());
        }
    } else {
        printf("Loading model: %s\n", resolved_model.c_str());
        std::string error = load_model_into_state(state, resolved_model);
        if (!error.empty()) {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
    }

    // --lora NAME=PATH: load PEFT adapters once; requests select by name.
    // An adapter needs its base model resident, so this combination is refused
    // at startup rather than silently dropping the adapters.
    if (!args.loras.empty() && !state.model_loaded()) {
        fprintf(stderr, "Error: --lora requires --model (adapters attach to a loaded base model)\n");
        return 1;
    }
    for (const auto& [name, path] : args.loras) {
        int32_t id = 0;
        if (imp_lora_load(state.ctx, path.c_str(), &id) != IMP_SUCCESS) {
            fprintf(stderr, "Failed to load LoRA adapter '%s' from %s\n", name.c_str(), path.c_str());
            return 1;
        }
        state.lora_ids[name] = id;
        printf("LoRA adapter loaded: %s (id=%d) from %s\n", name.c_str(), id, path.c_str());
    }

    // (svr was created + bound to the port above, before the model load.)

    // Limit request body size to 100 MiB (prevents DoS via large base64 images)
    svr.set_payload_max_length(static_cast<size_t>(100) * 1024 * 1024);

    // Connection-level limits (#1622). Every one of these was previously
    // whatever the build-time cpp-httplib defaulted to, which this repo cannot
    // even read: the library is fetched at a pinned tag, not vendored. A slow
    // reader holding a socket open costs a worker thread either way, so the
    // point is that the number is ours and is written down.
    //
    // The write timeout is the one that must not be tightened casually: a
    // streamed completion writes for as long as it generates, so 600 s is a
    // deliberate asymmetry against the 60 s read side.
    svr.set_read_timeout(args.read_timeout, 0);
    svr.set_write_timeout(args.write_timeout, 0);
    svr.set_keep_alive_max_count(args.keep_alive_max);

    // Store API key and limits in state
    state.api_key = args.api_key;
    state.metrics_require_auth = args.metrics_require_auth;
    state.max_concurrent = args.max_concurrent;
    state.request_timeout = args.request_timeout;
    state.rate_limiter.limit = args.rate_limit;

    // --max-input-tokens <n>: reject prompts whose tokenized length exceeds
    // <n> with HTTP 400 before prefill (0 = disabled).
    state.max_input_tokens = args.max_input_tokens;
    state.max_n = args.max_n;
    state.max_batch_items = args.max_batch_items;
    state.max_logit_bias = args.max_logit_bias;

    // --trusted-proxy a,b,c
    {
        const std::string& tp = args.trusted_proxies;
        size_t pos = 0;
        while (pos < tp.size()) {
            size_t comma = tp.find(',', pos);
            if (comma == std::string::npos)
                comma = tp.size();
            std::string one = tp.substr(pos, comma - pos);
            const size_t b = one.find_first_not_of(" \t");
            const size_t e = one.find_last_not_of(" \t");
            if (b != std::string::npos)
                state.rate_limiter.trusted_proxies.insert(one.substr(b, e - b + 1));
            pos = comma + 1;
        }
        if (!state.rate_limiter.trusted_proxies.empty())
            printf("Trusted proxies: %zu (X-Forwarded-For believed from these peers)\n",
                   state.rate_limiter.trusted_proxies.size());
    }
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

        // Skip auth/limits for health checks and CORS preflight. /metrics is
        // exempt by default so a stock Prometheus scrape works, but it leaks the
        // loaded model name, d_model and cumulative token counts — so
        // --metrics-require-auth folds it back under the api_key check (#1207).
        const bool metrics_exempt = (req.path == "/metrics" && !state.metrics_require_auth);
        if (req.path == "/health" || metrics_exempt || req.method == "OPTIONS")
            return httplib::Server::HandlerResponse::Unhandled;

        // Rate limiting (per-peer, everything but /health and /metrics)
        if (state.rate_limiter.limit > 0 && is_rate_limited_endpoint(req.path)) {
            const std::string ip = state.rate_limit_key(req.remote_addr,
                                                        req.get_header_value("X-Forwarded-For"));
            if (!state.check_rate_limit(ip)) {
                // Both dialects call this rate_limit_error; only the envelope
                // differs, and this site shipped the OpenAI one to every
                // endpoint (#1551).
                send_dialect_error(res, req.path, 429, "rate_limit_error", "rate_limit_error",
                                   "Rate limit exceeded");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        // Max concurrent requests
        if (state.max_concurrent > 0 && is_inference_endpoint(req.path)) {
            int queue = 0;
            {
                std::lock_guard<std::timed_mutex> lock(state.mtx);
                if (state.batching)
                    queue = state.batching->queue_depth();
            }
            if (queue >= state.max_concurrent) {
                // Anthropic's name for "too many in flight right now" is
                // overloaded_error (529 upstream; the status here stays 429,
                // which is what this server's own docs and clients expect).
                send_dialect_error(res, req.path, 429, "rate_limit_error", "overloaded_error",
                                   "Server overloaded, too many concurrent requests");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        // Enforce API key if configured. The constant-time compare lives in
        // api_key_matches()/bearer_token_matches() (utils.cpp) so it is
        // unit-testable (test-core). Accept both the OpenAI `Authorization:
        // Bearer` and the Anthropic `x-api-key` header so real Anthropic SDK
        // clients aren't 401'd on /v1/messages.
        if (!state.api_key.empty()) {
            std::string auth = req.get_header_value("Authorization");
            std::string xkey = req.get_header_value("x-api-key");
            if (!api_key_matches(auth, xkey, state.api_key)) {
                send_dialect_error(res, req.path, 401, "invalid_request_error", "authentication_error",
                                   "Invalid API key");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });

    // Web UI — embedded at build time (see cmake/embed_webui.cmake), so the
    // server has no asset directory to locate.
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(IMP_WEBUI_HTML, "text/html; charset=utf-8");
    });

    svr.Get("/health", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res, state);
    });

    svr.Get("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res, state);
    });

    // The only path-parameter route in this server (#1599). A model id can
    // contain a slash (a HuggingFace repo id), so the pattern is greedy.
    svr.Get(R"(/v1/models/(.+))", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_model_retrieve(req, res, state, req.matches[1].str());
    });

    // Context-window auto-detection probes for OpenAI-compatible clients:
    // /props is the llama.cpp shape, /info the TGI shape (/v1/models also
    // carries vLLM's max_model_len). A client written for any of the three can
    // read imp's context length without a hard-coded table.
    svr.Get("/props", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_props(req, res, state);
    });

    svr.Get("/info", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_info(req, res, state);
    });

    svr.Post("/v1/chat/completions", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_chat_completions(req, res, state);
    });

    svr.Post("/v1/responses", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res, state);
    });

    svr.Post("/v1/completions", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_completions(req, res, state);
    });

    // Anthropic-compatible Messages API. Supports both non-streaming and
    // native incremental SSE streaming (real per-token, not synthetic replay).
    svr.Post("/v1/messages", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res, state);
    });

    // Anthropic token counting (Claude Code uses it for context tracking /
    // auto-compaction). Tokenizes exactly like a real request, no generation.
    svr.Post("/v1/messages/count_tokens", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_count_tokens(req, res, state);
    });

    svr.Post("/v1/rerank", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_rerank(req, res, state);
    });
    // Cohere and TEI clients post to the unversioned path; vLLM serves both.
    svr.Post("/rerank", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_rerank(req, res, state);
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

    svr.Post("/admin/suspend", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_suspend(req, res, state);
    });

    svr.Post("/admin/resume", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_resume(req, res, state);
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
    svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
        try {
            std::rethrow_exception(std::move(ep));
        } catch (const nlohmann::json::exception& e) {
            send_dialect_error(res, req.path, 400, "invalid_request_error", "invalid_request_error",
                               e.what());
        } catch (const std::exception& e) {
            send_dialect_error(res, req.path, 500, "server_error", "api_error", e.what());
        } catch (...) {
            send_dialect_error(res, req.path, 500, "server_error", "api_error", "unknown internal error");
        }
    });

    // Any error response that would go out with an empty body gets the same
    // JSON envelope every handler uses. The reachable case is an unmatched
    // route: httplib answers 404 with zero bytes, so a client that does
    // `r.json()["error"]["message"]` on a typo'd path got a parse error instead
    // of a message. The mock CI tests against has always sent an envelope here
    // (#1302). Responses that already carry a body — i.e. everything a handler
    // produced — are left untouched: this callback runs for EVERY status >= 400.
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (!res.body.empty())
            return httplib::Server::HandlerResponse::Unhandled;
        const bool not_found = res.status == 404;
        // The method and the path are the client's bytes. Echoing them raw put
        // arbitrary input into a response body and into `.dump()`, which throws
        // json::type_error.316 on ill-formed UTF-8 - so a 404 for a path with a
        // stray 0x80 in it produced a 500 with an empty body instead (#1618).
        // Both halves are fixed: the echo is sanitised and truncated, and the
        // serialiser is the one that cannot throw.
        const std::string msg = not_found ? "Unknown endpoint: " + sanitize_for_echo(req.method, 16) + " " +
                                                sanitize_for_echo(req.path, 128)
                                          : "Request failed with status " + std::to_string(res.status);
        // api_error, not server_error: the latter is not an Anthropic error
        // type (#1556).
        const char* anthropic_type = not_found           ? "not_found_error"
                                     : res.status >= 500 ? "api_error"
                                                         : "invalid_request_error";
        const char* openai_type = res.status >= 500 ? "server_error" : "invalid_request_error";
        const int status = res.status;
        send_dialect_error(res, req.path, status, openai_type, anthropic_type, msg);
        return httplib::Server::HandlerResponse::Handled;
    });

    // Track failed requests via post-routing
    svr.set_post_routing_handler([&state](const httplib::Request&, httplib::Response& res) {
        if (res.status >= 500)
            state.metrics.requests_failed++;
        // 4xx is where this server puts every refusal it is designed to make
        // (tools/imp-server/CLAUDE.md), so counting only 5xx left the entire
        // designed error surface invisible (#1579). Separate series, because
        // "the server broke" and "the server refused" want different alerts.
        else if (res.status >= 400)
            state.metrics.requests_rejected++;
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
    if (state.rate_limiter.limit > 0)
        printf("Rate limit: %d req/min per peer\n", state.rate_limiter.limit);
    if (state.max_input_tokens > 0)
        printf("Max input tokens: %d\n", state.max_input_tokens);

    printf("Server listening on http://%s:%d\n", args.host.c_str(), args.port);
    printf("Endpoints:\n");
    printf("  GET    /                    web UI — open this in a browser\n");
    printf("  GET    /health\n");
    printf("  GET    /v1/models            (vLLM max_model_len + llama.cpp meta.n_ctx_train)\n");
    printf("  GET    /props               llama.cpp-compatible context probe (n_ctx)\n");
    printf("  GET    /info                TGI-compatible context probe (max_total_tokens)\n");
    printf("  POST   /v1/chat/completions\n");
    printf("  POST   /v1/responses          OpenAI Responses API (Agents SDK / Codex dialect)\n");
    printf("  POST   /v1/completions\n");
    printf("  POST   /v1/messages          Anthropic-compatible (streaming + non-streaming)\n");
    printf("  POST   /v1/messages/count_tokens\n");
    printf("  POST   /v1/embeddings\n");
    printf("  POST   /v1/rerank            (also /rerank) cross-encoder reranking\n");
    printf("  POST   /tokenize\n");
    printf("  POST   /detokenize\n");
    printf("  POST   /admin/suspend       Park weights in host RAM, free the GPU\n");
    printf("  POST   /admin/resume        Restore weights, serve again\n");
    printf("  GET    /metrics             Prometheus metrics\n");
    fflush(stdout);

    if (!svr.listen_after_bind()) {
        // listen_after_bind() returns false on stop() or bind failure
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
    imp_weights_snapshot_free(state.weight_snapshot);  // non-null only when suspended
    return 0;
}
