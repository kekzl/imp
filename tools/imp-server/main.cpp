#include "args.h"
#include "common/exit_codes.h"
#include "handlers.h"
#include "utils.h"
#include "webui_asset.h"  // generated: IMP_WEBUI_HTML
#include "model/hf_hub.h"
#include "runtime/config.h"
#include "core/process_diag.h"
#include "runtime/process_diag_install.h"

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

// is_inference_endpoint: routes gated by --max-concurrent admission control. /v1/messages and
// /v1/embeddings were once omitted, silently bypassing it (non-stream /v1/messages calls
// handle_chat_completions() directly, without re-entering pre-routing).
static bool is_inference_endpoint(const std::string& path) {
    return path == "/v1/chat/completions" || path == "/v1/completions" || path == "/v1/responses" ||
           path == "/v1/messages" || path == "/v1/embeddings" || path == "/v1/rerank" ||
           path == "/rerank";
}

// Set by the pre-routing hook when it entered the in-flight gate for this
// request, cleared by the post-routing hook; httplib runs both on the worker
// thread that serves the request (AUDIT_arch_2026 E-2).
static thread_local bool t_inflight_entered = false;

// is_rate_limited_endpoint: a WIDER set than --max-concurrent (which protects the engine) -
// --rate-limit stops a client hammering the process itself. #1615: tokenization (BPE walk on a
// server thread) and /admin/suspend (global state) were reachable at any rate; exemptions here are
// deliberate.
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

    // Loads imp.conf + --set overrides, stashed for Engine::init (replaces the RuntimeConfig::install
    // process-wide singleton). load_model_into_state re-stashes the same snapshot before each Engine
    // construction, since the server may load/swap models at runtime.
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
        state.tracer.init(state.runtime_config.server.otlp_endpoint,
                          state.runtime_config.server.otlp_service_name, imp_version());
        imp::set_pending_runtime_config(state.runtime_config);

        // --model is optional: model-less start still answers /health, /v1/models, /metrics, and the
        // first request naming a model in --models-dir auto-loads it (ensure_model_loaded); an
        // unresolvable name is 503, never a silent success. Lets CI run the shipping binary GPU-less (#1302).
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

    // Binds the listen socket before the (slow) model load so a port conflict fails in <1s (#760),
    // not after a full load. Routes register once the model is ready; listen_after_bind() then
    // starts accepting.
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

    // Connection-level limits (#1622): previously whatever cpp-httplib's build-time defaults were
    // (unreadable - the library is fetched at a pinned tag, not vendored). Write timeout (600s) is
    // deliberately asymmetric to the 60s read timeout: a streamed completion writes as long as it generates.
    svr.set_read_timeout(args.read_timeout, 0);
    svr.set_write_timeout(args.write_timeout, 0);
    svr.set_keep_alive_max_count(args.keep_alive_max);
    // Disables Nagle on accepted sockets (cpp-httplib defaults it OFF): the streaming path writes one
    // small SSE frame per token, and Nagle would delay it behind the peer's ACK (~40ms), adding ITL a
    // network client observes. Loopback and throughput are unaffected either way.
    svr.set_tcp_nodelay(true);
    // Worker pool sized to COVER CONCURRENT STREAMS, not cores: a streamed completion holds its
    // worker for the whole generation, and the httplib default (max(8,cores-1)) queued the tail
    // of a 32-stream burst 4-7s late (2026-08-25). +8 covers health/admin while every stream slot is held.
    // Task queue is bounded at exactly one pool's worth; past that httplib closes the connection at
    // once (AUDIT_arch_2026 E-2: an unbounded queue hung 9/10 requests with no timer instead of 429).
    svr.new_task_queue = [&args] {
        const size_t workers = static_cast<size_t>(args.max_concurrent) + 8;
        return new httplib::ThreadPool(workers, /*max_n=*/0, /*mqr=*/workers);
    };

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
    state.max_images = args.max_images;

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

        // /metrics is exempt from auth/limits by default (stock Prometheus scrape works out of the box),
        // but leaks model name, d_model, and cumulative token counts - --metrics-require-auth folds it
        // back under the api_key check (#1207).
        const bool metrics_exempt = (req.path == "/metrics" && !state.metrics_require_auth);
        if (req.path == "/health" || req.path == "/ready" || metrics_exempt || req.method == "OPTIONS")
            return httplib::Server::HandlerResponse::Unhandled;

        // Shutting down: a request httplib accepted before the listener stopped
        // is refused rather than started into the teardown (AUDIT_arch_2026 E-6).
        if (g_draining.load(std::memory_order_relaxed) && is_inference_endpoint(req.path)) {
            send_dialect_error(res, req.path, 503, "server_error", "overloaded_error",
                               "Server is shutting down");
            return httplib::Server::HandlerResponse::Handled;
        }

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

        // The in-flight depth read takes state.mtx, which a model swap or /admin/suspend can hold for
        // minutes: a blocking acquire here parked every arriving worker inside the load-shedding guard
        // until the thread pool was gone (AUDIT_arch_2026 F2-3). A lock timeout means "engine busy", 503.
        if (state.max_concurrent > 0 && is_inference_endpoint(req.path)) {
            int queue = 0;
            {
                std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
                if (!lock.owns_lock()) {
                    send_dialect_error(res, req.path, 503, "server_error", "overloaded_error",
                                       "Server busy (model swap or suspend in progress), retry shortly");
                    return httplib::Server::HandlerResponse::Handled;
                }
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
            // The depth read above is check-then-submit (a race let N workers all read "63 in flight" and
            // all get admitted, AUDIT_arch_2026 E-2). This gate counts admitted handlers atomically instead;
            // the post-routing hook on the same thread releases it.
            if (!state.inflight.try_enter(state.max_concurrent)) {
                send_dialect_error(res, req.path, 429, "rate_limit_error", "overloaded_error",
                                   "Server overloaded, too many concurrent requests");
                return httplib::Server::HandlerResponse::Handled;
            }
            t_inflight_entered = true;
        }

        // API-key enforcement uses a constant-time compare (api_key_matches/bearer_token_matches,
        // utils.cpp, unit-tested in test-core). Accepts both OpenAI `Authorization: Bearer` and
        // Anthropic `x-api-key` so either SDK works unmodified.
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

    svr.Get("/ready",
            [&state](const httplib::Request& req, httplib::Response& res) { handle_ready(req, res, state); });

    svr.Get("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res, state);
    });

    // The only path-parameter route in this server (#1599). A model id can
    // contain a slash (a HuggingFace repo id), so the pattern is greedy.
    svr.Get(R"(/v1/models/(.+))", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_model_retrieve(req, res, state, req.matches[1].str());
    });

    // /props (llama.cpp shape), /info (TGI shape), and /v1/models (vLLM max_model_len) all expose
    // context length so a client written for any of the three auto-detects it without a hardcoded table.
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

    // Global exception handler: every escaping exception becomes a JSON error envelope, never a bare
    // "500 Internal Server Error". json::exception (parse_error, or type_error.316 echoing bad bytes)
    // maps to 400; everything else is a genuine 500, still JSON. dump_safe cannot throw on bad bytes.
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

    // Wraps any >=400 response that would go out with an EMPTY body (e.g. an unmatched route's bare
    // httplib 404) in the standard JSON error envelope - a client doing
    // r.json()["error"]["message"] got a parse error instead (#1302). A body already present is untouched.
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (!res.body.empty())
            return httplib::Server::HandlerResponse::Unhandled;
        const bool not_found = res.status == 404;
        // Echoes method+path sanitized and truncated (#1618): raw client bytes fed straight into
        // .dump() threw json::type_error.316 on ill-formed UTF-8, so a 404 for a bad path produced a 500
        // with an empty body instead.
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
    svr.set_post_routing_handler([&state](const httplib::Request& req, httplib::Response& res) {
        // Leave the in-flight gate the pre-routing hook entered on this
        // thread (E-2). Conditional on the flag, not on the path: a request
        // the hook refused before entering must not be counted out.
        if (t_inflight_entered) {
            t_inflight_entered = false;
            state.inflight.leave();
        }
        // X-Request-Id is echoed on EVERY response, including refusals, so a client can join its trace
        // (has_header avoids double-setting it when a generation handler already did). sanitize_for_echo
        // turns CR/LF into '.' as the header-injection guard.
        if (!res.has_header("X-Request-Id")) {
            const std::string cid = req.get_header_value("X-Request-Id");
            if (!cid.empty())
                res.set_header("X-Request-Id", sanitize_for_echo(cid, 128));
        }
        if (res.status >= 500)
            state.metrics.requests_failed++;
        // 4xx is where this server puts every refusal it is designed to make (tools/imp-server/CLAUDE.md);
        // counting only 5xx left that whole surface invisible (#1579). Separate series: "the server
        // broke" and "the server refused" want different alerts.
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

    // listen_after_bind() returns false both on stop() and on a listen failure, and both used to exit
    // 0 - a supervisor restarting on non-zero never restarted a server that failed to listen (#1584).
    // g_server (nulled by the signal handler) distinguishes "asked to stop" from "could not serve".
    int exit_status = 0;
    if (!svr.listen_after_bind()) {
        if (!g_server.load(std::memory_order_relaxed)) {
            // Server was nulled by signal — clean shutdown
        } else {
            fprintf(stderr, "Failed to start server on %s:%d\n", args.host.c_str(), args.port);
            exit_status = imp::tools::exit_code_for(IMP_ERROR_INTERNAL);
        }
    }

    g_server.store(nullptr, std::memory_order_relaxed);
    g_draining.store(true, std::memory_order_relaxed);
    if (state.batching) {
        // Drains in-flight generations before engine teardown (stop() would cancel them) - same
        // contract/budget as a model swap. A drain that exhausts its budget falls through to cancel
        // (AUDIT_arch_2026 E-6); docker-compose.yml's stop_grace_period must cover this so
        // `docker stop` doesn't SIGKILL a draining server.
        {
            std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::seconds(5));
            if (lock.owns_lock()) {
                const int drain_ms = state.runtime_config.server.model_swap_drain_ms;
                if (!state.batching->pause(drain_ms))
                    fprintf(stderr, "shutdown: in-flight requests did not drain within %d ms, cancelling\n",
                            drain_ms);
            }
        }
        state.batching->stop();
        state.batching.reset();
    }
    imp_context_free(state.ctx);
    imp_model_free(state.model);
    imp_weights_snapshot_free(state.weight_snapshot);  // non-null only when suspended
    return exit_status;
}
