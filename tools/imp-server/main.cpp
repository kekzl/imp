#include "imp/imp.h"
#include "args.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

// Access internal engine from opaque context handle (same as imp-cli)
struct ImpModel_T {
    std::shared_ptr<imp::Model> model;
};

struct ImpContext_T {
    ImpModel model_handle = nullptr;
    std::unique_ptr<imp::Engine> engine;
    std::shared_ptr<imp::Request> active_request;
};

struct ServerState {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    imp::Tokenizer* tok = nullptr;
    imp::ChatTemplate chat_tpl;
    bool have_template = false;
    std::string model_name;
    std::mutex mtx;
    int default_max_tokens = 2048;
    int max_seq_len = 0;
    std::atomic<int> next_id{0};
};

// Graceful shutdown
static std::atomic<httplib::Server*> g_server{nullptr};

static void signal_handler(int /*sig*/) {
    fprintf(stderr, "\nShutting down...\n");
    if (auto* svr = g_server.exchange(nullptr, std::memory_order_relaxed))
        svr->stop();
}

static std::string make_completion_id(ServerState& state) {
    return "imp-" + std::to_string(state.next_id.fetch_add(1));
}

static int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Build a single SSE data line
static std::string sse_chunk(const std::string& id, int64_t created,
                             const std::string& model,
                             const json& delta,
                             const char* finish_reason) {
    json choice = {
        {"index", 0},
        {"delta", delta},
        {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}
    };
    json obj = {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({choice})}
    };
    return "data: " + obj.dump() + "\n\n";
}

static void handle_health(const httplib::Request& /*req*/, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
}

static void handle_models(const httplib::Request& /*req*/, httplib::Response& res,
                          ServerState& state) {
    json model_obj = {
        {"id", state.model_name},
        {"object", "model"},
        {"owned_by", "imp"}
    };
    json body = {
        {"object", "list"},
        {"data", json::array({model_obj})}
    };
    res.set_content(body.dump(), "application/json");
}

static void handle_chat_completions(const httplib::Request& req, httplib::Response& res,
                                    ServerState& state) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        res.status = 400;
        json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Extract parameters
    auto messages = body.value("messages", json::array());
    if (messages.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "messages array is required and must not be empty"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    float temperature = body.value("temperature", 0.7f);
    float top_p = body.value("top_p", 0.9f);
    int top_k = body.value("top_k", 40);
    int max_tokens = body.value("max_tokens", state.default_max_tokens);
    int seed = body.value("seed", -1);
    bool stream = body.value("stream", false);

    // Convert JSON messages to ChatMessage vector
    std::vector<imp::ChatMessage> chat_msgs;
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");
        std::string content = msg.value("content", "");
        chat_msgs.push_back({role, content});
    }

    // Try to acquire inference lock — return 503 if busy
    std::unique_lock<std::mutex> lock(state.mtx, std::try_to_lock);
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error", {{"message", "Server is busy processing another request. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Tokenize with chat template
    std::vector<int32_t> tokens;
    if (state.have_template) {
        tokens = state.chat_tpl.apply(*state.tok, chat_msgs);
    } else {
        // Concatenate all message content as raw text
        std::string raw;
        for (const auto& m : chat_msgs) raw += m.content + "\n";
        tokens = state.tok->encode(raw);
    }

    int n_prompt_tokens = static_cast<int>(tokens.size());

    // Validate prompt length against context window
    if (n_prompt_tokens >= state.max_seq_len) {
        res.status = 400;
        json error = {{"error", {{"message", "Prompt exceeds context window (" +
                                              std::to_string(n_prompt_tokens) + " tokens >= " +
                                              std::to_string(state.max_seq_len) + " max)"},
                                  {"type", "invalid_request_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    // Clamp max_tokens to remaining context window
    int remaining = state.max_seq_len - n_prompt_tokens;
    if (max_tokens > remaining) max_tokens = remaining;

    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();

    // Reset context and prefill
    ImpError err = imp_context_reset(state.ctx);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Context reset failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    err = imp_prefill(state.ctx, tokens.data(), n_prompt_tokens);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Prefill failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.max_tokens = max_tokens;
    params.seed = seed;

    std::string comp_id = make_completion_id(state);
    int64_t created = unix_timestamp();

    if (stream) {
        // SSE streaming response
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, params, comp_id, created, max_tokens, n_prompt_tokens, t_start](
                size_t /*offset*/, httplib::DataSink& sink) -> bool {

                // Send initial chunk with role
                json role_delta = {{"role", "assistant"}};
                std::string chunk = sse_chunk(comp_id, created, state.model_name,
                                              role_delta, nullptr);
                sink.write(chunk.data(), chunk.size());

                int n_output_tokens = 0;
                const char* finish = nullptr;

                for (int step = 0; step < max_tokens; step++) {
                    int32_t token = 0;
                    ImpError err = imp_decode_step(state.ctx, &params, &token);
                    if (err != IMP_SUCCESS) {
                        finish = "stop";
                        break;
                    }

                    // Check stop conditions
                    if (token == state.tok->eos_id()) {
                        finish = "stop";
                        break;
                    }
                    if (state.have_template) {
                        bool is_stop = false;
                        for (int32_t stop_id : state.chat_tpl.stop_token_ids()) {
                            if (token == stop_id) { is_stop = true; break; }
                        }
                        if (is_stop) {
                            finish = "stop";
                            break;
                        }
                    }

                    n_output_tokens++;
                    std::string piece = state.tok->decode_token(token);

                    json content_delta = {{"content", piece}};
                    std::string chunk = sse_chunk(comp_id, created, state.model_name,
                                                  content_delta, nullptr);
                    if (!sink.write(chunk.data(), chunk.size())) {
                        return false;  // Client disconnected
                    }
                }

                if (!finish) finish = "length";

                // Send final chunk with finish_reason
                json empty_delta = json::object();
                std::string final_chunk = sse_chunk(comp_id, created, state.model_name,
                                                    empty_delta, finish);
                sink.write(final_chunk.data(), final_chunk.size());

                // Send [DONE]
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();

                // Log request
                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                        comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

                return true;
            }
        );
    } else {
        // Non-streaming: decode all tokens, return complete response
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;

        for (int step = 0; step < max_tokens; step++) {
            int32_t token = 0;
            err = imp_decode_step(state.ctx, &params, &token);
            if (err != IMP_SUCCESS) {
                finish = "stop";
                break;
            }

            if (token == state.tok->eos_id()) {
                finish = "stop";
                break;
            }
            if (state.have_template) {
                bool is_stop = false;
                for (int32_t stop_id : state.chat_tpl.stop_token_ids()) {
                    if (token == stop_id) { is_stop = true; break; }
                }
                if (is_stop) {
                    finish = "stop";
                    break;
                }
            }

            output_ids.push_back(token);
        }

        if (!finish) finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string content = state.tok->decode(output_ids);

        // Log request
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

        json response = {
            {"id", comp_id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", state.model_name},
            {"choices", json::array({
                {
                    {"index", 0},
                    {"message", {{"role", "assistant"}, {"content", content}}},
                    {"finish_reason", finish}
                }
            })},
            {"usage", {
                {"prompt_tokens", n_prompt_tokens},
                {"completion_tokens", n_output_tokens},
                {"total_tokens", n_prompt_tokens + n_output_tokens}
            }}
        };

        res.set_content(response.dump(), "application/json");
    }
}

int main(int argc, char** argv) {
    ServerArgs args = parse_server_args(argc, argv);

    if (args.model_path.empty()) {
        print_server_usage(argv[0]);
        return 1;
    }

    printf("IMP Server %s\n", imp_version());
    printf("Loading model: %s\n", args.model_path.c_str());

    ServerState state;
    state.default_max_tokens = args.max_tokens;

    // Extract model name from path
    std::string path = args.model_path;
    size_t slash = path.find_last_of('/');
    state.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;

    // Load model
    ImpError err = imp_model_load(args.model_path.c_str(), IMP_FORMAT_GGUF, &state.model);
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error loading model: %s\n", imp_error_string(err));
        return 1;
    }

    // Create context
    ImpConfig config = imp_config_default();
    config.device_id = args.device;
    config.max_batch_size = 1;
    config.max_seq_len = 4096;
    config.gpu_layers = args.gpu_layers;
    if (args.ssm_fp16) config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs) config.enable_cuda_graphs = 0;

    err = imp_context_create(state.model, &config, &state.ctx);
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error creating context: %s\n", imp_error_string(err));
        imp_model_free(state.model);
        return 1;
    }

    // Set up tokenizer and chat template
    state.tok = state.model->model->tokenizer();
    const imp::ChatTemplate& engine_tpl = state.ctx->engine->chat_template();

    if (args.chat_template == "none") {
        // No template
    } else if (args.chat_template != "auto") {
        auto family = imp::ChatTemplate::parse_family(args.chat_template);
        if (family != imp::ChatTemplateFamily::RAW) {
            state.have_template = state.chat_tpl.init(family, *state.tok);
        }
    } else {
        if (!engine_tpl.is_raw()) {
            state.chat_tpl = engine_tpl;
            state.have_template = true;
        }
    }

    // Store max sequence length for token clamping
    state.max_seq_len = imp_model_max_seq_len(state.model);
    if (state.max_seq_len <= 0) state.max_seq_len = static_cast<int>(config.max_seq_len);

    if (state.have_template) {
        printf("Chat template: %s\n",
               imp::chat_template_family_name(state.chat_tpl.family()));
    } else {
        printf("No chat template (raw mode)\n");
    }

    // Set up HTTP server
    httplib::Server svr;

    // CORS headers on every response
    svr.set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    svr.Get("/health", handle_health);

    svr.Get("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res, state);
    });

    svr.Post("/v1/chat/completions",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_chat_completions(req, res, state);
        });

    // Graceful shutdown on SIGINT/SIGTERM
    g_server.store(&svr, std::memory_order_relaxed);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    printf("Server listening on http://%s:%d\n", args.host.c_str(), args.port);
    printf("Endpoints:\n");
    printf("  GET  /health\n");
    printf("  GET  /v1/models\n");
    printf("  POST /v1/chat/completions\n");
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
    imp_context_free(state.ctx);
    imp_model_free(state.model);
    return 0;
}
