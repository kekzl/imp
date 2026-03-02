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
                             const char* finish_reason,
                             const json& logprobs = nullptr) {
    json choice = {
        {"index", 0},
        {"delta", delta},
        {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}
    };
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
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
    float min_p = body.value("min_p", 0.0f);
    float typical_p = body.value("typical_p", 1.0f);
    float repetition_penalty = body.value("repetition_penalty", 1.0f);
    float frequency_penalty = body.value("frequency_penalty", 0.0f);
    float presence_penalty = body.value("presence_penalty", 0.0f);
    float dry_multiplier = body.value("dry_multiplier", 0.0f);
    float dry_base = body.value("dry_base", 1.75f);
    int dry_allowed_length = body.value("dry_allowed_length", 2);
    int dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    int mirostat = body.value("mirostat", 0);
    float mirostat_tau = body.value("mirostat_tau", 5.0f);
    float mirostat_eta = body.value("mirostat_eta", 0.1f);

    // Parse stop sequences (string or array of up to 4 strings)
    std::vector<std::string> stop_sequences;
    if (body.contains("stop") && !body["stop"].is_null()) {
        if (body["stop"].is_string()) {
            stop_sequences.push_back(body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                if (s.is_string()) {
                    stop_sequences.push_back(s.get<std::string>());
                    if (stop_sequences.size() >= 4) break;
                }
            }
        }
    }
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences) max_stop_len = std::max(max_stop_len, s.size());

    // Parse logprobs parameters
    bool req_logprobs = body.value("logprobs", false);
    int top_logprobs = body.value("top_logprobs", 0);
    if (top_logprobs < 0) top_logprobs = 0;
    if (top_logprobs > 20) top_logprobs = 20;

    // Parse response_format for JSON mode
    bool json_mode = false;
    if (body.contains("response_format") && body["response_format"].is_object()) {
        std::string fmt_type = body["response_format"].value("type", "text");
        if (fmt_type == "json_object") json_mode = true;
    }

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
    params.min_p = min_p;
    params.typical_p = typical_p;
    params.repetition_penalty = repetition_penalty;
    params.frequency_penalty = frequency_penalty;
    params.presence_penalty = presence_penalty;
    params.dry_multiplier = dry_multiplier;
    params.dry_base = dry_base;
    params.dry_allowed_length = dry_allowed_length;
    params.dry_penalty_last_n = dry_penalty_last_n;
    params.mirostat = mirostat;
    params.mirostat_tau = mirostat_tau;
    params.mirostat_eta = mirostat_eta;
    params.logprobs = req_logprobs ? 1 : 0;
    params.top_logprobs = top_logprobs;
    params.json_mode = json_mode ? 1 : 0;

    std::string comp_id = make_completion_id(state);
    int64_t created = unix_timestamp();

    if (stream) {
        // SSE streaming response
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, params, comp_id, created, max_tokens, n_prompt_tokens, t_start,
             stop_sequences, max_stop_len, req_logprobs](
                size_t /*offset*/, httplib::DataSink& sink) -> bool {

                // Save active request ref for logprobs access
                auto active_req = state.ctx->active_request;

                // Send initial chunk with role
                json role_delta = {{"role", "assistant"}};
                std::string chunk = sse_chunk(comp_id, created, state.model_name,
                                              role_delta, nullptr);
                sink.write(chunk.data(), chunk.size());

                int n_output_tokens = 0;
                const char* finish = nullptr;

                // Buffered output for stop sequence matching in streaming mode.
                // We hold back text until we're sure it doesn't contain a stop match.
                std::string pending_text;
                bool text_stop_matched = false;

                // Helper: flush confirmed text up to a byte position
                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0) return true;
                    std::string to_send = pending_text.substr(0, up_to);
                    pending_text = pending_text.substr(up_to);
                    json content_delta = {{"content", to_send}};
                    std::string sse = sse_chunk(comp_id, created, state.model_name,
                                                content_delta, nullptr);
                    return sink.write(sse.data(), sse.size());
                };

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

                    if (stop_sequences.empty()) {
                        // No stop sequences: stream directly
                        json content_delta = {{"content", piece}};
                        // Build per-token logprobs if requested
                        json lp_chunk = nullptr;
                        if (req_logprobs && active_req) {
                            size_t lp_idx = n_output_tokens - 1;
                            if (lp_idx < active_req->output_logprobs.size()) {
                                const auto& lp = active_req->output_logprobs[lp_idx];
                                json top_arr = json::array();
                                for (const auto& t : lp.top) {
                                    top_arr.push_back({
                                        {"token", t.text},
                                        {"logprob", t.logprob},
                                        {"bytes", nullptr}
                                    });
                                }
                                lp_chunk = {{"content", json::array({
                                    {{"token", lp.text},
                                     {"logprob", lp.logprob},
                                     {"bytes", nullptr},
                                     {"top_logprobs", top_arr}}
                                })}};
                            }
                        }
                        std::string chunk = sse_chunk(comp_id, created, state.model_name,
                                                      content_delta, nullptr, lp_chunk);
                        if (!sink.write(chunk.data(), chunk.size())) {
                            return false;
                        }
                    } else {
                        // Buffer text and check for stop matches
                        pending_text += piece;

                        // Check for complete stop match
                        bool stop_found = false;
                        for (const auto& stop : stop_sequences) {
                            auto pos = pending_text.find(stop);
                            if (pos != std::string::npos) {
                                // Flush text before the stop string
                                if (!flush_text(pos)) return false;
                                stop_found = true;
                                break;
                            }
                        }
                        if (stop_found) {
                            text_stop_matched = true;
                            finish = "stop";
                            break;
                        }

                        // Flush text that can't be part of a partial stop match.
                        // Keep only the last (max_stop_len - 1) chars as potential prefix.
                        if (pending_text.size() > max_stop_len) {
                            size_t safe = pending_text.size() - max_stop_len + 1;
                            if (!flush_text(safe)) return false;
                        }
                    }
                }

                // Flush any remaining buffered text (skip if text-level stop was matched)
                if (!pending_text.empty() && !text_stop_matched) {
                    json content_delta = {{"content", pending_text}};
                    std::string sse = sse_chunk(comp_id, created, state.model_name,
                                                content_delta, nullptr);
                    sink.write(sse.data(), sse.size());
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
        // Save request reference for logprobs access (active_request gets cleared on finish)
        auto active_req = state.ctx->active_request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;  // accumulated output for stop matching

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

            // Check text-level stop sequences
            if (!stop_sequences.empty()) {
                output_text += state.tok->decode_token(token);
                bool stop_found = false;
                for (const auto& stop : stop_sequences) {
                    auto pos = output_text.find(stop);
                    if (pos != std::string::npos) {
                        output_text = output_text.substr(0, pos);
                        stop_found = true;
                        break;
                    }
                }
                if (stop_found) {
                    finish = "stop";
                    break;
                }
            }
        }

        if (!finish) finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string content = !stop_sequences.empty()
            ? output_text : state.tok->decode(output_ids);

        // Log request
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

        // Build logprobs object if requested
        json logprobs_obj = nullptr;
        if (req_logprobs && active_req) {
            const auto& lp_data = active_req->output_logprobs;
            json content_logprobs = json::array();
            for (size_t idx = 0; idx < lp_data.size() && idx < output_ids.size(); idx++) {
                const auto& lp = lp_data[idx];
                json top_arr = json::array();
                for (const auto& t : lp.top) {
                    top_arr.push_back({
                        {"token", t.text},
                        {"logprob", t.logprob},
                        {"bytes", nullptr}
                    });
                }
                content_logprobs.push_back({
                    {"token", lp.text},
                    {"logprob", lp.logprob},
                    {"bytes", nullptr},
                    {"top_logprobs", top_arr}
                });
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        json choice = {
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", content}}},
            {"finish_reason", finish}
        };
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        json response = {
            {"id", comp_id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", state.model_name},
            {"choices", json::array({choice})},
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
    if (args.kv_fp8) config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (args.kv_int8) config.kv_cache_dtype = IMP_DTYPE_INT8;
    if (args.prefill_chunk_size > 0) config.prefill_chunk_size = args.prefill_chunk_size;

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
