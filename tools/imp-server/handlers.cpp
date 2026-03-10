#include "handlers.h"
#include "utils.h"
#include "tool_call.h"

#include "api/imp_internal.h"
#include "runtime/presets.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

// Graceful shutdown
std::atomic<httplib::Server*> g_server{nullptr};

void signal_handler(int /*sig*/) {
    fprintf(stderr, "\nShutting down...\n");
    if (auto* svr = g_server.exchange(nullptr, std::memory_order_relaxed))
        svr->stop();
}

std::string make_completion_id(ServerState& state) {
    return "imp-" + std::to_string(state.next_id.fetch_add(1));
}

int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void handle_health(const httplib::Request& /*req*/, httplib::Response& res,
                   ServerState& state) {
    json body = {
        {"status", "ok"},
        {"model_loaded", state.model_loaded()},
        {"queue_depth", state.batching ? state.batching->queue_depth() : 0}
    };
    res.set_content(body.dump(), "application/json");
}

// Recursively find all .gguf files in a directory, returning (filename, full_path) pairs.
// Resolves symlinks and rejects any path that escapes the base directory (path traversal).
std::vector<std::pair<std::string, std::string>> scan_gguf_files(const std::string& dir) {
    std::vector<std::pair<std::string, std::string>> results;
    if (dir.empty()) return results;
    std::error_code ec;
    auto base = std::filesystem::canonical(dir, ec);
    if (ec) return results;
    std::string base_prefix = base.string() + "/";

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file() && !entry.is_symlink()) continue;
        auto path = entry.path();
        if (path.extension() != ".gguf" || path.string().find(".no_exist") != std::string::npos)
            continue;
        // Resolve to real path and verify it stays under the base directory
        std::error_code ec2;
        auto real = std::filesystem::canonical(path, ec2);
        if (ec2) continue;
        std::string real_str = real.string();
        if (real_str.compare(0, base_prefix.size(), base_prefix) != 0) continue;
        results.emplace_back(path.filename().string(), real_str);
    }
    std::sort(results.begin(), results.end());
    return results;
}

void handle_models(const httplib::Request& /*req*/, httplib::Response& res,
                   ServerState& state) {
    json data = json::array();

    // Scan models directory for all available .gguf files
    auto available = scan_gguf_files(state.models_dir);
    if (!available.empty()) {
        for (const auto& [name, path] : available) {
            data.push_back({
                {"id", name},
                {"object", "model"},
                {"owned_by", "imp"},
                {"path", path}
            });
        }
    } else if (state.model_loaded()) {
        // Fallback: only show loaded model if no models_dir configured
        data.push_back({
            {"id", state.model_name},
            {"object", "model"},
            {"owned_by", "imp"}
        });
    }

    json body = {
        {"object", "list"},
        {"data", data}
    };
    res.set_content(body.dump(), "application/json");
}

// Find a GGUF file by name in models_dir. Returns full path or empty string.
std::string find_model_path(const ServerState& state, const std::string& name) {
    auto available = scan_gguf_files(state.models_dir);
    for (const auto& [fname, fpath] : available) {
        if (fname == name) return fpath;
    }
    return "";
}

// Build ImpConfig from default args + optional JSON overrides.
// Auto-detects optimal preset from model_path unless overridden.
ImpConfig build_config(const ServerArgs& args, const std::string& model_path,
                       const json& overrides) {
    ImpConfig config = imp_config_default();

    // Resolve preset: explicit flag/override > auto-detect from model filename
    const imp::PresetConfig* preset = nullptr;
    std::string preset_str = overrides.value("preset", args.preset);
    if (preset_str == "none") {
        // Explicitly disabled
    } else if (!preset_str.empty()) {
        preset = imp::find_preset(preset_str);
    } else if (!model_path.empty()) {
        preset = imp::detect_preset(model_path);
    }

    if (preset) {
        imp::apply_preset(preset, config);
        printf("Preset: %s\n", preset->description.c_str());
    }

    config.device_id = args.device;
    if (!preset) {
        config.max_batch_size = 8;  // Allow concurrent requests for continuous batching
        config.max_seq_len = overrides.value("max_seq_len", 4096);
    } else if (overrides.contains("max_seq_len")) {
        config.max_seq_len = overrides.value("max_seq_len", config.max_seq_len);
    }
    config.gpu_layers = args.gpu_layers;
    if (args.ssm_fp16) config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs) config.enable_cuda_graphs = 0;

    // KV cache dtype: explicit overrides only (preset already set optimal value)
    bool kv_fp8 = overrides.value("kv_fp8", args.kv_fp8);
    bool kv_int8 = overrides.value("kv_int8", args.kv_int8);
    if (kv_fp8) config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (kv_int8) config.kv_cache_dtype = IMP_DTYPE_INT8;

    int chunk = overrides.value("prefill_chunk_size", args.prefill_chunk_size);
    if (chunk > 0) config.prefill_chunk_size = chunk;

    int nvfp4 = overrides.value("decode_nvfp4", args.decode_nvfp4);
    if (nvfp4 != -1 || !preset)
        config.use_nvfp4_decode = nvfp4;

    if (args.mxfp4_prefill) config.use_mxfp4_prefill = 1;

    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();

    // Prefix caching: always on for server (reuses KV blocks across requests)
    config.use_prefix_caching = 1;

    return config;
}

// Load a model into ServerState. Caller must hold state.mtx.
// Returns error message on failure, empty string on success.
std::string load_model_into_state(ServerState& state, const std::string& path,
                                  const json& config_overrides) {
    // Stop batching engine before freeing context
    if (state.batching) {
        state.batching->stop();
        state.batching.reset();
    }

    // Free existing model/context
    if (state.ctx) { imp_context_free(state.ctx); state.ctx = nullptr; }
    if (state.model) { imp_model_free(state.model); state.model = nullptr; }
    state.tok = nullptr;
    state.have_template = false;
    state.model_name.clear();

    // Load model
    ImpError err = imp_model_load(path.c_str(), IMP_FORMAT_GGUF, &state.model);
    if (err != IMP_SUCCESS) {
        std::string msg = std::string("Failed to load model: ") + imp_error_string(err);
        state.model = nullptr;
        return msg;
    }

    // Create context (auto-detects preset from model path)
    ImpConfig config = build_config(state.default_args, path, config_overrides);
    err = imp_context_create(state.model, &config, &state.ctx);
    if (err != IMP_SUCCESS) {
        std::string msg = std::string("Failed to create context: ") + imp_error_string(err);
        imp_model_free(state.model);
        state.model = nullptr;
        return msg;
    }

    // Extract model name from path
    size_t slash = path.find_last_of('/');
    state.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;

    // Set up tokenizer and chat template
    state.tok = state.model->model->tokenizer();
    const imp::ChatTemplate& engine_tpl = state.ctx->engine->chat_template();

    std::string chat_tpl_name = config_overrides.value("chat_template",
                                                        state.default_args.chat_template);
    if (chat_tpl_name == "none") {
        // No template
    } else if (chat_tpl_name != "auto") {
        auto family = imp::ChatTemplate::parse_family(chat_tpl_name);
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

    // Detect thinking model (DeepSeek R1 etc.) by checking for <think> token in vocab
    state.think_start_id = state.tok->find_token("<think>");
    state.think_end_id = state.tok->find_token("</think>");
    state.is_think_model = (state.think_start_id >= 0);
    if (state.is_think_model) {
        printf("Reasoning model: <think>=%d, </think>=%d\n",
               state.think_start_id, state.think_end_id);
    }

    if (state.have_template) {
        printf("Chat template: %s\n",
               imp::chat_template_family_name(state.chat_tpl.family()));
    } else {
        printf("No chat template (raw mode)\n");
    }

    // Start the continuous batching engine
    state.batching = std::make_unique<BatchingEngine>();
    state.batching->start(state.ctx);
    printf("Continuous batching: started\n");

    state.metrics.model_loads_total++;
    return "";
}

void handle_chat_completions(const httplib::Request& req, httplib::Response& res,
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
    float top_p = body.value("top_p", 0.95f);
    int top_k = body.value("top_k", 40);
    int max_tokens = body.value("max_tokens", state.default_max_tokens);
    int seed = body.value("seed", -1);
    bool stream = body.value("stream", false);
    float min_p = body.value("min_p", 0.0f);
    float typical_p = body.value("typical_p", 1.0f);
    float repetition_penalty = body.value("repetition_penalty", 1.0f);
    float frequency_penalty = body.value("frequency_penalty", 0.0f);
    float presence_penalty = body.value("presence_penalty", 0.0f);
    int repeat_last_n = body.value("repeat_last_n", 0);
    float dry_multiplier = body.value("dry_multiplier", 0.0f);
    float dry_base = body.value("dry_base", 1.75f);
    int dry_allowed_length = body.value("dry_allowed_length", 2);
    int dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    int mirostat = body.value("mirostat", 0);
    float mirostat_tau = body.value("mirostat_tau", 5.0f);
    float mirostat_eta = body.value("mirostat_eta", 0.1f);
    float think_budget = body.value("think_budget", state.default_think_budget);

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

    // Parse stream_options for include_usage
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Parse tool calling parameters
    json tools = body.value("tools", json::array());
    json tool_choice = body.value("tool_choice", json("auto"));
    bool has_tools = !tools.empty() &&
        !(tool_choice.is_string() && tool_choice.get<std::string>() == "none");

    // Convert JSON messages to ChatMessage vector, extracting image data if present
    std::vector<imp::ChatMessage> chat_msgs;
    std::vector<uint8_t> image_data;  // decoded image bytes (if any)
    imp::ChatTemplateFamily tpl_family = state.have_template
        ? state.chat_tpl.family() : imp::ChatTemplateFamily::CHATML;

    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");

        if (role == "tool") {
            // Tool response message — format for the model
            std::string content = format_tool_response(tpl_family, msg);
            chat_msgs.push_back({"tool", content});
        } else if (role == "assistant" && msg.contains("tool_calls")) {
            // Assistant message with tool_calls — reconstruct model output format
            std::string content_str;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content_str = msg["content"].get<std::string>();
            }
            std::string reconstructed = reconstruct_tool_call_output(
                tpl_family, msg["tool_calls"], content_str);
            chat_msgs.push_back({"assistant", reconstructed});
        } else if (msg.contains("content") && msg["content"].is_array()) {
            // OpenAI multimodal format: content is array of parts
            std::string text_parts;
            for (const auto& part : msg["content"]) {
                std::string type = part.value("type", "");
                if (type == "text") {
                    if (!text_parts.empty()) text_parts += "\n";
                    text_parts += part.value("text", "");
                } else if (type == "image_url" && part.contains("image_url")) {
                    std::string url = part["image_url"].value("url", "");
                    if (url.rfind("data:", 0) == 0) {
                        // Data URI: data:image/...;base64,...
                        auto comma = url.find(',');
                        if (comma != std::string::npos) {
                            image_data = base64_decode(url.substr(comma + 1));
                        }
                    } else if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
                        // Remote URL: fetch image via HTTP
                        // Parse URL into host + path
                        bool is_https = (url.rfind("https://", 0) == 0);
                        std::string rest = url.substr(is_https ? 8 : 7);
                        auto slash = rest.find('/');
                        std::string host = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
                        std::string path_str = (slash != std::string::npos) ? rest.substr(slash) : "/";
                        if (is_https) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                            httplib::SSLClient cli(host);
                            cli.set_follow_location(true);
                            cli.set_connection_timeout(10);
                            auto img_res = cli.Get(path_str);
                            if (img_res && img_res->status == 200) {
                                image_data.assign(img_res->body.begin(), img_res->body.end());
                            }
#endif
                        } else {
                            httplib::Client cli(host);
                            cli.set_follow_location(true);
                            cli.set_connection_timeout(10);
                            auto img_res = cli.Get(path_str);
                            if (img_res && img_res->status == 200) {
                                image_data.assign(img_res->body.begin(), img_res->body.end());
                            }
                        }
                    }
                }
            }
            chat_msgs.push_back({role, text_parts});
        } else {
            std::string content;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content = msg["content"].get<std::string>();
            }
            chat_msgs.push_back({role, content});
        }
    }

    // Inject tool definitions into system message
    if (has_tools && state.have_template) {
        std::string tool_prompt = build_tool_prompt(tpl_family, tools, tool_choice);
        if (!tool_prompt.empty()) {
            // Find or create system message
            bool found_system = false;
            for (auto& m : chat_msgs) {
                if (m.role == "system") {
                    m.content += tool_prompt;
                    found_system = true;
                    break;
                }
            }
            if (!found_system) {
                std::string sys = state.chat_tpl.default_system_message();
                if (sys.empty()) sys = "You are a helpful assistant.";
                sys += tool_prompt;
                chat_msgs.insert(chat_msgs.begin(), {"system", sys});
            }
        }
    }

    // Log request received (structured)
    std::string req_id = make_completion_id(state);
    fprintf(stderr, "[%s] chat/completions: prompt_msgs=%zu stream=%s max_tokens=%d temp=%.2f\n",
            req_id.c_str(), messages.size(), stream ? "true" : "false", max_tokens, temperature);

    // Auto-load model if request specifies a different one (requires exclusive lock)
    std::string requested_model = body.value("model", "");
    if (!requested_model.empty() && requested_model != state.model_name) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
            if (!lock.owns_lock()) {
                res.status = 503;
                json err = {{"error", {{"message", "Server is busy loading a model. Please retry."},
                                        {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            fprintf(stderr, "[%s] auto-loading model: %s\n", req_id.c_str(), requested_model.c_str());
            fflush(stderr);
            std::string error = load_model_into_state(state, path);
            if (!error.empty()) {
                res.status = 500;
                json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            fprintf(stderr, "[%s] model loaded: %s\n", req_id.c_str(), state.model_name.c_str());
            fflush(stderr);
        }
    }

    // Check if a model is loaded
    if (!state.model_loaded()) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded. Use POST /v1/models to load one."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Handle vision: requires exclusive lock since it modifies engine state
    bool has_vision_request = !image_data.empty() && state.ctx->engine->has_vision();
    if (has_vision_request) {
        std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
        if (!lock.owns_lock()) {
            res.status = 503;
            json err = {{"error", {{"message", "Server is busy. Please retry."},
                                    {"type", "server_error"}}}};
            res.set_content(err.dump(), "application/json");
            return;
        }
        // Stop batching engine for exclusive vision access
        if (state.batching) state.batching->stop();

        state.ctx->engine->clear_image();
        if (!state.ctx->engine->set_image_from_memory(image_data.data(), image_data.size())) {
            if (state.batching) state.batching->start(state.ctx);
            res.status = 400;
            json error = {{"error", {{"message", "Failed to process image"},
                                      {"type", "invalid_request_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }
    }

    // Tokenize with chat template (with image tokens if vision is active)
    std::vector<int32_t> tokens;
    if (state.have_template && has_vision_request) {
        tokens = state.chat_tpl.apply_with_image(*state.tok, chat_msgs, 256);
    } else if (state.have_template) {
        tokens = state.chat_tpl.apply(*state.tok, chat_msgs);
    } else {
        // Concatenate all message content as raw text
        std::string raw;
        for (const auto& m : chat_msgs) raw += m.content + "\n";
        tokens = state.tok->encode(raw);
    }

    // Optionally append <think> token to trigger reasoning mode.
    bool enable_thinking = false;
    if (state.is_think_model && state.default_args.reasoning_format == "deepseek" &&
        state.think_start_id >= 0) {
        if (body.contains("enable_thinking")) {
            enable_thinking = body.value("enable_thinking", false);
        }
        if (enable_thinking) {
            tokens.push_back(state.think_start_id);
        }
    }

    int n_prompt_tokens = static_cast<int>(tokens.size());

    // Validate prompt length against context window
    if (n_prompt_tokens >= state.max_seq_len) {
        if (has_vision_request && state.batching) state.batching->start(state.ctx);
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

    // Create an imp::Request for the batching engine
    auto imp_req = std::make_shared<imp::Request>();
    imp_req->input_tokens = std::move(tokens);
    imp_req->max_tokens = max_tokens;
    imp_req->temperature = temperature;
    imp_req->top_p = top_p;
    imp_req->top_k = top_k;
    imp_req->seed = seed;
    imp_req->min_p = min_p;
    imp_req->typical_p = typical_p;
    imp_req->repetition_penalty = repetition_penalty;
    imp_req->frequency_penalty = frequency_penalty;
    imp_req->presence_penalty = presence_penalty;
    imp_req->repeat_last_n = repeat_last_n;
    imp_req->dry_multiplier = dry_multiplier;
    imp_req->dry_base = dry_base;
    imp_req->dry_allowed_length = dry_allowed_length;
    imp_req->dry_penalty_last_n = dry_penalty_last_n;
    imp_req->mirostat = mirostat;
    imp_req->mirostat_tau = mirostat_tau;
    imp_req->mirostat_eta = mirostat_eta;
    imp_req->logprobs = req_logprobs;
    imp_req->top_logprobs = top_logprobs;
    imp_req->json_mode = json_mode;
    imp_req->status = imp::RequestStatus::PENDING;

    // Create a ServerRequest wrapper and submit to the batching engine
    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = imp_req;

    // For vision requests, fall back to blocking mode since vision state
    // is per-engine (not per-request). Use the old C API path.
    if (has_vision_request) {
        // Vision path: use blocking C API (batching engine is stopped)
        ImpError err = imp_context_reset(state.ctx);
        if (err != IMP_SUCCESS) {
            state.ctx->engine->clear_image();
            state.batching->start(state.ctx);
            res.status = 500;
            json error = {{"error", {{"message", std::string("Context reset failed: ") + imp_error_string(err)},
                                      {"type", "server_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        err = imp_prefill(state.ctx, imp_req->input_tokens.data(), n_prompt_tokens);
        if (err != IMP_SUCCESS) {
            state.ctx->engine->clear_image();
            state.batching->start(state.ctx);
            res.status = 500;
            json error = {{"error", {{"message", std::string("Prefill failed: ") + imp_error_string(err)},
                                      {"type", "server_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        // After prefill, clear vision and restart batching engine
        // The rest of generation will use the old blocking decode path
        // (via imp_decode_step, which calls engine->step() directly)
        // This is safe because batching engine is stopped.

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
        params.repeat_last_n = repeat_last_n;
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

        // Blocking decode loop for vision requests
        std::vector<int32_t> output_ids;
        int32_t prefill_token = -1;
        if (state.ctx->active_request && !state.ctx->active_request->output_tokens.empty()) {
            prefill_token = state.ctx->active_request->output_tokens.back();
        }

        for (int step = -1; step < max_tokens; step++) {
            int32_t token = 0;
            if (step == -1) {
                if (prefill_token < 0) continue;
                token = prefill_token;
            } else {
                err = imp_decode_step(state.ctx, &params, &token);
                if (err != IMP_SUCCESS) break;
            }
            if (token == state.tok->eos_id()) break;
            if (state.have_template) {
                bool is_stop = false;
                for (int32_t stop_id : state.chat_tpl.stop_token_ids()) {
                    if (token == stop_id) { is_stop = true; break; }
                }
                if (is_stop) break;
            }
            output_ids.push_back(token);
        }

        state.ctx->engine->clear_image();
        state.batching->start(state.ctx);

        // Build simple non-streaming response for vision
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string content = state.tok->decode(output_ids);

        fprintf(stderr, "[%s] vision: %d prompt + %d completion tokens, %.1f ms\n",
                req_id.c_str(), n_prompt_tokens, n_output_tokens, ms);
        state.metrics.requests_total++;
        state.metrics.tokens_prompt_total += n_prompt_tokens;
        state.metrics.tokens_completion_total += n_output_tokens;
        state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

        json response = {
            {"id", req_id},
            {"object", "chat.completion"},
            {"created", unix_timestamp()},
            {"model", state.model_name},
            {"choices", json::array({{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", content}}},
                {"finish_reason", "stop"}
            }})},
            {"usage", {
                {"prompt_tokens", n_prompt_tokens},
                {"completion_tokens", n_output_tokens},
                {"total_tokens", n_prompt_tokens + n_output_tokens}
            }}
        };
        res.set_content(response.dump(), "application/json");
        return;
    }

    // Submit to batching engine for continuous batching
    if (!state.batching || !state.batching->is_running()) {
        res.status = 503;
        json err = {{"error", {{"message", "Inference engine not ready. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }
    state.batching->submit(server_req);

    std::string comp_id = req_id;
    int64_t created = unix_timestamp();

    if (stream) {
        // SSE streaming response
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, server_req, comp_id, created, max_tokens, n_prompt_tokens, t_start,
             stop_sequences, max_stop_len, req_logprobs, include_usage,
             enable_thinking, has_tools, tpl_family, think_budget](
                size_t /*offset*/, httplib::DataSink& sink) -> bool {

                // Active request ref for logprobs access
                auto active_req = server_req->request;

                // Pre-build SSE envelope templates for fast content/reasoning emission
                SSEChunkWriter sse_writer(comp_id, created, state.model_name);

                // Send initial chunk with role
                json role_delta = {{"role", "assistant"}};
                std::string chunk = sse_chunk(comp_id, created, state.model_name,
                                              role_delta, nullptr);
                sink.write(chunk.data(), chunk.size());

                int n_output_tokens = 0;
                const char* finish = nullptr;

                // Buffer for incomplete UTF-8 sequences across token boundaries
                std::string utf8_buf;

                // Buffered output for stop sequence matching in streaming mode.
                // We hold back text until we're sure it doesn't contain a stop match.
                std::string pending_text;
                bool text_stop_matched = false;

                // Tool call detection state machine for streaming
                enum class ToolPhase { CONTENT, TAG_SCANNING, TOOL_CALL_BODY };
                ToolPhase tool_phase = ToolPhase::CONTENT;
                std::string tool_tag_buf;           // buffer for partial tag match
                std::string tool_body_buf;           // buffer for tool call body
                std::string tool_close_tag;          // expected closing tag
                std::string tool_fn_name;            // Llama3: extracted function name from open tag
                std::vector<ParsedToolCall> stream_tool_calls;
                bool tool_calls_emitted = false;
                // The full accumulated output (only used when has_tools, for fallback)
                std::string full_output;

                // Reasoning content extraction (DeepSeek format)
                enum class ThinkPhase { SCAN, REASONING, CONTENT };
                bool use_reasoning = (state.default_args.reasoning_format == "deepseek"
                                      && state.is_think_model);
                ThinkPhase think_phase;
                if (enable_thinking) {
                    think_phase = ThinkPhase::REASONING;  // <think> in prefill -> start reasoning
                } else if (use_reasoning) {
                    think_phase = ThinkPhase::SCAN;       // model decides whether to think
                } else {
                    think_phase = ThinkPhase::CONTENT;    // no reasoning extraction
                }
                std::string reasoning_utf8_buf;
                std::string think_scan_buf;
                int think_scan_count = 0;
                int n_reasoning_tokens = 0;
                const int kThinkScanLimit = 8;

                // Helper: emit reasoning_content SSE chunk
                auto emit_reasoning = [&](const std::string& text) -> bool {
                    if (text.empty()) return true;
                    return sse_writer.write_reasoning(text, sink);
                };

                // Helper: flush confirmed text up to a byte position
                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0) return true;
                    bool ok = sse_writer.write_content(
                        pending_text.data(), up_to, sink);
                    pending_text.erase(0, up_to);
                    return ok;
                };

                for (;;) {
                    // Read next token from the batching engine
                    TokenEvent evt;
                    server_req->pop_token(evt);

                    if (evt.token_id < 0) {
                        // Finish event with no token
                        finish = evt.finish_reason ? evt.finish_reason : "stop";
                        break;
                    }

                    int32_t token = evt.token_id;

                    // Check stop conditions (EOS/stop tokens already detected by engine)
                    if (evt.is_last) {
                        // The engine marked this as the last token.
                        // Don't emit EOS/stop tokens — they're structural, not content.
                        if (token == state.tok->eos_id()) {
                            finish = evt.finish_reason ? evt.finish_reason : "stop";
                            break;
                        }
                        bool is_stop = false;
                        if (state.have_template) {
                            for (int32_t stop_id : state.chat_tpl.stop_token_ids()) {
                                if (token == stop_id) { is_stop = true; break; }
                            }
                        }
                        if (is_stop) {
                            finish = evt.finish_reason ? evt.finish_reason : "stop";
                            break;
                        }
                        // Not a stop token — emit it, then finish after this iteration
                        finish = evt.finish_reason ? evt.finish_reason : "length";
                    }

                    n_output_tokens++;
                    std::string piece = state.tok->decode_token(token);

                    // Reasoning content extraction (DeepSeek format)
                    if (think_phase == ThinkPhase::SCAN) {
                        if (token == state.think_start_id) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens++;
                            continue;
                        }
                        think_scan_buf += piece;
                        think_scan_count++;
                        if (think_scan_buf.find("<think>") != std::string::npos) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens += think_scan_count;
                            auto pos = think_scan_buf.find("<think>");
                            std::string after = think_scan_buf.substr(pos + 7);
                            think_scan_buf.clear();
                            if (!after.empty()) reasoning_utf8_buf += after;
                            continue;
                        }
                        if (think_scan_count == 1 && piece.empty()) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens++;
                            continue;
                        }
                        if (think_scan_count >= kThinkScanLimit) {
                            think_phase = ThinkPhase::CONTENT;
                            piece = think_scan_buf;
                            think_scan_buf.clear();
                        } else {
                            continue;
                        }
                    }

                    if (think_phase == ThinkPhase::REASONING) {
                        n_reasoning_tokens++;
                        // Think budget: cap reasoning at configured fraction of max_tokens
                        int think_limit = (think_budget > 0.0f)
                            ? static_cast<int>(max_tokens * think_budget) : max_tokens;
                        if (state.think_end_id >= 0 &&
                            n_reasoning_tokens >= think_limit &&
                            token != state.think_end_id) {
                            token = state.think_end_id;
                            piece = state.tok->decode_token(token);
                        }
                        if (token == state.think_end_id) {
                            if (!emit_reasoning(reasoning_utf8_buf)) return false;
                            reasoning_utf8_buf.clear();
                            think_phase = ThinkPhase::CONTENT;
                            continue;
                        }
                        // Skip duplicate <think> tokens while already reasoning
                        if (token == state.think_start_id) continue;
                        reasoning_utf8_buf += piece;
                        // Strip <think> text that appears via multi-token encoding
                        for (;;) {
                            auto tp = reasoning_utf8_buf.find("<think>");
                            if (tp == std::string::npos) break;
                            reasoning_utf8_buf.erase(tp, 7);
                        }
                        auto end_pos = reasoning_utf8_buf.find("</think>");
                        if (end_pos != std::string::npos) {
                            std::string before = reasoning_utf8_buf.substr(0, end_pos);
                            if (!emit_reasoning(before)) return false;
                            think_phase = ThinkPhase::CONTENT;
                            std::string after = reasoning_utf8_buf.substr(end_pos + 8);
                            reasoning_utf8_buf.clear();
                            auto start = after.find_first_not_of("\n\r\t ");
                            if (start != std::string::npos) {
                                piece = after.substr(start);
                            } else {
                                continue;
                            }
                        } else {
                            size_t complete = utf8_complete_len(reasoning_utf8_buf);
                            if (complete > 0) {
                                std::string to_emit = reasoning_utf8_buf.substr(0, complete);
                                reasoning_utf8_buf = reasoning_utf8_buf.substr(complete);
                                if (!emit_reasoning(to_emit)) return false;
                            }
                            continue;
                        }
                    }

                    // CONTENT phase: handle stray think tokens from confused models
                    if (use_reasoning) {
                        if (token == state.think_start_id) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens++;
                            continue;
                        }
                        if (token == state.think_end_id) {
                            n_reasoning_tokens++;
                            continue;
                        }
                        // Strip text-level think tags from content piece
                        for (;;) {
                            auto p = piece.find("<think>");
                            if (p != std::string::npos) { piece.erase(p, 7); continue; }
                            p = piece.find("</think>");
                            if (p != std::string::npos) { piece.erase(p, 8); continue; }
                            break;
                        }
                        if (piece.empty()) continue;
                    }

                    // CONTENT phase — with tool call tag detection
                    if (has_tools) full_output += piece;

                    // Tool call state machine (only active when tools are present)
                    if (has_tools && tool_phase == ToolPhase::TOOL_CALL_BODY) {
                        tool_body_buf += piece;
                        // Check for close tag
                        auto close_pos = tool_body_buf.find(tool_close_tag);
                        if (close_pos != std::string::npos) {
                            std::string body = tool_body_buf.substr(0, close_pos);
                            auto bs = body.find_first_not_of("\n\r\t ");
                            auto be = body.find_last_not_of("\n\r\t ");
                            if (bs != std::string::npos && be != std::string::npos)
                                body = body.substr(bs, be - bs + 1);

                            // Parse and emit tool call
                            try {
                                json j = json::parse(body);
                                ParsedToolCall tc;
                                tc.id = "call_imp_" + std::to_string(
                                    state.next_tool_call_id.fetch_add(1));
                                if (tpl_family == imp::ChatTemplateFamily::LLAMA3) {
                                    tc.name = tool_fn_name;
                                    tc.arguments = j.dump();
                                } else {
                                    tc.name = j.value("name", "");
                                    if (j.contains("arguments")) {
                                        tc.arguments = j["arguments"].dump();
                                    } else {
                                        json args = j;
                                        args.erase("name");
                                        tc.arguments = args.dump();
                                    }
                                }
                                if (!tc.name.empty()) {
                                    int idx = static_cast<int>(stream_tool_calls.size());
                                    // Emit name chunk
                                    json name_delta = {{"tool_calls", json::array({
                                        {{"index", idx}, {"id", tc.id},
                                         {"type", "function"},
                                         {"function", {{"name", tc.name}, {"arguments", ""}}}}
                                    })}};
                                    std::string sse = sse_chunk(comp_id, created,
                                        state.model_name, name_delta, nullptr);
                                    sink.write(sse.data(), sse.size());

                                    // Emit arguments chunk
                                    json args_delta = {{"tool_calls", json::array({
                                        {{"index", idx},
                                         {"function", {{"arguments", tc.arguments}}}}
                                    })}};
                                    sse = sse_chunk(comp_id, created,
                                        state.model_name, args_delta, nullptr);
                                    sink.write(sse.data(), sse.size());

                                    stream_tool_calls.push_back(std::move(tc));
                                    tool_calls_emitted = true;
                                }
                            } catch (...) {
                                // Malformed JSON — skip
                            }

                            // Check for more content after close tag
                            std::string after = tool_body_buf.substr(
                                close_pos + tool_close_tag.size());
                            tool_body_buf.clear();
                            tool_phase = ToolPhase::CONTENT;
                            // If there's remaining text, it might contain more tool calls
                            if (!after.empty()) {
                                auto ws = after.find_first_not_of("\n\r\t ");
                                if (ws != std::string::npos) {
                                    piece = after.substr(ws);
                                    // Fall through to CONTENT handling below
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        } else {
                            continue; // Still collecting body
                        }
                    }

                    if (has_tools && tool_phase == ToolPhase::TAG_SCANNING) {
                        tool_tag_buf += piece;
                        // ChatML: check for <tool_call>
                        if (tpl_family != imp::ChatTemplateFamily::LLAMA3) {
                            if (tool_tag_buf.size() >= 11) { // len("<tool_call>")
                                if (tool_tag_buf.find("<tool_call>") != std::string::npos) {
                                    auto pos = tool_tag_buf.find("<tool_call>");
                                    // Flush content before the tag
                                    std::string before = tool_tag_buf.substr(0, pos);
                                    if (!before.empty()) {
                                        json cd = {{"content", before}};
                                        std::string sse = sse_chunk(comp_id, created,
                                            state.model_name, cd, nullptr);
                                        sink.write(sse.data(), sse.size());
                                    }
                                    tool_body_buf = tool_tag_buf.substr(pos + 11);
                                    tool_close_tag = "</tool_call>";
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::TOOL_CALL_BODY;
                                    continue;
                                }
                                // Check if it's definitely not a tool_call tag
                                if (tool_tag_buf.find("<tool_call") == std::string::npos &&
                                    tool_tag_buf.find("<tool_c") == std::string::npos &&
                                    tool_tag_buf.find("<tool_") == std::string::npos &&
                                    tool_tag_buf.find("<tool") == std::string::npos &&
                                    tool_tag_buf.find("<too") == std::string::npos &&
                                    tool_tag_buf.find("<to") == std::string::npos &&
                                    tool_tag_buf.find("<t") == std::string::npos) {
                                    // Not a tool tag — flush as content
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                    // Fall through to content emission
                                } else {
                                    continue; // Still scanning
                                }
                            } else {
                                // Check partial match
                                const char* tc_tag = "<tool_call>";
                                bool could_match = true;
                                for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 11; ci++) {
                                    if (tool_tag_buf[ci] != tc_tag[ci]) {
                                        could_match = false;
                                        break;
                                    }
                                }
                                if (!could_match) {
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                } else {
                                    continue; // Still matching prefix
                                }
                            }
                        } else {
                            // Llama3: check for <function=
                            if (tool_tag_buf.size() >= 10) { // len("<function=")
                                auto fn_pos = tool_tag_buf.find("<function=");
                                if (fn_pos != std::string::npos) {
                                    auto gt = tool_tag_buf.find('>', fn_pos + 10);
                                    if (gt != std::string::npos) {
                                        std::string before = tool_tag_buf.substr(0, fn_pos);
                                        if (!before.empty()) {
                                            json cd = {{"content", before}};
                                            std::string sse = sse_chunk(comp_id, created,
                                                state.model_name, cd, nullptr);
                                            sink.write(sse.data(), sse.size());
                                        }
                                        tool_fn_name = tool_tag_buf.substr(fn_pos + 10,
                                            gt - (fn_pos + 10));
                                        tool_body_buf = tool_tag_buf.substr(gt + 1);
                                        tool_close_tag = "</function>";
                                        tool_tag_buf.clear();
                                        tool_phase = ToolPhase::TOOL_CALL_BODY;
                                        continue;
                                    } else {
                                        continue; // Still scanning for >
                                    }
                                }
                                // Check prefix match
                                const char* fn_tag = "<function=";
                                bool could_match = true;
                                for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 10; ci++) {
                                    if (tool_tag_buf[ci] != fn_tag[ci]) {
                                        could_match = false;
                                        break;
                                    }
                                }
                                if (!could_match) {
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                } else {
                                    continue;
                                }
                            } else {
                                const char* fn_tag = "<function=";
                                bool could_match = true;
                                for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 10; ci++) {
                                    if (tool_tag_buf[ci] != fn_tag[ci]) {
                                        could_match = false;
                                        break;
                                    }
                                }
                                if (!could_match) {
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                } else {
                                    continue;
                                }
                            }
                        }
                    }

                    // In CONTENT phase, check for start of tool call tag
                    if (has_tools && tool_phase == ToolPhase::CONTENT) {
                        // Look for < that might start a tool call tag
                        size_t lt_pos = piece.find('<');
                        if (lt_pos != std::string::npos) {
                            // Emit everything before the <
                            if (lt_pos > 0) {
                                std::string before = piece.substr(0, lt_pos);
                                if (stop_sequences.empty()) {
                                    utf8_buf += before;
                                } else {
                                    pending_text += before;
                                }
                            }
                            // Start tag scanning with the < and everything after
                            tool_tag_buf = piece.substr(lt_pos);
                            tool_phase = ToolPhase::TAG_SCANNING;
                            // Flush any buffered content before entering tag scan
                            if (stop_sequences.empty() && !utf8_buf.empty()) {
                                size_t complete = utf8_complete_len(utf8_buf);
                                if (complete > 0) {
                                    if (!sse_writer.write_content(
                                            utf8_buf.data(), complete, sink))
                                        return false;
                                    utf8_buf.erase(0, complete);
                                }
                            } else if (!stop_sequences.empty()) {
                                bool stop_found = false;
                                for (const auto& stop : stop_sequences) {
                                    auto pos = pending_text.find(stop);
                                    if (pos != std::string::npos) {
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
                                if (pending_text.size() > max_stop_len) {
                                    size_t safe = pending_text.size() - max_stop_len + 1;
                                    if (!flush_text(safe)) return false;
                                }
                            }
                            continue;
                        }
                    }

                    // Normal content emission (no tool tag detected)
                    if (stop_sequences.empty()) {
                        // No stop sequences: stream directly (with UTF-8 buffering)
                        utf8_buf += piece;
                        size_t complete = utf8_complete_len(utf8_buf);
                        if (complete > 0) {
                            if (req_logprobs && active_req) {
                                // Logprobs path: fall back to sse_chunk (rare)
                                std::string to_emit = utf8_buf.substr(0, complete);
                                utf8_buf.erase(0, complete);
                                json content_delta = {{"content", to_emit}};
                                json lp_chunk = nullptr;
                                size_t lp_idx = n_output_tokens - 1;
                                if (lp_idx < active_req->output_logprobs.size()) {
                                    const auto& lp = active_req->output_logprobs[lp_idx];
                                    json top_arr = json::array();
                                    for (const auto& t : lp.top) {
                                        top_arr.push_back({
                                            {"token", safe_token_json(t.text)},
                                            {"logprob", t.logprob},
                                            {"bytes", token_bytes_json(t.text)}
                                        });
                                    }
                                    lp_chunk = {{"content", json::array({
                                        {{"token", safe_token_json(lp.text)},
                                         {"logprob", lp.logprob},
                                         {"bytes", token_bytes_json(lp.text)},
                                         {"top_logprobs", top_arr}}
                                    })}};
                                }
                                std::string chunk = sse_chunk(comp_id, created,
                                    state.model_name, content_delta, nullptr, lp_chunk);
                                if (!sink.write(chunk.data(), chunk.size()))
                                    return false;
                            } else {
                                // Fast path: pre-formatted template
                                if (!sse_writer.write_content(
                                        utf8_buf.data(), complete, sink))
                                    return false;
                                utf8_buf.erase(0, complete);
                            }
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

                    // Break after processing the last non-EOS token from batching engine
                    if (finish) break;
                }

                // Flush scan buffer if we never left SCAN phase (model didn't think)
                if (think_phase == ThinkPhase::SCAN && !think_scan_buf.empty()) {
                    utf8_buf += think_scan_buf;
                    think_scan_buf.clear();
                }

                // Flush remaining reasoning buffer (model ended while still thinking)
                if (!reasoning_utf8_buf.empty()) {
                    emit_reasoning(reasoning_utf8_buf);
                    reasoning_utf8_buf.clear();
                }

                // If model exhausted tokens while still reasoning and never
                // produced content, emit a notice so the user sees something
                // instead of a blank response.
                if (think_phase == ThinkPhase::REASONING
                    && utf8_buf.empty() && pending_text.empty()) {
                    std::string notice =
                        "[Reasoning truncated — increase max_tokens for a complete answer]";
                    sse_writer.write_content(notice, sink);
                }

                // Handle incomplete tool call at end (max_tokens hit while in tag)
                if (tool_phase != ToolPhase::CONTENT && !tool_calls_emitted) {
                    // Partial tool call — emit as content, finish_reason stays "length"
                    std::string leftover;
                    if (!tool_tag_buf.empty()) leftover += tool_tag_buf;
                    if (!tool_body_buf.empty()) leftover += tool_body_buf;
                    if (!leftover.empty()) {
                        utf8_buf += leftover;
                    }
                }

                // Flush any remaining UTF-8 buffer (only if no tool calls were emitted)
                if (!utf8_buf.empty() && !text_stop_matched && !tool_calls_emitted) {
                    sse_writer.write_content(utf8_buf, sink);
                }

                // Flush any remaining buffered text (skip if text-level stop was matched)
                if (!pending_text.empty() && !text_stop_matched && !tool_calls_emitted) {
                    sse_writer.write_content(pending_text, sink);
                }

                if (!finish) {
                    finish = tool_calls_emitted ? "tool_calls" : "length";
                } else if (tool_calls_emitted && strcmp(finish, "stop") == 0) {
                    finish = "tool_calls";
                }

                // Send final chunk with finish_reason
                json empty_delta = json::object();
                std::string final_chunk = sse_chunk(comp_id, created, state.model_name,
                                                    empty_delta, finish);
                sink.write(final_chunk.data(), final_chunk.size());

                // Send usage chunk if requested
                if (include_usage) {
                    json usage = {
                        {"prompt_tokens", n_prompt_tokens},
                        {"completion_tokens", n_output_tokens},
                        {"total_tokens", n_prompt_tokens + n_output_tokens}
                    };
                    if (n_reasoning_tokens > 0) {
                        usage["completion_tokens_details"] = {
                            {"reasoning_tokens", n_reasoning_tokens}
                        };
                    }
                    json usage_obj = {
                        {"id", comp_id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", state.model_name},
                        {"choices", json::array()},
                        {"usage", usage}
                    };
                    std::string usage_chunk = "data: " + usage_obj.dump() + "\n\n";
                    sink.write(usage_chunk.data(), usage_chunk.size());
                }

                // Send [DONE]
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();

                // Log request
                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                        comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);
                state.metrics.requests_total++;
                state.metrics.tokens_prompt_total += n_prompt_tokens;
                state.metrics.tokens_completion_total += n_output_tokens;
                state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

                return true;
            }
        );
    } else {
        // Non-streaming: decode all tokens, return complete response
        auto active_req = server_req->request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;  // accumulated output for stop matching
        bool ns_in_think = false;   // non-streaming think budget tracking
        int ns_think_tokens = 0;

        for (;;) {
            // Read next token from the batching engine
            TokenEvent evt;
            server_req->pop_token(evt);

            if (evt.token_id < 0) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }

            int32_t token = evt.token_id;

            // Check stop conditions
            if (evt.is_last) {
                if (token == state.tok->eos_id()) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                bool is_stop = false;
                if (state.have_template) {
                    for (int32_t stop_id : state.chat_tpl.stop_token_ids()) {
                        if (token == stop_id) { is_stop = true; break; }
                    }
                }
                if (is_stop) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                finish = evt.finish_reason ? evt.finish_reason : "length";
            }

            // Think budget: cap reasoning at configured fraction of max_tokens (non-streaming)
            if (state.is_think_model && state.think_end_id >= 0 &&
                state.default_args.reasoning_format == "deepseek" &&
                think_budget > 0.0f) {
                if (token == state.think_start_id) {
                    ns_in_think = true;
                } else if (token == state.think_end_id) {
                    ns_in_think = false;
                } else if (ns_in_think) {
                    ns_think_tokens++;
                    int think_limit = static_cast<int>(max_tokens * think_budget);
                    if (ns_think_tokens >= think_limit) {
                        token = state.think_end_id;
                        ns_in_think = false;
                    }
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

            // Break after processing the last non-EOS token
            if (finish) break;
        }

        if (!finish) finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string content = !stop_sequences.empty()
            ? output_text : state.tok->decode(output_ids);

        // Extract reasoning content (DeepSeek format) or strip think blocks
        std::string reasoning_content;
        if (state.is_think_model &&
            state.default_args.reasoning_format == "deepseek") {
            auto [reasoning, cleaned] = extract_reasoning(content);
            reasoning_content = reasoning;
            content = cleaned;
        } else if (state.is_think_model &&
                   state.default_args.reasoning_format != "none") {
            strip_think_block(content);
        }

        // Log request
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);
        state.metrics.requests_total++;
        state.metrics.tokens_prompt_total += n_prompt_tokens;
        state.metrics.tokens_completion_total += n_output_tokens;
        state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

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
                        {"token", safe_token_json(t.text)},
                        {"logprob", t.logprob},
                        {"bytes", token_bytes_json(t.text)}
                    });
                }
                content_logprobs.push_back({
                    {"token", safe_token_json(lp.text)},
                    {"logprob", lp.logprob},
                    {"bytes", token_bytes_json(lp.text)},
                    {"top_logprobs", top_arr}
                });
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        // Parse tool calls from model output
        std::vector<ParsedToolCall> tool_calls;
        if (has_tools && strcmp(finish, "length") != 0) {
            auto [pre_content, parsed_calls] = parse_tool_calls(tpl_family, content, state.next_tool_call_id);
            if (!parsed_calls.empty()) {
                tool_calls = std::move(parsed_calls);
                content = pre_content;
                finish = "tool_calls";
            }
        }

        json msg = {{"role", "assistant"}};
        if (!tool_calls.empty()) {
            // content is null when only tool calls (no preceding text)
            msg["content"] = content.empty() ? json(nullptr) : json(content);
            json tc_array = json::array();
            for (const auto& tc : tool_calls) {
                tc_array.push_back({
                    {"id", tc.id},
                    {"type", "function"},
                    {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                });
            }
            msg["tool_calls"] = tc_array;
        } else {
            msg["content"] = content;
        }
        if (!reasoning_content.empty()) {
            msg["reasoning_content"] = reasoning_content;
        }

        json choice = {
            {"index", 0},
            {"message", msg},
            {"finish_reason", finish}
        };
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        json usage = {
            {"prompt_tokens", n_prompt_tokens},
            {"completion_tokens", n_output_tokens},
            {"total_tokens", n_prompt_tokens + n_output_tokens}
        };
        if (!reasoning_content.empty()) {
            // Use actual tracked count if available, otherwise estimate
            int n_reasoning_tokens = (ns_think_tokens > 0) ? ns_think_tokens + 2
                : static_cast<int>(state.tok->encode(reasoning_content).size()) + 2;
            usage["completion_tokens_details"] = {
                {"reasoning_tokens", n_reasoning_tokens}
            };
        }

        json response = {
            {"id", comp_id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", state.model_name},
            {"choices", json::array({choice})},
            {"usage", usage}
        };

        res.set_content(response.dump(), "application/json");
    }
}

void handle_completions(const httplib::Request& req, httplib::Response& res,
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

    // Extract prompt
    std::string prompt = body.value("prompt", "");
    if (prompt.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"prompt\" is required and must not be empty"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Extract parameters
    float temperature = body.value("temperature", 0.7f);
    float top_p = body.value("top_p", 0.95f);
    int top_k = body.value("top_k", 40);
    int max_tokens = body.value("max_tokens", state.default_max_tokens);
    int seed = body.value("seed", -1);
    bool stream = body.value("stream", false);
    bool echo = body.value("echo", false);
    float min_p = body.value("min_p", 0.0f);
    float typical_p = body.value("typical_p", 1.0f);
    float repetition_penalty = body.value("repetition_penalty", 1.0f);
    float frequency_penalty = body.value("frequency_penalty", 0.0f);
    float presence_penalty = body.value("presence_penalty", 0.0f);
    int repeat_last_n = body.value("repeat_last_n", 0);
    float dry_multiplier = body.value("dry_multiplier", 0.0f);
    float dry_base = body.value("dry_base", 1.75f);
    int dry_allowed_length = body.value("dry_allowed_length", 2);
    int dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    int mirostat = body.value("mirostat", 0);
    float mirostat_tau = body.value("mirostat_tau", 5.0f);
    float mirostat_eta = body.value("mirostat_eta", 0.1f);

    bool req_logprobs = body.value("logprobs", false);
    int top_logprobs = body.value("top_logprobs", 0);
    if (top_logprobs < 0) top_logprobs = 0;
    if (top_logprobs > 20) top_logprobs = 20;

    // Parse stop sequences
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

    // Parse stream_options for include_usage
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Log request received
    std::string req_id = make_completion_id(state);
    fprintf(stderr, "[%s] completions: prompt_len=%zu stream=%s max_tokens=%d temp=%.2f\n",
            req_id.c_str(), prompt.size(), stream ? "true" : "false", max_tokens, temperature);

    // Auto-load model if request specifies a different one (requires exclusive lock)
    std::string requested_model = body.value("model", "");
    if (!requested_model.empty() && requested_model != state.model_name) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
            if (!lock.owns_lock()) {
                res.status = 503;
                json err = {{"error", {{"message", "Server is busy loading a model. Please retry."},
                                        {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            fprintf(stderr, "[%s] auto-loading model: %s\n", req_id.c_str(), requested_model.c_str());
            fflush(stderr);
            std::string error = load_model_into_state(state, path);
            if (!error.empty()) {
                res.status = 500;
                json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            fprintf(stderr, "[%s] model loaded: %s\n", req_id.c_str(), state.model_name.c_str());
            fflush(stderr);
        }
    }

    if (!state.model_loaded()) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded. Use POST /v1/models to load one."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Tokenize raw prompt (no chat template)
    std::vector<int32_t> tokens = state.tok->encode(prompt);
    int n_prompt_tokens = static_cast<int>(tokens.size());

    if (n_prompt_tokens >= state.max_seq_len) {
        res.status = 400;
        json error = {{"error", {{"message", "Prompt exceeds context window (" +
                                              std::to_string(n_prompt_tokens) + " tokens >= " +
                                              std::to_string(state.max_seq_len) + " max)"},
                                  {"type", "invalid_request_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    int remaining = state.max_seq_len - n_prompt_tokens;
    if (max_tokens > remaining) max_tokens = remaining;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Create an imp::Request and submit to batching engine
    auto imp_req = std::make_shared<imp::Request>();
    imp_req->input_tokens = std::move(tokens);
    imp_req->max_tokens = max_tokens;
    imp_req->temperature = temperature;
    imp_req->top_p = top_p;
    imp_req->top_k = top_k;
    imp_req->seed = seed;
    imp_req->min_p = min_p;
    imp_req->typical_p = typical_p;
    imp_req->repetition_penalty = repetition_penalty;
    imp_req->frequency_penalty = frequency_penalty;
    imp_req->presence_penalty = presence_penalty;
    imp_req->repeat_last_n = repeat_last_n;
    imp_req->dry_multiplier = dry_multiplier;
    imp_req->dry_base = dry_base;
    imp_req->dry_allowed_length = dry_allowed_length;
    imp_req->dry_penalty_last_n = dry_penalty_last_n;
    imp_req->mirostat = mirostat;
    imp_req->mirostat_tau = mirostat_tau;
    imp_req->mirostat_eta = mirostat_eta;
    imp_req->logprobs = req_logprobs;
    imp_req->top_logprobs = top_logprobs;
    imp_req->status = imp::RequestStatus::PENDING;

    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = imp_req;

    if (!state.batching || !state.batching->is_running()) {
        res.status = 503;
        json err = {{"error", {{"message", "Inference engine not ready. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }
    state.batching->submit(server_req);

    std::string comp_id = req_id;
    int64_t created = unix_timestamp();

    if (stream) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, server_req, comp_id, created, max_tokens, n_prompt_tokens, t_start,
             stop_sequences, max_stop_len, echo, prompt, include_usage](
                size_t /*offset*/, httplib::DataSink& sink) -> bool {

                int n_output_tokens = 0;
                const char* finish = nullptr;

                // Echo prompt as first chunk if requested
                if (echo && !prompt.empty()) {
                    std::string chunk = sse_completion_chunk(comp_id, created,
                                                             state.model_name, prompt, nullptr);
                    sink.write(chunk.data(), chunk.size());
                }

                std::string utf8_buf;
                std::string pending_text;
                bool text_stop_matched = false;

                // Strip <think> blocks for completions (no reasoning_content field)
                bool think_strip = (state.is_think_model &&
                                    state.default_args.reasoning_format != "none");
                bool think_confirmed = think_strip;
                std::string think_buf;
                int think_tokens = 0;
                const int kThinkScanLimit = 8;

                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0) return true;
                    std::string to_send = pending_text.substr(0, up_to);
                    pending_text = pending_text.substr(up_to);
                    std::string sse = sse_completion_chunk(comp_id, created,
                                                           state.model_name, to_send, nullptr);
                    return sink.write(sse.data(), sse.size());
                };

                for (;;) {
                    TokenEvent evt;
                    server_req->pop_token(evt);

                    if (evt.token_id < 0) {
                        finish = evt.finish_reason ? evt.finish_reason : "stop";
                        break;
                    }

                    int32_t token = evt.token_id;

                    if (evt.is_last) {
                        if (token == state.tok->eos_id()) {
                            finish = evt.finish_reason ? evt.finish_reason : "stop";
                            break;
                        }
                        finish = evt.finish_reason ? evt.finish_reason : "length";
                    }

                    n_output_tokens++;
                    std::string piece = state.tok->decode_token(token);

                    // Strip <think>...</think> block for text completions
                    if (think_strip) {
                        think_buf += piece;
                        think_tokens++;

                        if (!think_confirmed) {
                            if (think_buf.find("<think>") != std::string::npos)
                                think_confirmed = true;
                            else if (think_tokens == 1 && piece.empty())
                                think_confirmed = true;
                        }

                        auto end_pos = think_buf.find("</think>");
                        if (end_pos != std::string::npos) {
                            think_strip = false;
                            std::string after = think_buf.substr(end_pos + 8);
                            think_buf.clear();
                            auto start = after.find_first_not_of("\n\r\t ");
                            piece = (start != std::string::npos) ? after.substr(start) : "";
                            if (piece.empty()) continue;
                        } else if (think_confirmed) {
                            continue;
                        } else if (think_tokens < kThinkScanLimit) {
                            continue;
                        } else {
                            think_strip = false;
                            piece = think_buf;
                            think_buf.clear();
                        }
                    }

                    if (stop_sequences.empty()) {
                        utf8_buf += piece;
                        size_t complete = utf8_complete_len(utf8_buf);
                        if (complete > 0) {
                            std::string to_emit = utf8_buf.substr(0, complete);
                            utf8_buf = utf8_buf.substr(complete);
                            std::string chunk = sse_completion_chunk(comp_id, created,
                                                                      state.model_name,
                                                                      to_emit, nullptr);
                            if (!sink.write(chunk.data(), chunk.size())) return false;
                        }
                    } else {
                        pending_text += piece;
                        bool stop_found = false;
                        for (const auto& stop : stop_sequences) {
                            auto pos = pending_text.find(stop);
                            if (pos != std::string::npos) {
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
                        if (pending_text.size() > max_stop_len) {
                            size_t safe = pending_text.size() - max_stop_len + 1;
                            if (!flush_text(safe)) return false;
                        }
                    }

                    if (finish) break;
                }

                // Flush think buffer: strip think blocks and emit remaining content
                if (!think_buf.empty()) {
                    strip_think_block(think_buf);
                    if (!think_buf.empty()) {
                        utf8_buf += think_buf;
                    }
                    think_buf.clear();
                }

                // Flush remaining buffers
                if (!utf8_buf.empty() && !text_stop_matched) {
                    std::string sse = sse_completion_chunk(comp_id, created,
                                                           state.model_name, utf8_buf, nullptr);
                    sink.write(sse.data(), sse.size());
                }
                if (!pending_text.empty() && !text_stop_matched) {
                    std::string sse = sse_completion_chunk(comp_id, created,
                                                           state.model_name, pending_text, nullptr);
                    sink.write(sse.data(), sse.size());
                }

                if (!finish) finish = "length";

                // Final chunk with finish_reason
                std::string final_chunk = sse_completion_chunk(comp_id, created,
                                                                state.model_name, "", finish);
                sink.write(final_chunk.data(), final_chunk.size());

                // Usage chunk if requested
                if (include_usage) {
                    json usage_obj = {
                        {"id", comp_id},
                        {"object", "text_completion"},
                        {"created", created},
                        {"model", state.model_name},
                        {"choices", json::array()},
                        {"usage", {
                            {"prompt_tokens", n_prompt_tokens},
                            {"completion_tokens", n_output_tokens},
                            {"total_tokens", n_prompt_tokens + n_output_tokens}
                        }}
                    };
                    std::string usage_chunk = "data: " + usage_obj.dump() + "\n\n";
                    sink.write(usage_chunk.data(), usage_chunk.size());
                }

                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();

                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                        comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);
                state.metrics.requests_total++;
                state.metrics.tokens_prompt_total += n_prompt_tokens;
                state.metrics.tokens_completion_total += n_output_tokens;
                state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

                return true;
            }
        );
    } else {
        // Non-streaming
        auto active_req = server_req->request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;

        for (;;) {
            TokenEvent evt;
            server_req->pop_token(evt);

            if (evt.token_id < 0) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }

            int32_t token = evt.token_id;

            if (evt.is_last) {
                if (token == state.tok->eos_id()) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                finish = evt.finish_reason ? evt.finish_reason : "length";
            }

            output_ids.push_back(token);

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

            if (finish) break;
        }

        if (!finish) finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string text = !stop_sequences.empty()
            ? output_text : state.tok->decode(output_ids);

        // Strip <think>...</think> for text completions (no reasoning_content field)
        if (state.is_think_model && state.default_args.reasoning_format != "none") {
            strip_think_block(text);
        }

        // Prepend prompt if echo requested
        if (echo) text = prompt + text;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);
        state.metrics.requests_total++;
        state.metrics.tokens_prompt_total += n_prompt_tokens;
        state.metrics.tokens_completion_total += n_output_tokens;
        state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

        // Build logprobs if requested
        json logprobs_obj = nullptr;
        if (req_logprobs && active_req) {
            const auto& lp_data = active_req->output_logprobs;
            json content_logprobs = json::array();
            for (size_t idx = 0; idx < lp_data.size() && idx < output_ids.size(); idx++) {
                const auto& lp = lp_data[idx];
                json top_arr = json::array();
                for (const auto& t : lp.top) {
                    top_arr.push_back({
                        {"token", safe_token_json(t.text)},
                        {"logprob", t.logprob},
                        {"bytes", token_bytes_json(t.text)}
                    });
                }
                content_logprobs.push_back({
                    {"token", safe_token_json(lp.text)},
                    {"logprob", lp.logprob},
                    {"bytes", token_bytes_json(lp.text)},
                    {"top_logprobs", top_arr}
                });
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        json choice = {
            {"index", 0},
            {"text", text},
            {"finish_reason", finish}
        };
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        json response = {
            {"id", comp_id},
            {"object", "text_completion"},
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

void handle_tokenize(const httplib::Request& req, httplib::Response& res,
                     ServerState& state) {
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

    std::string content = body.value("content", "");
    if (content.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"content\" is required"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (!state.model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    std::vector<int32_t> tokens(32768);
    int n_tokens = 0;
    ImpError err = imp_tokenize(state.model, content.c_str(), tokens.data(), &n_tokens, 32768);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Tokenize failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    tokens.resize(n_tokens);
    json response = {{"tokens", tokens}};
    res.set_content(response.dump(), "application/json");
}

void handle_detokenize(const httplib::Request& req, httplib::Response& res,
                       ServerState& state) {
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

    if (!body.contains("tokens") || !body["tokens"].is_array()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"tokens\" array is required"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (!state.model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    std::vector<int32_t> tokens = body["tokens"].get<std::vector<int32_t>>();
    std::vector<char> buf(tokens.size() * 32 + 256);
    ImpError err = imp_detokenize(state.model, tokens.data(), static_cast<int>(tokens.size()),
                                   buf.data(), buf.size());
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Detokenize failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    json response = {{"content", std::string(buf.data())}};
    res.set_content(response.dump(), "application/json");
}

void handle_load_model(const httplib::Request& req, httplib::Response& res,
                       ServerState& state) {
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

    std::string path = body.value("path", "");
    if (path.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"path\" is required"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    json config_overrides = body.value("config", json::object());

    // Block inference during model swap
    std::lock_guard<std::timed_mutex> lock(state.mtx);

    printf("Loading model: %s\n", path.c_str());
    fflush(stdout);

    std::string error = load_model_into_state(state, path, config_overrides);
    if (!error.empty()) {
        res.status = 500;
        json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    printf("Model loaded: %s\n", state.model_name.c_str());
    fflush(stdout);

    json response = {
        {"status", "loaded"},
        {"model", state.model_name}
    };
    res.set_content(response.dump(), "application/json");
}

void handle_unload_model(const httplib::Request& /*req*/, httplib::Response& res,
                         ServerState& state) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);

    // Stop batching engine before freeing context
    if (state.batching) {
        state.batching->stop();
        state.batching.reset();
    }

    if (state.ctx) { imp_context_free(state.ctx); state.ctx = nullptr; }
    if (state.model) { imp_model_free(state.model); state.model = nullptr; }
    state.tok = nullptr;
    state.have_template = false;
    state.model_name.clear();
    state.max_seq_len = 0;

    printf("Model unloaded\n");
    fflush(stdout);

    json response = {{"status", "unloaded"}};
    res.set_content(response.dump(), "application/json");
}

void handle_metrics(const httplib::Request& /*req*/, httplib::Response& res,
                    ServerState& state) {
    auto& m = state.metrics;
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - m.start_time).count();

    std::string out;
    out.reserve(1024);
    out += "# HELP imp_uptime_seconds Server uptime in seconds\n";
    out += "# TYPE imp_uptime_seconds gauge\n";
    out += "imp_uptime_seconds " + std::to_string(uptime) + "\n";
    out += "# HELP imp_requests_total Total inference requests\n";
    out += "# TYPE imp_requests_total counter\n";
    out += "imp_requests_total " + std::to_string(m.requests_total.load()) + "\n";
    out += "# HELP imp_requests_failed_total Failed inference requests\n";
    out += "# TYPE imp_requests_failed_total counter\n";
    out += "imp_requests_failed_total " + std::to_string(m.requests_failed.load()) + "\n";
    out += "# HELP imp_tokens_prompt_total Total prompt tokens processed\n";
    out += "# TYPE imp_tokens_prompt_total counter\n";
    out += "imp_tokens_prompt_total " + std::to_string(m.tokens_prompt_total.load()) + "\n";
    out += "# HELP imp_tokens_completion_total Total completion tokens generated\n";
    out += "# TYPE imp_tokens_completion_total counter\n";
    out += "imp_tokens_completion_total " + std::to_string(m.tokens_completion_total.load()) + "\n";
    out += "# HELP imp_last_request_duration_ms Duration of last request in milliseconds\n";
    out += "# TYPE imp_last_request_duration_ms gauge\n";
    out += "imp_last_request_duration_ms " + std::to_string(m.last_request_duration_ms.load()) + "\n";
    out += "# HELP imp_model_loads_total Total model loads\n";
    out += "# TYPE imp_model_loads_total counter\n";
    out += "imp_model_loads_total " + std::to_string(m.model_loads_total.load()) + "\n";
    out += "# HELP imp_model_loaded Whether a model is currently loaded\n";
    out += "# TYPE imp_model_loaded gauge\n";
    out += "imp_model_loaded " + std::string(state.model_loaded() ? "1" : "0") + "\n";
    out += "# HELP imp_queue_depth Current number of active and pending requests\n";
    out += "# TYPE imp_queue_depth gauge\n";
    out += "imp_queue_depth " + std::to_string(state.batching ? state.batching->queue_depth() : 0) + "\n";

    res.set_content(out, "text/plain; version=0.0.4; charset=utf-8");
}

// Convert IEEE 754 FP16 (uint16_t) to FP32 on host
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // Subnormal: normalize
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

void handle_embeddings(const httplib::Request& req, httplib::Response& res,
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

    // Collect inputs: "input" can be a string or array of strings
    std::vector<std::string> inputs;
    if (body.contains("input")) {
        if (body["input"].is_string()) {
            inputs.push_back(body["input"].get<std::string>());
        } else if (body["input"].is_array()) {
            for (const auto& item : body["input"]) {
                if (item.is_string()) {
                    inputs.push_back(item.get<std::string>());
                } else {
                    res.status = 400;
                    json err = {{"error", {{"message", "Each input must be a string"},
                                            {"type", "invalid_request_error"}}}};
                    res.set_content(err.dump(), "application/json");
                    return;
                }
            }
        } else {
            res.status = 400;
            json err = {{"error", {{"message", "\"input\" must be a string or array of strings"},
                                    {"type", "invalid_request_error"}}}};
            res.set_content(err.dump(), "application/json");
            return;
        }
    } else {
        res.status = 400;
        json err = {{"error", {{"message", "\"input\" is required"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (inputs.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"input\" must not be empty"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (!state.model_loaded()) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Acquire inference lock and pause batching engine for exclusive access
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(1));
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error", {{"message", "Server is busy processing another request. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Pause batching engine for exclusive C API access, restart on scope exit
    bool had_batching = (state.batching && state.batching->is_running());
    if (had_batching) state.batching->stop();
    auto restart_batching = [&] {
        if (had_batching && state.batching && state.ctx)
            state.batching->start(state.ctx);
    };
    // Use a simple scope guard
    struct ScopeGuard {
        std::function<void()> fn;
        ~ScopeGuard() { fn(); }
    } batching_guard{restart_batching};

    state.metrics.requests_total++;
    auto t0 = std::chrono::steady_clock::now();

    // Get model dimensions
    int d_model = imp_model_d_model(state.model);
    int total_prompt_tokens = 0;

    json data = json::array();

    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        const auto& text = inputs[input_idx];

        // Tokenize
        std::vector<int32_t> tokens(32768);
        int n_tokens = 0;
        ImpError err = imp_tokenize(state.model, text.c_str(),
                                     tokens.data(), &n_tokens, 32768);
        if (err != IMP_SUCCESS) {
            res.status = 500;
            json error = {{"error", {{"message",
                std::string("Tokenize failed: ") + imp_error_string(err)},
                {"type", "server_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }
        tokens.resize(n_tokens);

        if (n_tokens == 0) {
            res.status = 400;
            json error = {{"error", {{"message", "Input tokenizes to zero tokens"},
                                      {"type", "invalid_request_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        total_prompt_tokens += n_tokens;

        // Run prefill (forward pass without generation)
        err = imp_prefill(state.ctx, tokens.data(), n_tokens);
        if (err != IMP_SUCCESS) {
            imp_context_reset(state.ctx);
            res.status = 500;
            json error = {{"error", {{"message",
                std::string("Prefill failed: ") + imp_error_string(err)},
                {"type", "server_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Extract hidden states from the executor
        // hidden_ is [n_tokens, d_model] FP16 on GPU after forward_logits()
        auto* engine = state.ctx->engine.get();
        auto* executor = engine->executor();
        imp::Tensor hidden_view = executor->view_hidden(n_tokens);

        // Copy FP16 hidden states from GPU to host as uint16_t
        size_t n_elements = static_cast<size_t>(n_tokens) * d_model;
        std::vector<uint16_t> h_hidden(n_elements);
        cudaError_t cuda_err = cudaMemcpy(h_hidden.data(), hidden_view.data,
                                           n_elements * sizeof(uint16_t),
                                           cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            imp_context_reset(state.ctx);
            res.status = 500;
            json error = {{"error", {{"message",
                std::string("CUDA memcpy failed: ") + cudaGetErrorString(cuda_err)},
                {"type", "server_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Mean-pool across tokens: average all token hidden states
        std::vector<float> embedding(d_model, 0.0f);
        for (int t = 0; t < n_tokens; ++t) {
            for (int d = 0; d < d_model; ++d) {
                embedding[d] += fp16_to_fp32(h_hidden[t * d_model + d]);
            }
        }
        float inv_n = 1.0f / static_cast<float>(n_tokens);
        for (int d = 0; d < d_model; ++d) {
            embedding[d] *= inv_n;
        }

        // L2 normalize
        float norm_sq = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            norm_sq += embedding[d] * embedding[d];
        }
        float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
        for (int d = 0; d < d_model; ++d) {
            embedding[d] *= inv_norm;
        }

        data.push_back({
            {"object", "embedding"},
            {"embedding", embedding},
            {"index", input_idx}
        });

        // Reset context for next input
        imp_context_reset(state.ctx);
    }

    auto t1 = std::chrono::steady_clock::now();
    int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    state.metrics.last_request_duration_ms.store(duration_ms);
    state.metrics.tokens_prompt_total += total_prompt_tokens;

    // batching_guard restarts the batching engine automatically on scope exit

    json response = {
        {"object", "list"},
        {"data", data},
        {"model", body.value("model", state.model_name)},
        {"usage", {
            {"prompt_tokens", total_prompt_tokens},
            {"total_tokens", total_prompt_tokens}
        }}
    };
    res.set_content(response.dump(), "application/json");
}
