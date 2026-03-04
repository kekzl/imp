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
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

// Returns the number of leading bytes that form complete UTF-8 characters.
// Bytes from complete_len onwards are an incomplete trailing sequence.
static size_t utf8_complete_len(const std::string& s) {
    size_t len = s.size();
    if (len == 0) return 0;
    // Walk back from end to find the start of the last codepoint
    size_t i = len - 1;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80)
        --i;
    unsigned char lead = static_cast<unsigned char>(s[i]);
    int expected;
    if (lead < 0x80) expected = 1;
    else if ((lead & 0xE0) == 0xC0) expected = 2;
    else if ((lead & 0xF0) == 0xE0) expected = 3;
    else if ((lead & 0xF8) == 0xF0) expected = 4;
    else return i; // invalid byte — emit up to it
    if (i + static_cast<size_t>(expected) <= len) return len; // complete
    return i; // incomplete — emit up to start of this sequence
}

// Simple base64 decoder for image data URIs
static int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static std::vector<uint8_t> base64_decode(const std::string& encoded) {
    std::vector<uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    uint32_t accum = 0;
    int bits = 0;
    for (unsigned char c : encoded) {
        int val = b64_val(c);
        if (val < 0) continue;
        accum = (accum << 6) | static_cast<uint32_t>(val);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFF));
        }
    }
    return out;
}

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
    std::timed_mutex mtx;
    int default_max_tokens = 2048;
    int max_seq_len = 0;
    std::atomic<int> next_id{0};
    ServerArgs default_args;
    std::string models_dir;  // directory to scan for available .gguf files
    std::string api_key;     // if non-empty, require Bearer token auth
    bool model_loaded() const { return ctx != nullptr; }
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

static void handle_health(const httplib::Request& /*req*/, httplib::Response& res,
                          ServerState& state) {
    json body = {
        {"status", "ok"},
        {"model_loaded", state.model_loaded()}
    };
    res.set_content(body.dump(), "application/json");
}

// Recursively find all .gguf files in a directory, returning (filename, full_path) pairs
static std::vector<std::pair<std::string, std::string>> scan_gguf_files(const std::string& dir) {
    std::vector<std::pair<std::string, std::string>> results;
    if (dir.empty()) return results;
    std::error_code ec;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file() && !entry.is_symlink()) continue;
        auto path = entry.path();
        if (path.extension() == ".gguf" && path.string().find(".no_exist") == std::string::npos) {
            results.emplace_back(path.filename().string(), path.string());
        }
    }
    std::sort(results.begin(), results.end());
    return results;
}

static void handle_models(const httplib::Request& /*req*/, httplib::Response& res,
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

// Forward declaration (defined after build_config)
static std::string load_model_into_state(ServerState& state, const std::string& path,
                                         const json& config_overrides = json::object());

// Find a GGUF file by name in models_dir. Returns full path or empty string.
static std::string find_model_path(const ServerState& state, const std::string& name) {
    auto available = scan_gguf_files(state.models_dir);
    for (const auto& [fname, fpath] : available) {
        if (fname == name) return fpath;
    }
    return "";
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

    // Parse stream_options for include_usage
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Convert JSON messages to ChatMessage vector, extracting image data if present
    std::vector<imp::ChatMessage> chat_msgs;
    std::vector<uint8_t> image_data;  // decoded image bytes (if any)
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");
        if (msg.contains("content") && msg["content"].is_array()) {
            // OpenAI multimodal format: content is array of parts
            std::string text_parts;
            for (const auto& part : msg["content"]) {
                std::string type = part.value("type", "");
                if (type == "text") {
                    if (!text_parts.empty()) text_parts += "\n";
                    text_parts += part.value("text", "");
                } else if (type == "image_url" && part.contains("image_url")) {
                    std::string url = part["image_url"].value("url", "");
                    // Handle data URI: data:image/...;base64,...
                    auto comma = url.find(',');
                    if (url.rfind("data:", 0) == 0 && comma != std::string::npos) {
                        image_data = base64_decode(url.substr(comma + 1));
                    }
                }
            }
            chat_msgs.push_back({role, text_parts});
        } else {
            std::string content = msg.value("content", "");
            chat_msgs.push_back({role, content});
        }
    }

    // Acquire inference lock — wait up to 5 minutes (covers model load + generation)
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error", {{"message", "Server is busy processing another request. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Auto-load model if request specifies a different one
    std::string requested_model = body.value("model", "");
    if (!requested_model.empty() && requested_model != state.model_name) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            printf("Auto-loading model: %s\n", requested_model.c_str());
            fflush(stdout);
            std::string error = load_model_into_state(state, path);
            if (!error.empty()) {
                res.status = 500;
                json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            printf("Model loaded: %s\n", state.model_name.c_str());
            fflush(stdout);
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

    // Handle vision: clear previous, encode new image if present
    state.ctx->engine->clear_image();
    if (!image_data.empty() && state.ctx->engine->has_vision()) {
        if (!state.ctx->engine->set_image_from_memory(image_data.data(), image_data.size())) {
            res.status = 400;
            json error = {{"error", {{"message", "Failed to process image"},
                                      {"type", "invalid_request_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }
    }

    // Tokenize with chat template (with image tokens if vision is active)
    std::vector<int32_t> tokens;
    if (state.have_template && state.ctx->engine->has_vision_input()) {
        tokens = state.chat_tpl.apply_with_image(*state.tok, chat_msgs, 256);
    } else if (state.have_template) {
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
             stop_sequences, max_stop_len, req_logprobs, include_usage](
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

                // Buffer for incomplete UTF-8 sequences across token boundaries
                std::string utf8_buf;

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
                        // No stop sequences: stream directly (with UTF-8 buffering)
                        utf8_buf += piece;
                        size_t complete = utf8_complete_len(utf8_buf);
                        if (complete > 0) {
                            std::string to_emit = utf8_buf.substr(0, complete);
                            utf8_buf = utf8_buf.substr(complete);

                            json content_delta = {{"content", to_emit}};
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

                // Flush any remaining UTF-8 buffer
                if (!utf8_buf.empty() && !text_stop_matched) {
                    json content_delta = {{"content", utf8_buf}};
                    std::string sse = sse_chunk(comp_id, created, state.model_name,
                                                content_delta, nullptr);
                    sink.write(sse.data(), sse.size());
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

                // Send usage chunk if requested
                if (include_usage) {
                    json usage_obj = {
                        {"id", comp_id},
                        {"object", "chat.completion.chunk"},
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

// Build ImpConfig from default args + optional JSON overrides
static ImpConfig build_config(const ServerArgs& args, const json& overrides = json::object()) {
    ImpConfig config = imp_config_default();
    config.device_id = args.device;
    config.max_batch_size = 1;
    config.max_seq_len = overrides.value("max_seq_len", 4096);
    config.gpu_layers = args.gpu_layers;
    if (args.ssm_fp16) config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs) config.enable_cuda_graphs = 0;

    // KV cache dtype: check overrides first, then startup args
    bool kv_fp8 = overrides.value("kv_fp8", args.kv_fp8);
    bool kv_int8 = overrides.value("kv_int8", args.kv_int8);
    if (kv_fp8) config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (kv_int8) config.kv_cache_dtype = IMP_DTYPE_INT8;

    int chunk = overrides.value("prefill_chunk_size", args.prefill_chunk_size);
    if (chunk > 0) config.prefill_chunk_size = chunk;

    config.use_nvfp4_decode = overrides.value("decode_nvfp4", args.decode_nvfp4);

    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();

    return config;
}

// Load a model into ServerState. Caller must hold state.mtx.
// Returns error message on failure, empty string on success.
static std::string load_model_into_state(ServerState& state, const std::string& path,
                                         const json& config_overrides) {
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

    // Create context
    ImpConfig config = build_config(state.default_args, config_overrides);
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

    if (state.have_template) {
        printf("Chat template: %s\n",
               imp::chat_template_family_name(state.chat_tpl.family()));
    } else {
        printf("No chat template (raw mode)\n");
    }

    return "";
}

// Build a single SSE data line for text completions
static std::string sse_completion_chunk(const std::string& id, int64_t created,
                                         const std::string& model,
                                         const std::string& text,
                                         const char* finish_reason) {
    json choice = {
        {"index", 0},
        {"text", text},
        {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}
    };
    json obj = {
        {"id", id},
        {"object", "text_completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array({choice})}
    };
    return "data: " + obj.dump() + "\n\n";
}

static void handle_completions(const httplib::Request& req, httplib::Response& res,
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
    float top_p = body.value("top_p", 0.9f);
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

    // Acquire inference lock
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error", {{"message", "Server is busy processing another request. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Auto-load model if request specifies a different one
    std::string requested_model = body.value("model", "");
    if (!requested_model.empty() && requested_model != state.model_name) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            printf("Auto-loading model: %s\n", requested_model.c_str());
            fflush(stdout);
            std::string error = load_model_into_state(state, path);
            if (!error.empty()) {
                res.status = 500;
                json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            printf("Model loaded: %s\n", state.model_name.c_str());
            fflush(stdout);
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

    // Reset and prefill
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

    std::string comp_id = make_completion_id(state);
    int64_t created = unix_timestamp();

    if (stream) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, params, comp_id, created, max_tokens, n_prompt_tokens, t_start,
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

                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0) return true;
                    std::string to_send = pending_text.substr(0, up_to);
                    pending_text = pending_text.substr(up_to);
                    std::string sse = sse_completion_chunk(comp_id, created,
                                                           state.model_name, to_send, nullptr);
                    return sink.write(sse.data(), sse.size());
                };

                for (int step = 0; step < max_tokens; step++) {
                    int32_t token = 0;
                    ImpError err = imp_decode_step(state.ctx, &params, &token);
                    if (err != IMP_SUCCESS) {
                        finish = "stop";
                        break;
                    }

                    if (token == state.tok->eos_id()) {
                        finish = "stop";
                        break;
                    }

                    n_output_tokens++;
                    std::string piece = state.tok->decode_token(token);

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

                return true;
            }
        );
    } else {
        // Non-streaming
        auto active_req = state.ctx->active_request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;

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
        }

        if (!finish) finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string text = !stop_sequences.empty()
            ? output_text : state.tok->decode(output_ids);

        // Prepend prompt if echo requested
        if (echo) text = prompt + text;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

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

static void handle_tokenize(const httplib::Request& req, httplib::Response& res,
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

static void handle_detokenize(const httplib::Request& req, httplib::Response& res,
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

static void handle_load_model(const httplib::Request& req, httplib::Response& res,
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

static void handle_unload_model(const httplib::Request& /*req*/, httplib::Response& res,
                                ServerState& state) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);

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

int main(int argc, char** argv) {
    ServerArgs args = parse_server_args(argc, argv);

    printf("IMP Server %s\n", imp_version());

    ServerState state;
    state.default_max_tokens = args.max_tokens;
    state.default_args = args;

    // Set models directory (explicit flag, or infer from model path, or /models default)
    if (!args.models_dir.empty()) {
        state.models_dir = args.models_dir;
    } else if (!args.model_path.empty()) {
        auto parent = std::filesystem::path(args.model_path).parent_path().string();
        if (!parent.empty()) state.models_dir = parent;
    } else if (std::filesystem::is_directory("/models")) {
        state.models_dir = "/models";
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

    // Store API key in state
    state.api_key = args.api_key;

    // CORS headers + API key auth on every response
    svr.set_pre_routing_handler([&state](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

        // Skip auth for health checks and CORS preflight
        if (req.path == "/health" || req.method == "OPTIONS")
            return httplib::Server::HandlerResponse::Unhandled;

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

    svr.Post("/tokenize",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_tokenize(req, res, state);
        });

    svr.Post("/detokenize",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_detokenize(req, res, state);
        });

    // Graceful shutdown on SIGINT/SIGTERM
    g_server.store(&svr, std::memory_order_relaxed);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (!state.api_key.empty()) printf("API key: enabled\n");

    printf("Server listening on http://%s:%d\n", args.host.c_str(), args.port);
    printf("Endpoints:\n");
    printf("  GET    /health\n");
    printf("  GET    /v1/models\n");
    printf("  POST   /v1/models          Load/swap model\n");
    printf("  DELETE /v1/models          Unload model\n");
    printf("  POST   /v1/chat/completions\n");
    printf("  POST   /v1/completions\n");
    printf("  POST   /tokenize\n");
    printf("  POST   /detokenize\n");
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
