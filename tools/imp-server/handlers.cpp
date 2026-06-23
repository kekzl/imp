#include "handlers.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"

#include "api/imp_internal.h"
#include "memory/kv_cache.h"
#include "model/hf_hub.h"
#include "runtime/config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Chat completion context (bundles state for handle_chat_completions phases)
// ---------------------------------------------------------------------------
namespace {

// Body-parsed input parameters (no lock needed to populate).
struct ChatRequestParams {
    // Sampling
    float temperature = 0.7f, top_p = 0.95f, min_p = 0.0f, typical_p = 1.0f;
    float repetition_penalty = 1.05f;
    float frequency_penalty = 0.0f, presence_penalty = 0.0f;
    float dry_multiplier = 0.0f, dry_base = 1.75f;
    float mirostat_tau = 5.0f, mirostat_eta = 0.1f;
    float think_budget = 0.0f;
    int top_k = 40, max_tokens = 0, seed = -1, repeat_last_n = 0;
    int dry_allowed_length = 2, dry_penalty_last_n = 0, mirostat = 0;
    int n_completions = 1, top_logprobs = 0;
    bool stream = false, json_mode = false, req_logprobs = false, include_usage = false;
    bool top_p_explicit = false, top_k_explicit = false, rep_pen_explicit = false;
    // Pin the prompt's KV blocks against eviction (Anthropic cache_control →
    // mapped by anthropic_to_openai_body; also a direct llama.cpp-style
    // "cache_prompt" body field on the OpenAI route).
    bool cache_prompt = false;
    bool enable_thinking_requested = false;  // value of "enable_thinking" if present
    std::string lora_name;                   // "lora" body field (empty = base model)
    bool enable_thinking_set = false;        // true iff body contained "enable_thinking"
    // Stop sequences
    std::vector<std::string> stop_sequences;
    size_t max_stop_len = 0;
    // Logit bias / format
    std::vector<std::pair<int32_t, float>> logit_bias;
    std::string json_schema_str;
    // Tools
    nlohmann::json tools;
    nlohmann::json tool_choice;
    bool has_tools = false;
    // Messages + image
    std::vector<imp::ChatMessage> chat_msgs;
    std::vector<uint8_t> image_data;
    std::string requested_model;
};

// Lock-acquired engine state (populated under state.mtx).
struct ChatStateSnapshot {
    imp::Tokenizer* tok = nullptr;
    imp::ChatTemplate chat_tpl;
    bool have_template = false;
    std::string model_name;
    bool is_think_model = false;
    int32_t think_start_id = -1, think_end_id = -1;
    int32_t channel_open_id = -1, channel_close_id = -1, channel_newline_id = -1;
    int max_seq_len = 0;
    bool has_vision_request = false;
    std::vector<int32_t> stop_token_ids;
    imp::ChatTemplateFamily tpl_family = imp::ChatTemplateFamily::CHATML;
    std::vector<imp::ToolFunction> tool_defs;
    bool tools_via_jinja = false;
    bool enable_thinking = false, suppress_thinking = false;
    std::vector<int32_t> tokens;
    int n_prompt_tokens = 0;
};

// Top-level context bundling params + snap + transients.
struct ChatRequestContext {
    ChatRequestParams params;
    ChatStateSnapshot snap;
    std::string req_id;
    std::string comp_id;
    int64_t created = 0;
    std::chrono::high_resolution_clock::time_point t_start;
    std::chrono::system_clock::time_point t_log_start;
    std::string log_endpoint, log_client_ip, log_raw_body;
    bool log_skip = false;
    std::shared_ptr<imp::Request> imp_req;
    std::shared_ptr<ServerRequest> server_req;
};

// cache_creation_input_tokens (Anthropic): full prompt blocks newly written
// and pinned by this request — block-rounded prompt minus prefix-cache hits.
int cache_creation_tokens_(const std::shared_ptr<imp::Request>& req, int n_prompt_tokens) {
    if (!req || !req->pin_kv_prefix)
        return 0;
    int rounded = (n_prompt_tokens / imp::kKVBlockSize) * imp::kKVBlockSize;
    int creation = rounded - req->cached_tokens;
    return creation > 0 ? creation : 0;
}

}  // anonymous namespace

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
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void handle_health(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    bool loaded;
    int queue = 0;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        loaded = state.model_loaded();
        if (state.batching)
            queue = state.batching->queue_depth();
    }
    json body = {{"status", "ok"}, {"model_loaded", loaded}, {"queue_depth", queue}};
    res.set_content(dump_safe(body), "application/json");
}

// Recursively find all model files in a directory, returning (display_name, full_path) pairs.
// Finds both .gguf files and SafeTensors directories (containing model.safetensors[.index.json]).
// Resolves symlinks and rejects any path that escapes the base directory (path traversal).
std::vector<std::pair<std::string, std::string>> scan_model_files(const std::string& dir) {
    std::vector<std::pair<std::string, std::string>> results;
    if (dir.empty())
        return results;
    std::error_code ec;
    auto base = std::filesystem::canonical(dir, ec);
    if (ec)
        return results;
    std::string base_prefix = base.string() + "/";

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        const auto& path = entry.path();
        // GGUF files
        if ((entry.is_regular_file() || entry.is_symlink()) && path.extension() == ".gguf" &&
            path.string().find(".no_exist") == std::string::npos) {
            std::error_code ec2;
            auto real = std::filesystem::canonical(path, ec2);
            if (ec2)
                continue;
            std::string real_str = real.string();
            if (real_str.compare(0, base_prefix.size(), base_prefix) != 0)
                continue;
            results.emplace_back(path.filename().string(), real_str);
        }
        // SafeTensors directories (check for index or single file)
        if (entry.is_directory()) {
            std::string dpath = path.string();
            if (imp::is_safetensors_dir(dpath)) {
                std::error_code ec2;
                auto real = std::filesystem::canonical(path, ec2);
                if (ec2)
                    continue;
                std::string real_str = real.string();
                if (real_str.compare(0, base_prefix.size(), base_prefix) != 0)
                    continue;
                results.emplace_back(path.filename().string(), real_str);
            }
        }
    }
    std::sort(results.begin(), results.end());
    return results;
}

void handle_models(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    json data = json::array();

    // Snapshot state fields under lock
    bool loaded;
    std::string model_name;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        loaded = state.model_loaded();
        model_name = state.model_name;
    }

    // OpenAI semantics: expose only what this server can actually serve —
    // the loaded model. Listing the whole models directory invited clients
    // to request models the server then had to swap in mid-flight.
    if (loaded) {
        data.push_back({{"id", model_name}, {"object", "model"}, {"owned_by", "imp"}});
    }

    json body = {{"object", "list"}, {"data", data}};
    res.set_content(dump_safe(body), "application/json");
}

// Find a model by name in models_dir. Returns full path or empty string.
// Supports both GGUF files and SafeTensors directories.
// Also tries HuggingFace resolution if name looks like a repo ID (contains '/').
std::string find_model_path(const ServerState& state, const std::string& name) {
    // First try local models directory
    auto available = scan_model_files(state.models_dir);
    for (const auto& [fname, fpath] : available) {
        if (fname == name)
            return fpath;
    }

    // If it looks like a HuggingFace repo ID (contains '/'), try resolving
    if (name.find('/') != std::string::npos) {
        ImpModelFormat fmt;
        std::string resolved = imp::resolve_model_auto(name, fmt);
        if (!resolved.empty())
            return resolved;
    }

    return "";
}

// Serve only the loaded model (OpenAI semantics): requesting any other model
// name gets 404 model_not_found. Inference requests never trigger a model
// swap — the old auto-swap tore down the engine mid-stream, cancelling every
// in-flight request (and the whole process if the new model didn't fit).
// Switching models is an operator action: restart with a different --model.
//
// The one lifecycle action that remains on this path: if the server was
// started without a model, the first request's model is resolved from the
// models directory and loaded.
//
// Returns true if the requested model is loaded. Must be called with
// state.mtx held.
bool ensure_model_loaded(ServerState& state, const std::string& requested_model, httplib::Response& res) {
    if (!state.model_loaded()) {
        // No model loaded — try to load the requested one
        std::string path = find_model_path(state, requested_model);
        if (path.empty()) {
            res.status = 503;
            json err = {
                {"error",
                 {{"message", "No model loaded and '" + requested_model + "' not found in models directory"},
                  {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
        printf("[auto-load] Loading %s...\n", requested_model.c_str());
        fflush(stdout);
        std::string error = load_model_into_state(state, path, json::object());
        if (!error.empty()) {
            res.status = 500;
            json err = {{"error", {{"message", "Auto-load failed: " + error}, {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
        printf("[auto-load] %s loaded successfully\n", requested_model.c_str());
        fflush(stdout);
        return true;
    }

    if (requested_model == state.model_name) {
        return true;  // Already loaded
    }

    res.status = 404;
    json err = {{"error",
                 {{"message", "The model '" + requested_model + "' does not exist; this server is serving '" +
                                  state.model_name + "'"},
                  {"type", "invalid_request_error"},
                  {"param", "model"},
                  {"code", "model_not_found"}}}};
    res.set_content(dump_safe(err), "application/json");
    return false;
}

// Build ImpConfig from default args + optional JSON overrides.
// Engine auto-detects max_seq_len, max_batch_size, KV dtype, FP8 prefill, NVFP4 decode.
ImpConfig build_config(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg,
                       const std::string& model_path, const json& overrides, ImpModel model = nullptr) {
    (void)model_path;
    (void)model;
    ImpConfig config = imp_config_default();

    config.device_id = args.device;

    // max_seq_len / max_batch_size: 0 = auto-detect in engine.
    // Precedence for the batch size: per-request JSON override > --max-batch CLI
    // flag > [runtime] max_batch_size from imp.conf > 0 (engine auto-sizes from
    // the model's weight footprint; a >20 GiB MoE auto-picks 1). The imp.conf
    // value used to be dropped here — only the CLI arg seeded sizing — so
    // `[runtime] max_batch_size` silently affected nothing but the decode cap.
    if (overrides.contains("max_seq_len"))
        config.max_seq_len = overrides.value("max_seq_len", 0);
    int batch_cli_or_conf =
        args.max_batch_size > 0 ? args.max_batch_size : runtime_cfg.runtime.max_batch_size;
    config.max_batch_size = overrides.value("max_batch_size", batch_cli_or_conf);

    config.gpu_layers = args.gpu_layers;
    if (args.ssm_fp16)
        config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs)
        config.enable_cuda_graphs = 0;

    // KV cache dtype: explicit overrides only (engine auto-detects FP8 on sm_90+)
    bool kv_fp8 = overrides.value("kv_fp8", args.kv_fp8);
    bool kv_int8 = overrides.value("kv_int8", args.kv_int8);
    bool kv_int4 = overrides.value("kv_int4", args.kv_int4);
    bool kv_nvfp4 = overrides.value("kv_nvfp4", args.kv_nvfp4);
    bool kv_mxfp4 = overrides.value("kv_mxfp4", args.kv_mxfp4);
    bool kv_turboquant = overrides.value("kv_turboquant", args.kv_turboquant);
    bool kv_turboquant_lite = overrides.value("kv_turboquant_lite", args.kv_turboquant_lite);
    if (kv_fp8)
        config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (kv_int8)
        config.kv_cache_dtype = IMP_DTYPE_INT8;
    if (kv_int4)
        config.kv_cache_dtype = IMP_DTYPE_INT4;
    if (kv_nvfp4)
        config.kv_cache_dtype = IMP_DTYPE_NVFP4;
    if (kv_mxfp4)
        config.kv_cache_dtype = IMP_DTYPE_MXFP4_KV;
    if (kv_turboquant) {
        // DEPRECATED: TurboQuant retired — falls back to MXFP4-KV
        static bool warned_tq = false;
        if (!warned_tq) {
            fprintf(stderr, "[IMP WARN] --kv-turboquant is deprecated; TurboQuant has been retired. "
                            "Using --kv-mxfp4 instead.\n");
            warned_tq = true;
        }
        config.kv_cache_dtype = IMP_DTYPE_MXFP4_KV;
    }
    if (kv_turboquant_lite) {
        // DEPRECATED: TurboQuant Lite retired — falls back to MXFP4-KV
        static bool warned_tql = false;
        if (!warned_tql) {
            fprintf(stderr, "[IMP WARN] --kv-turboquant-lite is deprecated; TurboQuant has been retired. "
                            "Using --kv-mxfp4 instead.\n");
            warned_tql = true;
        }
        config.kv_cache_dtype = IMP_DTYPE_MXFP4_KV;
    }

    int chunk = overrides.value("prefill_chunk_size", args.prefill_chunk_size);
    // Default chunk = -1 → engine resolver picks per-arch default (512 for
    // full-attention + FP16/FP8 KV, 0 for Gemma-4 / hybrid / sub-byte KV).
    // Pass 0 via --prefill-chunk-size 0 to force single-chunk for all archs.
    config.prefill_chunk_size = chunk;

    int nvfp4 = overrides.value("decode_nvfp4", args.decode_nvfp4);
    config.use_nvfp4_decode = nvfp4;

    if (args.mxfp4_prefill)
        config.use_mxfp4_prefill = 1;
    if (args.dual_path_quant)
        config.dual_path_quant = 1;
    int min_kv = overrides.value("min_kv_tokens", args.min_kv_tokens);
    if (min_kv > 0)
        config.min_kv_tokens = min_kv;

    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();

    // Prefix caching: [server] prefix_cache, default ON since the #536/#538
    // stale-block-table fix (the historical "FP rounding / physical address"
    // off-by-default rationale was a misattribution of that bug —
    // PrefixCacheE2ETest is the ship gate). Disabled automatically for
    // recurrent (SSM/GDN) models in the engine.
    config.use_prefix_caching = runtime_cfg.server.prefix_cache ? 1 : 0;
    config.prefix_pin_budget_pct = runtime_cfg.server.prefix_pin_budget_pct;

    // Green Contexts: SM partitioning for concurrent prefill/decode (CUDA 13.1+)
    config.enable_green_contexts = runtime_cfg.server.green_contexts ? 1 : 0;
    if (!args.prefix_cache_path.empty()) {
        snprintf(config.prefix_cache_path, sizeof(config.prefix_cache_path), "%s",
                 args.prefix_cache_path.c_str());
    }

    return config;
}

// Load a model into ServerState. Caller must hold state.mtx.
// Returns error message on failure, empty string on success.
std::string load_model_into_state(ServerState& state, const std::string& path, const json& config_overrides) {
    // Stop batching engine before freeing context
    if (state.batching) {
        state.batching->stop();
        state.batching.reset();
    }

    // Free existing model/context
    if (state.ctx) {
        imp_context_free(state.ctx);
        state.ctx = nullptr;
    }
    if (state.model) {
        imp_model_free(state.model);
        state.model = nullptr;
    }
    state.tok = nullptr;
    state.have_template = false;
    state.model_name.clear();

    // Auto-detect format from path
    ImpModelFormat format = imp::is_safetensors_dir(path) ? IMP_FORMAT_SAFETENSORS : IMP_FORMAT_GGUF;

    // Load model
    ImpError err = imp_model_load(path.c_str(), format, &state.model);
    if (err != IMP_SUCCESS) {
        std::string msg = std::string("Failed to load model: ") + imp_error_string(err);
        state.model = nullptr;
        return msg;
    }

    // Create context (engine auto-detects config from model metadata).
    // Re-stash the runtime config so Engine::init's take_pending_runtime_config()
    // picks it up. The server may load a model at runtime (auto-load on first
    // request); each load rebuilds the Engine and consumes the pending slot.
    imp::set_pending_runtime_config(state.runtime_config);
    ImpConfig config = build_config(state.default_args, state.runtime_config, path, config_overrides,
                                    state.model);
    err = imp_context_create(state.model, &config, &state.ctx);
    if (err != IMP_SUCCESS) {
        std::string msg = std::string("Failed to create context: ") + imp_error_string(err);
        imp_model_free(state.model);
        state.model = nullptr;
        return msg;
    }

    // Extract model name from path. Strip trailing separators first so a
    // directory passed with a trailing slash (e.g. /models/Foo-NVFP4/) still
    // yields a non-empty id instead of "" — an empty id makes the model
    // unaddressable over the HTTP API (#756).
    std::string id_path = path;
    while (id_path.size() > 1 && (id_path.back() == '/' || id_path.back() == '\\'))
        id_path.pop_back();
    size_t slash = id_path.find_last_of('/');
    state.model_name = (slash != std::string::npos) ? id_path.substr(slash + 1) : id_path;

    // Set up tokenizer and chat template
    state.tok = state.model->model->tokenizer();
    const imp::ChatTemplate& engine_tpl = state.ctx->engine->chat_template();

    std::string chat_tpl_name = config_overrides.value("chat_template", state.default_args.chat_template);
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

    // Store max sequence length for prompt-length gating. Use the EFFECTIVE
    // context the engine actually allocated, not the model's declared max: the
    // engine VRAM-auto-sizes it and can land well below the model max (e.g.
    // ~4096 for a 14B on a tight budget). Gating on the model max let an
    // over-long prompt pass the length check and overrun the KV/position
    // buffers — a SIGSEGV instead of a clean 400.
    state.max_seq_len = imp_context_max_seq_len(state.ctx);
    if (state.max_seq_len <= 0)
        state.max_seq_len = imp_model_max_seq_len(state.model);
    if (state.max_seq_len <= 0)
        state.max_seq_len = config.max_seq_len;

    // Detect thinking model (DeepSeek R1, Qwen3 etc.) by checking for <think> token.
    // Only treat as think model if <think> is a special/added token (high vocab ID),
    // not a regular text piece. Nemotron has "<think>" at ID 12 as normal text.
    {
        int32_t ts = state.tok->find_token("<think>");
        int32_t te = state.tok->find_token("</think>");
        int vocab = state.tok->vocab_size();
        bool is_special = (ts >= 0 && ts > vocab * 99 / 100);
        state.think_start_id = is_special ? ts : -1;
        state.think_end_id = is_special ? te : -1;
        state.is_think_model = is_special;
        if (state.is_think_model) {
            printf("Reasoning model: <think>=%d, </think>=%d\n", state.think_start_id, state.think_end_id);
        }
    }

    // Detect Gemma-4 channel model: has <|channel> and <channel|> as dedicated tokens.
    // These wrap reasoning/answer headers like "<|channel>thought\n...<channel|>\n".
    // We strip the headers from the user-facing content stream.
    {
        int32_t co = state.tok->find_token("<|channel>");
        int32_t cc = state.tok->find_token("<channel|>");
        int32_t nl = state.tok->find_token("\n");
        if (co >= 0 && cc >= 0) {
            state.channel_open_id = co;
            state.channel_close_id = cc;
            state.channel_newline_id = nl;
            printf("Channel model: <|channel>=%d, <channel|>=%d, \\n=%d\n", co, cc, nl);
        }
    }

    if (state.have_template) {
        printf("Chat template: %s\n", imp::chat_template_family_name(state.chat_tpl.family()));
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

// Validate common sampling parameters. Returns false and sets error response if invalid.
bool validate_sampling_params(const json& body, httplib::Response& res) {
    // messages must be an array (for chat completions)
    if (body.contains("messages") && !body["messages"].is_null() && !body["messages"].is_array()) {
        res.status = 400;
        json err = {
            {"error", {{"message", "\"messages\" must be an array"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    if (body.contains("temperature")) {
        float t = body["temperature"].get<float>();
        if (t < 0.0f || t > 2.0f) {
            res.status = 400;
            json err = {{"error",
                         {{"message", "\"temperature\" must be between 0 and 2"},
                          {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
    }

    if (body.contains("top_p")) {
        float p = body["top_p"].get<float>();
        if (p < 0.0f || p > 1.0f) {
            res.status = 400;
            json err = {
                {"error",
                 {{"message", "\"top_p\" must be between 0 and 1"}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
    }

    if (body.contains("max_tokens") && !body["max_tokens"].is_null()) {
        int mt = body["max_tokens"].get<int>();
        if (mt < 1) {
            res.status = 400;
            json err = {
                {"error",
                 {{"message", "\"max_tokens\" must be at least 1"}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
    }

    if (body.contains("n")) {
        int n = body["n"].get<int>();
        if (n < 1 || n > 4) {
            res.status = 400;
            json err = {{"error",
                         {{"message", "\"n\" must be between 1 and 4."}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
    }

    return true;
}

// Set true on the calling thread when handle_messages is delegating to
// handle_chat_completions via a shim — suppresses inner request-log entries
// so the Anthropic call only logs once at the outer handler.
thread_local bool g_in_anthropic_shim = false;

// Write one JSONL line capturing this request: timing, endpoint, raw client
// body, token counts, finish reason, and (for non-streaming) the response.
// Streaming responses pass an empty `response_body` since per-chunk text is
// not accumulated.
static void log_request_jsonl(ServerState& state, bool skip,
                              const std::chrono::system_clock::time_point& t_start,
                              const std::string& req_id, const std::string& endpoint,
                              const std::string& client_ip, const std::string& raw_body,
                              double latency_ms, int prompt_tokens, int completion_tokens,
                              const char* finish_reason, const json& response_body) {
    if (skip || !state.request_logger.enabled)
        return;
    json record;
    record["ts_ms"] =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_start.time_since_epoch()).count();
    record["req_id"] = req_id;
    record["endpoint"] = endpoint;
    record["client_ip"] = client_ip;
    record["latency_ms"] = latency_ms;
    record["prompt_tokens"] = prompt_tokens;
    record["completion_tokens"] = completion_tokens;
    record["finish_reason"] = finish_reason ? finish_reason : "";
    try {
        record["request"] = json::parse(raw_body);
    } catch (...) {
        record["request"] = raw_body;
    }
    record["response"] = response_body;
    state.request_logger.log(record);
}

// Parses request body, validates params, builds chat_msgs from messages array.
// Populates ctx.params, ctx.log_*, ctx.req_id, ctx.snap.tpl_family (early best-
// effort snapshot used to format tool-role messages in the conversion loop).
// On parse/validation failure: sets res with 400 + error JSON and returns false.
// On success: returns true; caller proceeds to state snapshot + tokenize.
static bool parse_chat_request_params(
    const httplib::Request& req,
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx)
{
    // Capture inputs for opt-in JSONL request logging. Only used when
    // state.request_logger.enabled and the call is not an inner shim.
    ctx.t_log_start = std::chrono::system_clock::now();
    ctx.log_endpoint = req.path;
    ctx.log_client_ip = req.get_header_value("X-Forwarded-For");
    if (ctx.log_client_ip.empty())
        ctx.log_client_ip = req.remote_addr;
    ctx.log_raw_body = req.body;
    ctx.log_skip = g_in_anthropic_shim;

    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return false;
    }

    // Validate sampling parameters
    if (!validate_sampling_params(body, res))
        return false;

    // Extract parameters
    auto messages = body.value("messages", json::array());
    if (messages.empty()) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "messages array is required and must not be empty"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }
    // Bound the conversation length: each message is tokenized + template-expanded
    // on the host, so an unbounded array is a CPU/memory DoS within the body cap.
    constexpr size_t kMaxMessages = 10000;
    if (messages.size() > kMaxMessages) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "messages array exceeds maximum of 10000 entries"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    ctx.params.temperature = body.value("temperature", 0.7f);
    ctx.params.top_p_explicit = body.contains("top_p");
    ctx.params.top_k_explicit = body.contains("top_k");
    ctx.params.rep_pen_explicit = body.contains("repetition_penalty");
    // 1.05 default is mild — breaks pathological repetition loops on
    // verbose-think models (Qwen3.6-NVFP4 falling into "Wie wär es mit
    // diesem hier?" 40-iteration spirals on multi-turn sensitive prompts)
    // without disrupting structurally-repetitive valid output (JSON keys,
    // markdown lists, code idioms). Callers that need deterministic
    // sampling (validation harness, perf tests) can pass 1.0 explicitly.
    ctx.params.top_p = body.value("top_p", 0.95f);
    ctx.params.top_k = body.value("top_k", 40);
    ctx.params.max_tokens = body.value("max_tokens", state.default_max_tokens);
    ctx.params.seed = body.value("seed", -1);
    ctx.params.stream = body.value("stream", false);
    ctx.params.n_completions = body.value("n", 1);
    if (ctx.params.n_completions < 1)
        ctx.params.n_completions = 1;

    // Streaming with n > 1 is not supported
    if (ctx.params.stream && ctx.params.n_completions > 1) {
        res.status = 400;
        json err = {
            {"error",
             {{"message", "streaming with n > 1 is not supported"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    ctx.params.min_p = body.value("min_p", 0.0f);
    ctx.params.typical_p = body.value("typical_p", 1.0f);
    ctx.params.repetition_penalty = body.value("repetition_penalty", 1.05f);
    ctx.params.frequency_penalty = body.value("frequency_penalty", 0.0f);
    ctx.params.presence_penalty = body.value("presence_penalty", 0.0f);
    ctx.params.repeat_last_n = body.value("repeat_last_n", 0);
    ctx.params.dry_multiplier = body.value("dry_multiplier", 0.0f);
    ctx.params.dry_base = body.value("dry_base", 1.75f);
    ctx.params.dry_allowed_length = body.value("dry_allowed_length", 2);
    ctx.params.dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    ctx.params.mirostat = body.value("mirostat", 0);
    ctx.params.mirostat_tau = body.value("mirostat_tau", 5.0f);
    ctx.params.mirostat_eta = body.value("mirostat_eta", 0.1f);
    ctx.params.think_budget = body.value("think_budget", state.default_think_budget);

    // Parse stop sequences (string or array of up to 4 strings)
    if (body.contains("stop") && !body["stop"].is_null()) {
        if (body["stop"].is_string()) {
            ctx.params.stop_sequences.push_back(body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                if (s.is_string()) {
                    ctx.params.stop_sequences.push_back(s.get<std::string>());
                    if (ctx.params.stop_sequences.size() >= 4)
                        break;
                }
            }
        }
    }
    ctx.params.max_stop_len = 0;
    for (const auto& s : ctx.params.stop_sequences)
        ctx.params.max_stop_len = std::max(ctx.params.max_stop_len, s.size());

    // Parse logprobs parameters
    ctx.params.req_logprobs = body.value("logprobs", false);
    ctx.params.top_logprobs = body.value("top_logprobs", 0);
    if (ctx.params.top_logprobs < 0)
        ctx.params.top_logprobs = 0;
    if (ctx.params.top_logprobs > 20)
        ctx.params.top_logprobs = 20;

    // Parse response_format for JSON mode / JSON Schema
    if (body.contains("response_format") && body["response_format"].is_object()) {
        std::string fmt_type = body["response_format"].value("type", "text");
        if (fmt_type == "json_object") {
            ctx.params.json_mode = true;
        } else if (fmt_type == "json_schema") {
            ctx.params.json_mode = true;
            if (body["response_format"].contains("json_schema") &&
                body["response_format"]["json_schema"].is_object()) {
                auto& js = body["response_format"]["json_schema"];
                if (js.contains("schema") && js["schema"].is_object()) {
                    const auto& sch = js["schema"];
                    // Free-form object schema ({"type":"object"} without
                    // properties/enum) carries no structure the schema
                    // constrainer could enforce — its key phase would reject
                    // every token. Semantically this is json_object: leave
                    // json_schema_str empty so the whole request (scheduler
                    // included) takes the any-JSON constrainer path.
                    const bool free_form = sch.value("type", "") == "object" &&
                                           (!sch.contains("properties") ||
                                            sch["properties"].empty()) &&
                                           !sch.contains("enum");
                    if (!free_form) {
                        ctx.params.json_schema_str = dump_safe(sch);
                    }
                }
            }
        }
    }

    // Parse logit_bias: map of token_id (string) -> bias (float)
    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
        for (auto& [key, val] : body["logit_bias"].items()) {
            try {
                int32_t token_id = std::stoi(key);
                float bias = val.get<float>();
                ctx.params.logit_bias.emplace_back(token_id, bias);
            } catch (...) {
                // Skip invalid entries
            }
        }
    }

    // Parse stream_options for include_usage
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        ctx.params.include_usage = body["stream_options"].value("include_usage", false);
    }

    // Prompt KV pinning: Anthropic cache_control (mapped to "cache_prompt"
    // by anthropic_to_openai_body) or a direct llama.cpp-style field.
    ctx.params.cache_prompt = body.value("cache_prompt", false);

    // Parse tool calling parameters
    ctx.params.tools = body.value("tools", json::array());
    ctx.params.tool_choice = body.value("tool_choice", json("auto"));
    ctx.params.has_tools = !ctx.params.tools.empty() &&
                           !(ctx.params.tool_choice.is_string() &&
                             ctx.params.tool_choice.get<std::string>() == "none");

    // tools + response_format=json_schema/json_object: the engine-side gate
    // stays "no-mask" through tool-call bodies (see ConstraintManager::prepare
    // and PreambleGate::configure_with_tools), so we keep both signals set
    // and the gate decides at runtime which path the model takes. Tool-call
    // dialect comes from tpl_family, captured below into the request.

    // Snapshot template family (may be re-snapshotted under lock in the orchestrator)
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        ctx.snap.tpl_family = state.have_template ? state.chat_tpl.family() : imp::ChatTemplateFamily::CHATML;
    }

    // Convert JSON messages to ChatMessage vector, extracting image data if present
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");

        if (role == "tool") {
            // Tool response message — format for the model
            std::string content = format_tool_response(ctx.snap.tpl_family, msg);
            // Gemma's chat-template skips standalone role=tool messages and
            // expects tool_response markers to be glued onto the assistant
            // message that produced the call. Append to previous assistant
            // entry instead of pushing a fresh ChatMessage; ChatML/Llama3
            // templates render standalone tool messages so keep the push.
            if (ctx.snap.tpl_family == imp::ChatTemplateFamily::GEMMA && !ctx.params.chat_msgs.empty() &&
                ctx.params.chat_msgs.back().role == "assistant") {
                ctx.params.chat_msgs.back().content += content;
            } else {
                ctx.params.chat_msgs.push_back({"tool", content});
            }
        } else if (role == "assistant" && msg.contains("tool_calls")) {
            // Assistant message with tool_calls — reconstruct model output format
            std::string content_str;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content_str = msg["content"].get<std::string>();
            }
            std::string reconstructed = reconstruct_tool_call_output(ctx.snap.tpl_family, msg["tool_calls"],
                                                                     content_str);
            ctx.params.chat_msgs.push_back({"assistant", reconstructed});
        } else if (msg.contains("content") && msg["content"].is_array()) {
            // OpenAI multimodal format: content is array of parts
            std::string text_parts;
            for (const auto& part : msg["content"]) {
                std::string type = part.value("type", "");
                if (type == "text") {
                    if (!text_parts.empty())
                        text_parts += "\n";
                    text_parts += part.value("text", "");
                } else if (type == "image_url" && part.contains("image_url")) {
                    std::string url = part["image_url"].value("url", "");
                    if (url.rfind("data:", 0) == 0) {
                        // Data URI: data:image/...;base64,...
                        auto comma = url.find(',');
                        if (comma != std::string::npos) {
                            ctx.params.image_data = base64_decode(url.substr(comma + 1));
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
                                ctx.params.image_data.assign(img_res->body.begin(), img_res->body.end());
                            }
#endif
                        } else {
                            httplib::Client cli(host);
                            cli.set_follow_location(true);
                            cli.set_connection_timeout(10);
                            auto img_res = cli.Get(path_str);
                            if (img_res && img_res->status == 200) {
                                ctx.params.image_data.assign(img_res->body.begin(), img_res->body.end());
                            }
                        }
                    }
                }
            }
            ctx.params.chat_msgs.push_back({role, text_parts});
        } else {
            std::string content;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content = msg["content"].get<std::string>();
            }
            ctx.params.chat_msgs.push_back({role, content});
        }
    }

    // Log request received (structured)
    ctx.req_id = make_completion_id(state);
    fprintf(stderr, "[%s] chat/completions: prompt_msgs=%zu stream=%s max_tokens=%d temp=%.2f\n",
            ctx.req_id.c_str(), messages.size(), ctx.params.stream ? "true" : "false",
            ctx.params.max_tokens, ctx.params.temperature);

    // Validate model field (required per OpenAI spec)
    ctx.params.requested_model = body.value("model", "");
    if (ctx.params.requested_model.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"model\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    // Parse enable_thinking (only meaningful for think models; checked in orchestrator)
    ctx.params.enable_thinking_requested = body.value("enable_thinking", false);
    ctx.params.enable_thinking_set = body.contains("enable_thinking") && body["enable_thinking"].is_boolean();

    // Per-request LoRA adapter selection ("lora": "<name>"; absent/"" = base).
    ctx.params.lora_name = body.value("lora", std::string());

    return true;
}

// Acquires state.mtx lock, snapshots engine state into ctx.snap, sets up
// tool defs / vision lock / thinking detection, tokenizes the prompt with
// the chat template, validates prompt length, clamps max_tokens to remaining
// context, and starts timing. Returns true if OK; sets res with 400/503 and
// returns false on failure (model not loaded, prompt too long, vision lock
// timeout, image processing failure).
static bool snapshot_state_and_tokenize_(
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx)
{
    // Snapshot all state fields needed for request processing under lock.
    // This protects against concurrent model load/unload invalidating pointers.
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (!ensure_model_loaded(state, ctx.params.requested_model, res))
            return false;
        ctx.snap.tok = state.tok;
        ctx.snap.chat_tpl = state.chat_tpl;
        ctx.snap.have_template = state.have_template;
        ctx.snap.model_name = state.model_name;
        ctx.snap.is_think_model = state.is_think_model;
        ctx.snap.think_start_id = state.think_start_id;
        ctx.snap.think_end_id = state.think_end_id;
        ctx.snap.channel_open_id = state.channel_open_id;
        ctx.snap.channel_close_id = state.channel_close_id;
        ctx.snap.channel_newline_id = state.channel_newline_id;
        ctx.snap.max_seq_len = state.max_seq_len;
        ctx.snap.tpl_family = ctx.snap.have_template ? ctx.snap.chat_tpl.family() : imp::ChatTemplateFamily::CHATML;
        if (ctx.snap.have_template)
            ctx.snap.stop_token_ids = ctx.snap.chat_tpl.stop_token_ids();
        // Provisionally add <think> as a stop token. Removed below if the
        // request enables thinking. Without this, think-trained models at high
        // temp can hallucinate phantom turns ("Human\n<think>...").
        if (state.think_start_id >= 0) {
            ctx.snap.stop_token_ids.push_back(state.think_start_id);
        }
        ctx.snap.has_vision_request = !ctx.params.image_data.empty() && state.ctx && state.ctx->engine->has_vision();
    }

    // Channel models (Gemma-4) are more susceptible to sampling-driven
    // degeneration on casual prompts than DeepSeek-style reasoning models.
    // If the caller didn't specify a sampler parameter, tighten the default
    // to suppress the tail of the distribution. Qwen3 / DeepSeek / non-channel
    // models retain the 0.95 / 40 / 1.0 defaults.
    if (ctx.snap.channel_open_id >= 0) {
        if (!ctx.params.top_p_explicit)
            ctx.params.top_p = 0.9f;
        if (!ctx.params.top_k_explicit)
            ctx.params.top_k = 20;
        if (!ctx.params.rep_pen_explicit)
            ctx.params.repetition_penalty = 1.05f;
    }

    // Build tool definitions for Jinja2-native tool calling
    if (ctx.params.has_tools && ctx.snap.have_template && ctx.snap.chat_tpl.supports_tools()) {
        for (const auto& t : ctx.params.tools) {
            if (t.contains("function") && t["function"].is_object()) {
                imp::ToolFunction tf;
                tf.name = t["function"].value("name", "");
                tf.description = t["function"].value("description", "");
                if (t["function"].contains("parameters")) {
                    tf.parameters_json = dump_safe(t["function"]["parameters"]);
                }
                ctx.snap.tool_defs.push_back(std::move(tf));
            }
        }
    }
    // tools_via_jinja tracks whether we'll attempt the Jinja2 tools path
    ctx.snap.tools_via_jinja = !ctx.snap.tool_defs.empty();

    // Handle vision: requires exclusive lock since it modifies engine state
    if (ctx.snap.has_vision_request) {
        std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
        if (!lock.owns_lock()) {
            res.status = 503;
            json err = {{"error", {{"message", "Server is busy. Please retry."}, {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return false;
        }
        // Stop batching engine for exclusive vision access
        if (state.batching)
            state.batching->stop();

        state.ctx->engine->clear_image();
        if (!state.ctx->engine->set_image_from_memory(ctx.params.image_data.data(), ctx.params.image_data.size())) {
            if (state.batching)
                state.batching->start(state.ctx);
            res.status = 400;
            json error = {
                {"error", {{"message", "Failed to process image"}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return false;
        }
    }

    // Thinking mode default: ON for think models in plain chat. These models
    // are trained with the <think> prefix; serving them without it produces
    // bare reasoning that cannot be separated and leaks into user-visible
    // content ("Okay, let's see. The user is asking..." as the answer — the
    // recurring think-leak bug class). Exceptions, where entering reasoning
    // mode breaks the requested output format: structured output (json_mode)
    // and tool calls keep the old default OFF. An explicit "enable_thinking"
    // in the request always wins in both directions.
    // Template evidence guard: vocab-level <think> specials alone are not
    // proof of a think-trained model — Qwen3-*-Instruct-2507 ships the Qwen3
    // vocab (incl. <think>) but never opens a think block; defaulting it to
    // thinking traps the entire answer in reasoning_content (content "").
    // Default ON only when the chat template itself references thinking.
    // Models without a Jinja template keep the previous default (no evidence
    // either way); an explicit "enable_thinking" still wins in both cases.
    // Only a present-but-silent Jinja template counts as evidence AGAINST
    // thinking; hardcoded families / templateless runs keep the old default.
    const bool template_think_evidence = !ctx.snap.have_template ||
                                         !ctx.snap.chat_tpl.has_jinja() ||
                                         ctx.snap.chat_tpl.mentions_thinking();
    const bool thinking_default = ctx.snap.is_think_model && template_think_evidence &&
                                  !ctx.params.json_mode && !ctx.params.has_tools;
    const bool want_thinking = ctx.params.enable_thinking_set
                                   ? ctx.params.enable_thinking_requested
                                   : thinking_default;
    // think_budget is the fraction of max_tokens reserved for reasoning;
    // think_budget <= 0 means "no reasoning headroom" → disable thinking entirely
    // (documented "0 = disabled"). Folding it into enable_thinking keeps the two
    // flags consistent: without this, budget=0 left thinking ON yet never armed
    // the force-close, so the model reasoned to max_tokens and returned empty
    // content (#752). The Anthropic "disabled" path already zeroes the budget.
    const bool budget_disables_thinking = ctx.params.think_budget <= 0.0f;
    ctx.snap.enable_thinking = ctx.snap.is_think_model && ctx.snap.think_start_id >= 0 &&
                               want_thinking && !budget_disables_thinking;
    ctx.snap.suppress_thinking =
        ctx.snap.is_think_model && !ctx.snap.enable_thinking && budget_disables_thinking;

    // If thinking IS enabled, remove the provisional <think> stop token.
    if (ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0) {
        auto& ids = ctx.snap.stop_token_ids;
        ids.erase(std::remove(ids.begin(), ids.end(), ctx.snap.think_start_id), ids.end());
    }

    // Guard against hallucinated turn boundaries ("Human\n") that thinking
    // models emit at high temperature. Only inject if the caller didn't
    // already provide stop sequences (respect user intent).
    if (ctx.snap.is_think_model && ctx.params.stop_sequences.empty()) {
        ctx.params.stop_sequences.push_back("\nHuman");
    }

    // Tokenize with chat template (with image tokens if vision is active)
    if (ctx.snap.have_template && ctx.snap.has_vision_request) {
        ctx.snap.tokens = ctx.snap.chat_tpl.apply_with_image(*ctx.snap.tok, ctx.params.chat_msgs, 256, ctx.snap.suppress_thinking);
    } else if (ctx.snap.have_template && ctx.snap.tools_via_jinja) {
        std::string tc_str = ctx.params.tool_choice.is_string() ? ctx.params.tool_choice.get<std::string>() : "auto";
        ctx.snap.tokens = ctx.snap.chat_tpl.apply_with_tools(*ctx.snap.tok, ctx.params.chat_msgs, ctx.snap.tool_defs, tc_str, ctx.snap.suppress_thinking);
        // If Jinja2 tools render failed, fall back to text-based tool prompt injection
        if (ctx.snap.tokens.empty()) {
            IMP_LOG_INFO("Jinja2 tools path failed, falling back to text-based tool prompt");
            std::string tool_prompt = build_tool_prompt(ctx.snap.tpl_family, ctx.params.tools, ctx.params.tool_choice);
            if (!tool_prompt.empty()) {
                bool found_system = false;
                for (auto& m : ctx.params.chat_msgs) {
                    if (m.role == "system") {
                        m.content += tool_prompt;
                        found_system = true;
                        break;
                    }
                }
                if (!found_system) {
                    std::string sys = ctx.snap.chat_tpl.default_system_message();
                    if (sys.empty())
                        sys = "You are a helpful assistant.";
                    sys += tool_prompt;
                    ctx.params.chat_msgs.insert(ctx.params.chat_msgs.begin(), {"system", sys});
                }
            }
            ctx.snap.tokens = ctx.snap.chat_tpl.apply(*ctx.snap.tok, ctx.params.chat_msgs, ctx.snap.suppress_thinking);
        }
    } else if (ctx.snap.have_template) {
        // No tools, or no Jinja2 support — inject text-based tool prompt if tools present
        if (ctx.params.has_tools) {
            std::string tool_prompt = build_tool_prompt(ctx.snap.tpl_family, ctx.params.tools, ctx.params.tool_choice);
            if (!tool_prompt.empty()) {
                bool found_system = false;
                for (auto& m : ctx.params.chat_msgs) {
                    if (m.role == "system") {
                        m.content += tool_prompt;
                        found_system = true;
                        break;
                    }
                }
                if (!found_system) {
                    std::string sys = ctx.snap.chat_tpl.default_system_message();
                    if (sys.empty())
                        sys = "You are a helpful assistant.";
                    sys += tool_prompt;
                    ctx.params.chat_msgs.insert(ctx.params.chat_msgs.begin(), {"system", sys});
                }
            }
        }
        ctx.snap.tokens = ctx.snap.chat_tpl.apply(*ctx.snap.tok, ctx.params.chat_msgs, ctx.snap.suppress_thinking);
    } else {
        // Concatenate all message content as raw text
        std::string raw;
        for (const auto& m : ctx.params.chat_msgs)
            raw += m.content + "\n";
        ctx.snap.tokens = ctx.snap.tok->encode(raw);
    }

    // Detect chat-template-injected <think> prefix (Qwen3 / Qwen3.5 / Qwen3.6
    // / DeepSeek-R1 add `<think>\n` via add_generation_prompt by default). When
    // present, the model output starts mid-thinking with no opener — only a
    // closing `</think>` mid-stream. Matches vLLM's qwen3 reasoning_parser
    // auto-detection (see vllm/reasoning/qwen3_reasoning_parser.py docstring).
    // Treating these models as thinking-enabled lets the SSE stream emit
    // `reasoning_content` chunks until `</think>` is seen, then `content`.
    //
    // Detection is done over decoded text (not token-ID equality) because
    // Qwen3.6 ships `<think>`/`</think>` as `added_tokens` with `special=False`,
    // so the BPE tokenizer breaks them into 3 pieces (`<`, `think`, `>`)
    // rather than the single special-token id. vLLM's parser sidesteps this
    // by promoting them at AutoTokenizer load; imp's tokenizer doesn't, so
    // we match on the rendered string instead.
    auto prompt_tail_contains = [&](const char* needle, int max_tail_tokens) -> bool {
        int n = static_cast<int>(ctx.snap.tokens.size());
        int start = std::max(0, n - max_tail_tokens);
        std::string tail_text;
        for (int i = start; i < n; ++i) {
            tail_text += ctx.snap.tok->decode_token(ctx.snap.tokens[i]);
        }
        return tail_text.find(needle) != std::string::npos;
    };
    // No special-token requirement here: Nemotron-style models think at TEXT
    // level ("<think>" renders as plain text pieces, "</think>" closes it) —
    // when their chat template injects the prefix, the output is reasoning
    // from token 0 and must flow into reasoning_content, not content.
    //
    // Only an OPEN think prefix counts: when thinking is suppressed, Qwen3's
    // template injects a *closed* empty block `<think>\n\n</think>\n\n` (so the
    // model answers directly). That tail contains "<think>" too — re-enabling on
    // it would defeat suppression entirely (the model thinks despite the closed
    // block). Require "<think>" present AND no matching "</think>" in the tail.
    // Window 16 (not 8) so both tags of the adjacent closed block fall inside
    // the same tail — otherwise "<think>" could be in-window while "</think>"
    // just falls off, mis-reading a closed block as an open prefix.
    if (!ctx.snap.enable_thinking) {
        if (prompt_tail_contains("<think>", 16) && !prompt_tail_contains("</think>", 16)) {
            ctx.snap.enable_thinking = true;
        }
    }

    // Append <think>\n to trigger reasoning mode (matches llama.cpp behavior).
    // Without this prefix, think-trained models produce degenerate output.
    // Skip if the chat template already added it (Qwen3.x default path).
    if (ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0) {
        if (!prompt_tail_contains("<think>", 8)) {
            ctx.snap.tokens.push_back(ctx.snap.think_start_id);
            // Append newline after <think> — the model expects "\n" before reasoning
            auto nl_ids = ctx.snap.tok->encode("\n");
            ctx.snap.tokens.insert(ctx.snap.tokens.end(), nl_ids.begin(), nl_ids.end());
        }
    }

    ctx.snap.n_prompt_tokens = static_cast<int>(ctx.snap.tokens.size());

    // Server-side input-token limit (--max-input-tokens). Reject before
    // prefill so an oversized prompt never reaches the engine.
    if (state.max_input_tokens > 0 && ctx.snap.n_prompt_tokens > state.max_input_tokens) {
        if (ctx.snap.has_vision_request) {
            std::lock_guard<std::timed_mutex> lock(state.mtx);
            if (state.batching)
                state.batching->start(state.ctx);
        }
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds max input tokens (" +
                                        std::to_string(ctx.snap.n_prompt_tokens) + " > " +
                                        std::to_string(state.max_input_tokens) + ")"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return false;
    }

    // Validate prompt length against context window
    if (ctx.snap.n_prompt_tokens >= ctx.snap.max_seq_len) {
        if (ctx.snap.has_vision_request) {
            std::lock_guard<std::timed_mutex> lock(state.mtx);
            if (state.batching)
                state.batching->start(state.ctx);
        }
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds context window (" + std::to_string(ctx.snap.n_prompt_tokens) +
                                        " tokens >= " + std::to_string(ctx.snap.max_seq_len) + " max)"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return false;
    }

    // Per-request LoRA selection (#522): swap the engine-global adapter
    // before generation. Single-user semantics — the swap re-captures
    // decode graphs on the next step, so back-to-back requests with
    // different adapters work; concurrent mixed-adapter batches are out of
    // scope (imp is batch=1-first by mission).
    {
        int32_t want = 0;
        if (!ctx.params.lora_name.empty()) {
            auto it = state.lora_ids.find(ctx.params.lora_name);
            if (it == state.lora_ids.end()) {
                res.status = 400;
                json error = {{"error",
                               {{"message", "Unknown LoRA adapter '" + ctx.params.lora_name +
                                                "' (load at startup via --lora NAME=PATH)"},
                                {"type", "invalid_request_error"}}}};
                res.set_content(dump_safe(error), "application/json");
                return false;
            }
            want = it->second;
        }
        imp_lora_set(state.ctx, want);
    }

    // Clamp max_tokens to remaining context window
    int remaining = ctx.snap.max_seq_len - ctx.snap.n_prompt_tokens;
    if (ctx.params.max_tokens > remaining)
        ctx.params.max_tokens = remaining;

    // Start timing
    ctx.t_start = std::chrono::high_resolution_clock::now();

    return true;
}

// Vision-request blocking decode path: prefill via C API, sample tokens in a
// blocking loop until EOS/stop/max_tokens, build a non-streaming JSON response
// (vision doesn't support SSE — state is per-engine, not per-request).
// Caller must hold no lock on entry. Returns after sending the response.
static void handle_vision_chat_blocking_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx,
                                         const std::shared_ptr<imp::Request>& imp_req) {
    // Vision path: use blocking C API (batching engine is stopped)
    ImpError err = imp_context_reset(state.ctx);
    if (err != IMP_SUCCESS) {
        state.ctx->engine->clear_image();
        state.batching->start(state.ctx);
        res.status = 500;
        json error = {{"error",
                       {{"message", std::string("Context reset failed: ") + imp_error_string(err)},
                        {"type", "server_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    // Build params now so prefill's first-token sample also honours them
    ImpGenerateParams prefill_params = imp_generate_params_default();
    prefill_params.temperature = ctx.params.temperature;
    prefill_params.top_p = ctx.params.top_p;
    prefill_params.top_k = ctx.params.top_k;
    prefill_params.seed = ctx.params.seed;
    err = imp_prefill_with_params(state.ctx, imp_req->input_tokens.data(), ctx.snap.n_prompt_tokens,
                                  &prefill_params);
    if (err != IMP_SUCCESS) {
        state.ctx->engine->clear_image();
        state.batching->start(state.ctx);
        res.status = 500;
        json error = {{"error",
                       {{"message", std::string("Prefill failed: ") + imp_error_string(err)},
                        {"type", "server_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    // After prefill, clear vision and restart batching engine
    // The rest of generation will use the old blocking decode path
    // (via imp_decode_step, which calls engine->step() directly)
    // This is safe because batching engine is stopped.

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = ctx.params.temperature;
    params.top_p = ctx.params.top_p;
    params.top_k = ctx.params.top_k;
    params.max_tokens = ctx.params.max_tokens;
    params.seed = ctx.params.seed;
    params.min_p = ctx.params.min_p;
    params.typical_p = ctx.params.typical_p;
    params.repetition_penalty = ctx.params.repetition_penalty;
    params.frequency_penalty = ctx.params.frequency_penalty;
    params.presence_penalty = ctx.params.presence_penalty;
    params.repeat_last_n = ctx.params.repeat_last_n;
    params.dry_multiplier = ctx.params.dry_multiplier;
    params.dry_base = ctx.params.dry_base;
    params.dry_allowed_length = ctx.params.dry_allowed_length;
    params.dry_penalty_last_n = ctx.params.dry_penalty_last_n;
    params.mirostat = ctx.params.mirostat;
    params.mirostat_tau = ctx.params.mirostat_tau;
    params.mirostat_eta = ctx.params.mirostat_eta;
    params.logprobs = ctx.params.req_logprobs ? 1 : 0;
    params.top_logprobs = ctx.params.top_logprobs;
    params.json_mode = ctx.params.json_mode ? 1 : 0;

    // Blocking decode loop for vision requests
    std::vector<int32_t> output_ids;
    int32_t prefill_token = -1;
    if (state.ctx->active_request && !state.ctx->active_request->output_tokens.empty()) {
        prefill_token = state.ctx->active_request->output_tokens.back();
    }

    for (int step = -1; step < ctx.params.max_tokens; step++) {
        int32_t token = 0;
        if (step == -1) {
            if (prefill_token < 0)
                continue;
            token = prefill_token;
        } else {
            err = imp_decode_step(state.ctx, &params, &token);
            if (err != IMP_SUCCESS)
                break;
        }
        if (token == ctx.snap.tok->eos_id())
            break;
        if (ctx.snap.have_template) {
            bool is_stop = false;
            for (int32_t stop_id : ctx.snap.stop_token_ids) {
                if (token == stop_id) {
                    is_stop = true;
                    break;
                }
            }
            if (is_stop)
                break;
        }
        output_ids.push_back(token);
    }

    state.ctx->engine->clear_image();
    state.batching->start(state.ctx);

    // Build simple non-streaming response for vision
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - ctx.t_start).count();
    int n_output_tokens = static_cast<int>(output_ids.size());
    std::string content = ctx.snap.tok->decode(output_ids);

    fprintf(stderr, "[%s] vision: %d prompt + %d completion tokens, %.1f ms\n", ctx.req_id.c_str(),
            ctx.snap.n_prompt_tokens, n_output_tokens, ms);
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += ctx.snap.n_prompt_tokens;
    state.metrics.tokens_completion_total += n_output_tokens;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

    json response_for_log = {{"id", ctx.req_id},
                             {"object", "chat.completion"},
                             {"model", ctx.snap.model_name},
                             {"choices", json::array({{{"index", 0},
                                                        {"message", {{"role", "assistant"},
                                                                     {"content", content}}},
                                                        {"finish_reason", "stop"}}})}};
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, ctx.req_id, ctx.log_endpoint,
                      ctx.log_client_ip, ctx.log_raw_body, ms,
                      ctx.snap.n_prompt_tokens, n_output_tokens, "stop", response_for_log);

    json response = {{"id", ctx.req_id},
                     {"object", "chat.completion"},
                     {"created", unix_timestamp()},
                     {"model", ctx.snap.model_name},
                     {"choices", json::array({{{"index", 0},
                                               {"message", {{"role", "assistant"}, {"content", content}}},
                                               {"finish_reason", "stop"}}})},
                     {"usage",
                      {{"prompt_tokens", ctx.snap.n_prompt_tokens},
                       {"completion_tokens", n_output_tokens},
                       {"total_tokens", ctx.snap.n_prompt_tokens + n_output_tokens}}}};
    res.set_content(dump_safe(response), "application/json");
}

// Non-streaming chat completion: run n_completions independent generations
// sequentially via the batching engine, build the choices array with
// reasoning_content / tool_calls / logprobs as appropriate, send a single
// JSON response. Caller has already submitted server_req via state.batching.
static void nonstream_chat_response_(
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx,
    std::shared_ptr<imp::Request>& imp_req,
    std::shared_ptr<ServerRequest>& server_req,
    const std::vector<int32_t>& saved_tokens,
    const std::string& comp_id,
    int64_t created)
{
    // Helper to create an imp::Request with the given completion index
    auto make_imp_request = [&](int completion_idx) {
        auto req = std::make_shared<imp::Request>();
        req->input_tokens = saved_tokens;
        req->max_tokens = ctx.params.max_tokens;
        req->temperature = ctx.params.temperature;
        req->top_p = ctx.params.top_p;
        req->top_k = ctx.params.top_k;
        req->seed = (ctx.params.seed != -1) ? ctx.params.seed + completion_idx : -1;
        req->pin_kv_prefix = ctx.params.cache_prompt;
        req->min_p = ctx.params.min_p;
        req->typical_p = ctx.params.typical_p;
        req->repetition_penalty = ctx.params.repetition_penalty;
        req->frequency_penalty = ctx.params.frequency_penalty;
        req->presence_penalty = ctx.params.presence_penalty;
        req->repeat_last_n = ctx.params.repeat_last_n;
        req->dry_multiplier = ctx.params.dry_multiplier;
        req->dry_base = ctx.params.dry_base;
        req->dry_allowed_length = ctx.params.dry_allowed_length;
        req->dry_penalty_last_n = ctx.params.dry_penalty_last_n;
        req->mirostat = ctx.params.mirostat;
        req->mirostat_tau = ctx.params.mirostat_tau;
        req->mirostat_eta = ctx.params.mirostat_eta;
        req->logprobs = ctx.params.req_logprobs;
        req->top_logprobs = ctx.params.top_logprobs;
        req->json_mode = ctx.params.json_mode;
        req->json_schema = ctx.params.json_schema_str;
        req->has_tools = ctx.params.has_tools;
        req->tpl_family = ctx.snap.tpl_family;
        req->logit_bias = ctx.params.logit_bias;
        req->think_budget = ctx.params.think_budget;
        // Generation starts INSIDE the think block when the prompt carries the
        // <think> prefix (template-injected or server-appended). Without this
        // the engine's think-budget enforcement never sees an opener in the
        // output, counts zero reasoning tokens, and lets the model think until
        // max_tokens (content empty, finish=length).
        req->started_in_think = ctx.snap.enable_thinking;
        req->in_think_block = ctx.snap.enable_thinking;
        req->status = imp::RequestStatus::PENDING;
        return req;
    };

    // Non-streaming: decode all tokens, return complete response
    // For n > 1, run multiple independent generations sequentially
    json choices = json::array();
    int total_output_tokens = 0;

    for (int ci = 0; ci < ctx.params.n_completions; ci++) {
        // For subsequent completions, create a new request and submit it
        if (ci > 0) {
            imp_req = make_imp_request(ci);
            server_req = std::make_shared<ServerRequest>();
            server_req->request = imp_req;
            {
                std::lock_guard<std::timed_mutex> lock(state.mtx);
                if (!state.batching || !state.batching->is_running()) {
                    break;
                }
                state.batching->submit(server_req);
            }
        }

        auto active_req = server_req->request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;  // accumulated output for stop matching

        auto ns_request_start = std::chrono::steady_clock::now();
        for (;;) {
            // Check request timeout
            if (state.request_timeout > 0) {
                auto elapsed = std::chrono::steady_clock::now() - ns_request_start;
                if (elapsed > std::chrono::seconds(state.request_timeout)) {
                    server_req->cancel();
                    finish = "length";
                    break;
                }
            }

            // Read next token from the batching engine
            TokenEvent evt{};
            if (!server_req->pop_token(evt)) {
                continue;  // timeout — loop back to check request timeout
            }

            if (evt.token_id < 0) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }

            int32_t token = evt.token_id;

            // Silently drop structural stop tokens that slipped through.
            // The engine's think-block implicit-close passes ONE EOS-like
            // token through to recover from empty thinking; it must not
            // appear as user-visible content.
            if (!evt.is_last) {
                bool is_structural_stop = (token == ctx.snap.tok->eos_id());
                if (!is_structural_stop && ctx.snap.have_template) {
                    for (int32_t stop_id : ctx.snap.stop_token_ids) {
                        if (token == stop_id) {
                            is_structural_stop = true;
                            break;
                        }
                    }
                }
                if (is_structural_stop)
                    continue;
            }

            // Check stop conditions
            if (evt.is_last) {
                if (token == ctx.snap.tok->eos_id()) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                bool is_stop = false;
                if (ctx.snap.have_template) {
                    for (int32_t stop_id : ctx.snap.stop_token_ids) {
                        if (token == stop_id) {
                            is_stop = true;
                            break;
                        }
                    }
                }
                if (is_stop) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                finish = evt.finish_reason ? evt.finish_reason : "length";
            }

            output_ids.push_back(token);

            // Check text-level stop sequences
            if (!ctx.params.stop_sequences.empty()) {
                output_text += ctx.snap.tok->decode_token(token);
                bool stop_found = false;
                for (const auto& stop : ctx.params.stop_sequences) {
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
            if (finish)
                break;
        }

        if (!finish)
            finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        total_output_tokens += n_output_tokens;
        std::string content = !ctx.params.stop_sequences.empty() ? output_text : ctx.snap.tok->decode(output_ids);

        // Extract reasoning content (DeepSeek format) or strip think blocks.
        // enable_thinking also covers text-level thinkers (Nemotron) whose
        // template injects "<think>" as plain text — is_think_model is false
        // but the output is reasoning until the literal "</think>".
        std::string reasoning_content;
        if (ctx.snap.tpl_family == imp::ChatTemplateFamily::HARMONY) {
            // gpt-oss Harmony: split the <|channel|>analysis|final<|message|>…
            // blocks so the analysis channel becomes reasoning_content and the
            // final channel becomes the answer. Without this the raw Harmony
            // markup leaks verbatim into content (#760).
            auto segs = split_harmony_channels(content);
            content = std::move(segs.content);
            if (state.default_args.reasoning_format != "none")
                reasoning_content = std::move(segs.reasoning);
        } else if ((ctx.snap.is_think_model || ctx.snap.enable_thinking) &&
                   state.default_args.reasoning_format == "deepseek") {
            // Generation that started inside an injected <think> prefix
            // (chat-template or server-appended; see prompt_tail_contains
            // above) carries no opener in its output. If it also never
            // reached </think> — budget exhausted mid-think, or the model
            // stopped while reasoning — the WHOLE text is reasoning.
            // extract_reasoning() can't tell that from text alone and would
            // spill it into user-visible content (the streaming path gets
            // this right via its in-think state machine).
            if (ctx.snap.enable_thinking && content.find("</think>") == std::string::npos &&
                content.find("<think>") == std::string::npos) {
                reasoning_content = std::move(content);
                content.clear();
            } else {
                auto [reasoning, cleaned] = extract_reasoning(content);
                reasoning_content = reasoning;
                content = cleaned;
            }
        } else if (ctx.snap.is_think_model && state.default_args.reasoning_format != "none") {
            strip_think_block(content);
        }

        // Gemma-4 channel headers: structural "<|channel>NAME[<channel|>]…"
        // wraps both the chain-of-thought and the user-facing answer. Split
        // them so "thought" content goes to reasoning_content (OpenAI-
        // compat) and "final" content stays in content. Falls back to
        // strip-only if the request asked reasoning_format=none.
        if (ctx.snap.channel_open_id >= 0) {
            if (state.default_args.reasoning_format == "none") {
                strip_channel_headers(content);
            } else {
                auto segs = split_channel_segments(content);
                if (!segs.reasoning.empty() && reasoning_content.empty()) {
                    reasoning_content = std::move(segs.reasoning);
                }
                content = std::move(segs.content);
            }
        }

        // Build logprobs object if requested
        json logprobs_obj = nullptr;
        if (ctx.params.req_logprobs && active_req) {
            const auto& lp_data = active_req->output_logprobs;
            json content_logprobs = json::array();
            for (size_t idx = 0; idx < lp_data.size() && idx < output_ids.size(); idx++) {
                const auto& lp = lp_data[idx];
                json top_arr = json::array();
                for (const auto& t : lp.top) {
                    top_arr.push_back({{"token", safe_token_json(t.text)},
                                       {"logprob", t.logprob},
                                       {"bytes", token_bytes_json(t.text)}});
                }
                content_logprobs.push_back({{"token", safe_token_json(lp.text)},
                                            {"logprob", lp.logprob},
                                            {"bytes", token_bytes_json(lp.text)},
                                            {"top_logprobs", top_arr}});
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        // Parse tool calls from model output. Run even on finish=length:
        // the model may have emitted a complete tool_call and then kept
        // generating until the budget ran out (common before we hook the
        // family-specific close marker as a stop token). The parser is
        // tolerant of trailing garbage after the closing marker.
        std::vector<ParsedToolCall> tool_calls;
        std::string tool_validation_error;
        if (ctx.params.has_tools) {
            auto [pre_content, parsed_calls] = parse_tool_calls(ctx.snap.tpl_family, content,
                                                                state.next_tool_call_id);
            if (!parsed_calls.empty()) {
                tool_calls = std::move(parsed_calls);
                content = pre_content;
                finish = "tool_calls";
                // Validate parsed arguments against each tool's input schema.
                // A failure means the model hallucinated/garbled the call —
                // surface it rather than silently shipping bad arguments.
                for (auto& tc : tool_calls) {
                    validate_tool_call(tc, ctx.params.tools);
                    if (!tc.valid) {
                        if (!tool_validation_error.empty())
                            tool_validation_error += "; ";
                        tool_validation_error += tc.name + ": " + tc.error;
                    }
                }
            }
        }

        json msg = {{"role", "assistant"}};
        if (!tool_calls.empty()) {
            // content is null when only tool calls (no preceding text)
            msg["content"] = content.empty() ? json(nullptr) : json(content);
            json tc_array = json::array();
            for (const auto& tc : tool_calls) {
                json tc_json = {{"id", tc.id},
                                {"type", "function"},
                                {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}};
                if (!tc.valid)
                    tc_json["invalid_arguments"] = tc.error;
                tc_array.push_back(std::move(tc_json));
            }
            msg["tool_calls"] = tc_array;
        } else {
            msg["content"] = content;
        }
        if (!reasoning_content.empty()) {
            msg["reasoning_content"] = reasoning_content;
        }
        if (!tool_validation_error.empty()) {
            msg["tool_call_validation_error"] = tool_validation_error;
        }

        json choice = {{"index", ci}, {"message", msg}, {"finish_reason", finish}};
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        choices.push_back(choice);

        // Log each completion
        fprintf(stderr, "[%s] completion %d/%d: %d tokens\n", comp_id.c_str(), ci + 1,
                ctx.params.n_completions, n_output_tokens);
    }

    // Log aggregate request
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - ctx.t_start).count();
    fprintf(stderr, "[%s] %d prompt + %d completion tokens (%d choices), %.1f ms\n", comp_id.c_str(),
            ctx.snap.n_prompt_tokens, total_output_tokens, ctx.params.n_completions, ms);
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += ctx.snap.n_prompt_tokens;
    state.metrics.tokens_completion_total += total_output_tokens;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.request_duration.observe(ms / 1000.0);

    json usage = {{"prompt_tokens", ctx.snap.n_prompt_tokens},
                  {"completion_tokens", total_output_tokens},
                  {"total_tokens", ctx.snap.n_prompt_tokens + total_output_tokens}};
    // Prefix-cache reporting (OpenAI prompt_tokens_details; the Anthropic
    // converter maps these to cache_read/cache_creation_input_tokens).
    if (imp_req && (imp_req->cached_tokens > 0 || imp_req->pin_kv_prefix)) {
        json details = {{"cached_tokens", imp_req->cached_tokens}};
        int creation = cache_creation_tokens_(imp_req, ctx.snap.n_prompt_tokens);
        if (creation > 0)
            details["cache_creation_tokens"] = creation;
        usage["prompt_tokens_details"] = std::move(details);
        state.metrics.tokens_cached_total += imp_req->cached_tokens;
    }

    json response = {{"id", comp_id},      {"object", "chat.completion"},
                     {"created", created}, {"model", ctx.snap.model_name},
                     {"choices", choices}, {"usage", usage}};

    // Pull the final finish_reason from choice 0 for log correlation;
    // multi-completion requests still record only the aggregate.
    const char* nonstream_finish = nullptr;
    if (!choices.empty() && choices[0].contains("finish_reason") &&
        choices[0]["finish_reason"].is_string()) {
        nonstream_finish = choices[0]["finish_reason"].get_ref<const std::string&>().c_str();
    }
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, comp_id, ctx.log_endpoint,
                      ctx.log_client_ip, ctx.log_raw_body,
                      ms, ctx.snap.n_prompt_tokens, total_output_tokens, nonstream_finish, response);

    res.set_content(dump_safe(response), "application/json");
}

// Set up SSE chunked content provider for streaming chat completion.
// Captures state and ctx by reference for the chunked-provider lambda. ctx
// must outlive the SSE response (httplib invokes the chunked provider after
// this function returns; ctx is a stack-local in handle_chat_completions
// which keeps the request frame alive until the response is fully sent).
static bool run_chat_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                             const std::shared_ptr<ServerRequest>& server_req);

static void stream_chat_response_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx,
                                  const std::shared_ptr<ServerRequest>& server_req) {
    // SSE streaming response
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    ctx.comp_id = ctx.req_id;
    ctx.created = unix_timestamp();

    res.set_chunked_content_provider(
        "text/event-stream",
        [stream_ctx = ctx, &state, server_req](size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
            return run_chat_stream_(sink, stream_ctx, state, server_req);
        });
}

// Streaming chat response loop body. Extracted from the
// res.set_chunked_content_provider() lambda in stream_chat_response_ so
// the 760-LOC body is no longer a god-function nested four levels deep.
// httplib calls the lambda repeatedly until it returns false; the lambda
// just dispatches to this function. ctx is captured by value into the
// lambda (so it survives stream_chat_response_'s return); state is
// captured by reference (lives in the long-lived ServerState).
static bool run_chat_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                             const std::shared_ptr<ServerRequest>& server_req) {
    // Local aliases so the body reads unchanged from its previous
    // capture-list-based form. These were 30+ individual lambda captures
    // before this refactor.
    const std::string& comp_id          = ctx.comp_id;
    int64_t            created          = ctx.created;
    int                n_prompt_tokens  = ctx.snap.n_prompt_tokens;
    auto               t_start          = ctx.t_start;
    const auto&        stop_sequences   = ctx.params.stop_sequences;
    // Derive max_stop_len from the FINAL stop list, not ctx.params.max_stop_len:
    // the snapshot phase may inject server-side stops ("\nHuman" turn guard for
    // think models) AFTER request parsing computed max_stop_len. A stale 0 made
    // the partial-match holdback `size - max_stop_len + 1` flush one byte PAST
    // pending_text's end — emitting the std::string NUL terminator into every
    // SSE content delta ("4 ") and disabling cross-token stop matching.
    size_t             max_stop_len     = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());
    int                req_logprobs     = ctx.params.req_logprobs;
    bool               include_usage    = ctx.params.include_usage;
    bool               enable_thinking  = ctx.snap.enable_thinking;
    bool               has_tools        = ctx.params.has_tools;
    auto               tpl_family       = ctx.snap.tpl_family;
    float              think_budget     = ctx.params.think_budget;
    auto               snap_tok         = ctx.snap.tok;
    bool               snap_have_template = ctx.snap.have_template;
    const std::string& snap_model_name  = ctx.snap.model_name;
    bool               snap_is_think_model      = ctx.snap.is_think_model;
    int                snap_think_start_id      = ctx.snap.think_start_id;
    int                snap_think_end_id        = ctx.snap.think_end_id;
    int                snap_channel_open_id     = ctx.snap.channel_open_id;
    int                snap_channel_close_id    = ctx.snap.channel_close_id;
    int                snap_channel_newline_id  = ctx.snap.channel_newline_id;
    const auto&        snap_stop_token_ids      = ctx.snap.stop_token_ids;
    bool               log_skip         = ctx.log_skip;
    auto               t_log_start      = ctx.t_log_start;
    const std::string& log_endpoint     = ctx.log_endpoint;
    const std::string& log_client_ip    = ctx.log_client_ip;
    const std::string& log_raw_body     = ctx.log_raw_body;

    // Active request ref for logprobs access
    auto active_req = server_req->request;

    // Pre-build SSE envelope templates for fast content/reasoning emission
    SSEChunkWriter sse_writer(comp_id, created, snap_model_name);

    // Send initial chunk with role
    json role_delta = {{"role", "assistant"}};
    std::string chunk = sse_chunk(comp_id, created, snap_model_name, role_delta, nullptr);
    sink.write(chunk.data(), chunk.size());

    int n_output_tokens = 0;
    const char* finish = nullptr;
    double ttft_ms = 0.0;  // Time to first token

    // Buffer for incomplete UTF-8 sequences across token boundaries
    std::string utf8_buf;

    // Buffered output for stop sequence matching in streaming mode.
    // We hold back text until we're sure it doesn't contain a stop match.
    std::string pending_text;
    bool text_stop_matched = false;

    // Tool call detection state machine for streaming
    enum class ToolPhase { CONTENT, TAG_SCANNING, TOOL_CALL_BODY };
    ToolPhase tool_phase = ToolPhase::CONTENT;
    std::string tool_tag_buf;    // buffer for partial tag match
    std::string tool_body_buf;   // buffer for tool call body
    std::string tool_close_tag;  // expected closing tag
    std::string tool_fn_name;    // Llama3: extracted function name from open tag
    std::vector<ParsedToolCall> stream_tool_calls;
    bool tool_calls_emitted = false;
    // The full accumulated output (only used when has_tools, for fallback)
    std::string full_output;

    // Reasoning content extraction (DeepSeek format). enable_thinking also
    // covers text-level thinkers (Nemotron: template-injected "<think>" as
    // plain text, no special token — is_think_model is false but the output
    // starts mid-reasoning and exits via the literal "</think>").
    enum class ThinkPhase { SCAN, REASONING, CONTENT };
    bool use_reasoning = (state.default_args.reasoning_format == "deepseek" &&
                          (snap_is_think_model || enable_thinking));
    ThinkPhase think_phase;
    if (enable_thinking) {
        think_phase = ThinkPhase::REASONING;  // <think> in prefill -> start reasoning
    } else if (use_reasoning && think_budget > 0.0f) {
        think_phase = ThinkPhase::SCAN;  // model decides whether to think
    } else {
        think_phase = ThinkPhase::CONTENT;  // no reasoning extraction
    }
    std::string reasoning_utf8_buf;
    std::string think_scan_buf;
    int think_scan_count = 0;
    int n_reasoning_tokens = 0;
    bool content_started = (think_phase == ThinkPhase::CONTENT);
    int think_reentries = 0;
    const int kMaxThinkReentries = 1;
    const int kThinkScanLimit = 8;

    // Gemma-4 channel filter state: when we see <|channel> or <channel|>,
    // skip tokens until the next newline (the channel header).
    bool channel_header_active = false;

    // Helper: emit reasoning_content SSE chunk
    auto emit_reasoning = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        return sse_writer.write_reasoning(text, sink);
    };

    // gpt-oss Harmony streaming filter. The model emits
    //   <|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…
    // Route analysis/commentary channels to reasoning_content and the final
    // channel to content, stripping the control markers (which arrive as atomic
    // special-token pieces) and the <|start|>role plumbing. hm_buf holds the
    // current channel's bytes so a token that splits a multibyte char is not
    // emitted mid-codepoint (#760).
    const bool harmony = (tpl_family == imp::ChatTemplateFamily::HARMONY);
    // Harmony reasoning is its own mechanism (not the deepseek <think> path), so
    // it's gated on reasoning_format alone — emit reasoning_content unless the
    // caller explicitly asked for none.
    const bool hm_reasoning_on = (state.default_args.reasoning_format != "none");
    std::string hm_channel, hm_name, hm_buf;
    bool hm_in_msg = false, hm_reading_name = false;
    auto hm_flush = [&](bool force) -> bool {
        size_t complete = force ? hm_buf.size() : utf8_complete_len(hm_buf);
        if (complete == 0)
            return true;
        std::string chunk = hm_buf.substr(0, complete);
        hm_buf.erase(0, complete);
        if (hm_channel == "analysis" || hm_channel == "commentary")
            return hm_reasoning_on ? emit_reasoning(chunk) : true;
        return sse_writer.write_content(chunk.data(), chunk.size(), sink);
    };

    // Helper: flush confirmed text up to a byte position
    auto flush_text = [&](size_t up_to) {
        up_to = std::min(up_to, pending_text.size());  // never read past the buffer
        if (up_to == 0)
            return true;
        bool ok = sse_writer.write_content(pending_text.data(), up_to, sink);
        pending_text.erase(0, up_to);
        return ok;
    };

    auto request_start = std::chrono::steady_clock::now();
    for (;;) {
        // Terminate as soon as a finish reason has been recorded. The is_last
        // token sets `finish` and then falls through to the per-token
        // post-processing below, where a think/reasoning/channel `continue`
        // can skip the trailing `if (finish) break`. Re-checking here means the
        // stream always ends (and the terminal SSE frame is emitted) even when
        // the final token is swallowed by one of those paths (#755/#757).
        if (finish)
            break;

        // Check client disconnect
        if (!sink.is_writable()) {
            server_req->cancel();
            finish = "cancelled";
            break;
        }

        // Check request timeout
        if (state.request_timeout > 0) {
            auto elapsed = std::chrono::steady_clock::now() - request_start;
            if (elapsed > std::chrono::seconds(state.request_timeout)) {
                server_req->cancel();
                finish = "length";
                break;
            }
        }

        // Read next token from the batching engine (with timeout)
        TokenEvent evt{};
        if (!server_req->pop_token(evt)) {
            continue;  // timeout — loop back to check disconnect/timeout
        }

        if (evt.token_id < 0) {
            // Finish event with no token
            finish = evt.finish_reason ? evt.finish_reason : "stop";
            break;
        }

        int32_t token = evt.token_id;

        // Silently drop structural stop tokens that slipped through.
        // The engine's think-block implicit-close (Engine::should_stop)
        // passes ONE EOS-like token through to recover from empty
        // thinking. That token must not appear as user-visible content
        // (would render as "<|im_end|>" / "<|endoftext|>" in chat).
        if (!evt.is_last) {
            bool is_structural_stop = (token == snap_tok->eos_id());
            if (!is_structural_stop && snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids) {
                    if (token == stop_id) {
                        is_structural_stop = true;
                        break;
                    }
                }
            }
            if (is_structural_stop)
                continue;
        }

        // Check stop conditions (EOS/stop tokens already detected by engine)
        if (evt.is_last) {
            // The engine marked this as the last token.
            // Don't emit EOS/stop tokens — they're structural, not content.
            if (token == snap_tok->eos_id()) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            bool is_stop = false;
            if (snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids) {
                    if (token == stop_id) {
                        is_stop = true;
                        break;
                    }
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
        if (n_output_tokens == 1) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
        }
        std::string piece = snap_tok->decode_token(token);

        // gpt-oss Harmony channel routing (analysis/commentary -> reasoning,
        // final -> content). Markers arrive as atomic special-token pieces.
        if (harmony) {
            if (piece == "<|channel|>" || piece == "<|message|>" || piece == "<|end|>" ||
                piece == "<|return|>" || piece == "<|start|>") {
                if (hm_in_msg && !hm_flush(/*force=*/true))
                    return false;
                if (piece == "<|channel|>") {
                    hm_reading_name = true;
                    hm_in_msg = false;
                    hm_name.clear();
                } else if (piece == "<|message|>") {
                    size_t s = hm_name.find_first_not_of("\n\r\t ");
                    size_t e = hm_name.find_last_not_of("\n\r\t ");
                    hm_channel = (s == std::string::npos) ? std::string() : hm_name.substr(s, e - s + 1);
                    hm_reading_name = false;
                    hm_in_msg = true;
                } else {  // <|end|> / <|return|> / <|start|>: close the block
                    hm_in_msg = false;
                    hm_reading_name = false;
                    hm_channel.clear();
                }
                continue;
            }
            if (hm_reading_name) {  // channel name between <|channel|> and <|message|>
                hm_name += piece;
                continue;
            }
            if (!hm_in_msg)  // role text / inter-block plumbing
                continue;
            hm_buf += piece;
            if (!hm_flush(/*force=*/false))
                return false;
            continue;
        }

        // Gemma-4 channel filter: strip "<|channel>NAME\n" structural
        // headers from the content stream. `<channel|>` is the
        // channel-switch marker — strip the token but do NOT enter
        // the scan-until-newline mode, because Q5_K_M sometimes
        // emits the final answer directly after it with no newline
        // (observed: "<|channel>thought\n<channel|>5 + 3 = 8").
        if (snap_channel_open_id >= 0) {
            if (channel_header_active) {
                if (token == snap_channel_newline_id ||
                    (!piece.empty() && piece.back() == '\n')) {
                    channel_header_active = false;
                }
                continue;
            }
            if (token == snap_channel_open_id) {
                channel_header_active = true;
                continue;
            }
            if (token == snap_channel_close_id) {
                // Drop just the marker; the next token is body.
                continue;
            }
        }

        // Reasoning content extraction (DeepSeek format)
        if (think_phase == ThinkPhase::SCAN) {
            if (token == snap_think_start_id) {
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
                if (!after.empty())
                    reasoning_utf8_buf += after;
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
            // No forced </think> injection — let the model decide when
            // to stop thinking (like llama.cpp).  Forcing </think> via
            // token replacement corrupts the KV cache: the model sees
            // the original token, not </think>, so it keeps reasoning
            // while imp treats subsequent tokens as content.
            if (token == snap_think_end_id) {
                if (!emit_reasoning(reasoning_utf8_buf))
                    return false;
                reasoning_utf8_buf.clear();
                think_phase = ThinkPhase::CONTENT;
                continue;
            }
            // Skip duplicate <think> tokens while already reasoning
            if (token == snap_think_start_id)
                continue;
            reasoning_utf8_buf += piece;
            // Strip <think> text that appears via multi-token encoding
            for (;;) {
                auto tp = reasoning_utf8_buf.find("<think>");
                if (tp == std::string::npos)
                    break;
                reasoning_utf8_buf.erase(tp, 7);
            }
            auto end_pos = reasoning_utf8_buf.find("</think>");
            if (end_pos != std::string::npos) {
                std::string before = reasoning_utf8_buf.substr(0, end_pos);
                if (!emit_reasoning(before))
                    return false;
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
                // Keep a tail overlap so "</think>" spanning multiple
                // tokens can still be detected on the next iteration.
                // "</think>" is 8 bytes; we need at most 7 bytes of
                // overlap to catch any partial match at the boundary.
                constexpr size_t kOverlap = 7;
                size_t complete = utf8_complete_len(reasoning_utf8_buf);
                if (complete > kOverlap) {
                    size_t emit_end = complete - kOverlap;
                    // Walk emit_end back to a UTF-8 codepoint boundary —
                    // the 7-byte overlap is geared to literal "</think>"
                    // bytes, not codepoints, so it can land inside a
                    // multibyte char (German umlauts, CJK, emoji), which
                    // emits the lead byte alone and turns the trailing
                    // continuation byte into a U+FFFD on the next flush
                    // — visible to the user as "f��r" instead of "für".
                    while (emit_end > 0 &&
                           (static_cast<unsigned char>(reasoning_utf8_buf[emit_end]) & 0xC0) ==
                               0x80) {
                        --emit_end;
                    }
                    if (emit_end > 0) {
                        std::string to_emit = reasoning_utf8_buf.substr(0, emit_end);
                        reasoning_utf8_buf = reasoning_utf8_buf.substr(emit_end);
                        if (!emit_reasoning(to_emit))
                            return false;
                    }
                }
                continue;
            }
        }

        // Strip leading whitespace after </think> → CONTENT transition
        // (matches extract_reasoning behavior in non-streaming path)
        if (!content_started && think_phase == ThinkPhase::CONTENT) {
            auto ns = piece.find_first_not_of("\n\r\t ");
            if (ns == std::string::npos)
                continue;  // all whitespace
            piece = piece.substr(ns);
            content_started = true;
        }

        // CONTENT phase: handle stray think tokens from confused models
        if (use_reasoning) {
            if (token == snap_think_start_id) {
                if (think_reentries < kMaxThinkReentries) {
                    think_phase = ThinkPhase::REASONING;
                    n_reasoning_tokens++;
                    think_reentries++;
                }
                continue;  // always strip <think> from content
            }
            if (token == snap_think_end_id) {
                n_reasoning_tokens++;
                continue;
            }
            // Strip text-level think tags from content piece
            for (;;) {
                auto p = piece.find("<think>");
                if (p != std::string::npos) {
                    piece.erase(p, 7);
                    continue;
                }
                p = piece.find("</think>");
                if (p != std::string::npos) {
                    piece.erase(p, 8);
                    continue;
                }
                break;
            }
            if (piece.empty())
                continue;
        }

        // CONTENT phase — with tool call tag detection
        if (has_tools)
            full_output += piece;

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
                    tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                    if (tpl_family == imp::ChatTemplateFamily::LLAMA3) {
                        tc.name = tool_fn_name;
                        tc.arguments = dump_safe(j);
                    } else {
                        tc.name = j.value("name", "");
                        if (j.contains("arguments")) {
                            tc.arguments = dump_safe(j["arguments"]);
                        } else {
                            json args = j;
                            args.erase("name");
                            tc.arguments = dump_safe(args);
                        }
                    }
                    if (!tc.name.empty()) {
                        int idx = static_cast<int>(stream_tool_calls.size());
                        // Emit name chunk
                        json name_delta = {
                            {"tool_calls",
                             json::array(
                                 {{{"index", idx},
                                   {"id", tc.id},
                                   {"type", "function"},
                                   {"function", {{"name", tc.name}, {"arguments", ""}}}}})}};
                        std::string sse = sse_chunk(comp_id, created, snap_model_name, name_delta,
                                                    nullptr);
                        sink.write(sse.data(), sse.size());

                        // Emit arguments incrementally as partial-JSON deltas
                        // (Task 6) so OpenAI streaming clients see the tool
                        // arguments grow rather than land in one block.
                        constexpr size_t kArgChunk = 48;
                        const std::string& full_args = tc.arguments;
                        if (full_args.empty()) {
                            json args_delta = {
                                {"tool_calls",
                                 json::array({{{"index", idx},
                                               {"function", {{"arguments", ""}}}}})}};
                            sse = sse_chunk(comp_id, created, snap_model_name, args_delta, nullptr);
                            sink.write(sse.data(), sse.size());
                        }
                        for (size_t aoff = 0; aoff < full_args.size(); aoff += kArgChunk) {
                            size_t an = std::min(kArgChunk, full_args.size() - aoff);
                            json args_delta = {
                                {"tool_calls",
                                 json::array(
                                     {{{"index", idx},
                                       {"function", {{"arguments", full_args.substr(aoff, an)}}}}})}};
                            sse = sse_chunk(comp_id, created, snap_model_name, args_delta, nullptr);
                            sink.write(sse.data(), sse.size());
                        }

                        stream_tool_calls.push_back(std::move(tc));
                        tool_calls_emitted = true;
                    }
                } catch (...) {
                    // Malformed JSON — skip
                }

                // Check for more content after close tag
                std::string after = tool_body_buf.substr(close_pos + tool_close_tag.size());
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
                continue;  // Still collecting body
            }
        }

        if (has_tools && tool_phase == ToolPhase::TAG_SCANNING) {
            tool_tag_buf += piece;
            // ChatML: check for <tool_call>
            if (tpl_family != imp::ChatTemplateFamily::LLAMA3) {
                if (tool_tag_buf.size() >= 11) {  // len("<tool_call>")
                    if (tool_tag_buf.find("<tool_call>") != std::string::npos) {
                        auto pos = tool_tag_buf.find("<tool_call>");
                        // Flush content before the tag
                        std::string before = tool_tag_buf.substr(0, pos);
                        if (!before.empty()) {
                            json cd = {{"content", before}};
                            std::string sse = sse_chunk(comp_id, created, snap_model_name, cd,
                                                        nullptr);
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
                        continue;  // Still scanning
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
                        continue;  // Still matching prefix
                    }
                }
            } else {
                // Llama3: check for <function=
                if (tool_tag_buf.size() >= 10) {  // len("<function=")
                    auto fn_pos = tool_tag_buf.find("<function=");
                    if (fn_pos != std::string::npos) {
                        auto gt = tool_tag_buf.find('>', fn_pos + 10);
                        if (gt != std::string::npos) {
                            std::string before = tool_tag_buf.substr(0, fn_pos);
                            if (!before.empty()) {
                                json cd = {{"content", before}};
                                std::string sse = sse_chunk(comp_id, created, snap_model_name, cd,
                                                            nullptr);
                                sink.write(sse.data(), sse.size());
                            }
                            tool_fn_name = tool_tag_buf.substr(fn_pos + 10, gt - (fn_pos + 10));
                            tool_body_buf = tool_tag_buf.substr(gt + 1);
                            tool_close_tag = "</function>";
                            tool_tag_buf.clear();
                            tool_phase = ToolPhase::TOOL_CALL_BODY;
                            continue;
                        } else {
                            continue;  // Still scanning for >
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
                        if (!sse_writer.write_content(utf8_buf.data(), complete, sink))
                            return false;
                        utf8_buf.erase(0, complete);
                    }
                } else if (!stop_sequences.empty()) {
                    auto d = imp::stream::holdback_decision(pending_text, max_stop_len,
                                                            stop_sequences);
                    if (!flush_text(d.flush_len))
                        return false;
                    if (d.complete_match) {
                        text_stop_matched = true;
                        finish = "stop";
                        break;
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
                            top_arr.push_back({{"token", safe_token_json(t.text)},
                                               {"logprob", t.logprob},
                                               {"bytes", token_bytes_json(t.text)}});
                        }
                        lp_chunk = {
                            {"content", json::array({{{"token", safe_token_json(lp.text)},
                                                      {"logprob", lp.logprob},
                                                      {"bytes", token_bytes_json(lp.text)},
                                                      {"top_logprobs", top_arr}}})}};
                    }
                    std::string chunk = sse_chunk(comp_id, created, snap_model_name,
                                                  content_delta, nullptr, lp_chunk);
                    if (!sink.write(chunk.data(), chunk.size()))
                        return false;
                } else {
                    // Fast path: pre-formatted template
                    if (!sse_writer.write_content(utf8_buf.data(), complete, sink))
                        return false;
                    utf8_buf.erase(0, complete);
                }
            }
        } else {
            // Buffer text and check for stop matches via the pure holdback
            // pipeline (stream_pipeline.h). It returns the safe-to-emit prefix
            // and whether a complete stop sequence is present.
            pending_text += piece;
            auto d = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(d.flush_len))
                return false;
            if (d.complete_match) {
                text_stop_matched = true;
                finish = "stop";
                break;
            }
        }

        // Break after processing the last non-EOS token from batching engine
        if (finish)
            break;
    }

    // Flush scan buffer if we never left SCAN phase (model didn't think)
    // Harmony: flush the final channel's tail (the final block usually ends at
    // EOS/<|return|> with no trailing <|end|>). The other buffers below stay
    // empty for harmony, so they're no-ops.
    if (harmony && !hm_buf.empty())
        hm_flush(/*force=*/true);

    if (think_phase == ThinkPhase::SCAN && !think_scan_buf.empty()) {
        utf8_buf += think_scan_buf;
        think_scan_buf.clear();
    }

    // Flush remaining reasoning buffer (model ended while still thinking)
    if (!reasoning_utf8_buf.empty()) {
        emit_reasoning(reasoning_utf8_buf);
        reasoning_utf8_buf.clear();
    }

    // If the model exhausted tokens while still reasoning and never
    // produced content, emit a notice so the user sees something
    // instead of a blank response. Only fire this when max_tokens
    // was actually the cause (finish == "length") — a model that
    // naturally hit EOS during thinking will already have its
    // reasoning_content delivered, and the notice would be
    // misleading ("increase max_tokens" doesn't help when the model
    // chose to stop).
    if (think_phase == ThinkPhase::REASONING && utf8_buf.empty() && pending_text.empty() &&
        finish && std::strcmp(finish, "length") == 0) {
        std::string notice = "[Reasoning truncated — increase max_tokens for a complete answer]";
        sse_writer.write_content(notice, sink);
    }

    // Handle incomplete tool call at end (max_tokens hit while in tag)
    if (tool_phase != ToolPhase::CONTENT && !tool_calls_emitted) {
        // Partial tool call — emit as content, finish_reason stays "length"
        std::string leftover;
        if (!tool_tag_buf.empty())
            leftover += tool_tag_buf;
        if (!tool_body_buf.empty())
            leftover += tool_body_buf;
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
    std::string final_chunk = sse_chunk(comp_id, created, snap_model_name, empty_delta, finish);
    sink.write(final_chunk.data(), final_chunk.size());

    // Send usage chunk if requested
    if (include_usage) {
        json usage = {{"prompt_tokens", n_prompt_tokens},
                      {"completion_tokens", n_output_tokens},
                      {"total_tokens", n_prompt_tokens + n_output_tokens}};
        // Report prefix cache hit (OpenAI-compatible prompt_tokens_details)
        if (active_req && (active_req->cached_tokens > 0 || active_req->pin_kv_prefix)) {
            json details = {{"cached_tokens", active_req->cached_tokens}};
            int creation = cache_creation_tokens_(active_req, n_prompt_tokens);
            if (creation > 0)
                details["cache_creation_tokens"] = creation;
            usage["prompt_tokens_details"] = std::move(details);
        }
        if (n_reasoning_tokens > 0) {
            usage["completion_tokens_details"] = {{"reasoning_tokens", n_reasoning_tokens}};
        }
        json usage_obj = {{"id", comp_id},
                          {"object", "chat.completion.chunk"},
                          {"created", created},
                          {"model", snap_model_name},
                          {"choices", json::array()},
                          {"usage", usage}};
        std::string usage_chunk = "data: " + dump_safe(usage_obj) + "\n\n";
        sink.write(usage_chunk.data(), usage_chunk.size());
    }

    // Send [DONE]
    std::string done = "data: [DONE]\n\n";
    sink.write(done.data(), done.size());
    sink.done();

    // Log request with TTFT and cache hit info
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
    fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms, cached=%d)\n",
            comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms, ttft_ms, cached);
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += n_prompt_tokens;
    state.metrics.tokens_completion_total += n_output_tokens;
    state.metrics.tokens_cached_total += cached;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.last_ttft_ms = static_cast<int64_t>(ttft_ms);
    state.metrics.request_duration.observe(ms / 1000.0);
    if (n_output_tokens > 0)
        state.metrics.ttft.observe(ttft_ms / 1000.0);

    // Streaming response content is not accumulated across SSE
    // chunks, so the JSONL `response` field stays null. The
    // request body, token counts, finish reason, and latency
    // still reflect everything the client did.
    log_request_jsonl(state, log_skip, t_log_start, comp_id, log_endpoint, log_client_ip,
                      log_raw_body, ms, n_prompt_tokens, n_output_tokens, finish, json());

    return true;
}

void handle_chat_completions(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    ChatRequestContext ctx;
    if (!parse_chat_request_params(req, res, state, ctx))
        return;
    if (!snapshot_state_and_tokenize_(res, state, ctx))
        return;

    // Save input tokens for potential reuse with n > 1
    std::vector<int32_t> saved_tokens = ctx.snap.tokens;

    // Helper to create an imp::Request with the given completion index
    auto make_imp_request = [&](int completion_idx) {
        auto req = std::make_shared<imp::Request>();
        req->input_tokens = saved_tokens;
        req->max_tokens = ctx.params.max_tokens;
        req->temperature = ctx.params.temperature;
        req->top_p = ctx.params.top_p;
        req->top_k = ctx.params.top_k;
        req->seed = (ctx.params.seed != -1) ? ctx.params.seed + completion_idx : -1;
        req->pin_kv_prefix = ctx.params.cache_prompt;
        req->min_p = ctx.params.min_p;
        req->typical_p = ctx.params.typical_p;
        req->repetition_penalty = ctx.params.repetition_penalty;
        req->frequency_penalty = ctx.params.frequency_penalty;
        req->presence_penalty = ctx.params.presence_penalty;
        req->repeat_last_n = ctx.params.repeat_last_n;
        req->dry_multiplier = ctx.params.dry_multiplier;
        req->dry_base = ctx.params.dry_base;
        req->dry_allowed_length = ctx.params.dry_allowed_length;
        req->dry_penalty_last_n = ctx.params.dry_penalty_last_n;
        req->mirostat = ctx.params.mirostat;
        req->mirostat_tau = ctx.params.mirostat_tau;
        req->mirostat_eta = ctx.params.mirostat_eta;
        req->logprobs = ctx.params.req_logprobs;
        req->top_logprobs = ctx.params.top_logprobs;
        req->json_mode = ctx.params.json_mode;
        req->json_schema = ctx.params.json_schema_str;
        req->has_tools = ctx.params.has_tools;
        req->tpl_family = ctx.snap.tpl_family;
        req->logit_bias = ctx.params.logit_bias;
        req->think_budget = ctx.params.think_budget;
        // Generation starts INSIDE the think block when the prompt carries the
        // <think> prefix (template-injected or server-appended). Without this
        // the engine's think-budget enforcement never sees an opener in the
        // output, counts zero reasoning tokens, and lets the model think until
        // max_tokens (content empty, finish=length).
        req->started_in_think = ctx.snap.enable_thinking;
        req->in_think_block = ctx.snap.enable_thinking;
        // Stream requests stay on per-step decode for real per-token SSE (#754).
        req->stream = ctx.params.stream;
        req->status = imp::RequestStatus::PENDING;
        return req;
    };

    // Create first request
    auto imp_req = make_imp_request(0);

    // Create a ServerRequest wrapper and submit to the batching engine
    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = imp_req;

    // For vision requests, fall back to blocking mode since vision state
    // is per-engine (not per-request). Use the old C API path.
    if (ctx.snap.has_vision_request) {
        handle_vision_chat_blocking_(res, state, ctx, imp_req);
        return;
    }

    // Submit to batching engine for continuous batching
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (!state.batching || !state.batching->is_running()) {
            res.status = 503;
            json err = {
                {"error",
                 {{"message", "Inference engine not ready. Please retry."}, {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }
        state.batching->submit(server_req);
    }

    std::string comp_id = ctx.req_id;
    int64_t created = unix_timestamp();

    if (ctx.params.stream) {
        stream_chat_response_(res, state, ctx, server_req);
    } else {
        nonstream_chat_response_(res, state, ctx, imp_req, server_req, saved_tokens, comp_id, created);
    }
}

void handle_completions(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    // Validate sampling parameters
    if (!validate_sampling_params(body, res))
        return;

    // Extract prompt
    std::string prompt = body.value("prompt", "");
    if (prompt.empty()) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "\"prompt\" is required and must not be empty"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
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
    float repetition_penalty = body.value("repetition_penalty", 1.05f);
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
    if (top_logprobs < 0)
        top_logprobs = 0;
    if (top_logprobs > 20)
        top_logprobs = 20;

    // Parse stop sequences
    std::vector<std::string> stop_sequences;
    if (body.contains("stop") && !body["stop"].is_null()) {
        if (body["stop"].is_string()) {
            stop_sequences.push_back(body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                if (s.is_string()) {
                    stop_sequences.push_back(s.get<std::string>());
                    if (stop_sequences.size() >= 4)
                        break;
                }
            }
        }
    }
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());

    // Parse logit_bias: map of token_id (string) -> bias (float)
    std::vector<std::pair<int32_t, float>> logit_bias;
    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
        for (auto& [key, val] : body["logit_bias"].items()) {
            try {
                int32_t token_id = std::stoi(key);
                float bias = val.get<float>();
                logit_bias.emplace_back(token_id, bias);
            } catch (...) {
                // Skip invalid entries
            }
        }
    }

    // Parse stream_options for include_usage
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Log request received
    std::string req_id = make_completion_id(state);
    fprintf(stderr, "[%s] completions: prompt_len=%zu stream=%s max_tokens=%d temp=%.2f\n", req_id.c_str(),
            prompt.size(), stream ? "true" : "false", max_tokens, temperature);

    // Validate model field (required per OpenAI spec)
    std::string requested_model = body.value("model", "");
    if (requested_model.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"model\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Snapshot state fields under lock for thread-safe access
    imp::Tokenizer* snap_tok;
    std::string snap_model_name;
    bool snap_is_think_model;
    int32_t snap_channel_open_id;
    int snap_max_seq_len;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (!ensure_model_loaded(state, requested_model, res))
            return;
        snap_tok = state.tok;
        snap_model_name = state.model_name;
        snap_is_think_model = state.is_think_model;
        snap_channel_open_id = state.channel_open_id;
        snap_max_seq_len = state.max_seq_len;
    }

    // Tokenize raw prompt (no chat template)
    std::vector<int32_t> tokens = snap_tok->encode(prompt);
    int n_prompt_tokens = static_cast<int>(tokens.size());

    // Server-side input-token limit (--max-input-tokens). Reject pre-prefill.
    if (state.max_input_tokens > 0 && n_prompt_tokens > state.max_input_tokens) {
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds max input tokens (" + std::to_string(n_prompt_tokens) +
                                        " > " + std::to_string(state.max_input_tokens) + ")"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    if (n_prompt_tokens >= snap_max_seq_len) {
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds context window (" + std::to_string(n_prompt_tokens) +
                                        " tokens >= " + std::to_string(snap_max_seq_len) + " max)"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    int remaining = snap_max_seq_len - n_prompt_tokens;
    if (max_tokens > remaining)
        max_tokens = remaining;

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
    imp_req->logit_bias = std::move(logit_bias);
    imp_req->think_budget = body.value("think_budget", state.default_think_budget);
    imp_req->pin_kv_prefix = body.value("cache_prompt", false);
    // Stream requests stay on per-step decode for real per-token SSE (#754).
    imp_req->stream = stream;
    imp_req->status = imp::RequestStatus::PENDING;

    auto server_req = std::make_shared<ServerRequest>();
    server_req->request = imp_req;

    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (!state.batching || !state.batching->is_running()) {
            res.status = 503;
            json err = {
                {"error",
                 {{"message", "Inference engine not ready. Please retry."}, {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }
        state.batching->submit(server_req);
    }

    std::string comp_id = req_id;
    int64_t created = unix_timestamp();

    if (stream) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, server_req, comp_id, created, n_prompt_tokens, t_start, stop_sequences,
             max_stop_len, echo, prompt, include_usage, snap_tok, snap_model_name,
             snap_is_think_model](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                int n_output_tokens = 0;
                const char* finish = nullptr;

                // Echo prompt as first chunk if requested
                if (echo && !prompt.empty()) {
                    std::string chunk = sse_completion_chunk(comp_id, created, snap_model_name, prompt,
                                                             nullptr);
                    sink.write(chunk.data(), chunk.size());
                }

                std::string utf8_buf;
                std::string pending_text;
                bool text_stop_matched = false;

                // Strip <think> blocks for completions (no reasoning_content field).
                // think_confirmed starts FALSE so a raw /v1/completions prompt
                // (no chat template → no injected <think>) streams incrementally
                // instead of buffering every token into think_buf waiting for a
                // </think> that never comes (#760: completions stream arrived as
                // one frame). It flips true only if a real <think> opener shows
                // up in the first kThinkScanLimit tokens, so genuine think blocks
                // are still stripped.
                bool think_strip = (snap_is_think_model && state.default_args.reasoning_format != "none");
                bool think_confirmed = false;
                std::string think_buf;
                int think_tokens = 0;
                const int kThinkScanLimit = 8;

                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0)
                        return true;
                    std::string to_send = pending_text.substr(0, up_to);
                    pending_text = pending_text.substr(up_to);
                    std::string sse = sse_completion_chunk(comp_id, created, snap_model_name, to_send,
                                                           nullptr);
                    return sink.write(sse.data(), sse.size());
                };

                auto request_start_c = std::chrono::steady_clock::now();
                for (;;) {
                    // #757: the is_last token sets `finish` then falls through
                    // to think-stripping, which `continue`s on every swallowed
                    // token — bypassing the trailing `if (finish) break`. For a
                    // think-capable model whose final token lands inside the
                    // think buffer the loop would otherwise spin on pop_token
                    // until the client gives up (0 bytes, never terminates).
                    // Break here so the buffers flush and [DONE] is sent.
                    if (finish)
                        break;

                    // Check client disconnect
                    if (!sink.is_writable()) {
                        server_req->cancel();
                        finish = "cancelled";
                        break;
                    }

                    // Check request timeout
                    if (state.request_timeout > 0) {
                        auto elapsed = std::chrono::steady_clock::now() - request_start_c;
                        if (elapsed > std::chrono::seconds(state.request_timeout)) {
                            server_req->cancel();
                            finish = "length";
                            break;
                        }
                    }

                    TokenEvent evt{};
                    if (!server_req->pop_token(evt)) {
                        continue;
                    }

                    if (evt.token_id < 0) {
                        finish = evt.finish_reason ? evt.finish_reason : "stop";
                        break;
                    }

                    int32_t token = evt.token_id;

                    if (evt.is_last) {
                        if (token == snap_tok->eos_id()) {
                            finish = evt.finish_reason ? evt.finish_reason : "stop";
                            break;
                        }
                        finish = evt.finish_reason ? evt.finish_reason : "length";
                    }

                    n_output_tokens++;
                    std::string piece = snap_tok->decode_token(token);

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
                            if (piece.empty())
                                continue;
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
                            std::string chunk = sse_completion_chunk(comp_id, created, snap_model_name,
                                                                     to_emit, nullptr);
                            if (!sink.write(chunk.data(), chunk.size()))
                                return false;
                        }
                    } else {
                        pending_text += piece;
                        auto d = imp::stream::holdback_decision(pending_text, max_stop_len,
                                                                stop_sequences);
                        if (!flush_text(d.flush_len))
                            return false;
                        if (d.complete_match) {
                            text_stop_matched = true;
                            finish = "stop";
                            break;
                        }
                    }

                    if (finish)
                        break;
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
                    std::string sse = sse_completion_chunk(comp_id, created, snap_model_name, utf8_buf,
                                                           nullptr);
                    sink.write(sse.data(), sse.size());
                }
                if (!pending_text.empty() && !text_stop_matched) {
                    std::string sse = sse_completion_chunk(comp_id, created, snap_model_name, pending_text,
                                                           nullptr);
                    sink.write(sse.data(), sse.size());
                }

                if (!finish)
                    finish = "length";

                // Final chunk with finish_reason
                std::string final_chunk = sse_completion_chunk(comp_id, created, snap_model_name, "", finish);
                sink.write(final_chunk.data(), final_chunk.size());

                // Usage chunk if requested
                if (include_usage) {
                    json usage_obj = {{"id", comp_id},
                                      {"object", "text_completion"},
                                      {"created", created},
                                      {"model", snap_model_name},
                                      {"choices", json::array()},
                                      {"usage",
                                       {{"prompt_tokens", n_prompt_tokens},
                                        {"completion_tokens", n_output_tokens},
                                        {"total_tokens", n_prompt_tokens + n_output_tokens}}}};
                    std::string usage_chunk = "data: " + dump_safe(usage_obj) + "\n\n";
                    sink.write(usage_chunk.data(), usage_chunk.size());
                }

                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();

                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n", comp_id.c_str(),
                        n_prompt_tokens, n_output_tokens, ms);
                state.metrics.requests_total++;
                state.metrics.tokens_prompt_total += n_prompt_tokens;
                state.metrics.tokens_completion_total += n_output_tokens;
                state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);

                return true;
            });
    } else {
        // Non-streaming
        auto active_req = server_req->request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;

        auto ns_comp_start = std::chrono::steady_clock::now();
        for (;;) {
            if (state.request_timeout > 0) {
                auto elapsed = std::chrono::steady_clock::now() - ns_comp_start;
                if (elapsed > std::chrono::seconds(state.request_timeout)) {
                    server_req->cancel();
                    finish = "length";
                    break;
                }
            }

            TokenEvent evt{};
            if (!server_req->pop_token(evt)) {
                continue;
            }

            if (evt.token_id < 0) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }

            int32_t token = evt.token_id;

            if (evt.is_last) {
                if (token == snap_tok->eos_id()) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                finish = evt.finish_reason ? evt.finish_reason : "length";
            }

            output_ids.push_back(token);

            if (!stop_sequences.empty()) {
                output_text += snap_tok->decode_token(token);
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

            if (finish)
                break;
        }

        if (!finish)
            finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string text = !stop_sequences.empty() ? output_text : snap_tok->decode(output_ids);

        // Strip <think>...</think> for text completions (no reasoning_content field)
        if (snap_is_think_model && state.default_args.reasoning_format != "none") {
            strip_think_block(text);
        }
        if (snap_channel_open_id >= 0) {
            strip_channel_headers(text);
        }

        // Prepend prompt if echo requested
        if (echo)
            text = prompt + text;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n", comp_id.c_str(), n_prompt_tokens,
                n_output_tokens, ms);
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
                    top_arr.push_back({{"token", safe_token_json(t.text)},
                                       {"logprob", t.logprob},
                                       {"bytes", token_bytes_json(t.text)}});
                }
                content_logprobs.push_back({{"token", safe_token_json(lp.text)},
                                            {"logprob", lp.logprob},
                                            {"bytes", token_bytes_json(lp.text)},
                                            {"top_logprobs", top_arr}});
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        json choice = {{"index", 0}, {"text", text}, {"finish_reason", finish}};
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        json response = {{"id", comp_id},
                         {"object", "text_completion"},
                         {"created", created},
                         {"model", snap_model_name},
                         {"choices", json::array({choice})},
                         {"usage",
                          {{"prompt_tokens", n_prompt_tokens},
                           {"completion_tokens", n_output_tokens},
                           {"total_tokens", n_prompt_tokens + n_output_tokens}}}};

        res.set_content(dump_safe(response), "application/json");
    }
}

void handle_tokenize(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    std::string content = body.value("content", "");
    if (content.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"content\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Snapshot model pointer under lock
    ImpModel snap_model;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        snap_model = state.model;
    }
    if (!snap_model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"}, {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    std::vector<int32_t> tokens(32768);
    int n_tokens = 0;
    ImpError err = imp_tokenize(snap_model, content.c_str(), tokens.data(), &n_tokens, 32768);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error",
                       {{"message", std::string("Tokenize failed: ") + imp_error_string(err)},
                        {"type", "server_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    tokens.resize(n_tokens);
    json response = {{"tokens", tokens}};
    res.set_content(dump_safe(response), "application/json");
}

void handle_detokenize(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    if (!body.contains("tokens") || !body["tokens"].is_array()) {
        res.status = 400;
        json err = {
            {"error", {{"message", "\"tokens\" array is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Snapshot model pointer under lock
    ImpModel snap_model;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        snap_model = state.model;
    }
    if (!snap_model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"}, {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Bound the array before allocating buf = tokens.size()*32 + 256: a ~100 MiB
    // JSON array of small ints would otherwise force a multi-GB host allocation.
    if (body["tokens"].size() > 1000000) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "tokens array exceeds maximum of 1000000 entries"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    std::vector<int32_t> tokens = body["tokens"].get<std::vector<int32_t>>();
    std::vector<char> buf(tokens.size() * 32 + 256);
    ImpError err = imp_detokenize(snap_model, tokens.data(), static_cast<int>(tokens.size()), buf.data(),
                                  buf.size());
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error",
                       {{"message", std::string("Detokenize failed: ") + imp_error_string(err)},
                        {"type", "server_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return;
    }

    json response = {{"content", std::string(buf.data())}};
    res.set_content(dump_safe(response), "application/json");
}

void handle_metrics(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    auto& m = state.metrics;
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                                   m.start_time)
                      .count();

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
    out += "# HELP imp_tokens_cached_total Total prompt tokens served from prefix cache\n";
    out += "# TYPE imp_tokens_cached_total counter\n";
    out += "imp_tokens_cached_total " + std::to_string(m.tokens_cached_total.load()) + "\n";
    out += "# HELP imp_last_ttft_ms Time to first token of last request in milliseconds\n";
    out += "# TYPE imp_last_ttft_ms gauge\n";
    out += "imp_last_ttft_ms " + std::to_string(m.last_ttft_ms.load()) + "\n";

    // Latency histograms (Prometheus histogram: cumulative _bucket{le=...},
    // plus _sum and _count). Buckets are in seconds.
    auto emit_histogram = [&out](const char* name, const char* help, const LatencyHistogram& h) {
        out += "# HELP ";
        out += name;
        out += ' ';
        out += help;
        out += "\n# TYPE ";
        out += name;
        out += " histogram\n";
        for (int i = 0; i < LatencyHistogram::kNumBuckets; ++i) {
            char le[32];
            std::snprintf(le, sizeof(le), "%g", LatencyHistogram::kBounds[i]);
            out += name;
            out += "_bucket{le=\"";
            out += le;
            out += "\"} ";
            out += std::to_string(h.buckets[i].load());
            out += "\n";
        }
        out += name;
        out += "_bucket{le=\"+Inf\"} ";
        out += std::to_string(h.count.load());
        out += "\n";
        char sum[48];
        std::snprintf(sum, sizeof(sum), "%g", h.sum_us.load() / 1e6);
        out += name;
        out += "_sum ";
        out += sum;
        out += "\n";
        out += name;
        out += "_count ";
        out += std::to_string(h.count.load());
        out += "\n";
    };
    emit_histogram("imp_request_duration_seconds", "Request end-to-end latency in seconds",
                   m.request_duration);
    emit_histogram("imp_ttft_seconds", "Time to first token in seconds", m.ttft);
    out += "# HELP imp_model_loaded Whether a model is currently loaded\n";
    out += "# TYPE imp_model_loaded gauge\n";
    bool loaded;
    int queue = 0;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        loaded = state.model_loaded();
        if (state.batching)
            queue = state.batching->queue_depth();
    }
    out += "imp_model_loaded " + std::string(loaded ? "1" : "0") + "\n";
    out += "# HELP imp_queue_depth Current number of active and pending requests\n";
    out += "# TYPE imp_queue_depth gauge\n";
    out += "imp_queue_depth " + std::to_string(queue) + "\n";

    res.set_content(out, "text/plain; version=0.0.4; charset=utf-8");
}

// Convert IEEE 754 FP16 (uint16_t) to FP32 on host
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // Subnormal: normalize
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
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

void handle_embeddings(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
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
                    json err = {
                        {"error",
                         {{"message", "Each input must be a string"}, {"type", "invalid_request_error"}}}};
                    res.set_content(dump_safe(err), "application/json");
                    return;
                }
            }
        } else {
            res.status = 400;
            json err = {{"error",
                         {{"message", "\"input\" must be a string or array of strings"},
                          {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }
    } else {
        res.status = 400;
        json err = {{"error", {{"message", "\"input\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    if (inputs.empty()) {
        res.status = 400;
        json err = {
            {"error", {{"message", "\"input\" must not be empty"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Acquire inference lock and pause batching engine for exclusive access
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(1));
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error",
                     {{"message", "Server is busy processing another request. Please retry."},
                      {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    if (!state.model_loaded()) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"}, {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Pause the batching engine for exclusive C-API access (imp_prefill drives
    // engine->step() directly, which must not race the worker). pause() lets
    // in-flight generations FINISH before parking the worker — stop() would
    // cancel them, so any chat running concurrently with this embeddings call
    // returned an empty `finish_reason:"cancelled"` completion (the "0
    // completion tokens" wedge). We hold state.mtx, so no new chat is admitted
    // while paused; resume() unparks on scope exit.
    bool had_batching = (state.batching && state.batching->is_running());
    if (had_batching) {
        if (!state.batching->pause()) {
            res.status = 503;
            json err = {{"error",
                         {{"message", "Server busy draining in-flight requests. Please retry."},
                          {"type", "server_error"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }
    }
    auto restart_batching = [&] {
        if (had_batching && state.batching)
            state.batching->resume();
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
        ImpError err = imp_tokenize(state.model, text.c_str(), tokens.data(), &n_tokens, 32768);
        if (err != IMP_SUCCESS) {
            res.status = 500;
            json error = {{"error",
                           {{"message", std::string("Tokenize failed: ") + imp_error_string(err)},
                            {"type", "server_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return;
        }
        tokens.resize(n_tokens);

        if (n_tokens == 0) {
            res.status = 400;
            json error = {
                {"error",
                 {{"message", "Input tokenizes to zero tokens"}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return;
        }

        // Reject over-long inputs before prefill. Two independent bounds:
        //   1. the engine's allocated context (state.max_seq_len), and
        //   2. the executor's single-pass hidden capacity (max_tokens()): the
        //      embeddings path mean-pools EVERY token's hidden state, which only
        //      works when the whole input is prefilled in one pass. A longer
        //      input is chunked and hidden_ keeps only the last chunk, so
        //      view_hidden(n) would slice [0,n) out of a [max_tokens,*] buffer
        //      and abort the whole server (Tensor::slice IMP_CHECK). max_tokens
        //      can be far below max_seq_len (e.g. 4096 vs a 32768 context).
        int embed_cap = state.max_seq_len;
        if (state.ctx && state.ctx->engine && state.ctx->engine->executor()) {
            int hid = state.ctx->engine->executor()->max_tokens();
            if (hid > 0 && (embed_cap <= 0 || hid < embed_cap))
                embed_cap = hid;
        }
        if (embed_cap > 0 && n_tokens > embed_cap) {
            send_json_error(res, 400, "invalid_request_error",
                            "Input exceeds the embedding context (" + std::to_string(n_tokens) +
                                " tokens > " + std::to_string(embed_cap) + " max)");
            return;
        }

        total_prompt_tokens += n_tokens;

        // Run prefill (forward pass without generation)
        err = imp_prefill(state.ctx, tokens.data(), n_tokens);
        if (err != IMP_SUCCESS) {
            imp_context_reset(state.ctx);
            res.status = 500;
            json error = {{"error",
                           {{"message", std::string("Prefill failed: ") + imp_error_string(err)},
                            {"type", "server_error"}}}};
            res.set_content(dump_safe(error), "application/json");
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
        cudaError_t cuda_err = cudaMemcpy(h_hidden.data(), hidden_view.data, n_elements * sizeof(uint16_t),
                                          cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            imp_context_reset(state.ctx);
            res.status = 500;
            json error = {{"error",
                           {{"message", std::string("CUDA memcpy failed: ") + cudaGetErrorString(cuda_err)},
                            {"type", "server_error"}}}};
            res.set_content(dump_safe(error), "application/json");
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

        data.push_back({{"object", "embedding"}, {"embedding", embedding}, {"index", input_idx}});

        // Reset context for next input
        imp_context_reset(state.ctx);
    }

    auto t1 = std::chrono::steady_clock::now();
    int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    state.metrics.last_request_duration_ms.store(duration_ms);
    state.metrics.tokens_prompt_total += total_prompt_tokens;

    // batching_guard restarts the batching engine automatically on scope exit

    json response = {{"object", "list"},
                     {"data", data},
                     {"model", body.value("model", state.model_name)},
                     {"usage",
                      {{"prompt_tokens", total_prompt_tokens}, {"total_tokens", total_prompt_tokens}}}};
    res.set_content(dump_safe(response), "application/json");
}

// ===========================================================================
// Anthropic /v1/messages — native SSE streaming
// ===========================================================================
//
// Non-streaming reuses the OpenAI code path (Anthropic→OpenAI body in,
// OpenAI→Anthropic response out). Streaming drives the same real per-token
// batching-engine loop the OpenAI streaming path uses (pop_token), but emits
// native Anthropic SSE events incrementally so TTFT == real first-token
// latency rather than full-generation latency.
// ---------------------------------------------------------------------------

namespace {

// Anthropic SSE event writer. Emits "event: <name>\ndata: <json>\n\n".
struct AnthropicSSE {
    httplib::DataSink& sink;
    bool emit(const char* event_name, const json& payload) const {
        std::string buf = "event: ";
        buf += event_name;
        buf += "\ndata: ";
        buf += dump_safe(payload);
        buf += "\n\n";
        return sink.write(buf.data(), buf.size());
    }
};

// Tracks which content block (if any) is currently open in the stream so we
// can close it before opening one of a different kind. Anthropic requires a
// content_block_start before deltas and a content_block_stop after.
enum class AnthBlock { NONE, THINKING, TEXT, TOOL_USE };

// Drives the real token loop and emits native Anthropic SSE events. Mirrors
// the token-handling of run_chat_stream_ (reasoning extraction, channel
// filter, tool-call tag state machine) but maps it onto Anthropic blocks:
//   reasoning -> thinking block (thinking_delta)
//   content   -> text block (text_delta)
//   tool call -> tool_use block (input_json_delta, chunked)
static bool run_anthropic_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                                  const std::shared_ptr<ServerRequest>& server_req,
                                  const std::string& anth_model, const std::string& msg_id) {
    AnthropicSSE out{sink};

    const auto& stop_sequences = ctx.params.stop_sequences;
    // Derive from the FINAL stop list (server-injected stops update it late) —
    // see the matching comment in the chat streaming path.
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());
    bool enable_thinking = ctx.snap.enable_thinking;
    bool has_tools = ctx.params.has_tools;
    auto tpl_family = ctx.snap.tpl_family;
    float think_budget = ctx.params.think_budget;
    auto snap_tok = ctx.snap.tok;
    bool snap_have_template = ctx.snap.have_template;
    bool snap_is_think_model = ctx.snap.is_think_model;
    int snap_think_start_id = ctx.snap.think_start_id;
    int snap_think_end_id = ctx.snap.think_end_id;
    int snap_channel_open_id = ctx.snap.channel_open_id;
    int snap_channel_close_id = ctx.snap.channel_close_id;
    int snap_channel_newline_id = ctx.snap.channel_newline_id;
    const auto& snap_stop_token_ids = ctx.snap.stop_token_ids;
    auto active_req = server_req->request;
    auto t_start = ctx.t_start;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;

    // ---- message_start ----------------------------------------------------
    {
        json msg = {
            {"id", msg_id},
            {"type", "message"},
            {"role", "assistant"},
            {"content", json::array()},
            {"model", anth_model},
            {"stop_reason", nullptr},
            {"stop_sequence", nullptr},
            {"usage", {{"input_tokens", n_prompt_tokens}, {"output_tokens", 0}}},
        };
        if (!out.emit("message_start", json{{"type", "message_start"}, {"message", std::move(msg)}}))
            return false;
    }

    int block_index = -1;
    AnthBlock open_block = AnthBlock::NONE;

    auto stop_block = [&]() -> bool {
        if (open_block == AnthBlock::NONE)
            return true;
        bool ok = out.emit("content_block_stop",
                           json{{"type", "content_block_stop"}, {"index", block_index}});
        open_block = AnthBlock::NONE;
        return ok;
    };
    auto start_text_block = [&]() -> bool {
        if (open_block == AnthBlock::TEXT)
            return true;
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::TEXT;
        return out.emit("content_block_start",
                        json{{"type", "content_block_start"},
                             {"index", block_index},
                             {"content_block", {{"type", "text"}, {"text", ""}}}});
    };
    auto start_thinking_block = [&]() -> bool {
        if (open_block == AnthBlock::THINKING)
            return true;
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::THINKING;
        return out.emit("content_block_start",
                        json{{"type", "content_block_start"},
                             {"index", block_index},
                             {"content_block", {{"type", "thinking"}, {"thinking", ""}}}});
    };
    auto emit_text = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_text_block())
            return false;
        return out.emit("content_block_delta",
                        json{{"type", "content_block_delta"},
                             {"index", block_index},
                             {"delta", {{"type", "text_delta"}, {"text", text}}}});
    };
    auto emit_thinking = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_thinking_block())
            return false;
        return out.emit("content_block_delta",
                        json{{"type", "content_block_delta"},
                             {"index", block_index},
                             {"delta", {{"type", "thinking_delta"}, {"thinking", text}}}});
    };

    // gpt-oss Harmony streaming filter (analysis/commentary -> thinking block,
    // final -> text block); markers arrive as atomic special-token pieces. See
    // the matching filter in run_chat_stream_ (#760).
    const bool harmony = (ctx.snap.tpl_family == imp::ChatTemplateFamily::HARMONY);
    std::string hm_channel, hm_name, hm_buf;
    bool hm_in_msg = false, hm_reading_name = false;
    auto hm_flush = [&](bool force) -> bool {
        size_t complete = force ? hm_buf.size() : utf8_complete_len(hm_buf);
        if (complete == 0)
            return true;
        std::string chunk = hm_buf.substr(0, complete);
        hm_buf.erase(0, complete);
        if (hm_channel == "analysis" || hm_channel == "commentary")
            return emit_thinking(chunk);
        return emit_text(chunk);
    };
    // Open a tool_use block and stream its arguments as chunked
    // input_json_delta events (Task 6: incremental arg deltas).
    auto emit_tool_use = [&](const ParsedToolCall& tc) -> bool {
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::TOOL_USE;
        namespace anth = imp_server::anthropic;
        if (!out.emit("content_block_start",
                      json{{"type", "content_block_start"},
                           {"index", block_index},
                           {"content_block",
                            {{"type", "tool_use"},
                             {"id", anth::tool_call_id_to_anthropic(tc.id)},
                             {"name", tc.name},
                             {"input", json::object()}}}}))
            return false;
        const std::string& args = tc.arguments;
        constexpr size_t kChunk = 48;
        for (size_t off = 0; off < args.size(); off += kChunk) {
            size_t n = std::min(kChunk, args.size() - off);
            if (!out.emit("content_block_delta",
                          json{{"type", "content_block_delta"},
                               {"index", block_index},
                               {"delta",
                                {{"type", "input_json_delta"},
                                 {"partial_json", args.substr(off, n)}}}}))
                return false;
        }
        return stop_block();
    };

    int n_output_tokens = 0;
    const char* finish = nullptr;
    double ttft_ms = 0.0;

    std::string utf8_buf;        // confirmed-UTF8 content buffer
    std::string pending_text;    // stop-sequence holdback
    bool text_stop_matched = false;

    // Tool call detection state machine (same shape as run_chat_stream_).
    enum class ToolPhase { CONTENT, TAG_SCANNING, TOOL_CALL_BODY };
    ToolPhase tool_phase = ToolPhase::CONTENT;
    std::string tool_tag_buf, tool_body_buf, tool_close_tag, tool_fn_name;
    std::vector<ParsedToolCall> stream_tool_calls;
    bool tool_calls_emitted = false;

    // Reasoning extraction (DeepSeek <think>). enable_thinking also covers
    // text-level thinkers (Nemotron) — see the chat streaming path.
    enum class ThinkPhase { SCAN, REASONING, CONTENT };
    bool use_reasoning = (state.default_args.reasoning_format == "deepseek" &&
                          (snap_is_think_model || enable_thinking));
    ThinkPhase think_phase;
    if (enable_thinking)
        think_phase = ThinkPhase::REASONING;
    else if (use_reasoning && think_budget > 0.0f)
        think_phase = ThinkPhase::SCAN;
    else
        think_phase = ThinkPhase::CONTENT;
    std::string reasoning_utf8_buf, think_scan_buf;
    int think_scan_count = 0;
    bool content_started = (think_phase == ThinkPhase::CONTENT);
    int think_reentries = 0;
    const int kMaxThinkReentries = 1;
    const int kThinkScanLimit = 8;
    bool channel_header_active = false;

    auto flush_text = [&](size_t up_to) -> bool {
        if (up_to == 0)
            return true;
        bool ok = emit_text(pending_text.substr(0, up_to));
        pending_text.erase(0, up_to);
        return ok;
    };

    auto request_start = std::chrono::steady_clock::now();
    for (;;) {
        // #755: when a thinking model exhausts its token budget the final
        // (is_last) token lands inside the REASONING phase, which `continue`s
        // and skips the trailing `if (finish) break`. The loop would then spin
        // on pop_token forever — message_delta/message_stop never emitted, so
        // the Anthropic client (SDK or curl -N) hangs indefinitely. Breaking
        // here guarantees the terminal events are always sent.
        if (finish)
            break;

        if (!sink.is_writable()) {
            server_req->cancel();
            finish = "cancelled";
            break;
        }
        if (state.request_timeout > 0) {
            auto elapsed = std::chrono::steady_clock::now() - request_start;
            if (elapsed > std::chrono::seconds(state.request_timeout)) {
                server_req->cancel();
                finish = "length";
                break;
            }
        }

        TokenEvent evt{};
        if (!server_req->pop_token(evt))
            continue;

        if (evt.token_id < 0) {
            finish = evt.finish_reason ? evt.finish_reason : "stop";
            break;
        }
        int32_t token = evt.token_id;

        if (!evt.is_last) {
            bool is_structural_stop = (token == snap_tok->eos_id());
            if (!is_structural_stop && snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids)
                    if (token == stop_id) {
                        is_structural_stop = true;
                        break;
                    }
            }
            if (is_structural_stop)
                continue;
        }
        if (evt.is_last) {
            if (token == snap_tok->eos_id()) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            bool is_stop = false;
            if (snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids)
                    if (token == stop_id) {
                        is_stop = true;
                        break;
                    }
            }
            if (is_stop) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            finish = evt.finish_reason ? evt.finish_reason : "length";
        }

        n_output_tokens++;
        if (n_output_tokens == 1)
            ttft_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t_start)
                          .count();
        std::string piece = snap_tok->decode_token(token);

        // gpt-oss Harmony channel routing (analysis/commentary -> thinking,
        // final -> text). Markers arrive as atomic special-token pieces.
        if (harmony) {
            if (piece == "<|channel|>" || piece == "<|message|>" || piece == "<|end|>" ||
                piece == "<|return|>" || piece == "<|start|>") {
                if (hm_in_msg && !hm_flush(/*force=*/true))
                    return false;
                if (piece == "<|channel|>") {
                    hm_reading_name = true;
                    hm_in_msg = false;
                    hm_name.clear();
                } else if (piece == "<|message|>") {
                    size_t s = hm_name.find_first_not_of("\n\r\t ");
                    size_t e = hm_name.find_last_not_of("\n\r\t ");
                    hm_channel = (s == std::string::npos) ? std::string() : hm_name.substr(s, e - s + 1);
                    hm_reading_name = false;
                    hm_in_msg = true;
                } else {
                    hm_in_msg = false;
                    hm_reading_name = false;
                    hm_channel.clear();
                }
                continue;
            }
            if (hm_reading_name) {
                hm_name += piece;
                continue;
            }
            if (!hm_in_msg)
                continue;
            hm_buf += piece;
            if (!hm_flush(/*force=*/false))
                return false;
            continue;
        }

        // Gemma-4 channel filter.
        if (snap_channel_open_id >= 0) {
            if (channel_header_active) {
                if (token == snap_channel_newline_id || (!piece.empty() && piece.back() == '\n'))
                    channel_header_active = false;
                continue;
            }
            if (token == snap_channel_open_id) {
                channel_header_active = true;
                continue;
            }
            if (token == snap_channel_close_id)
                continue;
        }

        // Reasoning extraction.
        if (think_phase == ThinkPhase::SCAN) {
            if (token == snap_think_start_id) {
                think_phase = ThinkPhase::REASONING;
                continue;
            }
            think_scan_buf += piece;
            think_scan_count++;
            if (think_scan_buf.find("<think>") != std::string::npos) {
                think_phase = ThinkPhase::REASONING;
                auto pos = think_scan_buf.find("<think>");
                std::string after = think_scan_buf.substr(pos + 7);
                think_scan_buf.clear();
                if (!after.empty())
                    reasoning_utf8_buf += after;
                continue;
            }
            if (think_scan_count == 1 && piece.empty()) {
                think_phase = ThinkPhase::REASONING;
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
            if (token == snap_think_end_id) {
                size_t complete = utf8_complete_len(reasoning_utf8_buf);
                if (!emit_thinking(reasoning_utf8_buf.substr(0, complete)))
                    return false;
                reasoning_utf8_buf.clear();
                think_phase = ThinkPhase::CONTENT;
                continue;
            }
            if (token == snap_think_start_id)
                continue;
            reasoning_utf8_buf += piece;
            for (;;) {
                auto tp = reasoning_utf8_buf.find("<think>");
                if (tp == std::string::npos)
                    break;
                reasoning_utf8_buf.erase(tp, 7);
            }
            auto end_pos = reasoning_utf8_buf.find("</think>");
            if (end_pos != std::string::npos) {
                if (!emit_thinking(reasoning_utf8_buf.substr(0, end_pos)))
                    return false;
                think_phase = ThinkPhase::CONTENT;
                std::string after = reasoning_utf8_buf.substr(end_pos + 8);
                reasoning_utf8_buf.clear();
                auto start = after.find_first_not_of("\n\r\t ");
                if (start != std::string::npos)
                    piece = after.substr(start);
                else
                    continue;
            } else {
                constexpr size_t kOverlap = 7;
                size_t complete = utf8_complete_len(reasoning_utf8_buf);
                if (complete > kOverlap) {
                    size_t emit_end = complete - kOverlap;
                    while (emit_end > 0 &&
                           (static_cast<unsigned char>(reasoning_utf8_buf[emit_end]) & 0xC0) == 0x80)
                        --emit_end;
                    if (emit_end > 0) {
                        std::string to_emit = reasoning_utf8_buf.substr(0, emit_end);
                        reasoning_utf8_buf = reasoning_utf8_buf.substr(emit_end);
                        if (!emit_thinking(to_emit))
                            return false;
                    }
                }
                continue;
            }
        }

        if (!content_started && think_phase == ThinkPhase::CONTENT) {
            auto ns = piece.find_first_not_of("\n\r\t ");
            if (ns == std::string::npos)
                continue;
            piece = piece.substr(ns);
            content_started = true;
        }

        if (use_reasoning) {
            if (token == snap_think_start_id) {
                if (think_reentries < kMaxThinkReentries) {
                    think_phase = ThinkPhase::REASONING;
                    think_reentries++;
                }
                continue;
            }
            if (token == snap_think_end_id) {
                continue;
            }
            for (;;) {
                auto p = piece.find("<think>");
                if (p != std::string::npos) {
                    piece.erase(p, 7);
                    continue;
                }
                p = piece.find("</think>");
                if (p != std::string::npos) {
                    piece.erase(p, 8);
                    continue;
                }
                break;
            }
            if (piece.empty())
                continue;
        }

        // Tool call body accumulation.
        if (has_tools && tool_phase == ToolPhase::TOOL_CALL_BODY) {
            tool_body_buf += piece;
            auto close_pos = tool_body_buf.find(tool_close_tag);
            if (close_pos != std::string::npos) {
                std::string body = tool_body_buf.substr(0, close_pos);
                auto bs = body.find_first_not_of("\n\r\t ");
                auto be = body.find_last_not_of("\n\r\t ");
                if (bs != std::string::npos && be != std::string::npos)
                    body = body.substr(bs, be - bs + 1);
                try {
                    json j = json::parse(body);
                    ParsedToolCall tc;
                    tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                    if (tpl_family == imp::ChatTemplateFamily::LLAMA3) {
                        tc.name = tool_fn_name;
                        tc.arguments = dump_safe(j);
                    } else {
                        tc.name = j.value("name", "");
                        if (j.contains("arguments"))
                            tc.arguments = dump_safe(j["arguments"]);
                        else {
                            json args = j;
                            args.erase("name");
                            tc.arguments = dump_safe(args);
                        }
                    }
                    if (!tc.name.empty()) {
                        validate_tool_call(tc, ctx.params.tools);
                        if (!tc.valid) {
                            fprintf(stderr, "[%s] tool-call arg validation failed: %s: %s\n",
                                    msg_id.c_str(), tc.name.c_str(), tc.error.c_str());
                        }
                        if (!emit_tool_use(tc))
                            return false;
                        stream_tool_calls.push_back(std::move(tc));
                        tool_calls_emitted = true;
                    }
                } catch (...) {
                    // Malformed tool-call JSON — skip this block and continue streaming
                }
                std::string after = tool_body_buf.substr(close_pos + tool_close_tag.size());
                tool_body_buf.clear();
                tool_phase = ToolPhase::CONTENT;
                if (!after.empty()) {
                    auto ws = after.find_first_not_of("\n\r\t ");
                    if (ws != std::string::npos)
                        piece = after.substr(ws);
                    else
                        continue;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        }

        if (has_tools && tool_phase == ToolPhase::TAG_SCANNING) {
            tool_tag_buf += piece;
            if (tpl_family != imp::ChatTemplateFamily::LLAMA3) {
                if (tool_tag_buf.size() >= 11) {
                    auto pos = tool_tag_buf.find("<tool_call>");
                    if (pos != std::string::npos) {
                        std::string before = tool_tag_buf.substr(0, pos);
                        if (!before.empty() && !emit_text(before))
                            return false;
                        tool_body_buf = tool_tag_buf.substr(pos + 11);
                        tool_close_tag = "</tool_call>";
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::TOOL_CALL_BODY;
                        continue;
                    }
                    const char* tc_tag = "<tool_call>";
                    bool could_match = true;
                    for (size_t k = 0; k < tool_tag_buf.size() && k < 11; k++)
                        if (tool_tag_buf[k] != tc_tag[k]) {
                            could_match = false;
                            break;
                        }
                    if (!could_match) {
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                    } else {
                        continue;
                    }
                } else {
                    const char* tc_tag = "<tool_call>";
                    bool could_match = true;
                    for (size_t k = 0; k < tool_tag_buf.size() && k < 11; k++)
                        if (tool_tag_buf[k] != tc_tag[k]) {
                            could_match = false;
                            break;
                        }
                    if (!could_match) {
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                    } else {
                        continue;
                    }
                }
            } else {
                if (tool_tag_buf.size() >= 10) {
                    auto fn_pos = tool_tag_buf.find("<function=");
                    if (fn_pos != std::string::npos) {
                        auto gt = tool_tag_buf.find('>', fn_pos + 10);
                        if (gt != std::string::npos) {
                            std::string before = tool_tag_buf.substr(0, fn_pos);
                            if (!before.empty() && !emit_text(before))
                                return false;
                            tool_fn_name = tool_tag_buf.substr(fn_pos + 10, gt - (fn_pos + 10));
                            tool_body_buf = tool_tag_buf.substr(gt + 1);
                            tool_close_tag = "</function>";
                            tool_tag_buf.clear();
                            tool_phase = ToolPhase::TOOL_CALL_BODY;
                            continue;
                        } else {
                            continue;
                        }
                    }
                }
                const char* fn_tag = "<function=";
                bool could_match = true;
                for (size_t k = 0; k < tool_tag_buf.size() && k < 10; k++)
                    if (tool_tag_buf[k] != fn_tag[k]) {
                        could_match = false;
                        break;
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

        if (has_tools && tool_phase == ToolPhase::CONTENT) {
            size_t lt_pos = piece.find('<');
            if (lt_pos != std::string::npos) {
                if (lt_pos > 0) {
                    std::string before = piece.substr(0, lt_pos);
                    if (stop_sequences.empty())
                        utf8_buf += before;
                    else
                        pending_text += before;
                }
                tool_tag_buf = piece.substr(lt_pos);
                tool_phase = ToolPhase::TAG_SCANNING;
                if (stop_sequences.empty() && !utf8_buf.empty()) {
                    size_t complete = utf8_complete_len(utf8_buf);
                    if (complete > 0) {
                        if (!emit_text(utf8_buf.substr(0, complete)))
                            return false;
                        utf8_buf.erase(0, complete);
                    }
                } else if (!stop_sequences.empty()) {
                    auto d = imp::stream::holdback_decision(pending_text, max_stop_len,
                                                            stop_sequences);
                    if (!flush_text(d.flush_len))
                        return false;
                    if (d.complete_match) {
                        text_stop_matched = true;
                        finish = "stop";
                        break;
                    }
                }
                continue;
            }
        }

        // Normal content emission.
        if (stop_sequences.empty()) {
            utf8_buf += piece;
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                if (!emit_text(utf8_buf.substr(0, complete)))
                    return false;
                utf8_buf.erase(0, complete);
            }
        } else {
            pending_text += piece;
            auto d = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(d.flush_len))
                return false;
            if (d.complete_match) {
                text_stop_matched = true;
                finish = "stop";
                break;
            }
        }

        if (finish)
            break;
    }

    // Flush trailing buffers.
    // Harmony: flush the final channel's tail (ends at EOS/<|return|> with no
    // trailing <|end|>); the other buffers below stay empty for harmony.
    if (harmony && !hm_buf.empty())
        hm_flush(/*force=*/true);

    if (think_phase == ThinkPhase::SCAN && !think_scan_buf.empty()) {
        utf8_buf += think_scan_buf;
        think_scan_buf.clear();
    }
    if (!reasoning_utf8_buf.empty()) {
        emit_thinking(reasoning_utf8_buf);
        reasoning_utf8_buf.clear();
    }
    if (tool_phase != ToolPhase::CONTENT && !tool_calls_emitted) {
        std::string leftover = tool_tag_buf + tool_body_buf;
        if (!leftover.empty())
            utf8_buf += leftover;
    }
    if (!utf8_buf.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(utf8_buf);
    if (!pending_text.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(pending_text);

    // Close any block still open.
    stop_block();

    if (!finish)
        finish = tool_calls_emitted ? "tool_calls" : "length";
    else if (tool_calls_emitted && strcmp(finish, "stop") == 0)
        finish = "tool_calls";

    // Map finish_reason -> Anthropic stop_reason.
    std::string stop_reason;
    if (strcmp(finish, "stop") == 0)
        stop_reason = "end_turn";
    else if (strcmp(finish, "length") == 0)
        stop_reason = "max_tokens";
    else if (strcmp(finish, "tool_calls") == 0)
        stop_reason = "tool_use";
    else if (strcmp(finish, "cancelled") == 0)
        stop_reason = "end_turn";
    else
        stop_reason = finish;

    // ---- message_delta + message_stop ------------------------------------
    // Cache accounting is only known after prefill ran, so it rides on the
    // final usage update instead of message_start.
    json delta_usage = {{"output_tokens", n_output_tokens}};
    {
        int cached_now = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
        int creation = cache_creation_tokens_(active_req, n_prompt_tokens);
        if (cached_now > 0 || creation > 0) {
            delta_usage["input_tokens"] = n_prompt_tokens - cached_now;
            delta_usage["cache_read_input_tokens"] = cached_now;
            delta_usage["cache_creation_input_tokens"] = creation;
        }
    }
    out.emit("message_delta",
             json{{"type", "message_delta"},
                  {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
                  {"usage", std::move(delta_usage)}});
    out.emit("message_stop", json{{"type", "message_stop"}});
    sink.done();

    // Metrics + log.
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += n_prompt_tokens;
    state.metrics.tokens_completion_total += n_output_tokens;
    state.metrics.tokens_cached_total += cached;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.last_ttft_ms = static_cast<int64_t>(ttft_ms);
    state.metrics.request_duration.observe(ms / 1000.0);
    if (n_output_tokens > 0)
        state.metrics.ttft.observe(ttft_ms / 1000.0);
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, msg_id, ctx.log_endpoint, ctx.log_client_ip,
                      ctx.log_raw_body, ms, n_prompt_tokens, n_output_tokens, finish, json());
    fprintf(stderr, "[%s] messages stream: %d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms)\n",
            msg_id.c_str(), n_prompt_tokens, n_output_tokens, ms, ttft_ms);
    return true;
}

}  // namespace

void handle_messages(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    namespace anth = imp_server::anthropic;

    // Capture original Anthropic request data for opt-in JSONL logging.
    const auto t_log_start = std::chrono::system_clock::now();
    const std::string log_endpoint = req.path;
    std::string log_client_ip = req.get_header_value("X-Forwarded-For");
    if (log_client_ip.empty())
        log_client_ip = req.remote_addr;
    const std::string log_raw_body = req.body;

    json anth_body;
    try {
        anth_body = json::parse(req.body);
    } catch (const std::exception& e) {
        res.status = 400;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "invalid_request_error"},
                      {"message", std::string("Invalid JSON: ") + e.what()}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    if (!anth_body.is_object()) {
        res.status = 400;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "invalid_request_error"}, {"message", "Request body must be a JSON object"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // Anthropic requires max_tokens — if it's missing, supply a sane default
    // matching the server's chat-completions default (handled downstream).
    std::string anth_model = anth_body.value("model", "");
    const bool want_stream = anth_body.value("stream", false);

    // Transform -> OpenAI body.
    json oai_body;
    try {
        oai_body = anth::anthropic_to_openai_body(anth_body);
    } catch (const std::exception& e) {
        res.status = 400;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "invalid_request_error"},
                      {"message", std::string("Failed to transform Anthropic body: ") + e.what()}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    // ---- Real streaming path -------------------------------------------
    // For stream=true we drive the same per-token batching-engine loop the
    // OpenAI streaming path uses and emit native Anthropic SSE events as
    // tokens arrive — TTFT is real first-token latency, not full-gen latency.
    if (want_stream) {
        // Build the chat request context from the transformed OpenAI body.
        httplib::Request shim_req = req;
        json shim_body = oai_body;
        shim_body["stream"] = true;
        shim_req.body = dump_safe(shim_body);
        shim_req.headers.erase("Content-Length");
        shim_req.headers.erase("content-length");

        ChatRequestContext ctx;
        g_in_anthropic_shim = true;  // suppress inner request-log (we log here)
        bool ok = parse_chat_request_params(shim_req, res, state, ctx) &&
                  snapshot_state_and_tokenize_(res, state, ctx);
        g_in_anthropic_shim = false;
        if (!ok) {
            // parse/snapshot set an OpenAI-shaped error on res; re-wrap as
            // an Anthropic error envelope.
            json parsed;
            try {
                parsed = json::parse(res.body);
            } catch (...) {
                parsed = {{"error", {{"message", res.body}, {"type", "invalid_request_error"}}}};
            }
            json out = {{"type", "error"},
                        {"error", parsed.value("error",
                                               json{{"type", "invalid_request_error"}, {"message", "bad request"}})}};
            res.set_content(dump_safe(out), "application/json");
            return;
        }

        // Restore Anthropic logging context (parse_chat_request_params set
        // these from the shim request; we log the outer Anthropic request).
        ctx.log_skip = false;
        ctx.log_endpoint = log_endpoint;
        ctx.log_client_ip = log_client_ip;
        ctx.log_raw_body = log_raw_body;
        ctx.t_log_start = t_log_start;

        // Vision streaming is unsupported (state is per-engine, not per-req).
        // snapshot_state_and_tokenize_ stops the batching engine for exclusive
        // vision access — restart it before bailing out.
        if (ctx.snap.has_vision_request) {
            {
                std::lock_guard<std::timed_mutex> lock(state.mtx);
                if (state.batching)
                    state.batching->start(state.ctx);
            }
            res.status = 400;
            json err = {{"type", "error"},
                        {"error",
                         {{"type", "invalid_request_error"},
                          {"message", "Streaming is not supported for vision/image requests"}}}};
            res.set_content(dump_safe(err), "application/json");
            return;
        }

        auto imp_req = std::make_shared<imp::Request>();
        imp_req->input_tokens = ctx.snap.tokens;
        imp_req->max_tokens = ctx.params.max_tokens;
        imp_req->temperature = ctx.params.temperature;
        imp_req->top_p = ctx.params.top_p;
        imp_req->top_k = ctx.params.top_k;
        imp_req->seed = ctx.params.seed;
        imp_req->pin_kv_prefix = ctx.params.cache_prompt;
        // This is the streaming /v1/messages path — stay on per-step decode so
        // SSE is real per-token rather than one burst at generation end (#754).
        imp_req->stream = true;
        imp_req->min_p = ctx.params.min_p;
        imp_req->typical_p = ctx.params.typical_p;
        imp_req->repetition_penalty = ctx.params.repetition_penalty;
        imp_req->frequency_penalty = ctx.params.frequency_penalty;
        imp_req->presence_penalty = ctx.params.presence_penalty;
        imp_req->repeat_last_n = ctx.params.repeat_last_n;
        imp_req->dry_multiplier = ctx.params.dry_multiplier;
        imp_req->dry_base = ctx.params.dry_base;
        imp_req->dry_allowed_length = ctx.params.dry_allowed_length;
        imp_req->dry_penalty_last_n = ctx.params.dry_penalty_last_n;
        imp_req->mirostat = ctx.params.mirostat;
        imp_req->mirostat_tau = ctx.params.mirostat_tau;
        imp_req->mirostat_eta = ctx.params.mirostat_eta;
        imp_req->logprobs = ctx.params.req_logprobs;
        imp_req->top_logprobs = ctx.params.top_logprobs;
        imp_req->json_mode = ctx.params.json_mode;
        imp_req->json_schema = ctx.params.json_schema_str;
        imp_req->has_tools = ctx.params.has_tools;
        imp_req->tpl_family = ctx.snap.tpl_family;
        imp_req->logit_bias = ctx.params.logit_bias;
        imp_req->think_budget = ctx.params.think_budget;
        imp_req->status = imp::RequestStatus::PENDING;

        auto server_req = std::make_shared<ServerRequest>();
        server_req->request = imp_req;
        {
            std::lock_guard<std::timed_mutex> lock(state.mtx);
            if (!state.batching || !state.batching->is_running()) {
                res.status = 503;
                json err = {{"type", "error"},
                            {"error",
                             {{"type", "server_error"}, {"message", "Inference engine not ready. Please retry."}}}};
                res.set_content(dump_safe(err), "application/json");
                return;
            }
            state.batching->submit(server_req);
        }

        std::string msg_id = anth::make_message_id(static_cast<uint64_t>(state.next_id.fetch_add(1)));
        ctx.t_start = std::chrono::high_resolution_clock::now();

        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [stream_ctx = std::move(ctx), &state, server_req, anth_model, msg_id](
                size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
                return run_anthropic_stream_(sink, stream_ctx, state, server_req, anth_model, msg_id);
            });
        return;
    }

    // ---- Non-streaming path: reuse the OpenAI handler via a shim --------
    // httplib::Request is a plain struct, safe to copy. Force stream=false on
    // the inner OpenAI call — we re-serialize the response as Anthropic JSON.
    httplib::Request shim_req = req;
    json shim_body = oai_body;
    shim_body["stream"] = false;
    shim_req.body = dump_safe(shim_body);
    shim_req.headers.erase("Content-Length");
    shim_req.headers.erase("content-length");

    httplib::Response shim_res;
    g_in_anthropic_shim = true;
    handle_chat_completions(shim_req, shim_res, state);
    g_in_anthropic_shim = false;

    // Propagate error envelopes (transform them to Anthropic error shape).
    // httplib::Response defaults status to -1 and auto-promotes to 200 only
    // at send time; any other non-200 code set by handle_chat_completions is
    // a real error we should forward.
    const bool is_error = shim_res.status >= 400;
    if (is_error) {
        res.status = shim_res.status;
        json parsed;
        try {
            parsed = json::parse(shim_res.body);
        } catch (...) {
            parsed = {{"error", {{"message", shim_res.body}, {"type", "server_error"}}}};
        }
        json out = {{"type", "error"},
                    {"error", parsed.value("error", json{{"type", "server_error"}, {"message", "unknown"}})}};
        res.set_content(dump_safe(out), "application/json");
        return;
    }

    json oai_response;
    try {
        oai_response = json::parse(shim_res.body);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"type", "error"},
                    {"error",
                     {{"type", "server_error"},
                      {"message", std::string("Upstream returned non-JSON: ") + e.what()}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    json anth_response = anth::openai_to_anthropic_response(oai_response, anth_model);

    // JSONL log — built from Anthropic shapes so /v1/messages clients see
    // exactly what they sent and what they got back.
    {
        auto t_end = std::chrono::system_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_log_start).count();
        int prompt_t = oai_response.value("usage", json::object()).value("prompt_tokens", 0);
        int completion_t = oai_response.value("usage", json::object()).value("completion_tokens", 0);
        std::string stop_reason = anth_response.value("stop_reason", "");
        std::string req_id = anth_response.value("id", make_completion_id(state));
        log_request_jsonl(state, /*skip=*/false, t_log_start, req_id, log_endpoint, log_client_ip,
                          log_raw_body, ms, prompt_t, completion_t,
                          stop_reason.empty() ? nullptr : stop_reason.c_str(), anth_response);
    }

    // Non-streaming requests are fully assembled above (the want_stream path
    // returned earlier with a native incremental SSE stream).
    res.status = 200;
    res.set_content(dump_safe(anth_response), "application/json");
}
