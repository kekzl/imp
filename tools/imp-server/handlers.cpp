#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"
#include "model/hf_hub.h"
#include "runtime/config.h"

#include <algorithm>
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
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void handle_health(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    // Never block unbounded on state.mtx: a long /v1/embeddings call holds it
    // across its whole computation, so an unbounded lock here would hang a
    // liveness probe (#889). Grab it with a short timeout and fall back to the
    // lock-free status snapshot on contention — a held lock means a request is
    // in flight, i.e. the server is alive and cannot be faulted right now.
    bool loaded = false;
    int queue = -1;
    bool faulted = false;
    std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
    if (lock.owns_lock()) {
        loaded = state.model_loaded();
        if (state.batching) {
            queue = state.batching->queue_depth();
            faulted = state.batching->faulted();
        }
        lock.unlock();
    } else {
        loaded = state.model_status_snapshot().loaded;  // queue_depth stays -1 (unknown)
    }
    json body = {{"status", faulted ? "unhealthy" : "ok"},
                 {"model_loaded", loaded},
                 {"queue_depth", queue},
                 // Suspended is HEALTHY (deliberate operator state, HTTP 200):
                 // the model is parked in host RAM, POST /admin/resume serves again.
                 {"suspended", state.suspended.load()}};
    if (faulted)
        res.status = 503;  // let orchestrators restart a wedged server (#874)
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

    // Snapshot state fields under a bounded lock (see #889 — do not block
    // behind a long /v1/embeddings holder). On contention, fall back to the
    // lock-free status snapshot rather than hang.
    bool loaded = false;
    std::string model_name;
    int max_seq_len = 0;
    {
        std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
        if (lock.owns_lock()) {
            loaded = state.model_loaded();
            model_name = state.model_name;
            max_seq_len = state.max_seq_len;
        } else {
            ServerState::ObsStatus snap = state.model_status_snapshot();
            loaded = snap.loaded;
            model_name = snap.model_name;
            max_seq_len = state.max_seq_len;  // plain int, set once at load
        }
    }

    // Expose what this server can actually serve. That used to mean the loaded
    // model alone, because listing the directory invited clients to request a
    // model the server then had to swap in mid-flight. With server.model_swap
    // those requests are served safely (drain + restore-on-failure), so the
    // rest of the directory is listed too — a harness cannot pick a model it
    // cannot see. The loaded one comes first and is the only one whose context
    // window is known; the others report none until they are loaded.
    //
    // The context window is not part of OpenAI's model object, but every major
    // OpenAI-compatible server bolts it on so clients can auto-detect it. We
    // follow both live conventions on the same object: vLLM's `max_model_len`
    // and llama.cpp's `meta.n_ctx_train`. See also GET /props and GET /info.
    if (loaded) {
        json model = {{"id", model_name},
                      {"object", "model"},
                      {"created", unix_timestamp()},
                      {"owned_by", "imp"},
                      {"loaded", true}};
        if (max_seq_len > 0) {
            model["max_model_len"] = max_seq_len;               // vLLM convention
            model["meta"] = {{"n_ctx_train", max_seq_len}};     // llama.cpp convention
        }
        data.push_back(std::move(model));
    }

    if (state.runtime_config.server.model_swap) {
        for (const auto& [fname, fpath] : scan_model_files(state.models_dir)) {
            (void)fpath;
            if (loaded && fname == model_name)
                continue;
            data.push_back({{"id", fname},
                            {"object", "model"},
                            {"created", unix_timestamp()},
                            {"owned_by", "imp"},
                            {"loaded", false}});
        }
    }

    json body = {{"object", "list"}, {"data", data}};
    res.set_content(dump_safe(body), "application/json");
}

// Snapshot {loaded, model_name, max_seq_len} for the context-length probes
// below, using the same bounded-lock / fall-back-to-snapshot discipline as the
// other observability endpoints (#889).
static void snapshot_ctx(ServerState& state, bool& loaded, std::string& model_name,
                         int& max_seq_len) {
    std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
    if (lock.owns_lock()) {
        loaded = state.model_loaded();
        model_name = state.model_name;
    } else {
        ServerState::ObsStatus snap = state.model_status_snapshot();
        loaded = snap.loaded;
        model_name = snap.model_name;
    }
    max_seq_len = state.max_seq_len;  // plain int, set once at load
}

// GET /props — llama.cpp-compatible context probe. llama.cpp clients read the
// context window from `default_generation_settings.n_ctx` (and a top-level
// `n_ctx`); we mirror both so an auto-detect path written for llama.cpp works
// unchanged against imp.
void handle_props(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    bool loaded = false;
    std::string model_name;
    int max_seq_len = 0;
    snapshot_ctx(state, loaded, model_name, max_seq_len);

    json body = {{"model_path", model_name},
                 {"total_slots", state.max_concurrent},
                 {"n_ctx", max_seq_len},
                 {"default_generation_settings", {{"n_ctx", max_seq_len}}}};
    res.set_content(dump_safe(body), "application/json");
}

// GET /info — Text-Generation-Inference-compatible context probe. TGI clients
// read `max_total_tokens` (context window) and `max_input_tokens` (largest
// prompt). We expose both so a TGI-shaped auto-detect path works against imp.
void handle_info(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    bool loaded = false;
    std::string model_name;
    int max_seq_len = 0;
    snapshot_ctx(state, loaded, model_name, max_seq_len);

    // TGI's max_input_tokens is the prompt cap; if the operator pinned one via
    // --max-input-tokens, honor it, otherwise it is (context - 1) to leave room
    // for at least one generated token.
    int max_input = state.max_input_tokens > 0 ? state.max_input_tokens
                    : max_seq_len > 0          ? max_seq_len - 1
                                               : 0;
    json body = {{"model_id", model_name},
                 {"max_total_tokens", max_seq_len},
                 {"max_input_tokens", max_input}};
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

// Resolve the model a request asks for, loading or swapping as needed.
//
// A first-generation auto-swap was removed once because it tore the engine down
// mid-stream (cancelling every in-flight request) and could leave the process
// with no model when the new one did not fit. Both causes are now closed, so
// swapping is back under `server.model_swap` (default on): in-flight
// generations DRAIN before teardown and are never cancelled (the /admin/suspend
// contract), and a failed load restores the previous model instead of leaving
// the server empty. What is not solved — and cannot be on 32 GB — is serving
// two models at once: a swap is serial and the requesting call pays one model
// load. Set server.model_swap=false for the strict single-model contract.
//
// Unknown names never load anything: the name must resolve inside the models
// directory (or as a HuggingFace repo id), otherwise this is a 404.
//
// Returns true if the requested model is loaded. Must be called with
// state.mtx held.
bool ensure_model_loaded(ServerState& state, const std::string& requested_model, httplib::Response& res) {
    // Suspended (/admin/suspend): the GPU is deliberately free — do NOT
    // auto-load a cold copy. Inference waits for POST /admin/resume.
    if (state.suspended.load()) {
        res.status = 503;
        json err = {{"error",
                     {{"message", "server suspended (weights parked in host RAM); POST /admin/resume "
                                  "to serve again"},
                      {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }
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

    // A different model was asked for. If it resolves inside the models
    // directory, swap to it rather than refusing: agent harnesses drive a big
    // model beside a small one, and 32 GB serves one at a time. An unresolvable
    // name falls through to the 404 below, so a typo never triggers a load.
    if (state.runtime_config.server.model_swap) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            // Drain first: in-flight generations finish, they are never
            // cancelled (same contract as /admin/suspend). state.mtx is held,
            // so no new request can be submitted while we tear down. On
            // timeout nothing has been touched and the current model keeps
            // serving.
            const int drain_ms = state.runtime_config.server.model_swap_drain_ms;
            if (state.batching && !state.batching->pause(drain_ms)) {
                state.batching->resume();
                res.status = 503;
                json err = {{"error",
                             {{"message", "model swap to '" + requested_model +
                                              "' aborted: in-flight requests did not drain within " +
                                              std::to_string(drain_ms) + "ms — still serving '" +
                                              state.model_name + "'"},
                              {"type", "server_error"}}}};
                res.set_content(dump_safe(err), "application/json");
                return false;
            }
            const std::string previous = state.model_name;
            const std::string previous_path = state.loaded_model_path;
            printf("[model-swap] %s -> %s\n", previous.c_str(), requested_model.c_str());
            fflush(stdout);
            // load_model_into_state owns the teardown of the previous model and
            // builds a fresh batching engine, so the paused worker goes with it.
            std::string error = load_model_into_state(state, path, json::object());
            if (!error.empty()) {
                // The previous model is already gone at this point. Put it back,
                // or the server is left serving nothing — that (not the swap
                // itself) is what made the historical auto-swap unsafe.
                std::string restore = "not attempted";
                if (!previous_path.empty()) {
                    std::string rerr = load_model_into_state(state, previous_path, json::object());
                    restore = rerr.empty() ? "previous model restored" : "restore ALSO failed: " + rerr;
                }
                printf("[model-swap] failed: %s (%s)\n", error.c_str(), restore.c_str());
                fflush(stdout);
                res.status = 503;
                json err = {{"error",
                             {{"message", "model swap to '" + requested_model + "' failed: " + error +
                                              " — " + restore},
                              {"type", "server_error"}}}};
                res.set_content(dump_safe(err), "application/json");
                return false;
            }
            printf("[model-swap] now serving %s\n", requested_model.c_str());
            fflush(stdout);
            return true;
        }
    }

    res.status = 404;
    json err = {{"error",
                 {{"message",
                   "The model '" + requested_model + "' is not available; this server is serving '" +
                       state.model_name + "'" +
                       (state.runtime_config.server.model_swap
                            ? " and '" + requested_model + "' was not found in the models directory"
                            : " and model swapping is disabled (server.model_swap=false)")},
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

    // Hard per-process VRAM cap for multi-server-per-GPU deployments.
    // Precedence: --vram-budget CLI flag > [runtime] vram_budget_mb from
    // imp.conf (the engine bridges the imp.conf key itself when this is 0).
    if (args.vram_budget_mb > 0)
        config.vram_budget_mb = args.vram_budget_mb;

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
    // Publish "not loaded" for the whole load window; republished as loaded
    // once the context is up and the name is set (#889 observability snapshot).
    state.publish_model_status(false, "");

    // Auto-detect format from path
    ImpModelFormat format = imp::is_safetensors_dir(path) ? IMP_FORMAT_SAFETENSORS : IMP_FORMAT_GGUF;

    // Load model. speculative.mtp_k > 0 opts into loading the MTP draft-head
    // sidecar (~1.6 GiB VRAM; SafeTensors models shipping
    // model_mtp.safetensors) so MTP drafting can be enabled below.
    const int mtp_k = state.runtime_config.speculative.mtp_k;
    ImpError err = imp_model_load_ex(path.c_str(), format, /*load_mtp_head=*/mtp_k > 0 ? 1 : 0,
                                     &state.model);
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

    // MTP drafting for the verify loop (speculative.mtp_k). Best-effort: a
    // model without a loadable head just runs without MTP drafts.
    if (mtp_k > 0) {
        ImpError mtp_err = imp_enable_mtp_spec_decode(state.ctx, mtp_k);
        if (mtp_err != IMP_SUCCESS)
            fprintf(stderr, "Warning: speculative.mtp_k=%d requested but MTP enable failed (%s); "
                            "continuing without MTP drafts\n",
                    mtp_k, imp_error_string(mtp_err));
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
    state.loaded_model_path = path;  // remembered for /admin/resume
    state.publish_model_status(true, state.model_name);

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

    if (body.contains("max_completion_tokens") && !body["max_completion_tokens"].is_null()) {
        int mt = body["max_completion_tokens"].get<int>();
        if (mt < 1) {
            res.status = 400;
            json err = {{"error",
                         {{"message", "\"max_completion_tokens\" must be at least 1"},
                          {"type", "invalid_request_error"}}}};
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
