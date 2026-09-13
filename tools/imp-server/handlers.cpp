#include "runtime/engine.h"
#include "handlers.h"
#include "handlers_internal.h"
#include "model_name_policy.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"

#include "api/imp_internal.h"
#include "common/mtp_auto.h"
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
std::atomic<bool> g_draining{false};

void signal_handler(int /*sig*/) {
    fprintf(stderr, "\nShutting down...\n");
    // Order matters: the flag first, so a request that httplib accepted before
    // stop() lands gets a 503 from pre-routing rather than a generation that
    // the teardown below will cancel.
    g_draining.store(true, std::memory_order_relaxed);
    if (auto* svr = g_server.exchange(nullptr, std::memory_order_relaxed))
        svr->stop();
}

std::string make_completion_id(ServerState& state) {
    return "imp-" + std::to_string(state.next_id.fetch_add(1));
}

// Request id: ties a response/error body to its JSONL request-log line (#1561). Shares the
// completion-id counter so the two ids cannot collide.
std::string make_request_id(ServerState& state) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "req_imp_%016llx",
                  static_cast<unsigned long long>(state.next_id.fetch_add(1)));
    return std::string(buf);
}

int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void handle_health(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    // Never block unbounded on state.mtx (a long /v1/embeddings call holds it, #889): use a short
    // timeout and fall back to the lock-free status snapshot on contention.
    bool loaded = false;
    int queue = -1;
    bool faulted = false;
    bool kv_floored = false;
    int kv_blocks = -1;
    int kv_block_size = -1;
    int kv_ceiling = -1;
    bool kv_growable = false;
    double kv_bandwidth = 0.0;
    std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
    if (lock.owns_lock()) {
        loaded = state.model_loaded();
        if (state.batching) {
            queue = state.batching->queue_depth();
            faulted = state.batching->faulted();
        }
        // Reports KV capacity, not occupancy (invariant I7): callers need it before sending a long
        // prompt; harnesses need it to refuse measuring a clamped server.
        if (state.ctx && state.ctx->engine) {
            kv_floored = state.ctx->engine->kv_pool_floored();
            if (const auto* kv = state.ctx->engine->kv_cache()) {
                kv_blocks = kv->total_blocks();
                kv_block_size = kv->block_size();
                kv_ceiling = kv->ceiling_blocks();
                kv_growable = kv->growable();
                kv_bandwidth = kv->residency_gbps();
            }
        }
        lock.unlock();
    } else {
        loaded = state.model_status_snapshot().loaded;  // queue_depth stays -1 (unknown)
    }
    // can_grow: a growable pool below its ceiling heals as the card frees (wait, do not restart);
    // distinguish from a pool stuck at its ceiling, which never will.
    const bool can_grow = kv_ceiling > kv_blocks;
    const std::string unservable = health_unservable_reason(faulted, kv_floored && !can_grow, kv_blocks,
                                                            kv_block_size);
    json body = {{"status", unservable.empty() ? "ok" : "unhealthy"},
                 {"model_loaded", loaded},
                 {"queue_depth", queue},
                 // Suspended is HEALTHY (deliberate operator state, HTTP 200):
                 // the model is parked in host RAM, POST /admin/resume serves again.
                 {"suspended", state.suspended.load()}};
    if (kv_blocks >= 0) {
        body["kv_blocks_total"] = kv_blocks;
        body["kv_block_size"] = kv_block_size;
        body["kv_capacity_tokens"] = static_cast<long long>(kv_blocks) * kv_block_size;
        // The ceiling a growable pool may still reach. Equal to the total for a
        // fixed pool, so a caller can compare the two rather than test a flag.
        body["kv_ceiling_blocks"] = kv_ceiling;
        // kv_pool_growable disambiguates ceiling==total: a fixed pool at capacity vs a growable pool
        // already at its ceiling look identical otherwise, and only one of them ever heals.
        body["kv_pool_growable"] = kv_growable;
        // kv_pool_bandwidth_gbps, GB/s measured at pool init (read+write). ~1500 = resident on this
        // card, ~240 = WDDM-spilled to host memory at that fraction of throughput. 0 = not measured.
        body["kv_pool_bandwidth_gbps"] = kv_bandwidth;
    }
    // #1537: name why an MTP head this checkpoint ships was declined at load (mtp_k=auto is a
    // per-server regime decision) - a container operator never sees the startup log.
    if (state.model && state.model->model && state.model->model->mtp_head_available_unloaded_) {
        body["mtp_head_available"] = true;
        body["mtp_head_hint"] =
            state.runtime_config.speculative.mtp_k == 0 && state.resolved_max_batch_size != 1
                ? "this checkpoint ships an MTP head; speculative.mtp_k=auto left it unloaded "
                  "because this server takes concurrent requests (MTP drafts for one request at "
                  "a time, and the head's 0.79 GiB comes out of the batch slot budget). "
                  "--set speculative.mtp_k=2 --set speculative.ngram=false forces it: measured "
                  "+27-30% single-stream decode on Qwen3.8-27B-NVFP4 thinking chats (2026-08-27)."
                : "this checkpoint ships an MTP head that is not loaded "
                  "(speculative.mtp_k=0). --set speculative.mtp_k=2 --set "
                  "speculative.ngram=false measured +27-30% single-stream decode on "
                  "Qwen3.8-27B-NVFP4 thinking chats (2026-08-27, adaptive chain "
                  "depth), for the head's VRAM (0.79 GiB).";
    }

    if (!unservable.empty()) {
        // A client has to tell a permanent 503 from a transient one, or it
        // retries a condition retrying cannot fix and burns its budget doing
        // it. The code is the stable half; the detail is for the operator.
        body["code"] = health_unservable_code(faulted, kv_floored && !can_grow);
        body["detail"] = unservable;
        res.status = 503;  // let orchestrators restart a wedged server (#874)
    }
    res.set_content(dump_safe(body), "application/json");
}

// Readiness must be a status code, not a JSON field: an orchestrator keyed on the code got
// traffic during model-less/suspend/swap states because /health said 200 for all three
// (AUDIT_arch_2026 E-5). No mutex: the swap holds it for the whole load, when this must answer.
void handle_ready(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    const char* code = nullptr;
    if (g_draining.load(std::memory_order_relaxed))
        code = "draining";
    else if (state.batching && state.batching->faulted())
        code = "engine_faulted";  // process-wide CUDA context, a swap does not clear it
    else if (state.swapping.load())
        code = "swapping";
    else if (state.suspended.load())
        code = "suspended";
    else if (!state.model_status_snapshot().loaded)
        code = "no_model";
    json body = {{"ready", code == nullptr},
                 {"model_loaded", state.model_status_snapshot().loaded},
                 {"suspended", state.suspended.load()}};
    if (code) {
        body["code"] = code;
        res.status = 503;
    }
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
    // kv_capacity_tokens = what the pool can actually hold, not max_seq_len (the plan); they can
    // differ hugely on a tight card (#1542). Growable pools report their ceiling, not current commit.
    long long kv_capacity_tokens = -1;
    {
        std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
        if (lock.owns_lock()) {
            loaded = state.model_loaded();
            model_name = state.model_name;
            max_seq_len = state.max_seq_len;
            if (state.ctx && state.ctx->engine) {
                if (const auto* kv = state.ctx->engine->kv_cache())
                    kv_capacity_tokens = kv_capacity_ceiling_tokens(kv->total_blocks(), kv->ceiling_blocks(),
                                                                    kv->block_size());
            }
        } else {
            ServerState::ObsStatus snap = state.model_status_snapshot();
            loaded = snap.loaded;
            model_name = snap.model_name;
            max_seq_len = state.max_seq_len;  // plain int, set once at load
        }
    }
    max_seq_len = servable_context_tokens(max_seq_len, kv_capacity_tokens);

    // Lists every model in the models dir (not just the loaded one) now that server.model_swap
    // makes swap-in-flight safe. Loaded model first with known context window; others report none.
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

// GET /v1/models/{id}: client.models.retrieve(...) 404'd for lack of a path-param route (#1599).
void handle_model_retrieve(const httplib::Request& req, httplib::Response& res, ServerState& state,
                           const std::string& model_id) {
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
            max_seq_len = state.max_seq_len;
        }
    }

    if (loaded && model_id == model_name) {
        json model = {{"id", model_name},
                      {"object", "model"},
                      {"created", unix_timestamp()},
                      {"owned_by", "imp"},
                      {"loaded", true}};
        if (max_seq_len > 0) {
            model["max_model_len"] = max_seq_len;
            model["meta"] = {{"n_ctx_train", max_seq_len}};
        }
        res.set_content(dump_safe(model), "application/json");
        return;
    }

    // A model on disk that this server would swap in. Same answer the list
    // endpoint gives for it, so the two cannot disagree.
    if (state.runtime_config.server.model_swap) {
        for (const auto& [fname, fpath] : scan_model_files(state.models_dir)) {
            (void)fpath;
            if (fname != model_id)
                continue;
            json model = {{"id", fname},
                          {"object", "model"},
                          {"created", unix_timestamp()},
                          {"owned_by", "imp"},
                          {"loaded", false}};
            res.set_content(dump_safe(model), "application/json");
            return;
        }
    }

    // OpenAI answers a missing model with 404 and a typed envelope naming the
    // parameter, not the generic unmatched-route body.
    res.status = 404;
    json err = {{"error",
                 {{"message", "The model '" + sanitize_for_echo(model_id, 128) + "' does not exist"},
                  {"type", "invalid_request_error"},
                  {"param", "model"},
                  {"code", "model_not_found"}}}};
    res.set_content(dump_safe(err), "application/json");
}

// Snapshot {loaded, model_name, max_seq_len} for the context-length probes
// below, using the same bounded-lock / fall-back-to-snapshot discipline as the
// other observability endpoints (#889).
static void snapshot_ctx(ServerState& state, bool& loaded, std::string& model_name,
                         int& max_seq_len) {
    long long kv_capacity_tokens = -1;
    std::unique_lock<std::timed_mutex> lock(state.mtx, kObservabilityLockTimeout);
    if (lock.owns_lock()) {
        loaded = state.model_loaded();
        model_name = state.model_name;
        if (state.ctx && state.ctx->engine) {
            if (const auto* kv = state.ctx->engine->kv_cache())
                kv_capacity_tokens = kv_capacity_ceiling_tokens(kv->total_blocks(), kv->ceiling_blocks(),
                                                                kv->block_size());
        }
    } else {
        ServerState::ObsStatus snap = state.model_status_snapshot();
        loaded = snap.loaded;
        model_name = snap.model_name;
    }
    max_seq_len = state.max_seq_len;  // plain int, set once at load
    // All three probes must answer the same question with the same number
    // (docs/usage.md says so), so the pool clamp applies here too (#1542).
    max_seq_len = servable_context_tokens(max_seq_len, kv_capacity_tokens);
}

// GET /props (llama.cpp-compatible): mirrors default_generation_settings.n_ctx and top-level
// n_ctx so llama.cpp auto-detect clients work unchanged against imp.
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

// find_model_path: name resolves only as a models_dir basename or HF cache id, never as a raw
// filesystem path (AUDIT_arch_2026 F2-1: {"model":"/any/x.gguf"} loaded and evicted the resident model).
std::string find_model_path(const ServerState& state, const std::string& name) {
    using imp_server::ModelNameKind;
    switch (imp_server::classify_model_name(name)) {
        case ModelNameKind::Basename: {
            auto available = scan_model_files(state.models_dir);
            for (const auto& [fname, fpath] : available) {
                if (fname == name)
                    return fpath;
            }
            return "";
        }
        case ModelNameKind::HfRepoId: {
            const std::string cache = imp::hf_cache_dir();
            if (cache.empty())
                return "";
            ImpModelFormat fmt;
            std::string resolved = imp::resolve_model_auto(name, fmt);
            // resolve_model_path() tries the id as a local path first; a
            // relative "org/repo" directory under the CWD would satisfy it.
            if (resolved.empty() || !imp_server::path_within(cache, resolved))
                return "";
            return resolved;
        }
        case ModelNameKind::Rejected:
            return "";
    }
    return "";
}

// ensure_model_loaded: resolves/loads/swaps as needed. server.model_swap (default on) drains
// in-flight generations before teardown and restores the prior model on a failed load. Unknown
// names 404 without loading anything. Caller must hold state.mtx.
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
        std::string error = load_model_into_state(state, path);
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

    // A different resolvable model triggers a swap rather than a refusal (agent harnesses run a big
    // model beside a small one); an unresolvable name falls through to 404, never a load.
    if (state.runtime_config.server.model_swap) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            // Swap drains in-flight generations first (never cancelled, same contract as /admin/suspend).
            // state.mtx held throughout; on drain timeout nothing is torn down and the current model keeps
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
            std::string error = load_model_into_state(state, path);
            if (!error.empty()) {
                // The previous model is already gone at this point. Put it back,
                // or the server is left serving nothing — that (not the swap
                // itself) is what made the historical auto-swap unsafe.
                std::string restore = "not attempted";
                if (!previous_path.empty()) {
                    std::string rerr = load_model_into_state(state, previous_path);
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

int resolve_max_batch_size(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg) {
    return args.max_batch_size > 0 ? args.max_batch_size : runtime_cfg.runtime.max_batch_size;
}

// Build ImpConfig from the server args + imp.conf.
// Engine auto-detects max_seq_len, max_batch_size, KV dtype, FP8 prefill, NVFP4 decode.
ImpConfig build_config(const ServerArgs& args, const imp::RuntimeConfig& runtime_cfg,
                       const std::string& model_path) {
    (void)model_path;
    ImpConfig config = imp_config_default();

    config.device_id = args.device;

    // max_batch_size precedence: --max-batch CLI > [runtime] max_batch_size (imp.conf) > 0 (engine
    // auto-sizes from weight footprint; >20 GiB MoE auto-picks 1). All paths must resolve here.
    config.max_batch_size = resolve_max_batch_size(args, runtime_cfg);

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

    // KV cache dtype: explicit CLI pins only (engine resolves `auto` itself)
    if (args.kv_fp8)
        config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (args.kv_int8)
        config.kv_cache_dtype = IMP_DTYPE_INT8;
    if (args.kv_int4)
        config.kv_cache_dtype = IMP_DTYPE_INT4;
    if (args.kv_nvfp4)
        config.kv_cache_dtype = IMP_DTYPE_NVFP4;
    if (args.kv_mxfp4)
        config.kv_cache_dtype = IMP_DTYPE_MXFP4_KV;

    // Default chunk = -1 → engine resolver picks per-arch default (512 for
    // full-attention + FP16/FP8 KV, 0 for Gemma-4 / hybrid / sub-byte KV).
    // Pass 0 via --prefill-chunk-size 0 to force single-chunk for all archs.
    config.prefill_chunk_size = args.prefill_chunk_size;

    config.use_nvfp4_decode = args.decode_nvfp4;

    if (args.mxfp4_prefill)
        config.use_mxfp4_prefill = 1;
    if (args.dual_path_quant)
        config.dual_path_quant = 1;
    if (args.min_kv_tokens > 0)
        config.min_kv_tokens = args.min_kv_tokens;

    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();

    // [server] prefix_cache defaults ON since the #536/#538 stale-block-table fix; PrefixCacheE2ETest
    // is the ship gate. Auto-disabled for recurrent (SSM/GDN) models in the engine.
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
std::string load_model_into_state(ServerState& state, const std::string& path) {
    // GET /ready answers 503 "swapping" for the whole of this function: the
    // resident engine goes away on the next lines and the new one is not up
    // until the last. Cleared on every exit path.
    state.swapping.store(true);
    struct SwapScope {
        std::atomic<bool>& flag;
        ~SwapScope() { flag.store(false); }
    } swap_scope{state.swapping};

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

    // speculative.mtp_k=auto engages only for a single-stream server: read the RESOLVED batch size
    // (imp.conf/--set/--max-batch all count), not the raw CLI flag, or auto declines wrongly
    // (2026-08-29: cost 88.1 -> 117.2 tok/s on such a config).
    state.resolved_max_batch_size = resolve_max_batch_size(state.default_args, state.runtime_config);
    int mtp_k = imp::tools::mtp_auto_request_k(state.runtime_config, state.resolved_max_batch_size);
    ImpError err = imp_model_load_ex(path.c_str(), format, /*load_mtp_head=*/mtp_k > 0 ? 1 : 0,
                                     &state.model);
    if (err != IMP_SUCCESS) {
        std::string msg = std::string("Failed to load model: ") + imp_error_string(err);
        state.model = nullptr;
        return msg;
    }
    // Resolve the pair against what the load produced, BEFORE the pending
    // config is stashed for Engine::init below.
    imp::tools::mtp_auto_finalize(state.runtime_config, mtp_k,
                                  state.model->model->mtp_.has_value() &&
                                      state.model->model->mtp_->loaded);
    // The enable call below takes the RESOLVED depth, not the requested one.
    mtp_k = state.runtime_config.speculative.mtp_k;

    // Re-stash the runtime config before each Engine construction so take_pending_runtime_config()
    // picks it up: a runtime model load rebuilds the Engine and consumes the pending slot.
    imp::set_pending_runtime_config(state.runtime_config);
    ImpConfig config = build_config(state.default_args, state.runtime_config, path);
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
    // head_present: true when the checkpoint ships MTP tensors this load declined to upload, so a
    // concurrency-driven decline is distinguishable from "no head at all".
    state.armed_mtp_k.store(state.ctx ? state.ctx->engine->mtp_spec_decode_k() : 0,
                            std::memory_order_relaxed);
    state.mtp_head_loaded.store(state.model->model->mtp_.has_value() &&
                                    state.model->model->mtp_->loaded,
                                std::memory_order_relaxed);
    state.mtp_head_present.store(state.mtp_head_loaded.load(std::memory_order_relaxed) ||
                                     state.model->model->mtp_head_available_unloaded_,
                                 std::memory_order_relaxed);

    // Strip trailing path separators first: a directory path ending in "/" must still yield a
    // non-empty model id (#756), or the model becomes unaddressable over the HTTP API.
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

    std::string chat_tpl_name = state.default_args.chat_template;
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

    // Gate prompt length on the engine's EFFECTIVE allocated context, not the model's declared max:
    // the engine VRAM-auto-sizes lower, and gating on the model max let an over-long prompt overrun
    // the KV/position buffers (SIGSEGV instead of a 400).
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

    if (!validate_constraints(body, res))
        return false;

    if (!validate_content_parts(body, res))
        return false;

    if (!validate_tool_choice(body, res))
        return false;

    return true;
}

