// /admin/suspend: pause() drains in-flight, D2H-snapshots post-upload weight buffers, tears
// down model+engine, then imp_gpu_release ([suspend] device_reset default resets the device).
// /admin/resume: reload from the snapshot; only weights stay warm, KV/graphs/cuBLAS rebuild.

#include "handlers.h"
#include "sampling_fields.h"
#include "utils.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <cuda_runtime.h>

namespace {

json admin_error(const std::string& message, const char* type) {
    return {{"error", {{"message", message}, {"type", type}}}};
}

int64_t steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

int suspend_locked(ServerState& state, json& body) {
    if (state.suspended.load()) {
        body = {{"suspended", true}, {"note", "already suspended"}};
        return 200;
    }
    if (!state.model_loaded()) {
        body = admin_error("no model loaded — nothing to suspend", "invalid_request_error");
        return 409;
    }

    size_t vram_free_before = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free_before, &vram_total);

    // Drain: let in-flight generations FINISH (never cancels), then park the
    // worker. We hold state.mtx so no new request can be submitted meanwhile
    // (documented pause() contract, same as the embeddings exclusive window).
    if (state.batching && !state.batching->pause(/*timeout_ms=*/60000)) {
        state.batching->resume();
        body = admin_error("suspend aborted: in-flight requests did not drain "
                           "within 60s — server keeps serving",
                           "server_error");
        return 503;
    }

    // Snapshot while model + engine are still fully alive. On failure nothing
    // has been torn down — unpark the worker and keep serving.
    ImpWeightSnapshot snap = nullptr;
    const size_t headroom_mb =
        static_cast<size_t>(std::max(0, state.runtime_config.suspend.host_ram_headroom_mb));
    ImpError err = imp_weights_snapshot_capture(state.model, headroom_mb, &snap);
    if (err != IMP_SUCCESS) {
        if (state.batching)
            state.batching->resume();
        body = admin_error(std::string("suspend failed (nothing torn down): ") + imp_error_string(err) +
                               " — see server log for details",
                           "server_error");
        return (err == IMP_ERROR_OUT_OF_MEMORY) ? 507 : (err == IMP_ERROR_UNSUPPORTED) ? 501 : 500;
    }

    const std::string model_name = state.model_name;

    // Full teardown — mirrors load_model_into_state's unload half.
    if (state.batching) {
        state.batching->stop();
        state.batching.reset();
    }
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
    // Engine ids died with the context; names, server ids and paths stay for resume.
    for (auto& [name, entry] : state.loras)
        entry.engine_id = 0;
    state.publish_model_status(false, model_name);

    const bool device_reset = state.runtime_config.suspend.device_reset;
    // Measure the reclaim BEFORE imp_gpu_release when resetting: any CUDA call
    // after cudaDeviceReset would lazily re-create the primary context.
    size_t vram_free_after = 0;
    if (!device_reset)
        cudaMemGetInfo(&vram_free_after, &vram_total);
    imp_gpu_release(device_reset ? 1 : 0);
    if (!device_reset) {
        size_t f = 0;
        if (cudaMemGetInfo(&f, &vram_total) == cudaSuccess)
            vram_free_after = f;
    }

    state.weight_snapshot = snap;
    state.suspended.store(true);

    printf("[suspend] %s suspended to host RAM (%.2f GiB snapshot)%s\n", model_name.c_str(),
           imp_weights_snapshot_bytes(snap) / (1024.0 * 1024.0 * 1024.0),
           device_reset ? ", CUDA context reset" : "");
    fflush(stdout);

    body = {{"suspended", true},
            {"model", model_name},
            {"snapshot_bytes", imp_weights_snapshot_bytes(snap)},
            {"vram_free_before", vram_free_before},
            {"device_reset", device_reset}};
    if (!device_reset)
        body["vram_free_after"] = vram_free_after;
    return 200;
}

int resume_locked(ServerState& state, json& body) {
    if (!state.suspended.load()) {
        body = {{"suspended", false}, {"note", "not suspended"}};
        return 200;
    }

    if (state.weight_snapshot)
        imp_weights_snapshot_arm(state.weight_snapshot);

    const auto t0 = std::chrono::steady_clock::now();
    std::string error = load_model_into_state(state, state.loaded_model_path);
    if (!error.empty()) {
        // Still suspended; the snapshot stays owned so a retry can re-arm it.
        body = admin_error("resume failed: " + error + " — still suspended", "server_error");
        return 500;
    }
    const auto resume_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    // Re-load every registered adapter (--lora and /admin/lora/load); server ids stay stable.
    for (auto it = state.loras.begin(); it != state.loras.end();) {
        int32_t engine_id = 0;
        if (imp_lora_load(state.ctx, it->second.path.c_str(), &engine_id) != IMP_SUCCESS) {
            fprintf(stderr, "[resume] warning: failed to re-load LoRA adapter '%s' from %s, dropped\n",
                    it->first.c_str(), it->second.path.c_str());
            it = state.loras.erase(it);
            continue;
        }
        it->second.engine_id = engine_id;
        ++it;
    }

    const int warm_hits = imp_weights_snapshot_hits(state.weight_snapshot);
    imp_weights_snapshot_free(state.weight_snapshot);
    state.weight_snapshot = nullptr;
    state.suspended.store(false);
    state.idle_suspended.store(false);
    state.touch_activity();  // the idle clock restarts at resume, not at the last request

    printf("[resume] %s resumed in %lld ms (%d uploads restored warm)\n", state.model_name.c_str(),
           static_cast<long long>(resume_ms), warm_hits);
    fflush(stdout);

    body = {{"suspended", false},
            {"model", state.model_name},
            {"resume_ms", resume_ms},
            {"warm_hits", warm_hits}};
    return 200;
}

bool resume_if_idle_locked(ServerState& state, httplib::Response& res) {
    if (!state.suspended.load() || !state.idle_suspended.load())
        return true;
    json body;
    const int status = resume_locked(state, body);
    if (status != 200) {
        res.status = 503;
        res.set_content(dump_safe(admin_error("idle auto-resume failed: " +
                                                  body["error"]["message"].get<std::string>(),
                                              "server_error")),
                        "application/json");
        return false;
    }
    printf("[idle-unload] resumed on request in %lld ms\n",
           static_cast<long long>(body.value("resume_ms", int64_t{0})));
    fflush(stdout);
    return true;
}

bool idle_unload_tick(ServerState& state) {
    const int ttl_s = state.idle_unload_seconds.load();
    if (ttl_s <= 0 || state.suspended.load() || state.swapping.load() || g_draining.load())
        return false;
    // Never wait for the lock: a holder is a request or an admin call, i.e. not idle.
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::try_to_lock);
    if (!lock.owns_lock())
        return false;
    if (!state.model_loaded() || state.suspended.load())
        return false;
    // Queued or generating work (a long stream) is activity: the clock starts when it ends.
    if (state.inflight.inflight() > 0 || (state.batching && state.batching->queue_depth() > 0)) {
        state.touch_activity();
        return false;
    }
    if (steady_now_ms() - state.last_activity_ms.load() < static_cast<int64_t>(ttl_s) * 1000)
        return false;

    // Set first: a request blocked on mtx must see an idle suspend, never a manual one.
    state.idle_suspended.store(true);
    json body;
    const int status = suspend_locked(state, body);
    if (status != 200) {
        state.idle_suspended.store(false);
        const std::string why = body.contains("error") ? body["error"]["message"].get<std::string>() : "";
        if (status == 501) {
            // The model cannot be snapshotted; retrying every TTL would pause the worker for nothing.
            state.idle_unload_seconds.store(0);
            fprintf(stderr, "[idle-unload] disabled: %s\n", why.c_str());
        } else {
            state.touch_activity();  // retry after another full TTL
            fprintf(stderr, "[idle-unload] suspend failed (HTTP %d), retry in %d s: %s\n", status, ttl_s,
                    why.c_str());
        }
        return false;
    }
    printf("[idle-unload] suspended after %d s idle\n", ttl_s);
    fflush(stdout);
    return true;
}

void handle_suspend(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);
    json body;
    res.status = suspend_locked(state, body);
    // An operator suspend is sticky, also over an idle one: only POST /admin/resume ends it.
    if (res.status == 200)
        state.idle_suspended.store(false);
    res.set_content(dump_safe(body), "application/json");
}

void handle_resume(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);
    json body;
    res.status = resume_locked(state, body);
    res.set_content(dump_safe(body), "application/json");
}

namespace {

// Parses the admin body as a JSON object; on failure answers 400 and returns false.
bool parse_admin_body(const httplib::Request& req, httplib::Response& res, json& out) {
    out = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
    if (!out.is_object()) {
        send_json_error(res, 400, "invalid_request_error", "request body must be a JSON object");
        return false;
    }
    return true;
}

// Runs fn with the batching worker parked (drained, never cancelled); the adapter table and the
// executor's adapter pointer are only touched with no generation in flight.
template <typename Fn>
bool with_engine_parked(ServerState& state, httplib::Response& res, Fn&& fn) {
    if (state.batching && !state.batching->pause(state.runtime_config.server.model_swap_drain_ms)) {
        state.batching->resume();
        send_json_error(res, 503, "server_error",
                        "LoRA change aborted: in-flight requests did not drain, retry shortly");
        return false;
    }
    fn();
    if (state.batching)
        state.batching->resume();
    return true;
}

}  // namespace

void handle_lora_load(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    json body;
    if (!parse_admin_body(req, res, body))
        return;
    if (!body.contains("path") || !body["path"].is_string() || body["path"].get<std::string>().empty()) {
        send_json_error(res, 400, "invalid_request_error", "'path' (non-empty string) is required", "path");
        return;
    }
    const std::string path = body["path"].get<std::string>();
    std::string name;
    if (body.contains("name")) {
        if (!body["name"].is_string() || body["name"].get<std::string>().empty()) {
            send_json_error(res, 400, "invalid_request_error", "'name' must be a non-empty string", "name");
            return;
        }
        name = body["name"].get<std::string>();
    } else {
        std::filesystem::path p(path);
        if (!p.has_filename())
            p = p.parent_path();
        name = p.stem().string();
        if (name.empty()) {
            send_json_error(res, 400, "invalid_request_error", "cannot derive a name from 'path'; pass 'name'",
                            "name");
            return;
        }
    }

    std::lock_guard<std::timed_mutex> lock(state.mtx);
    if (state.suspended.load() && !state.idle_suspended.load()) {
        send_json_error(res, 409, "invalid_request_error",
                        "server suspended by POST /admin/suspend; POST /admin/resume first");
        return;
    }
    if (!resume_if_idle_locked(state, res))
        return;
    if (!state.model_loaded()) {
        send_json_error(res, 409, "invalid_request_error",
                        "no model loaded: a LoRA adapter attaches to a loaded base model");
        return;
    }
    if (auto it = state.loras.find(name); it != state.loras.end()) {
        send_json_error(res, 409, "invalid_request_error",
                        "LoRA adapter '" + name + "' is already loaded (id " + std::to_string(it->second.id) +
                            "); unload it first or pass another 'name'",
                        "name", "lora_already_loaded");
        return;
    }

    ImpError err = IMP_SUCCESS;
    int32_t engine_id = 0;
    if (!with_engine_parked(state, res, [&] { err = imp_lora_load(state.ctx, path.c_str(), &engine_id); }))
        return;
    if (err != IMP_SUCCESS) {
        send_json_error(res, 400, "invalid_request_error",
                        "LoRA adapter load failed for '" + path + "': " + imp_error_string(err) +
                            " (missing path, not a PEFT adapter, or shapes do not match the model; "
                            "see server log)",
                        "path", "lora_load_failed");
        return;
    }
    const int32_t id = state.next_lora_id++;
    state.loras[name] = ServerState::LoraEntry{id, engine_id, path};
    printf("[lora] loaded '%s' (id=%d) from %s\n", name.c_str(), id, path.c_str());
    fflush(stdout);
    res.set_content(dump_safe(json{{"id", id}, {"name", name}, {"path", path}, {"loaded", true}}),
                    "application/json");
}

void handle_lora_unload(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    json body;
    if (!parse_admin_body(req, res, body))
        return;
    const bool by_id = body.contains("id");
    if (by_id && !body["id"].is_number_integer()) {
        send_json_error(res, 400, "invalid_request_error", "'id' must be an integer", "id");
        return;
    }
    if (!by_id && !(body.contains("name") && body["name"].is_string())) {
        send_json_error(res, 400, "invalid_request_error", "'id' (integer) or 'name' (string) is required",
                        "id");
        return;
    }

    std::lock_guard<std::timed_mutex> lock(state.mtx);
    auto it = state.loras.end();
    if (by_id) {
        const int64_t want = body["id"].get<int64_t>();
        for (auto i = state.loras.begin(); i != state.loras.end(); ++i)
            if (i->second.id == want)
                it = i;
    } else {
        it = state.loras.find(body["name"].get<std::string>());
    }
    if (it == state.loras.end()) {
        const std::string what = by_id ? "id " + std::to_string(body["id"].get<int64_t>())
                                       : "'" + body["name"].get<std::string>() + "'";
        send_json_error(res, 404, "invalid_request_error", "LoRA adapter " + what + " is not loaded",
                        by_id ? "id" : "name", "lora_not_found");
        return;
    }

    // Suspended: nothing is on the device, dropping the entry is the whole unload.
    if (it->second.engine_id > 0 && state.ctx) {
        ImpError err = IMP_SUCCESS;
        if (!with_engine_parked(state, res, [&] { err = imp_lora_unload(state.ctx, it->second.engine_id); }))
            return;
        if (err != IMP_SUCCESS) {
            send_json_error(res, 500, "server_error",
                            std::string("LoRA adapter unload failed: ") + imp_error_string(err));
            return;
        }
    }
    const std::string name = it->first;
    const int32_t id = it->second.id;
    state.loras.erase(it);
    printf("[lora] unloaded '%s' (id=%d)\n", name.c_str(), id);
    fflush(stdout);
    res.set_content(dump_safe(json{{"id", id}, {"name", name}, {"unloaded", true}}), "application/json");
}

void handle_session_close(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (!session_id_valid(id)) {
        send_json_error(res, 400, "invalid_request_error", kSessionIdRule, "session_id");
        return;
    }
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (state.batching)
            state.batching->close_session(id);  // no engine (model-less, suspended): nothing is pinned
    }
    res.set_content(dump_safe(json{{"id", id}, {"object", "session"}, {"closed", true}}), "application/json");
}
