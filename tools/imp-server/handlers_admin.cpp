// /admin/suspend + /admin/resume — suspend the loaded model to host RAM so the
// GPU is fully free for other workloads, then resume serving in seconds.
//
// Suspend: drain in-flight requests (BatchingEngine::pause, same exclusive-
// access handshake as embeddings), D2H-snapshot the post-upload weight buffers
// (imp_weights_snapshot_capture), tear down engine + model (all VRAM freed),
// then imp_gpu_release() — with [suspend] device_reset (default) that includes
// cudaDeviceReset, so nvidia-smi shows ~0 MiB for this process.
//
// Resume: arm the snapshot and run the normal load path
// (load_model_into_state). The weight upload restores buffer bytes from the
// snapshot instead of re-reading + re-converting; everything else (KV cache,
// CUDA graphs, cuBLAS handles) is rebuilt fresh. Sessions/KV do not survive —
// only the weights stay warm.

#include "handlers.h"
#include "utils.h"

#include <chrono>
#include <cuda_runtime.h>

void handle_suspend(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);

    if (state.suspended.load()) {
        json body = {{"suspended", true}, {"note", "already suspended"}};
        res.set_content(dump_safe(body), "application/json");
        return;
    }
    if (!state.model_loaded()) {
        res.status = 409;
        json err = {{"error", {{"message", "no model loaded — nothing to suspend"},
                               {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }

    size_t vram_free_before = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free_before, &vram_total);

    // Drain: let in-flight generations FINISH (never cancels), then park the
    // worker. We hold state.mtx so no new request can be submitted meanwhile
    // (documented pause() contract, same as the embeddings exclusive window).
    if (state.batching && !state.batching->pause(/*timeout_ms=*/60000)) {
        state.batching->resume();
        res.status = 503;
        json err = {{"error", {{"message", "suspend aborted: in-flight requests did not drain "
                                           "within 60s — server keeps serving"},
                               {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
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
        res.status = (err == IMP_ERROR_OUT_OF_MEMORY) ? 507
                     : (err == IMP_ERROR_UNSUPPORTED) ? 501
                                                      : 500;
        json jerr = {{"error",
                      {{"message", std::string("suspend failed (nothing torn down): ") +
                                       imp_error_string(err) + " — see server log for details"},
                       {"type", "server_error"}}}};
        res.set_content(dump_safe(jerr), "application/json");
        return;
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
    state.lora_ids.clear();  // adapter ids died with the context; re-loaded at resume
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

    json body = {{"suspended", true},
                 {"model", model_name},
                 {"snapshot_bytes", imp_weights_snapshot_bytes(snap)},
                 {"vram_free_before", vram_free_before},
                 {"device_reset", device_reset}};
    if (!device_reset)
        body["vram_free_after"] = vram_free_after;
    res.set_content(dump_safe(body), "application/json");
}

void handle_resume(const httplib::Request& /*req*/, httplib::Response& res, ServerState& state) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);

    if (!state.suspended.load()) {
        json body = {{"suspended", false}, {"note", "not suspended"}};
        res.set_content(dump_safe(body), "application/json");
        return;
    }

    if (state.weight_snapshot)
        imp_weights_snapshot_arm(state.weight_snapshot);

    const auto t0 = std::chrono::steady_clock::now();
    std::string error = load_model_into_state(state, state.loaded_model_path);
    if (!error.empty()) {
        // Still suspended; the snapshot stays owned so a retry can re-arm it.
        res.status = 500;
        json err = {{"error", {{"message", "resume failed: " + error + " — still suspended"},
                               {"type", "server_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return;
    }
    const auto resume_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    // Re-load --lora adapters: main() only does this at startup.
    for (const auto& [name, path] : state.default_args.loras) {
        int32_t id = 0;
        if (imp_lora_load(state.ctx, path.c_str(), &id) != IMP_SUCCESS) {
            fprintf(stderr, "[resume] warning: failed to re-load LoRA adapter '%s' from %s\n",
                    name.c_str(), path.c_str());
            continue;
        }
        state.lora_ids[name] = id;
    }

    const int warm_hits = imp_weights_snapshot_hits(state.weight_snapshot);
    imp_weights_snapshot_free(state.weight_snapshot);
    state.weight_snapshot = nullptr;
    state.suspended.store(false);

    printf("[resume] %s resumed in %lld ms (%d uploads restored warm)\n", state.model_name.c_str(),
           static_cast<long long>(resume_ms), warm_hits);
    fflush(stdout);

    json body = {{"suspended", false},
                 {"model", state.model_name},
                 {"resume_ms", resume_ms},
                 {"warm_hits", warm_hits}};
    res.set_content(dump_safe(body), "application/json");
}
