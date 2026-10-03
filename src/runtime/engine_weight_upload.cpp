// Engine init phase: weight upload to VRAM via Model::upload_weights_gpu
// (src/model/weight_upload.cpp, pre-dequant via executor_pre_dequant.cpp). Also
// reserves L2 cache, tunes cudaMallocAsync, sizes the expert-upload VRAM reserve, wires StreamingLLM / host-offload guards, and hashes the model identity for the persisted prefix cache.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "core/cuda_raii.h"
#include "core/logging.h"
#include "memory/ssm_state_size.h"
#include "model/layer_host_keep.h"
#include "memory/vram_query.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <utility>
#include <vector>

namespace imp {

// Stable model identity hash: FNV-1a over config scalars + weight bytes
// (LM head, embeddings, layer-0/mid Q proj) - distinguishes same-shape fine-tunes.
// Gates the persisted prefix cache only (cold path, twice per process).
uint64_t Engine::model_fingerprint_() const {
    auto fnv = [](uint64_t h, const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 0x100000001b3ULL;
        }
        return h;
    };
    uint64_t h = 0xcbf29ce484222325ULL;
    const auto& c = model_->config();
    const uint32_t ids[] = {
        static_cast<uint32_t>(std::to_underlying(c.arch)), static_cast<uint32_t>(c.n_layers),
        static_cast<uint32_t>(c.n_heads), static_cast<uint32_t>(c.n_kv_heads),
        static_cast<uint32_t>(c.d_model), static_cast<uint32_t>(c.d_ff),
        static_cast<uint32_t>(c.vocab_size), static_cast<uint32_t>(c.head_dim),
        static_cast<uint32_t>(c.n_experts), static_cast<uint32_t>(c.n_experts_active),
        static_cast<uint32_t>(c.is_nvfp4_prequant), static_cast<uint32_t>(c.is_mxfp4_prequant)};
    h = fnv(h, ids, sizeof(ids));
    h = fnv(h, &c.rope_theta, sizeof(c.rope_theta));

    auto sample = [&](const Tensor& t) {
        if (!t.data || t.dropped_source)  // freed after load (FP8 LM head, GDN sources)
            return;
        size_t n = std::min<size_t>(t.nbytes(), 512);
        if (n == 0)
            return;
        std::vector<uint8_t> buf(n);
        if (t.on_device) {
            if (cudaMemcpy(buf.data(), t.data, n, cudaMemcpyDeviceToHost) != cudaSuccess)
                return;
        } else {
            std::memcpy(buf.data(), t.data, n);
        }
        h = fnv(h, buf.data(), n);
    };
    sample(model_->output_proj());
    sample(model_->token_embedding());
    sample(model_->layer(0).wq);
    if (c.n_layers > 1)
        sample(model_->layer(c.n_layers / 2).wq);
    return h;
}

namespace {
// Bytes of model on disk, the one comparison that can tell a weight upload
// from a card someone else is on. A checkpoint directory sums its weight
// shards (config/tokenizer JSON is noise); 0 = could not tell.
size_t model_source_bytes(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (path.empty())
        return 0;
    if (fs::is_regular_file(path, ec))
        return static_cast<size_t>(fs::file_size(path, ec));
    if (!fs::is_directory(path, ec))
        return 0;
    size_t total = 0;
    for (const auto& e : fs::directory_iterator(path, ec)) {
        if (ec)
            return 0;
        if (!e.is_regular_file(ec))
            continue;
        const std::string ext = e.path().extension().string();
        if (ext == ".safetensors" || ext == ".gguf" || ext == ".bin")
            total += static_cast<size_t>(e.file_size(ec));
    }
    return ec ? 0 : total;
}

// cudaLimitPersistingL2CacheSize is a per-primary-context device limit that
// survives Engine teardown, so a previous model's reservation persists.
// Reset it first, or the next access-policy-window set can return cudaErrorInvalidValue and poison the stream.
void reserve_persisting_l2() {
    cudaDeviceProp prop{};
    size_t max_persist = 0;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        max_persist = prop.persistingL2CacheMaxSize;
    else
        IMP_LOG_WARN("L2 persisting cache: device query failed, nothing reserved");
    if (max_persist > 0) {
        if (cudaCtxResetPersistingL2Cache() != cudaSuccess)
            IMP_LOG_DEBUG("L2 persisting cache: reset unsupported");
        (void)cudaGetLastError();
        size_t reserve = max_persist * 3 / 4;
        if (const cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve); e == cudaSuccess)
            IMP_LOG_INFO("L2 persisting cache: reserved %zu MB / %zu MB total", reserve >> 20,
                         max_persist >> 20);
        else
            IMP_LOG_WARN("L2 persisting cache: reserve %zu MB failed: %s", reserve >> 20, cudaGetErrorString(e));
    }
}

// Tunes the default cudaMallocAsync pool to retain freed memory instead of
// calling cuMemUnmap on every free (default threshold 0): many paths (prefill
// metadata, MoE scratch, spec block tables, vision staging) use this pool; KV cache and workspaces own their memory separately (VRAMAllocator/cudaMalloc).
void retain_async_pool_memory() {
    cudaMemPool_t default_pool = nullptr;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetDefaultMemPool(&default_pool, dev) == cudaSuccess &&
        default_pool != nullptr) {
        uint64_t threshold = UINT64_MAX;
        IMP_CUDA_CHECK_LOG(cudaMemPoolSetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
}

// The weight-upload reserve as a function of max_batch_size (memory/upload_reserve.h):
// workspace + capped KV + SSM/GDN state slots + recurrent snapshot store + 256 MiB safety.
UploadReserve upload_reserve_for(const Model& model, const EngineConfig& cfg, const RuntimeConfig& rc,
                                 size_t workspace_bytes, int reserved_state_slots) {
    const auto& mcfg = model.config();
    UploadReserve r;
    r.fixed_bytes = workspace_bytes + (256ULL << 20);
    int head_dim_est = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
    int est_bs = cfg.kv_block_size > 0 ? cfg.kv_block_size : kKVBlockSize;
    int blocks_per_seq = (cfg.max_seq_len + est_bs - 1) / est_bs;
    int n_attn = 0;
    for (int i = 0; i < mcfg.n_layers; i++)
        if (model.layer(i).wq.data != nullptr)
            n_attn++;
    if (n_attn == 0)
        n_attn = mcfg.n_layers;
    // Packing- and scale-aware K+V per-layer block bytes (#942): raw dtype_size()
    // is 0 for NVFP4/MXFP4_KV and zeroed the KV headroom.
    r.kv_bytes_per_seq = static_cast<size_t>(blocks_per_seq) * n_attn *
                         kv_block_bytes_per_layer(cfg.kv_cache_dtype, est_bs, mcfg.n_kv_heads, head_dim_est);
    size_t total_vram = 0, f = 0;
    // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
    (void)vram_budget_mem_get_info(&f, &total_vram);
    // MoE > 16 experts: all-GPU experts (decode fast path, graph capture) win over a big KV reserve.
    r.kv_cap_bytes = total_vram / ((mcfg.n_experts > 16) ? 10 : 5);
    if (mcfg.ssm_inner_size <= 0)
        return r;
    int n_ssm = 0;
    for (int i = 0; i < mcfg.n_layers; i++)
        if (model.layer(i).ssm_in.data != nullptr)
            n_ssm++;
    // memory/ssm_state_size.h, the header SSMState::init reads (MEMORY.md D14/D15).
    const int n_heads = mcfg.ssm_dt_rank;
    const SsmStateGeometry geom{n_ssm,
                                mcfg.ssm_conv_channels(),
                                mcfg.ssm_conv_kernel,
                                n_heads,
                                (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0,
                                mcfg.ssm_state_size,
                                cfg.ssm_state_dtype,
                                model.ple_state_bytes()};
    r.state_bytes_per_slot = ssm_bytes_per_slot(geom);
    r.reserved_state_slots = reserved_state_slots;
    // Recurrent-snapshot store (hybrid prefix caching): pre-allocated eagerly at KV-cache init.
    if (cfg.use_prefix_caching || rc.server.prefix_cache)
        r.fixed_bytes += static_cast<size_t>(std::max(rc.server.recurrent_snapshot_mb, 0)) << 20;
    return r;
}

// Applies the batch weight upload fitted (#2393) to EngineConfig, the runtime config and the
// admission cap, as clamp_max_batch_to_plan_ does. No-op when the configured batch fit.
void apply_upload_batch_fit(int fitted, int& engine_batch, int& runtime_batch, Scheduler* sched) {
    if (fitted < 1 || fitted >= engine_batch)
        return;
    engine_batch = fitted;
    runtime_batch = std::min(runtime_batch, fitted);
    if (sched)
        sched->clamp_max_batch_size(fitted);
}
}  // namespace

bool Engine::init_weights() {
    const auto& mcfg = model_->config();

    // Initialize graph executor (Phase 1: compute sizes, no GPU allocation)
    executor_ = std::make_unique<GraphExecutor>();
    executor_->set_vram_allocator(&vram_alloc_);
    // Fill the dispatch snapshot HERE, not earlier: init_resolve_* still mutates
    // runtime_config_ (deterministic_gemm, kv dtype, fp8 prefill, quant flags)
    // before this runs; filling earlier hands exec/ pre-resolution values, the wrong kernel running with no other symptom.
    dispatch_policy_.kv_cache = runtime_config_.kv_cache;
    dispatch_policy_.attention = runtime_config_.attention;
    dispatch_policy_.moe = runtime_config_.moe;
    dispatch_policy_.gdn = runtime_config_.gdn;
    dispatch_policy_.gemm = runtime_config_.gemm;
    dispatch_policy_.generation = runtime_config_.generation;
    dispatch_policy_.speculative = runtime_config_.speculative;
    dispatch_policy_.ffn = runtime_config_.ffn;
    dispatch_policy_.diagnostics = runtime_config_.diagnostics;
    executor_->set_dispatch_policy(dispatch_policy_);
    // Before init(): allocate_workspaces() sizes the sparse decode budget in
    // BLOCKS, and the conversion from a token count needs the real block size.
    executor_->set_kv_block_size(config_.kv_block_size);
    {
        int eff_batch = config_.max_batch_size;
        if (!executor_->init(*model_, config_.compute_dtype, config_.use_pdl, eff_batch, config_.max_seq_len,
                             config_.use_fp8_prefill, config_.use_nvfp4_decode, config_.use_mxfp4_prefill))
            return false;

        if (config_.dual_path_quant) {
            executor_->set_dual_path_quant(true);
            IMP_LOG_INFO("Dual-path quant: attention weights → FP8, FFN weights → NVFP4");
        }

        if (runtime_config_.calibration.enabled) {
            executor_->enable_calibration();
            IMP_LOG_INFO("Activation calibration ON — collecting per-channel activation magnitudes.");
            // The collector allocates an accumulator the first time it sees a
            // weight, which a graph capture rejects outright.
            demote_graphs_(GraphDemotionReason::CalibrationActive);
        }

        if (config_.streaming_kv_enabled) {
            // Every KV dtype's decode kernels skip the -1 sentinels evict_middle_blocks
            // leaves (#1704); the quantised kernels attend every live block under sinks.
            int n_sinks = (config_.streaming_kv_n_sinks > 0) ? config_.streaming_kv_n_sinks : 4;
            int win = (config_.streaming_kv_window > 0) ? config_.streaming_kv_window
                                                        : model_->config().sliding_window;
            executor_->set_streaming_kv(n_sinks, win);
            if (n_sinks > 0 && win > 0) {
                IMP_LOG_INFO("StreamingLLM smart KV cache enabled: %d sinks + %d-token window", n_sinks, win);
                // Block-table contents change every step once eviction begins;
                // a captured graph would replay stale pointers, and re-capturing
                // per step negates the graph's win, so disable graphs entirely.
                demote_graphs_(GraphDemotionReason::StreamingKvConfigured);
            } else {
                IMP_LOG_WARN(
                    "StreamingLLM enabled but no sliding window configured "
                    "(n_sinks=%d, window=%d) - disabling streaming.",
                    n_sinks, win);
                config_.streaming_kv_enabled = false;
            }
        }
    }

    reserve_persisting_l2();
    retain_async_pool_memory();

    // Compute VRAM reserve for expert weight upload
    model_->upload_batch_fit_ = UploadBatchFit{upload_reserve_for(*model_, config_, runtime_config_,
                                                                  executor_->workspace_estimate(),
                                                                  spec_mc_reserved_slots_()),
                                               config_.max_batch_size, config_.max_batch_size};
    const UploadReserve& upload_reserve = model_->upload_batch_fit_.reserve;
    const size_t expert_reserve = upload_reserve_bytes(upload_reserve, config_.max_batch_size);
    IMP_LOG_INFO("Expert upload reserve: %.2f MiB (workspace=%.2f, kv=%.2f, ssm+safety=rest)",
                 expert_reserve / (1024.0 * 1024.0), executor_->workspace_estimate() / (1024.0 * 1024.0),
                 upload_reserve_kv_bytes(upload_reserve, config_.max_batch_size) / (1024.0 * 1024.0));

    // Upload weights
    size_t free_before = 0, total_before = 0;
    const bool mem_before_ok = cudaMemGetInfo(&free_before, &total_before) == cudaSuccess;
    IMP_LOG_INFO("GPU memory before weight upload: %zu MiB free / %zu MiB total",
                 free_before / (1024UL * 1024), total_before / (1024UL * 1024));

    // RAII stream/event: upload_weights_gpu() can throw (internal errors throw
    // per project convention), and a raw cudaStream_t/cudaEvent_t held across it
    // would leak on the unwind. The wrappers destroy on scope exit / exception.
    CudaStream upload_stream_raii;
    if (!upload_stream_raii.create(cudaStreamNonBlocking))
        IMP_LOG_WARN("Failed to create weight-upload stream; uploading on the main stream.");
    cudaStream_t upload_stream = upload_stream_raii.get();  // may be null
    // --gpu-layers: planned host layers keep their matmuls on host (#2298), dense models only.
    model_->upload_host_layers_ = gpu_layers_host_plan(*model_, config_.gpu_layers);

    if (!model_->upload_weights_gpu(config_.compute_dtype, upload_stream ? upload_stream : stream_,
                                    expert_reserve, runtime_config_.warm_cache.enabled,
                                    runtime_config_.warm_cache.dir)) {
        IMP_LOG_ERROR("Weight upload failed. Try a smaller quantization.");
        return false;
    }
    apply_upload_batch_fit(model_->upload_batch_fit_.fitted, config_.max_batch_size,
                           runtime_config_.runtime.max_batch_size, scheduler_.get());
    model_->upload_batch_fit_ = {};

    if (upload_stream) {
        CudaEvent upload_done;
        if (upload_done.record(upload_stream))
            IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream_, upload_done.get()));
    }

    size_t free_after = 0, total_after = 0;
    // A failed query on either side: no occupancy verdict.
    const bool mem_ok = cudaMemGetInfo(&free_after, &total_after) == cudaSuccess && mem_before_ok;
    // A free-VRAM delta is not a weight size: on WSL2 the driver reports the
    // whole card as free until a process allocates, so a server started beside
    // a neighbour only learns of it THROUGH its own upload (MEMORY.md B8).
    const size_t upload_consumed = free_before - free_after;
    const size_t on_disk = model_source_bytes(model_->source_path());
    IMP_LOG_INFO(
        "GPU memory after weight upload: %zu MiB free / %zu MiB total (upload consumed "
        "%zu MiB of device free)",
        free_after / (1024UL * 1024), total_after / (1024UL * 1024), upload_consumed / (1024UL * 1024));
    // One-sided on purpose: consuming less than the checkpoint is ordinary
    // (host-resident experts, dropped sources), but consuming a quarter more
    // cannot be weights, and this is the only moment occupancy is observable.
    if (mem_ok && upload_exceeds_checkpoint(upload_consumed, on_disk)) {
        IMP_LOG_WARN(
            "Weight upload consumed %zu MiB of device free VRAM for a %zu MiB checkpoint. The "
            "excess is not weights: another process is holding the card (on WSL2 it is invisible "
            "until this upload), or the upload spilled to host memory. This process will still "
            "load and serve, and every number it produces — pool size, tokens/s — will be a "
            "statement about the card it shared. Free the card and restart before measuring "
            "anything here.",
            upload_consumed / (1024UL * 1024), on_disk / (1024UL * 1024));
    }

    // runtime.cuda_graphs=="never" is checked HERE, unconditionally, not nested
    // in the MoE block below: SSM/GDN models have no experts, so nesting it there
    // silently ignored an explicit "never" for exactly those models.
    // Runs first so its reason wins in demote_graphs_ (keeps the first reason).
    // Unknown values now warn instead of silently resolving to "auto" (AUDIT.md G1).
    {
        const std::string& g = runtime_config_.runtime.cuda_graphs;
        if (g == "never") {
            demote_graphs_(GraphDemotionReason::ConfigNever);
        } else if (g != "auto" && g != "always") {
            IMP_LOG_WARN(
                "runtime.cuda_graphs: unknown value '%s' (expected auto|always|never) - keeping "
                "graphs enabled. Nothing was disabled by this setting.",
                g.c_str());
        }
    }

    // Check for host-resident expert weights.
    // gpt-oss is exempt: its MXFP4 experts are kept host-resident through
    // upload but converted to on-device NVFP4 + CUTLASS-grouped at pre_dequant,
    // not host-offloaded at decode; treating them as on-host would read post-convert device pointers as host MXFP4 (garbage) and wrongly disable CUDA graphs.
    if (mcfg.n_experts > 0 && !model_->profile().experts_convert_at_predequant) {
        for (int i = 0; i < mcfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            // Packed-tensor path (most MoE archs) OR per-expert 2D views
            // (DeepSeek-V2 SafeTensors: only expert_w_up[e], no expert_*_packed).
            // Either being host-resident means the decode MoE uses the D2H-sync host path, which is NOT graph-capturable.
            bool packed_host = L.expert_up_packed.data && !L.expert_up_packed.on_device;
            bool view_host = !L.expert_w_up.empty() && L.expert_w_up[0].data &&
                             !L.expert_w_up[0].on_device;
            if (packed_host || view_host) {
                experts_on_host_ = true;
                break;
            }
        }
        // Device expert cache (moe.device_expert_cache + pin_host_experts, NVFP4 prequant):
        // routing, residency and the miss copies resolve on the device, so the captured
        // decode replays correctly. init_device_expert_cache() demotes later if a
        // host-resident layer stays on the host path.
        // pin_host_experts is no longer part of the condition: host-resident NVFP4 experts
        // are pinned unless the host is out of RAM (weight_upload.cpp), and a failed pin
        // demotes through init_device_expert_cache() below.
        const bool device_cache_planned = runtime_config_.moe.device_expert_cache && mcfg.is_nvfp4_prequant;
        if (experts_on_host_ && config_.use_cuda_graphs && device_cache_planned) {
            IMP_LOG_INFO("Experts on host: CUDA graphs stay on, the device expert cache serves decode "
                         "(demoted after init if it cannot cover every host-resident layer)");
        } else if (experts_on_host_ && config_.use_cuda_graphs) {
            // `moe.allow_graphs_under_offload` skips this guard but buys nothing:
            // every MoE path serving host-resident experts reads routing on the
            // host, and moe_host_args_capture_guard throws unconditionally under
            // capture. Kept as the escape hatch for the day routing + residency resolve device-side (docs/roadmap.md).
            if (runtime_config_.moe.allow_graphs_under_offload) {
                IMP_LOG_WARN(
                    "moe.allow_graphs_under_offload=true: the graphs-off guard is skipped, but "
                    "capture will still ABORT on every attempt and fall back to per-step decode "
                    "— the MoE decode path reads routing on the host, which is not capturable. "
                    "This flag currently changes nothing except adding failed capture attempts.");
            } else {
                demote_graphs_(GraphDemotionReason::ExpertsOnHost);
                IMP_LOG_INFO(
                    "  Tip: if model+KV fits in VRAM, set IMP_EXPERT_OVERHEAD_PCT=10 "
                    "(default 30) to upload ALL experts and re-enable CUDA graphs "
                    "(+~180%% decode on Qwen 3.6 35B Q4_K_M).");
                // Deliberately no longer advertised as an alternative: it
                // does not deliver captured decode (see the branch above).
            }
        }
        // MoE decode fast path is fully device-side (no D2H memcpy): graph-safe.
        // Only MoE prefill paths use D2H sync for expert_offsets, but prefill is
        // never captured in CUDA graphs.
    }

    // Capacity for the mandatory native-NVFP4 decode caches comes from the plan,
    // not a live query: vram_budget floors phase 3 at the measured demand
    // (mandatory_sf_bytes / mandatory_moe_bytes), per invariant I4.

    // Phase 2: allocate GPU workspace
    (void)executor_->allocate_workspaces(experts_on_host_);

    // Layer offloading: one forward at a time owns the two staging slots, none captured.
    if (layer_offload_blocks_graphs(config_.gpu_layers)) {
        offload_mgr_ = std::make_unique<LayerOffloadManager>();
        runtime_config_.runtime.prefill_overlap = false;
        demote_graphs_(GraphDemotionReason::LayerOffload);
        if (!offload_mgr_->init(model_.get(), config_.gpu_layers)) {
            IMP_LOG_ERROR("Layer offloading init failed for --gpu-layers %d: %s", config_.gpu_layers,
                          layer_streaming_failure(*model_));
            offload_mgr_.reset();
            return false;
        }
    }

    // Every phase that reads the checkpoint has run: release the pages the
    // loaders faulted in (MAP_POPULATE + MADV_WILLNEED), since nothing else
    // drops them. `docker stats` reports the cgroup, not this host-resident RSS.
    // Pages only, not the mapping: host-resident experts and offloaded layers still point into it and refault from the page cache.
    const size_t dropped = model_->release_weight_pages();
    if (dropped > 0)
        IMP_LOG_INFO(
            "weight file pages released after upload: %.2f GiB advised away "
            "(host-resident weights refault on demand)",
            static_cast<double>(dropped) / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace imp
