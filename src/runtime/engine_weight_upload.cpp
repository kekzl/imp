// Engine init phase: weight upload to VRAM via Model::upload_weights_gpu
// (src/model/weight_upload.cu, pre-dequant via executor_pre_dequant.cu). Also
// reserves L2 cache, tunes cudaMallocAsync, sizes the expert-upload VRAM reserve, wires StreamingLLM / host-offload guards.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "core/cuda_raii.h"
#include "core/logging.h"
#include "memory/ssm_state_size.h"
#include "memory/vram_query.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <utility>

namespace imp {

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
            // Streaming is only safe for the FP16 GQA decode kernel: quantized
            // variants don't yet skip -1 sentinels in their block tables. Refuse
            // non-FP16 KV caches so evict_middle_blocks never runs an unsupported path.
            if (config_.kv_cache_dtype != QType::F16) {
                IMP_LOG_WARN(
                    "StreamingLLM smart KV cache requires FP16 KV cache "
                    "(requested %d) — disabling streaming.",
                    std::to_underlying(config_.kv_cache_dtype));
                config_.streaming_kv_enabled = false;
            } else {
                int n_sinks = (config_.streaming_kv_n_sinks > 0) ? config_.streaming_kv_n_sinks : 4;
                int win = (config_.streaming_kv_window > 0) ? config_.streaming_kv_window
                                                            : model_->config().sliding_window;
                executor_->set_streaming_kv(n_sinks, win);
                if (n_sinks > 0 && win > 0) {
                    IMP_LOG_INFO("StreamingLLM smart KV cache enabled: %d sinks + %d-token window", n_sinks,
                                 win);
                    // Block-table contents change every step once eviction begins;
                    // a captured graph would replay stale pointers, and re-capturing
                    // per step negates the graph's win, so disable graphs entirely.
                    demote_graphs_(GraphDemotionReason::StreamingKvConfigured);
                } else {
                    IMP_LOG_WARN(
                        "StreamingLLM enabled but no sliding window configured "
                        "(n_sinks=%d, window=%d) — disabling streaming.",
                        n_sinks, win);
                    config_.streaming_kv_enabled = false;
                }
            }
        }
    }

    // cudaLimitPersistingL2CacheSize is a per-primary-context device limit that
    // survives Engine teardown, so a previous model's reservation persists.
    // Reset it first, or the next access-policy-window set can return cudaErrorInvalidValue and poison the stream.
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        size_t max_persist = prop.persistingL2CacheMaxSize;
        if (max_persist > 0) {
            cudaCtxResetPersistingL2Cache();
            (void)cudaGetLastError();
            size_t reserve = max_persist * 3 / 4;
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve);
            IMP_LOG_INFO("L2 persisting cache: reserved %zu MB / %zu MB total", reserve >> 20,
                         max_persist >> 20);
        }
    }

    // Tunes the default cudaMallocAsync pool to retain freed memory instead of
    // calling cuMemUnmap on every free (default threshold 0): many paths (prefill
    // metadata, MoE scratch, spec block tables, vision staging) use this pool; KV cache and workspaces own their memory separately (VRAMAllocator/cudaMalloc).
    {
        cudaMemPool_t default_pool = nullptr;
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetDefaultMemPool(&default_pool, dev) == cudaSuccess && default_pool != nullptr) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
    }

    // Compute VRAM reserve for expert weight upload
    size_t expert_reserve = executor_->workspace_estimate();
    {
        int head_dim_est = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
        int est_bs = config_.kv_block_size > 0 ? config_.kv_block_size : kKVBlockSize;
        int blocks_per_seq = (config_.max_seq_len + est_bs - 1) / est_bs;
        int n_attn = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).wq.data != nullptr)
                n_attn++;
        if (n_attn == 0)
            n_attn = mcfg.n_layers;
        // Packing- and scale-aware K+V per-layer block bytes (#942): the raw
        // dtype_size() this used to multiply by returns 0 for NVFP4/MXFP4_KV,
        // which zeroed the KV headroom out of the expert-offload decision.
        size_t kv_est = static_cast<size_t>(blocks_per_seq * config_.max_batch_size) * n_attn *
                        kv_block_bytes_per_layer(config_.kv_cache_dtype, est_bs, mcfg.n_kv_heads,
                                                 head_dim_est);
        {
            size_t total_vram = 0, f = 0;
            vram_budget_mem_get_info(&f, &total_vram);
            // For large MoE models (128 experts), prefer fitting all experts on GPU
            // over reserving huge KV cache. All-GPU experts enable the decode fast
            // path (dp4a GEMV, no D2H sync) and CUDA graph capture.
            size_t vram_frac = (mcfg.n_experts > 16) ? 10 : 5;
            kv_est = std::min(kv_est, total_vram / vram_frac);
        }
        expert_reserve += kv_est;

        if (mcfg.ssm_inner_size > 0) {
            int n_ssm = 0;
            for (int i = 0; i < mcfg.n_layers; i++)
                if (model_->layer(i).ssm_in.data != nullptr)
                    n_ssm++;
            // Uses memory/ssm_state_size.h, the same header SSMState::init reads:
            // an inline formula here undercounted (missing 256-byte alignment and
            // verify slots), under-charging in the direction that oversubscribes the card (MEMORY.md D14/D15).
            const int n_heads = mcfg.ssm_dt_rank;
            const SsmStateGeometry geom{n_ssm,
                                        mcfg.ssm_conv_channels(),
                                        mcfg.ssm_conv_kernel,
                                        n_heads,
                                        (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0,
                                        mcfg.ssm_state_size,
                                        config_.ssm_state_dtype};
            expert_reserve += ssm_pool_bytes(geom, config_.max_batch_size, spec_mc_reserved_slots_());
            // Recurrent-snapshot store (hybrid prefix caching): its buffers
            // are pre-allocated eagerly at KV-cache init, so the offload
            // decision must leave room for them.
            if (config_.use_prefix_caching || runtime_config_.server.prefix_cache) {
                expert_reserve +=
                    static_cast<size_t>(std::max(runtime_config_.server.recurrent_snapshot_mb, 0)) << 20;
            }
        }

        size_t safety = 256ULL * 1024 * 1024;  // base safety
        // Only add safety for features that will actually allocate VRAM.
        // On tight VRAM models (Nemotron-30B), every MiB matters for expert coverage.
        expert_reserve += safety;

        IMP_LOG_INFO("Expert upload reserve: %.2f MiB (workspace=%.2f, kv=%.2f, ssm+safety=rest)",
                     expert_reserve / (1024.0 * 1024.0), executor_->workspace_estimate() / (1024.0 * 1024.0),
                     kv_est / (1024.0 * 1024.0));
    }

    // Upload weights
    size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);
    IMP_LOG_INFO("GPU memory before weight upload: %zu MiB free / %zu MiB total",
                 free_before / (1024UL * 1024), total_before / (1024UL * 1024));

    // RAII stream/event: upload_weights_gpu() can throw (internal errors throw
    // per project convention), and a raw cudaStream_t/cudaEvent_t held across it
    // would leak on the unwind. The wrappers destroy on scope exit / exception.
    CudaStream upload_stream_raii;
    if (!upload_stream_raii.create(cudaStreamNonBlocking))
        IMP_LOG_WARN("Failed to create weight-upload stream; uploading on the main stream.");
    cudaStream_t upload_stream = upload_stream_raii.get();  // may be null

    if (!model_->upload_weights_gpu(config_.compute_dtype, upload_stream ? upload_stream : stream_,
                                    expert_reserve, runtime_config_.warm_cache.enabled,
                                    runtime_config_.warm_cache.dir)) {
        IMP_LOG_ERROR("Weight upload failed. Try a smaller quantization.");
        return false;
    }

    if (upload_stream) {
        CudaEvent upload_done;
        if (upload_done.record(upload_stream))
            IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream_, upload_done.get()));
    }

    size_t free_after = 0, total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);
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
    if (upload_exceeds_checkpoint(upload_consumed, on_disk)) {
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
    if (mcfg.n_experts > 0 && !model_->profile().is_gpt_oss) {
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
        if (experts_on_host_ && config_.use_cuda_graphs) {
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

    // Layer offloading
    if (config_.gpu_layers >= 0) {
        offload_mgr_ = std::make_unique<LayerOffloadManager>();
        if (!offload_mgr_->init(model_.get(), config_.gpu_layers)) {
            IMP_LOG_WARN("Layer offloading init failed, continuing without it");
            offload_mgr_.reset();
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
