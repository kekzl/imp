// Engine init phase: weight upload to VRAM.
// Drives Model::upload_weights_gpu (which in turn calls upload_weight +
// upload_expert_weights from src/model/weight_upload.cu and triggers
// pre-dequant via src/exec/executor_pre_dequant.cu's pre_dequant_weights()
// orchestrator). Also reserves the L2 persisting cache, tunes the default
// cudaMallocAsync pool, computes the expert-upload VRAM reserve, and wires
// up StreamingLLM / host-offload guards before kv-cache init runs.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. Method remains Engine::* with declaration in engine.h.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "core/cuda_raii.h"
#include "core/logging.h"
#include "memory/vram_query.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace imp {

bool Engine::init_weights() {
    const auto& mcfg = model_->config();

    // Initialize graph executor (Phase 1: compute sizes, no GPU allocation)
    executor_ = std::make_unique<GraphExecutor>();
    executor_->set_vram_allocator(&vram_alloc_);
    // Fill the dispatch snapshot HERE, not earlier: init_resolve_* still mutates
    // runtime_config_ (deterministic_gemm, kv dtype policy, fp8 prefill, quant
    // flags) and those run before init_weights(). Filling before them would hand
    // exec/ pre-resolution values, which is exactly the class of bug that has no
    // symptom other than the wrong kernel running.
    dispatch_policy_.kv_cache = runtime_config_.kv_cache;
    dispatch_policy_.attention = runtime_config_.attention;
    dispatch_policy_.moe = runtime_config_.moe;
    dispatch_policy_.gdn = runtime_config_.gdn;
    dispatch_policy_.gemm = runtime_config_.gemm;
    dispatch_policy_.generation = runtime_config_.generation;
    dispatch_policy_.speculative = runtime_config_.speculative;
    dispatch_policy_.ffn = runtime_config_.ffn;
    dispatch_policy_.diagnostics = runtime_config_.diagnostics;
    executor_->set_runtime_config(dispatch_policy_);
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
            // Streaming is only safe for the FP16 GQA decode kernel — the
            // quantized variants don't yet skip -1 sentinels in their block
            // tables. Refuse to enable streaming with non-FP16 KV caches so
            // we never call evict_middle_blocks for an unsupported path.
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
                    // Block-table contents change every step once eviction
                    // begins; a CUDA graph captured against an old table
                    // would replay stale pointers. Re-capturing per step
                    // negates the graph's win, so disable graphs entirely.
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

    // Reserve L2 persisting cache for decode GEMV. cudaLimitPersistingL2CacheSize
    // is a per-primary-context device limit that survives Engine teardown, so a
    // previously-loaded model in this process leaves its reservation (and any
    // persisting cache lines) behind. Reset the persisting lines first so this
    // model starts from a clean L2 persistence state — otherwise the next
    // per-forward access-policy-window set can return cudaErrorInvalidValue and
    // poison the stream (cross-model GDN→Gemma-4 garbage repro).
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

    // Tune the default cudaMallocAsync pool so it retains freed memory instead
    // of returning it to the driver. Many paths (prefill metadata, MoE scratch,
    // spec decoder block tables, vision staging) use cudaMallocAsync with the
    // default pool; the default threshold is 0, which calls cuMemUnmap on every
    // free. Setting UINT64_MAX keeps allocations for re-use — the KV cache and
    // workspaces own their memory via VRAMAllocator/plain cudaMalloc, so the
    // default-pool footprint stays small.
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
            int conv_ch = mcfg.ssm_conv_channels();
            int n_heads = mcfg.ssm_dt_rank;
            int hd_ssm = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            int n_ssm = 0;
            for (int i = 0; i < mcfg.n_layers; i++)
                if (model_->layer(i).ssm_in.data != nullptr)
                    n_ssm++;
            expert_reserve += static_cast<size_t>(n_ssm) * config_.max_batch_size *
                              (static_cast<unsigned long>(conv_ch) * std::max(mcfg.ssm_conv_kernel - 1, 0) *
                                   sizeof(float) +
                               static_cast<size_t>(n_heads) * hd_ssm * mcfg.ssm_state_size *
                                   dtype_size(config_.ssm_state_dtype));
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
    IMP_LOG_INFO("GPU memory after weight upload: %zu MiB free / %zu MiB total (weights ~%zu MiB)",
                 free_after / (1024UL * 1024), total_after / (1024UL * 1024),
                 (free_before - free_after) / (1024UL * 1024));

    // Check for host-resident expert weights.
    // gpt-oss is exempt: its MXFP4 experts are intentionally kept host-resident
    // through upload (weight_upload's carve-out) and converted to on-device
    // NVFP4 + CUTLASS-grouped at pre_dequant. They are NOT host-offloaded at
    // decode, so treating them as on-host here would wrongly enable the LRU
    // host-offload path (reading the post-convert device pointers as host
    // MXFP4 → garbage) and disable CUDA graphs.
    if (mcfg.n_experts > 0 && !model_->profile().is_gpt_oss) {
        for (int i = 0; i < mcfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            // Packed-tensor path (most MoE archs) OR per-expert 2D views
            // (DeepSeek-V2 SafeTensors builds no expert_*_packed — only
            // expert_w_up[e]). Either being host-resident means the decode
            // MoE uses the D2H-sync host path, which is NOT graph-capturable.
            bool packed_host = L.expert_up_packed.data && !L.expert_up_packed.on_device;
            bool view_host = !L.expert_w_up.empty() && L.expert_w_up[0].data &&
                             !L.expert_w_up[0].on_device;
            if (packed_host || view_host) {
                experts_on_host_ = true;
                break;
            }
        }
        if (experts_on_host_ && config_.use_cuda_graphs) {
            // The opt-in `moe.allow_graphs_under_offload` flag skips this
            // guard. It does NOT currently buy captured decode, and the old
            // wording here ("correctness depends on prefetch coverage")
            // oversold it: capture never gets far enough for that to be the
            // question. Every MoE path that serves host-resident experts reads
            // routing on the host — the serial fallback and, since #1370, the
            // slot-pool decode path — and `moe_host_args_capture_guard` throws
            // unconditionally under capture. Measured 2026-08-11: three capture
            // attempts, three aborts, per-step decode throughout.
            //
            // The flag is kept because it is the escape hatch the day that
            // changes: making this capturable means resolving routing AND
            // expert residency device-side, and residency needs a host-issued
            // H2D on a miss. See docs/roadmap.md.
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
        if (runtime_config_.runtime.cuda_graphs == "never")
            demote_graphs_(GraphDemotionReason::ConfigNever);
        // MoE decode fast path is fully device-side (no D2H memcpy) — graph-safe.
        // Only MoE prefill paths use D2H sync for expert_offsets, but prefill is
        // never captured in CUDA graphs.
    }

    // The mandatory native-NVFP4 decode caches used to be protected by a
    // physical balloon here: a cudaMalloc taken right after the upload purely so
    // every later live-free reader saw fewer bytes, released again before the
    // cache build. It is gone (AUDIT B61/B62). Its premise expired when A7 step
    // 6.4 moved the cache build BEFORE the KV pool — the comment still claimed
    // the caches were built last — and measured across five configs it bound on
    // exactly one, where it cost 5x KV capacity (16 vs 87 blocks) to buy coverage
    // whose absence changed neither throughput nor correctness. The guarantee it
    // provided now comes from the plan instead: vram_budget floors phase 3 at the
    // measured demand (mandatory_sf_bytes / mandatory_moe_bytes), which is what
    // I4 asks for — capacity decided up front, not hidden from a live query.

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

    return true;
}

}  // namespace imp
