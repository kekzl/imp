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
#include "core/cuda_raii.h"
#include "core/logging.h"

#include <algorithm>
#include <cstdint>

namespace imp {

bool Engine::init_weights() {
    const auto& mcfg = model_->config();

    // Initialize graph executor (Phase 1: compute sizes, no GPU allocation)
    executor_ = std::make_unique<GraphExecutor>();
    executor_->set_vram_allocator(&memory_manager_.vram_allocator());
    executor_->set_runtime_config(runtime_config_);
    {
        int eff_batch = config_.max_batch_size;
        if (!executor_->init(*model_, config_.compute_dtype, config_.use_pdl, eff_batch, config_.max_seq_len,
                             config_.use_fp8_prefill, config_.use_nvfp4_decode, config_.use_mxfp4_prefill))
            return false;

        if (config_.dual_path_quant) {
            executor_->set_dual_path_quant(true);
            IMP_LOG_INFO("Dual-path quant: attention weights → FP8, FFN weights → NVFP4");
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
                    static_cast<int>(config_.kv_cache_dtype));
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
                    if (config_.use_cuda_graphs) {
                        IMP_LOG_INFO(
                            "Disabling CUDA Graphs while StreamingLLM is active "
                            "(block table mutates per decode step).");
                        config_.use_cuda_graphs = false;
                    }
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
    // workspaces already own their memory through the DeviceAllocator, so the
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
        size_t elem_sz = dtype_size(config_.kv_cache_dtype);
        int est_bs = config_.kv_block_size > 0 ? config_.kv_block_size : kKVBlockSize;
        int blocks_per_seq = (config_.max_seq_len + est_bs - 1) / est_bs;
        int n_attn = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).wq.data != nullptr)
                n_attn++;
        if (n_attn == 0)
            n_attn = mcfg.n_layers;
        size_t kv_block_bytes = static_cast<size_t>(est_bs) * mcfg.n_kv_heads * head_dim_est * elem_sz;
        size_t kv_est = static_cast<size_t>(blocks_per_seq * config_.max_batch_size) * 2 * n_attn *
                        kv_block_bytes;
        {
            size_t total_vram = 0, f = 0;
            cudaMemGetInfo(&f, &total_vram);
            // For large MoE models (128 experts), prefer fitting all experts on GPU
            // over reserving huge KV cache. All-GPU experts enable the decode fast
            // path (dp4a GEMV, no D2H sync) and CUDA graph capture.
            size_t vram_frac = (mcfg.n_experts > 16) ? 10 : 5;
            kv_est = std::min(kv_est, total_vram / vram_frac);
        }
        expert_reserve += kv_est;

        if (mcfg.ssm_inner_size > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
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
                                    expert_reserve)) {
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
            // Phase 5: the opt-in `moe.allow_graphs_under_offload` flag skips
            // this guard so the user can experiment with captured decode
            // under host-offload. Correctness is conditional on prefetch
            // coverage matching router selection — captured cudaMemcpyAsync
            // nodes have fixed (src host ptr, dst slot) pairs that don't
            // adapt to per-token routing changes. See config.h doc for the
            // architectural caveat; Phase 5.1+ refactors dispatch kernels
            // to read the device mirror at runtime so the captured graph
            // adapts correctly.
            if (runtime_config_.moe.allow_graphs_under_offload) {
                IMP_LOG_WARN(
                    "CUDA graphs ENABLED under host-offload "
                    "(moe.allow_graphs_under_offload=true). EXPERIMENTAL: output "
                    "correctness depends on prefetch coverage matching router "
                    "selection. Set moe.prefetch_top_k high enough that the "
                    "captured top-K covers the router's hot set.");
            } else {
                IMP_LOG_INFO("Disabling CUDA graphs: expert weights on host");
                IMP_LOG_INFO(
                    "  Tip: if model+KV fits in VRAM, set IMP_EXPERT_OVERHEAD_PCT=10 "
                    "(default 30) to upload ALL experts and re-enable CUDA graphs "
                    "(+~180%% decode on Qwen 3.6 35B Q4_K_M).");
                IMP_LOG_INFO(
                    "  Or set moe.allow_graphs_under_offload=true (experimental) "
                    "to opt into captured decode under host-offload.");
                config_.use_cuda_graphs = false;
            }
        }
        if (runtime_config_.runtime.cuda_graphs == "never" && config_.use_cuda_graphs) {
            IMP_LOG_INFO("Disabling CUDA graphs: runtime.cuda_graphs=never");
            config_.use_cuda_graphs = false;
        }
        // MoE decode fast path is fully device-side (no D2H memcpy) — graph-safe.
        // Only MoE prefill paths use D2H sync for expert_offsets, but prefill is
        // never captured in CUDA graphs.
    }

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
