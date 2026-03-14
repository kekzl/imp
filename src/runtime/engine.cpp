#include "runtime/engine.h"
#include "runtime/speculative.h"
#include "runtime/self_speculative.h"
#include "runtime/batch.h"
#include "memory/kv_cache.h"
#include "model/gguf_loader.h"
#include "model/chat_template.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/sampling.h"
#include "compute/attention.h"
#include "vision/vision_loader.h"
#include "vision/image_processor.h"
#include "core/logging.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace imp {

Engine::~Engine() {
    // Save prefix cache to disk before shutdown
    if (kv_manager_ && !config_.prefix_cache_path.empty() &&
        kv_manager_->prefix_caching_enabled()) {
        kv_manager_->save_prefix_cache(config_.prefix_cache_path, stream_);
    }

    gemm_cleanup();
    gemm_grouped_cleanup();
    if (async_graph_runner_.is_setup()) {
        async_graph_runner_.cleanup();
    }
    if (async_d_block_tables_) {
        cudaFree(async_d_block_tables_);
        async_d_block_tables_ = nullptr;
    }
    if (d_penalty_tokens_) {
        cudaFree(d_penalty_tokens_);
        d_penalty_tokens_ = nullptr;
    }
    if (d_vision_embeddings_) {
        cudaFree(d_vision_embeddings_);
        d_vision_embeddings_ = nullptr;
    }
    if (h_sample_pinned_) {
        cudaFreeHost(h_sample_pinned_);
        h_sample_pinned_ = nullptr;
    }
    // stream_, prefill_done_, decode_done_ cleaned up by CudaStream/CudaEvent RAII
}

cudaStream_t Engine::prefill_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available())
           ? green_ctx_.prefill_stream() : stream_;
}

cudaStream_t Engine::decode_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available())
           ? green_ctx_.decode_stream() : stream_;
}

void Engine::reset_ssm_state(int seq_id) {
    if (ssm_state_) {
        ssm_state_->reset_sequence(seq_id % ssm_state_->max_sequences(), stream_);
    }
}

size_t Engine::effective_free_vram() const {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return 0;
    }
    if (config_.vram_budget_mb > 0) {
        size_t budget = config_.vram_budget_mb * 1024ULL * 1024;
        size_t used = total_mem - free_mem;
        free_mem = (budget > used) ? (budget - used) : 0;
    }
    return free_mem;
}

bool Engine::init(std::shared_ptr<Model> model, const EngineConfig& config) {
    if (!model) {
        return false;
    }

    model_ = std::move(model);
    config_ = config;

    const auto& mcfg = model_->config();

    // Resolve NVFP4 decode auto mode based on GPU compute capability.
    // Skip for small dense models (d_model < 4096): FP4 quantization error accumulates
    // over layers and degrades output quality, while the decode speed gain is negligible
    // (small models are already memory-bandwidth-light).
    // Exception: MoE models benefit regardless of d_model — expert weights dominate VRAM
    // and sparse activation limits error accumulation.
    if (config_.use_nvfp4_decode < 0) {
        int sm = get_device_sm_version();
        if (mcfg.d_model < 4096 && mcfg.n_experts == 0) {
            config_.use_nvfp4_decode = 0;
            IMP_LOG_INFO("NVFP4 decode: auto → disabled (d_model=%d < 4096, precision risk)", mcfg.d_model);
        } else if (sm >= 120) {
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO("NVFP4 decode: auto → mode %d (sm_%d)", config_.use_nvfp4_decode, sm);
        } else if (sm >= 90) {
            config_.use_nvfp4_decode = 1;
            IMP_LOG_INFO("NVFP4 decode: auto → mode %d (sm_%d)", config_.use_nvfp4_decode, sm);
        } else {
            config_.use_nvfp4_decode = 0;
            IMP_LOG_INFO("NVFP4 decode: auto → disabled (sm_%d < sm_90)", sm);
        }
    }

    // --- Pre-allocate cuBLAS/cuBLASLt workspace while GPU memory is plentiful ---
    gemm_init();

    // --- Initialize scheduler ---
    scheduler_ = std::make_unique<Scheduler>(config_.max_batch_size);

    // --- Initialize graph executor (Phase 1: compute sizes, no GPU allocation) ---
    // GPU workspace allocation is deferred to AFTER weight upload to maximize
    // VRAM available for expert layers during upload.
    executor_ = std::make_unique<GraphExecutor>();
    {
        // Self-speculative verify needs logits for K+1 tokens in one pass.
        // Ensure max_batch_size (which sizes the logits buffer) is large enough.
        int eff_batch = config_.max_batch_size;
        if (config_.enable_self_speculative) {
            eff_batch = std::max(eff_batch, config_.self_spec_k + 1);
        }
        if (!executor_->init(*model_, config_.compute_dtype, config_.use_pdl,
                             eff_batch, config_.max_seq_len,
                             config_.use_fp8_prefill, config_.use_nvfp4_decode,
                             config_.use_mxfp4_prefill)) {
            return false;
        }
    }

    // --- Reserve L2 persisting cache for decode GEMV ---
    // KV cache reads use streaming loads (__ldcs / cp.async.cg) that hint L2 to
    // evict those lines first. Without a persisting reservation, the hardware has
    // no set-aside region — this call enables the streaming/persisting distinction.
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t max_persist = prop.persistingL2CacheMaxSize;
        if (max_persist > 0) {
            size_t reserve = max_persist * 3 / 4;  // 75% of L2 (72 MB on RTX 5090)
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve);
            IMP_LOG_INFO("L2 persisting cache: reserved %zu MB / %zu MB total",
                         reserve >> 20, max_persist >> 20);
        }
    }

    // --- Create CUDA stream ---
    // Non-blocking stream avoids implicit synchronization with the default stream,
    // preventing hidden stalls when other CUDA work is in flight.
    stream_.create(cudaStreamNonBlocking);

    // --- Upload model weights to GPU ---
    // Compute dynamic reserve: workspace + KV cache + SSM state + safety margin.
    // The workspace is not allocated yet, so all VRAM above this reserve is
    // available for expert weight upload — fitting more layers on GPU.
    size_t expert_reserve = executor_->workspace_estimate();
    {
        // Add estimated KV cache + SSM state footprint
        int head_dim_est = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
        size_t elem_sz = dtype_size(config_.kv_cache_dtype);
        // Minimum KV: enough for max_seq_len * max_batch_size
        int blocks_per_seq = (config_.max_seq_len + kKVBlockSize - 1) / kKVBlockSize;
        int n_attn = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).wq.data != nullptr) n_attn++;
        if (n_attn == 0) n_attn = mcfg.n_layers;
        size_t kv_block_bytes = static_cast<size_t>(kKVBlockSize) * mcfg.n_kv_heads * head_dim_est * elem_sz;
        size_t kv_est = static_cast<size_t>(blocks_per_seq * config_.max_batch_size) * 2 * n_attn * kv_block_bytes;
        expert_reserve += kv_est;

        // SSM state estimate
        if (mcfg.ssm_inner_size > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int hd_ssm = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            int n_ssm = 0;
            for (int i = 0; i < mcfg.n_layers; i++)
                if (model_->layer(i).ssm_in.data != nullptr) n_ssm++;
            size_t ssm_est = static_cast<size_t>(n_ssm) * config_.max_batch_size *
                             (conv_ch * std::max(mcfg.ssm_conv_kernel - 1, 0) * sizeof(float) +
                              n_heads * hd_ssm * mcfg.ssm_state_size * dtype_size(config_.ssm_state_dtype));
            expert_reserve += ssm_est;
        }

        // Safety margin for FP16 cache, CUDA graph driver memory, and runtime overhead.
        // On WSL2/WDDM, exceeding physical VRAM silently spills to shared system memory,
        // causing massive slowdowns. 768 MiB covers driver internals + graph capture +
        // cuBLAS growth + misc stream-ordered allocations during inference.
        expert_reserve += 768ULL * 1024 * 1024;

        IMP_LOG_INFO("Expert upload reserve: %.2f MiB (workspace=%.2f, kv=%.2f, ssm+safety=rest)",
                     expert_reserve / (1024.0 * 1024.0),
                     executor_->workspace_estimate() / (1024.0 * 1024.0),
                     kv_est / (1024.0 * 1024.0));
    }

    {
        size_t free_before = 0, total_before = 0;
        cudaMemGetInfo(&free_before, &total_before);
        IMP_LOG_INFO("GPU memory before weight upload: %zu MiB free / %zu MiB total",
                     free_before / (1024 * 1024), total_before / (1024 * 1024));

        // Use a separate upload stream so H2D transfers can overlap with
        // workspace allocation and other init work on stream_.
        cudaStream_t upload_stream = nullptr;
        cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking);

        if (!model_->upload_weights_gpu(config_.compute_dtype,
                                         upload_stream ? upload_stream : stream_,
                                         expert_reserve)) {
            IMP_LOG_ERROR("Weight upload failed. Model may be too large for GPU. "
                          "Try a smaller quantization (e.g. Q4_K_M instead of Q6_K).");
            if (upload_stream) cudaStreamDestroy(upload_stream);
            return false;
        }

        // Record event on upload stream so main stream can wait before using weights
        cudaEvent_t upload_done = nullptr;
        if (upload_stream) {
            cudaEventCreate(&upload_done);
            cudaEventRecord(upload_done, upload_stream);
        }

        size_t free_after = 0, total_after = 0;
        cudaMemGetInfo(&free_after, &total_after);
        IMP_LOG_INFO("GPU memory after weight upload: %zu MiB free / %zu MiB total "
                     "(weights used ~%zu MiB)",
                     free_after / (1024 * 1024), total_after / (1024 * 1024),
                     (free_before - free_after) / (1024 * 1024));

        // Ensure upload completes before any weight access on main stream
        if (upload_done) {
            cudaStreamWaitEvent(stream_, upload_done);
            cudaEventDestroy(upload_done);
        }
        if (upload_stream) cudaStreamDestroy(upload_stream);
    }

    // --- Check if any expert weights ended up on host ---
    bool experts_on_host = false;
    if (mcfg.n_experts > 0) {
        for (int i = 0; i < mcfg.n_layers; i++) {
            const auto& ly = model_->layer(i);
            if (ly.expert_up_packed.data && !ly.expert_up_packed.on_device) {
                experts_on_host = true;
                break;
            }
        }
        if (experts_on_host) {
            experts_on_host_ = true;
            if (config_.use_cuda_graphs) {
                IMP_LOG_INFO("Disabling CUDA graphs: expert weights on host (H2D not graph-capturable)");
                config_.use_cuda_graphs = false;
            }
        }
    }

    // --- Initialize graph executor (Phase 2: allocate GPU workspace) ---
    executor_->allocate_workspaces(experts_on_host);

    // --- Initialize layer offloading if configured ---
    if (config_.gpu_layers >= 0) {
        offload_mgr_ = std::make_unique<LayerOffloadManager>();
        if (!offload_mgr_->init(model_.get(), config_.gpu_layers)) {
            IMP_LOG_WARN("Layer offloading init failed, continuing without it");
            offload_mgr_.reset();
        }
    }

    // --- Initialize KV cache (AFTER weights + workspace so cudaMemGetInfo is accurate) ---
    int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
    int max_blocks = 0;

    // Count attention layers and build KV layer mapping for hybrid models.
    // Only attention layers need KV cache entries — SSM/MoE-only layers don't.
    int n_attn_layers = 0;
    std::vector<int> kv_layer_map(mcfg.n_layers, -1);
    for (int i = 0; i < mcfg.n_layers; i++) {
        if (model_->layer(i).wq.data != nullptr) {
            kv_layer_map[i] = n_attn_layers++;
        }
    }
    if (n_attn_layers == 0) {
        n_attn_layers = mcfg.n_layers;  // fallback: all layers have attention
        for (int i = 0; i < mcfg.n_layers; i++) kv_layer_map[i] = i;
    }
    int n_kv_layers = n_attn_layers;
    IMP_LOG_INFO("KV cache layers: %d attention out of %d total", n_kv_layers, mcfg.n_layers);

    // Calculate how many blocks are actually needed for the configured workload.
    int blocks_per_seq = (config_.max_seq_len + kKVBlockSize - 1) / kKVBlockSize;
    int needed_blocks = blocks_per_seq * config_.max_batch_size;

    if (config_.kv_cache_max_blocks > 0) {
        max_blocks = config_.kv_cache_max_blocks;
    } else {
        // Auto-size: allocate what's needed plus headroom, bounded by free VRAM.
        // Pre-calculate SSM footprint so we don't starve SSM state allocation.
        size_t single_block_bytes;
        if (config_.kv_cache_dtype == DType::INT4) {
            // INT4: 0.5 bytes per element (2 elements packed per byte)
            single_block_bytes = static_cast<size_t>(kKVBlockSize) *
                                 mcfg.n_kv_heads * head_dim / 2;
        } else {
            single_block_bytes = static_cast<size_t>(kKVBlockSize) *
                                 mcfg.n_kv_heads * head_dim * dtype_size(config_.kv_cache_dtype);
        }
        size_t per_block_total = single_block_bytes * 2 * n_kv_layers;
        // INT8/INT4 scale overhead: one half per head per token per block (K+V)
        if (config_.kv_cache_dtype == DType::INT8 ||
            config_.kv_cache_dtype == DType::INT4) {
            size_t scale_per_block = static_cast<size_t>(kKVBlockSize) *
                                     mcfg.n_kv_heads * sizeof(half);
            per_block_total += scale_per_block * 2 * n_kv_layers;
        }

        // Estimate SSM state footprint for hybrid models
        size_t ssm_footprint = 0;
        if (mcfg.ssm_inner_size > 0) {
            int n_ssm_layers = 0;
            for (int i = 0; i < mcfg.n_layers; i++) {
                if (model_->layer(i).ssm_in.data != nullptr) n_ssm_layers++;
            }
            if (n_ssm_layers > 0) {
                int conv_channels = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
                int n_heads = mcfg.ssm_dt_rank;
                int head_dim_ssm = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
                size_t ssm_elem = dtype_size(config_.ssm_state_dtype);
                // conv_state: [n_ssm_layers * max_batch * conv_channels * (kernel-1)]
                ssm_footprint += static_cast<size_t>(n_ssm_layers) * config_.max_batch_size *
                                 conv_channels * std::max(mcfg.ssm_conv_kernel - 1, 0) * sizeof(float);
                // h_state: [n_ssm_layers * max_batch * n_heads * head_dim_ssm * state_size]
                ssm_footprint += static_cast<size_t>(n_ssm_layers) * config_.max_batch_size *
                                 n_heads * head_dim_ssm * mcfg.ssm_state_size * ssm_elem;
            }
        }

        // Target: 2x headroom for batching / prefix caching.
        // In NVFP4-only mode (2), skip headroom — every MiB freed goes to
        // weight caching, which directly improves decode throughput.
        int target_blocks = (config_.use_nvfp4_decode == 2)
                            ? needed_blocks
                            : needed_blocks * 2;

        size_t free_mem = effective_free_vram();
        if (free_mem == 0 || per_block_total == 0) {
            max_blocks = needed_blocks;
        } else {
            // Reserve space for SSM state + 256 MiB safety margin, then cap KV
            // at a fraction of what remains to leave room for weight caching.
            // In NVFP4-only mode (2), use only 10% for KV — each extra MiB of
            // NVFP4 weight cache directly improves decode tok/s, while excess
            // KV blocks sit unused for typical context lengths.
            constexpr size_t kSafetyMarginBytes = 256ULL * 1024 * 1024;
            size_t reserved = ssm_footprint + kSafetyMarginBytes;
            size_t available = (free_mem > reserved) ? (free_mem - reserved) : 0;
            double kv_fraction = (config_.use_nvfp4_decode == 2) ? 0.1 : 0.8;
            size_t kv_budget = static_cast<size_t>(available * kv_fraction);
            int budget_blocks = static_cast<int>(kv_budget / per_block_total);
            max_blocks = std::min(target_blocks, budget_blocks);
        }

        // In NVFP4-only mode, KV budget takes priority over needed_blocks —
        // excess KV wastes VRAM that weight caching needs for decode throughput.
        // Sequences exceeding the KV capacity will fail gracefully at runtime.
        if (config_.use_nvfp4_decode != 2) {
            max_blocks = std::max(max_blocks, needed_blocks);
        }
        max_blocks = std::max(max_blocks, 16);
    }

    {
        DType kv_dtype = config_.kv_cache_dtype;
        size_t elem_size = dtype_size(kv_dtype);
        size_t block_bytes = static_cast<size_t>(kKVBlockSize) *
                             mcfg.n_kv_heads * head_dim * elem_size;
        size_t total_kv = static_cast<size_t>(n_kv_layers) * max_blocks * 2 * block_bytes;
        IMP_LOG_INFO("KV cache: %d blocks (%.0f tokens), %.2f MiB, dtype=%s "
                     "(layers=%d/%d, kv_heads=%d, head_dim=%d, block_size=%d)",
                     max_blocks,
                     static_cast<double>(max_blocks) * kKVBlockSize,
                     static_cast<double>(total_kv) / (1024.0 * 1024.0),
                     kv_dtype == DType::FP8_E4M3 ? "FP8_E4M3" :
                     kv_dtype == DType::INT8 ? "INT8" :
                     kv_dtype == DType::INT4 ? "INT4" : "FP16",
                     n_kv_layers, mcfg.n_layers, mcfg.n_kv_heads, head_dim, kKVBlockSize);
    }

    auto kv_cache = std::make_unique<KVCache>(
        n_kv_layers, mcfg.n_kv_heads, head_dim,
        config_.kv_cache_dtype, max_blocks);

    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));

    if (config_.use_prefix_caching) {
        kv_manager_->set_prefix_caching_enabled(true);
        IMP_LOG_INFO("Prefix caching enabled");

        // Restore prefix cache from disk if available
        if (!config_.prefix_cache_path.empty()) {
            int restored = kv_manager_->load_prefix_cache(config_.prefix_cache_path, stream_);
            if (restored > 0) {
                IMP_LOG_INFO("Restored %d prefix cache blocks from %s",
                             restored, config_.prefix_cache_path.c_str());
            }
        }
    }

    // Pass KV layer mapping to executor for correct cache indexing
    executor_->set_kv_layer_map(std::move(kv_layer_map));

    // Pass layer offload manager to executor (if enabled)
    if (offload_mgr_) {
        executor_->set_offload_manager(offload_mgr_.get());
    }

    // Wire up scheduler with KV cache manager for memory-aware scheduling
    scheduler_->set_kv_manager(kv_manager_.get());

    // --- Initialize SSM state for Mamba2 hybrid models ---
    if (mcfg.ssm_inner_size > 0) {
        // Count SSM layers
        int n_ssm_layers = 0;
        for (int i = 0; i < mcfg.n_layers; i++) {
            if (model_->layer(i).ssm_in.data != nullptr) n_ssm_layers++;
        }
        if (n_ssm_layers > 0) {
            int conv_channels = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int head_dim_ssm = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;

            ssm_state_ = std::make_unique<SSMState>();
            if (!ssm_state_->init(n_ssm_layers, config_.max_batch_size,
                                   conv_channels, mcfg.ssm_conv_kernel,
                                   n_heads, head_dim_ssm, mcfg.ssm_state_size,
                                   config_.ssm_state_dtype)) {
                IMP_LOG_WARN("Failed to init SSM state, continuing without it");
                ssm_state_.reset();
            }
        }
    }

    // Pre-dequantize quantized weights to FP16 for fast prefill GEMM.
    // Done eagerly at init time so the first real-world prefill isn't penalized
    // by ~380ms of dequant overhead (previously this was lazy on first prefill).
    // Decode (n=1) uses raw quantized dp4a GEMV and doesn't need FP16 cache.
    // Compute weight cache budget: free VRAM minus runtime reserve.
    // All fixed allocations (weights, KV, SSM, workspace) are done by now.
    size_t cache_budget = effective_free_vram();
    constexpr size_t kCacheReserveMiB = 1024;  // CUDA graphs + cuBLAS + driver
    cache_budget = (cache_budget > kCacheReserveMiB * 1024ULL * 1024)
                   ? (cache_budget - kCacheReserveMiB * 1024ULL * 1024) : 0;
    executor_->pre_dequant_weights(stream_, cache_budget);
    dequant_done_ = true;
    cudaStreamSynchronize(stream_);

    if (config_.use_fp8_prefill) {
        IMP_LOG_INFO("Weight cache: FP8 E4M3 (2x prefill throughput on sm_120)");
    }

    // --- Pre-allocate decode batch pool for stable CUDA Graph pointers ---
    {
        int max_blocks_per_seq = blocks_per_seq;
        decode_batch_pool_.allocate(config_.max_batch_size, max_blocks_per_seq);
    }

    // --- Report total GPU memory usage ---
    {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            size_t used = total_mem - free_mem;
            IMP_LOG_INFO("GPU memory: %.0f MiB used / %.0f MiB total (%.0f MiB free)",
                         used / (1024.0 * 1024.0),
                         total_mem / (1024.0 * 1024.0),
                         free_mem / (1024.0 * 1024.0));
        }
    }

    // --- Initialize green contexts if requested ---
    if (config_.use_green_contexts) {
        if (!green_ctx_.init(0, config_.green_ctx_prefill_ratio)) {
            // Non-fatal: fall back to regular streams
        }
    }

    // --- Initialize speculative decoding if configured ---
    if (config_.enable_speculative) {
        if (!init_speculative()) {
            IMP_LOG_WARN("Speculative decoding init failed, continuing without it");
            config_.enable_speculative = false;
        }
    }

    // --- Initialize self-speculative decoding if configured ---
    if (config_.enable_self_speculative) {
        // Self-spec is incompatible with CUDA graphs (batch shape changes between draft/verify)
        if (config_.use_cuda_graphs) {
            IMP_LOG_INFO("Disabling CUDA graphs: self-speculative decoding active");
            config_.use_cuda_graphs = false;
        }
        self_spec_decoder_ = std::make_unique<SelfSpeculativeDecoder>();
        SelfSpecConfig ssc;
        ssc.spec_k = config_.self_spec_k;
        ssc.exit_layer = config_.self_spec_exit_layer;
        ssc.skip_n = config_.self_spec_skip_n;
        if (!self_spec_decoder_->init(executor_.get(), kv_manager_.get(),
                                       kv_cache_raw_, mcfg.n_layers, ssc)) {
            IMP_LOG_WARN("Self-speculative init failed, continuing without it");
            self_spec_decoder_.reset();
            config_.enable_self_speculative = false;
        }
    }

    // --- Initialize chat template from tokenizer metadata ---
    {
        Tokenizer* tok = model_->tokenizer();
        if (tok) {
            auto family = ChatTemplate::detect_family(tok->chat_template_str());
            if (family == ChatTemplateFamily::RAW) {
                family = ChatTemplate::default_family_for_arch(model_->config().arch);
                if (family != ChatTemplateFamily::RAW) {
                    IMP_LOG_INFO("No chat template in metadata, using %s default for %s",
                                 chat_template_family_name(family),
                                 model_arch_name(model_->config().arch));
                }
            }
            if (family != ChatTemplateFamily::RAW) {
                chat_template_.init(family, *tok);
            }
        }
    }

    // --- Initialize vision encoder if mmproj path provided ---
    if (!config_.mmproj_path.empty()) {
        vision_model_ = load_vision_gguf(config_.mmproj_path);
        if (!vision_model_) {
            IMP_LOG_ERROR("Failed to load vision model: %s", config_.mmproj_path.c_str());
            return false;
        }

        int lm_d = vision_model_->lm_d_model > 0 ? vision_model_->lm_d_model : mcfg.d_model;
        vision_encoder_ = std::make_unique<VisionEncoder>();
        if (!vision_encoder_->init(*vision_model_, lm_d, stream_)) {
            IMP_LOG_ERROR("Failed to init vision encoder");
            vision_encoder_.reset();
            vision_model_.reset();
            return false;
        }

        // Allocate device buffer for vision embeddings
        int n_img_tokens = vision_model_->config.num_image_tokens;
        size_t emb_bytes = static_cast<size_t>(n_img_tokens) * lm_d * sizeof(half);
        if (cudaMalloc(&d_vision_embeddings_, emb_bytes) != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate vision embedding buffer (%zu bytes)", emb_bytes);
            vision_encoder_.reset();
            vision_model_.reset();
            return false;
        }

        // Resolve vision special token IDs
        Tokenizer* tok = model_->tokenizer();
        if (tok) {
            // Try well-known IDs first, fall back to vocab search
            vision_soft_token_id_ = tok->find_token("<image_soft_token>");
            if (vision_soft_token_id_ < 0) {
                // Gemma-3: <image_soft_token> is token 262144
                if (mcfg.vocab_size > 262144) {
                    vision_soft_token_id_ = 262144;
                }
            }
            vision_boi_id_ = tok->find_token("<start_of_image>");
            if (vision_boi_id_ < 0 && mcfg.vocab_size > 255999)
                vision_boi_id_ = 255999;
            vision_eoi_id_ = tok->find_token("<end_of_image>");
            if (vision_eoi_id_ < 0 && mcfg.vocab_size > 256000)
                vision_eoi_id_ = 256000;
            IMP_LOG_INFO("Vision tokens: soft=%d, boi=%d, eoi=%d",
                         vision_soft_token_id_, vision_boi_id_, vision_eoi_id_);
        }

        IMP_LOG_INFO("Vision encoder ready: %d image tokens -> %d-dim embeddings",
                     n_img_tokens, lm_d);
    }

    // Allocate pinned host buffer for graph-captured sampling and prefill event sync
    if (!h_sample_pinned_) {
        cudaError_t err = cudaHostAlloc(&h_sample_pinned_, sizeof(int32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostAlloc for sample buffer failed: %s",
                         cudaGetErrorString(err));
            if (config_.use_cuda_graphs) config_.use_cuda_graphs = false;
            h_sample_pinned_ = nullptr;
        }
    }

    // Lightweight event for decode spin-poll sync (no timing overhead)
    if (!decode_done_) {
        decode_done_.create(cudaEventDisableTiming);
    }

    // Warmup: run one prefill+decode step to prime cuBLAS algorithm selection,
    // NVFP4 kernels, and attention dispatch.  Without this, the first real
    // request on some models (Qwen3) produces incorrect output due to
    // non-deterministic kernel state on first invocation.
    {
        Tokenizer* tok = model_->tokenizer();
        int32_t warmup_id = tok ? tok->bos_id() : 1;
        if (warmup_id < 0) warmup_id = 1;

        // Use a multi-token prompt to prime both prefill GEMM shapes and decode
        // GEMV paths.  Single-token warmup only primes decode; the first prefill
        // with a real prompt would still trigger cuBLAS autotuning.
        auto req = std::make_shared<Request>();
        req->id = next_request_id_++;
        req->input_tokens.resize(16, warmup_id);
        req->max_tokens = 4;
        req->temperature = 0.0f;
        req->ignore_eos = true;
        scheduler_->add_request(req);

        // Run prefill + decode steps to prime all kernel paths
        for (int i = 0; i < 8 && req->status != RequestStatus::FINISHED; i++) {
            step();
        }

        // Clean up warmup state
        kv_manager_->free_sequence(req->id);
        req->status = RequestStatus::CANCELLED;
        decode_graph_runner_.invalidate();
        decode_batch_pool_.reset_upload_cache();
        if (async_graph_runner_.is_setup()) {
            async_graph_runner_.cleanup();
        }
        if (async_d_block_tables_) {
            cudaFree(async_d_block_tables_);
            async_d_block_tables_ = nullptr;
        }
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;
        cudaStreamSynchronize(stream_);
        IMP_LOG_INFO("Warmup complete");
    }

    return true;
}

bool Engine::init_speculative() {
    if (config_.draft_model_path.empty()) {
        IMP_LOG_ERROR("Speculative decoding enabled but no draft model path provided");
        return false;
    }

    // Load draft model via GGUF loader (returns unique_ptr, convert to shared)
    auto draft_unique = load_gguf(config_.draft_model_path);
    if (!draft_unique) {
        IMP_LOG_ERROR("Failed to load draft model: %s", config_.draft_model_path.c_str());
        return false;
    }
    draft_model_ = std::move(draft_unique);

    // Upload draft model weights
    if (!draft_model_->upload_weights_gpu(config_.compute_dtype, stream_)) {
        IMP_LOG_ERROR("Failed to upload draft model weights");
        return false;
    }

    // Create draft KV cache (smaller, matching draft model dimensions)
    const auto& dcfg = draft_model_->config();
    int head_dim = dcfg.head_dim > 0 ? dcfg.head_dim : (dcfg.d_model / dcfg.n_heads);
    int draft_max_blocks = std::max(kv_cache_raw_->total_blocks() / 4, 64);
    auto draft_kv = std::make_unique<KVCache>(
        dcfg.n_layers, dcfg.n_kv_heads, head_dim,
        config_.compute_dtype, draft_max_blocks);
    draft_kv_manager_ = std::make_unique<KVCacheManager>(std::move(draft_kv));

    // Create draft executor
    auto draft_exec = std::make_unique<GraphExecutor>();
    if (!draft_exec->init(*draft_model_, config_.compute_dtype, config_.use_pdl)) {
        IMP_LOG_ERROR("Failed to init draft executor");
        return false;
    }

    // Create speculative decoder
    spec_decoder_ = std::make_unique<SpeculativeDecoder>();
    SpeculativeConfig spec_cfg;
    spec_cfg.spec_k = config_.spec_k;
    if (!spec_decoder_->init(executor_.get(), draft_model_, std::move(draft_exec),
                              kv_manager_.get(), draft_kv_manager_.get(), spec_cfg)) {
        IMP_LOG_ERROR("Failed to init speculative decoder");
        return false;
    }

    IMP_LOG_INFO("Speculative decoding enabled: draft=%s, k=%d",
                 config_.draft_model_path.c_str(), config_.spec_k);
    return true;
}

bool Engine::set_draft_model(const std::string& path, int spec_k) {
    if (path.empty()) {
        IMP_LOG_ERROR("set_draft_model: empty path");
        return false;
    }
    if (spec_decoder_) {
        IMP_LOG_ERROR("set_draft_model: draft model already set");
        return false;
    }
    config_.draft_model_path = path;
    config_.spec_k = spec_k;
    config_.enable_speculative = true;
    return init_speculative();
}

bool Engine::set_image(const std::string& path) {
    if (!vision_encoder_) {
        IMP_LOG_ERROR("set_image: no vision model loaded (missing --mmproj)");
        return false;
    }

    ImageData img;
    if (!load_and_preprocess_image(path, vision_model_->config.image_size,
                                    vision_model_->config.image_mean,
                                    vision_model_->config.image_std, img)) {
        return false;
    }

    // Upload pixels to GPU
    int n_pixels = 3 * img.width * img.height;
    half* d_pixels = nullptr;
    if (cudaMalloc(&d_pixels, n_pixels * sizeof(half)) != cudaSuccess) {
        IMP_LOG_ERROR("set_image: cudaMalloc failed for %d pixels", n_pixels);
        return false;
    }
    cudaMemcpyAsync(d_pixels, img.pixels.data(), n_pixels * sizeof(half),
                    cudaMemcpyHostToDevice, stream_);

    // Encode — sync before freeing d_pixels (encode runs async on stream_)
    bool ok = vision_encoder_->encode(d_pixels, d_vision_embeddings_, stream_);
    cudaStreamSynchronize(stream_);
    cudaFree(d_pixels);

    if (ok) {
        has_vision_input_ = true;
        IMP_LOG_INFO("Vision: encoded image -> %d tokens", vision_model_->config.num_image_tokens);
    }
    return ok;
}

bool Engine::set_image_from_memory(const uint8_t* data, size_t len) {
    if (!vision_encoder_) {
        IMP_LOG_ERROR("set_image_from_memory: no vision model loaded");
        return false;
    }

    ImageData img;
    if (!load_and_preprocess_image_from_memory(data, len,
                                                vision_model_->config.image_size,
                                                vision_model_->config.image_mean,
                                                vision_model_->config.image_std, img)) {
        return false;
    }

    int n_pixels = 3 * img.width * img.height;
    half* d_pixels = nullptr;
    if (cudaMalloc(&d_pixels, n_pixels * sizeof(half)) != cudaSuccess) {
        IMP_LOG_ERROR("set_image_from_memory: cudaMalloc failed for %d pixels", n_pixels);
        return false;
    }
    cudaMemcpyAsync(d_pixels, img.pixels.data(), n_pixels * sizeof(half),
                    cudaMemcpyHostToDevice, stream_);

    bool ok = vision_encoder_->encode(d_pixels, d_vision_embeddings_, stream_);
    cudaStreamSynchronize(stream_);
    cudaFree(d_pixels);

    if (ok) {
        has_vision_input_ = true;
        IMP_LOG_INFO("Vision: encoded image from memory -> %d tokens",
                     vision_model_->config.num_image_tokens);
    }
    return ok;
}

void Engine::clear_image() {
    has_vision_input_ = false;
}

bool Engine::step() {
    // ====================================================================
    // Fast path: async conditional graph loop completed on GPU.
    // All tokens were generated at full GPU speed (no per-step host
    // overhead). We deliver them one per step() call from the buffer.
    // ====================================================================
    if (async_graph_runner_.is_setup() && async_graph_req_) {
        auto& req = async_graph_req_;

        // First entry: sync on GPU completion and collect all tokens.
        // WSL2's GPU-PV delays mapped memory writes, so we must sync
        // before reading the ring buffer (polling without sync is unreliable).
        if (async_pending_tokens_.empty() && async_pending_cursor_ == 0) {
            cudaStream_t dec_stream = decode_stream();
            async_pending_tokens_ = async_graph_runner_.wait_and_get_tokens(dec_stream);
        }

        // Deliver one token per step() call
        int32_t token = -1;
        if (async_pending_cursor_ < static_cast<int>(async_pending_tokens_.size())) {
            token = async_pending_tokens_[async_pending_cursor_++];
        }

        bool generation_done = false;
        if (token >= 0) {
            req->output_tokens.push_back(token);

            // Check stop conditions (respect ignore_eos for bench mode)
            Tokenizer* tok = model_->tokenizer();
            bool is_stop = false;
            if (!req->ignore_eos) {
                is_stop = (token == tok->eos_id());
                for (int32_t stop_id : chat_template_.stop_token_ids()) {
                    if (token == stop_id) { is_stop = true; break; }
                }
            }
            generation_done = is_stop ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (!generation_done) return true;  // more graph tokens to deliver
        }

        // Save request before clearing async state (req is a reference to
        // async_graph_req_ which we're about to null).
        auto saved_req = async_graph_req_;

        // Clean up async graph state
        async_graph_runner_.cleanup();
        if (async_d_block_tables_) {
            cudaFree(async_d_block_tables_);
            async_d_block_tables_ = nullptr;
        }
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;

        if (generation_done) {
            // Stop/max_tokens reached — request is truly finished
            saved_req->status = RequestStatus::FINISHED;
            // Register block hashes for prefix caching before freeing.
            if (kv_manager_->prefix_caching_enabled()) {
                kv_manager_->register_block_hashes(
                    saved_req->id, saved_req->input_tokens.data(),
                    static_cast<int>(saved_req->input_tokens.size()));
            }
            kv_manager_->free_sequence(saved_req->id);
            return scheduler_->has_pending() || scheduler_->active_count() > 0;
        }

        // Graph exhausted its pre-allocated tokens but generation isn't done.
        // Fall through to regular per-step decode below.
        IMP_LOG_DEBUG("AsyncGraphLoop: graph tokens exhausted, continuing with step decode");
    }

    // Clean up stale async graph state
    if (async_graph_req_ && !async_graph_runner_.is_setup()) {
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;
    }
    // 1. Call scheduler to get prefill and decode batches
    std::vector<std::shared_ptr<Request>> prefill_batch;
    std::vector<std::shared_ptr<Request>> decode_batch;
    scheduler_->schedule(prefill_batch, decode_batch);

    if (prefill_batch.empty() && decode_batch.empty()) {
        return false;
    }

    // ====================================================================
    // 2. Process prefill requests (per-request, no cross-sequence batching)
    // ====================================================================
    cudaStream_t pf_stream = prefill_stream();

    for (auto& req : prefill_batch) {
        int total_input = static_cast<int>(req->input_tokens.size());
        int offset = req->prefill_offset;

        // Determine chunk boundaries
        int chunk_len = total_input - offset;
        bool is_last_chunk = true;
        if (config_.prefill_chunk_size > 0 && chunk_len > config_.prefill_chunk_size) {
            chunk_len = config_.prefill_chunk_size;
            is_last_chunk = false;
        }

        // Context length covers all tokens up to end of this chunk
        int ctx_len = offset + chunk_len;

        // Resize workspace for this chunk's token count
        executor_->resize_workspace(chunk_len, pf_stream);

        int num_blocks = (ctx_len + kKVBlockSize - 1) / kKVBlockSize;

        // Allocate KV cache blocks, using prefix caching when enabled.
        int prefix_reused = 0;
        int existing = static_cast<int>(kv_manager_->block_table(req->id).size());

        if (kv_manager_->prefix_caching_enabled() && existing == 0 && offset == 0) {
            // First chunk of a fresh sequence — try content-addressed prefix match.
            // Allocate all blocks for the full input at once (not just this chunk).
            int total_blocks_needed = (total_input + kKVBlockSize - 1) / kKVBlockSize;
            prefix_reused = kv_manager_->allocate_blocks_with_prefix(
                req->id, req->input_tokens.data(), total_input);
            if (prefix_reused < 0) {
                // Allocation failed — try eviction.
                while (kv_manager_->num_free_blocks() < total_blocks_needed) {
                    int evicted = kv_manager_->evict_lru();
                    if (evicted < 0) break;
                }
                prefix_reused = kv_manager_->allocate_blocks_with_prefix(
                    req->id, req->input_tokens.data(), total_input);
                if (prefix_reused < 0) {
                    req->status = RequestStatus::CANCELLED;
                    continue;
                }
            }

            // Skip prefill for tokens covered by reused blocks.
            if (prefix_reused > 0) {
                int skip_tokens = prefix_reused * kKVBlockSize;
                // Must keep at least 1 token for the forward pass.
                if (skip_tokens >= total_input) {
                    skip_tokens = (total_input / kKVBlockSize) * kKVBlockSize;
                    if (skip_tokens >= total_input) {
                        skip_tokens = total_input - 1;
                    }
                }
                if (skip_tokens > offset) {
                    IMP_LOG_INFO("PrefixCache: seq %d skipping %d/%d prefill tokens (%d blocks reused)",
                                 req->id, skip_tokens, total_input, prefix_reused);
                    offset = skip_tokens;
                    req->prefill_offset = offset;
                    // Recalculate chunk boundaries with new offset.
                    chunk_len = total_input - offset;
                    is_last_chunk = true;
                    if (config_.prefill_chunk_size > 0 && chunk_len > config_.prefill_chunk_size) {
                        chunk_len = config_.prefill_chunk_size;
                        is_last_chunk = false;
                    }
                    ctx_len = offset + chunk_len;
                    // Re-resize workspace for the smaller chunk.
                    executor_->resize_workspace(chunk_len, pf_stream);
                }
            }
        } else {
            // Normal incremental allocation.
            int additional = num_blocks - existing;
            if (additional > 0) {
                if (!kv_manager_->allocate_blocks(req->id, additional)) {
                    while (kv_manager_->num_free_blocks() < additional) {
                        int evicted = kv_manager_->evict_lru();
                        if (evicted < 0) break;
                    }
                    if (!kv_manager_->allocate_blocks(req->id, additional)) {
                        req->status = RequestStatus::CANCELLED;
                        continue;
                    }
                }
            }
        }

        const auto& block_table = kv_manager_->block_table(req->id);

        // Positions for this chunk start at offset
        std::vector<int> positions(chunk_len);
        for (int i = 0; i < chunk_len; i++) {
            positions[i] = offset + i;
        }

        // Upload to device
        int32_t* d_token_ids = nullptr;
        int* d_positions = nullptr;
        int* d_block_tables = nullptr;
        int* d_context_lens = nullptr;

        auto check = [&req](cudaError_t err, const char* op) {
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("Engine::step prefill %s failed: %s", op, cudaGetErrorString(err));
                req->status = RequestStatus::CANCELLED;
            }
            return err == cudaSuccess;
        };
        if (!check(cudaMallocAsync(&d_token_ids, chunk_len * sizeof(int32_t), pf_stream), "malloc token_ids") ||
            !check(cudaMallocAsync(&d_positions, chunk_len * sizeof(int), pf_stream), "malloc positions") ||
            !check(cudaMallocAsync(&d_block_tables, block_table.size() * sizeof(int), pf_stream), "malloc block_tables") ||
            !check(cudaMallocAsync(&d_context_lens, sizeof(int), pf_stream), "malloc context_lens")) {
            if (d_token_ids) cudaFreeAsync(d_token_ids, pf_stream);
            if (d_positions) cudaFreeAsync(d_positions, pf_stream);
            if (d_block_tables) cudaFreeAsync(d_block_tables, pf_stream);
            if (d_context_lens) cudaFreeAsync(d_context_lens, pf_stream);
            continue;
        }

        // Upload chunk tokens (starting from offset)
        check(cudaMemcpyAsync(d_token_ids, req->input_tokens.data() + offset,
                        chunk_len * sizeof(int32_t),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy token_ids");
        check(cudaMemcpyAsync(d_positions, positions.data(),
                        chunk_len * sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy positions");
        check(cudaMemcpyAsync(d_block_tables, block_table.data(),
                        block_table.size() * sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy block_tables");
        check(cudaMemcpyAsync(d_context_lens, &ctx_len, sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy context_lens");

        // Build InferenceState (single-sequence prefill)
        InferenceState state;
        state.token_ids = d_token_ids;
        state.positions = d_positions;
        state.n_tokens = chunk_len;
        state.kv_cache = kv_cache_raw_;
        state.block_tables = d_block_tables;
        state.context_lens = d_context_lens;
        state.max_context_len = ctx_len;
        state.n_sequences = 1;
        state.max_blocks_per_seq = 0;  // flat single-seq block table
        state.is_prefill = true;
        state.temperature = req->temperature;
        state.top_p = req->top_p;
        state.top_k = req->top_k;
        state.seed = req->seed;
        state.min_p = req->min_p;
        state.typical_p = req->typical_p;
        state.repetition_penalty = req->repetition_penalty;
        state.frequency_penalty = req->frequency_penalty;
        state.presence_penalty = req->presence_penalty;
        state.repeat_last_n = req->repeat_last_n;
        state.dry_multiplier = req->dry_multiplier;
        state.dry_base = req->dry_base;
        state.dry_allowed_length = req->dry_allowed_length;
        state.dry_penalty_last_n = req->dry_penalty_last_n;
        if (req->dry_multiplier > 0.0f && !req->output_tokens.empty())
            state.host_penalty_tokens = req->output_tokens.data();
        state.mirostat = req->mirostat;
        state.mirostat_tau = req->mirostat_tau;
        state.mirostat_eta = req->mirostat_eta;
        state.mirostat_mu = req->mirostat_mu;

        // JSON mode: lazily init constrainer and set on state
        if (req->json_mode && req->json_schema.empty()) {
            if (!json_constrainer_) {
                json_constrainer_ = std::make_unique<JsonConstrainer>();
                Tokenizer* jtok = model_->tokenizer();
                if (!json_constrainer_->init(*jtok)) {
                    IMP_LOG_ERROR("Failed to initialize JSON constrainer");
                    json_constrainer_.reset();
                }
            }
            if (json_constrainer_) {
                json_constrainer_->reset();
                state.json_constrainer = json_constrainer_.get();
            }
        }

        // Schema-constrained JSON mode
        if (!req->json_schema.empty()) {
            auto schema = parse_json_schema(req->json_schema);
            if (schema) {
                schema_constrainer_ = std::make_unique<SchemaConstrainer>();
                Tokenizer* stok = model_->tokenizer();
                if (schema_constrainer_->init(*stok, std::move(schema))) {
                    state.schema_constrainer = schema_constrainer_.get();
                } else {
                    IMP_LOG_ERROR("Failed to initialize schema constrainer");
                    schema_constrainer_.reset();
                }
            } else {
                IMP_LOG_ERROR("Failed to parse JSON schema");
            }
        }

        // Upload penalty token history if penalties are active
        bool needs_penalties = (req->repetition_penalty != 1.0f ||
                                req->frequency_penalty != 0.0f ||
                                req->presence_penalty != 0.0f);
        if (needs_penalties && !req->output_tokens.empty()) {
            size_t n = req->output_tokens.size();
            if (n > d_penalty_tokens_capacity_) {
                if (d_penalty_tokens_) cudaFree(d_penalty_tokens_);
                d_penalty_tokens_capacity_ = std::max(n, (size_t)256);
                if (cudaMalloc(&d_penalty_tokens_, d_penalty_tokens_capacity_ * sizeof(int32_t)) != cudaSuccess) {
                    IMP_LOG_ERROR("cudaMalloc failed for penalty tokens (%zu)", d_penalty_tokens_capacity_);
                    d_penalty_tokens_ = nullptr;
                    d_penalty_tokens_capacity_ = 0;
                }
            }
            if (d_penalty_tokens_) {
                cudaMemcpyAsync(d_penalty_tokens_, req->output_tokens.data(),
                                n * sizeof(int32_t), cudaMemcpyHostToDevice, pf_stream);
                state.penalty_tokens = d_penalty_tokens_;
                state.n_penalty_tokens = static_cast<int>(n);
            }
        }

        // SSM state for hybrid models
        if (ssm_state_) {
            state.ssm_state = ssm_state_.get();
            // Use request ID mod max_sequences as SSM sequence slot
            state.ssm_seq_id = req->id % ssm_state_->max_sequences();
            // Only reset on first chunk
            if (offset == 0) {
                ssm_state_->reset_sequence(state.ssm_seq_id, pf_stream);
            }
        }

        // Vision embeddings: set on first chunk only (image tokens are at the start)
        if (has_vision_input_ && vision_encoder_ && offset == 0) {
            state.vision_embeddings = d_vision_embeddings_;
            state.vision_token_id = vision_soft_token_id_;
            state.n_vision_tokens = vision_model_->config.num_image_tokens;
        }

        if (!is_last_chunk) {
            // Intermediate chunk: run forward to fill KV cache, discard logits
            Tensor logits_out;
            executor_->forward_logits(state, logits_out, pf_stream);

            cudaFreeAsync(d_token_ids, pf_stream);
            cudaFreeAsync(d_positions, pf_stream);
            cudaFreeAsync(d_block_tables, pf_stream);
            cudaFreeAsync(d_context_lens, pf_stream);

            // Advance offset, stay in PREFILLING
            req->prefill_offset = offset + chunk_len;
            IMP_LOG_DEBUG("Chunked prefill: req %d chunk [%d, %d) of %d",
                          req->id, offset, offset + chunk_len, total_input);
        } else {
            // Last chunk: run forward + sample
            int32_t next_token;
            bool use_event_sync = (h_sample_pinned_ != nullptr &&
                                   executor_->d_sample_result() != nullptr &&
                                   (state.temperature <= 0.0f || state.top_k == 1) &&
                                   !req->logprobs &&
                                   !state.json_constrainer &&
                                   !state.schema_constrainer);

            Tensor prefill_logits_out;  // retained for logprobs extraction

            if (use_event_sync) {
                Tensor logits_out;
                executor_->forward_logits(state, logits_out, pf_stream);
                Tensor last_logits = logits_out.slice(0, 1);
                int64_t vocab_shape[1] = {last_logits.shape[1]};
                last_logits = last_logits.reshape(1, vocab_shape);
                sample_greedy_device(last_logits, executor_->d_sample_result(),
                                      h_sample_pinned_, pf_stream);

                if (!prefill_done_) prefill_done_.create();
                cudaEventRecord(prefill_done_, pf_stream);

                cudaFreeAsync(d_token_ids, pf_stream);
                cudaFreeAsync(d_positions, pf_stream);
                cudaFreeAsync(d_block_tables, pf_stream);
                cudaFreeAsync(d_context_lens, pf_stream);

                cudaEventSynchronize(prefill_done_);
                next_token = *h_sample_pinned_;
            } else if (req->logprobs) {
                // forward_logits + sample separately to retain logits access
                executor_->forward_logits(state, prefill_logits_out, pf_stream);
                auto sampled = executor_->sample_from_logits(prefill_logits_out, state, pf_stream);
                next_token = sampled[0];

                cudaFreeAsync(d_token_ids, pf_stream);
                cudaFreeAsync(d_positions, pf_stream);
                cudaFreeAsync(d_block_tables, pf_stream);
                cudaFreeAsync(d_context_lens, pf_stream);
            } else {
                next_token = executor_->forward(state, pf_stream);

                cudaFreeAsync(d_token_ids, pf_stream);
                cudaFreeAsync(d_positions, pf_stream);
                cudaFreeAsync(d_block_tables, pf_stream);
                cudaFreeAsync(d_context_lens, pf_stream);
            }

            // Mirostat v2: write back updated mu to request
            if (req->mirostat == 2)
                req->mirostat_mu = state.mirostat_mu;

            // Extract logprobs for first token if requested
            if (req->logprobs && prefill_logits_out.data != nullptr) {
                int vocab_size = static_cast<int>(prefill_logits_out.shape[prefill_logits_out.ndim - 1]);
                executor_->ensure_logits_pinned(vocab_size);

                // Get last row (for prefill, forward_logits returns last token's logits)
                const float* d_logits = static_cast<const float*>(prefill_logits_out.data);

                cudaMemcpyAsync(executor_->h_logits_pinned(), d_logits,
                                vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, pf_stream);
                cudaStreamSynchronize(pf_stream);

                LogprobResult lp_result;
                compute_logprobs_cpu(executor_->h_logits_pinned(), vocab_size,
                                     next_token, req->top_logprobs, &lp_result);

                Tokenizer* ptok = model_->tokenizer();
                TokenLogprobInfo info;
                info.logprob = lp_result.sampled_logprob;
                info.text = ptok->decode_token(next_token);
                info.top.reserve(lp_result.top.size());
                for (const auto& [tid, tlp] : lp_result.top) {
                    info.top.push_back({tid, tlp, ptok->decode_token(tid)});
                }
                req->output_logprobs.push_back(std::move(info));
            }

            req->output_tokens.push_back(next_token);

            Tokenizer* tok = model_->tokenizer();
            IMP_LOG_DEBUG("Prefill -> token %d (ctx=%d): id=%d [%s]",
                          (int)req->output_tokens.size(), req->context_len(),
                          next_token, tok->decode_token(next_token).c_str());

            // Check EOS and chat template stop tokens
            bool is_stop = false;
            if (!req->ignore_eos) {
                is_stop = (next_token == tok->eos_id());
                for (int32_t stop_id : chat_template_.stop_token_ids()) {
                    if (next_token == stop_id) { is_stop = true; break; }
                }
            }

            // Update JSON/schema constrainer FSM with the first token
            if (schema_constrainer_) {
                schema_constrainer_->update(next_token);
            } else if (req->json_mode && json_constrainer_) {
                json_constrainer_->update(next_token);
            }

            if (is_stop ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
                req->status = RequestStatus::FINISHED;
                // Register block hashes before freeing so blocks can be cached.
                if (kv_manager_->prefix_caching_enabled()) {
                    kv_manager_->register_block_hashes(
                        req->id, req->input_tokens.data(),
                        static_cast<int>(req->input_tokens.size()));
                }
                kv_manager_->free_sequence(req->id);
                if (req->json_mode && json_constrainer_) json_constrainer_->reset();
            } else {
                req->status = RequestStatus::DECODING;
                // Register block hashes so future sequences can reuse prefix blocks.
                if (kv_manager_->prefix_caching_enabled()) {
                    kv_manager_->register_block_hashes(
                        req->id, req->input_tokens.data(),
                        static_cast<int>(req->input_tokens.size()));
                }
            }
        }

        kv_manager_->touch(req->id);
    }

    // ====================================================================
    // 3. Process decode requests (BATCHED)
    // ====================================================================
    if (!decode_batch.empty()) {
        cudaStream_t dec_stream = decode_stream();

        // 3a. Pre-process: allocate new blocks where needed
        std::vector<std::shared_ptr<Request>> valid_decode;
        valid_decode.reserve(decode_batch.size());

        for (auto& req : decode_batch) {
            int ctx_len = req->context_len();
            int blocks_needed = (ctx_len + kKVBlockSize - 1) / kKVBlockSize;
            const auto& block_table = kv_manager_->block_table(req->id);
            int blocks_have = static_cast<int>(block_table.size());

            if (blocks_needed > blocks_have) {
                int new_block = kv_manager_->append_block(req->id);
                if (new_block < 0) {
                    int evicted = kv_manager_->evict_lru();
                    if (evicted >= 0) {
                        new_block = kv_manager_->append_block(req->id);
                    }
                    if (new_block < 0) {
                        req->status = RequestStatus::CANCELLED;
                        continue;
                    }
                }
            }
            valid_decode.push_back(req);
        }

        if (!valid_decode.empty()) {
            // ── Self-speculative decode shortcut (single-sequence only) ──
            if (self_spec_decoder_ && config_.enable_self_speculative &&
                valid_decode.size() == 1) {
                auto& req = valid_decode[0];
                int32_t last_token = req->output_tokens.empty()
                    ? req->input_tokens.back()
                    : req->output_tokens.back();
                int position = req->context_len() - 1;

                auto spec_tokens = self_spec_decoder_->step(
                    last_token, position, req->id,
                    req->temperature, req->top_p, req->top_k, req->seed,
                    dec_stream);

                Tokenizer* tok = model_->tokenizer();
                for (int32_t t : spec_tokens) {
                    req->output_tokens.push_back(t);

                    IMP_LOG_DEBUG("SelfSpec decode (ctx=%d): id=%d [%s]",
                                  req->context_len(), t,
                                  tok ? tok->decode_token(t).c_str() : "?");

                    bool is_stop = false;
                    if (!req->ignore_eos && tok) {
                        is_stop = (t == tok->eos_id());
                        for (int32_t stop_id : chat_template_.stop_token_ids()) {
                            if (t == stop_id) { is_stop = true; break; }
                        }
                    }
                    if (is_stop ||
                        static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
                        req->status = RequestStatus::FINISHED;
                        kv_manager_->free_sequence(req->id);
                        break;
                    }
                }
                kv_manager_->touch(req->id);

                // Skip normal batched decode
                goto decode_done;
            }

            // Resize workspace for decode batch size (much smaller than prefill)
            executor_->resize_workspace(static_cast<int>(valid_decode.size()), dec_stream);

            // 3b. Build batched decode using BatchBuilder
            BatchBuilder builder;
            builder.reset();

            int max_ctx = 0;
            for (auto& req : valid_decode) {
                int ctx_len = req->context_len();
                max_ctx = std::max(max_ctx, ctx_len);

                int32_t last_token = req->output_tokens.empty()
                    ? req->input_tokens.back()
                    : req->output_tokens.back();
                int position = ctx_len - 1;

                const auto& bt = kv_manager_->block_table(req->id);
                builder.add_decode_sequence(last_token, position,
                                            bt.data(), static_cast<int>(bt.size()),
                                            ctx_len);
            }

            Batch batch = builder.build();

            // 3c. Upload to GPU using pre-allocated pool (stable pointers for CUDA Graph)
            GPUBatch gpu_batch;
            if (decode_batch_pool_.is_allocated()) {
                // Pad max_blocks_per_seq to pool capacity so the block_table
                // stride is stable across tokens. This prevents CUDA graph
                // invalidation every time a new KV cache block is allocated.
                int pool_max = decode_batch_pool_.max_blocks_per_seq();
                if (batch.max_blocks_per_seq < pool_max) {
                    // Re-pad the 2D block_table to the pool's stride (reuse member buffer)
                    int n_seq = batch.n_sequences;
                    int old_stride = batch.max_blocks_per_seq;
                    size_t needed = static_cast<size_t>(n_seq) * pool_max;
                    padded_block_table_.resize(needed);
                    std::memset(padded_block_table_.data(), 0, needed * sizeof(int));
                    for (int s = 0; s < n_seq; s++) {
                        for (int b = 0; b < old_stride; b++) {
                            padded_block_table_[s * pool_max + b] = batch.block_tables[s * old_stride + b];
                        }
                    }
                    batch.block_tables = padded_block_table_;
                    batch.max_blocks_per_seq = pool_max;
                }
                gpu_batch = decode_batch_pool_.upload_into_pool(batch, dec_stream);
            } else {
                gpu_batch.upload(batch, dec_stream);
            }

            // 3d. Build batched InferenceState
            InferenceState state;
            state.token_ids = gpu_batch.d_token_ids;
            state.positions = gpu_batch.d_positions;
            state.n_tokens = gpu_batch.total_tokens;  // = n_sequences for decode
            state.n_sequences = gpu_batch.n_sequences;
            state.max_blocks_per_seq = gpu_batch.max_blocks_per_seq;
            state.kv_cache = kv_cache_raw_;
            state.block_tables = gpu_batch.d_block_tables;
            state.context_lens = gpu_batch.d_context_lens;
            state.max_context_len = max_ctx;
            state.is_prefill = false;
            state.temperature = valid_decode[0]->temperature;
            state.top_p = valid_decode[0]->top_p;
            state.top_k = valid_decode[0]->top_k;
            state.seed = -1;
            state.min_p = valid_decode[0]->min_p;
            state.typical_p = valid_decode[0]->typical_p;
            state.repetition_penalty = valid_decode[0]->repetition_penalty;
            state.frequency_penalty = valid_decode[0]->frequency_penalty;
            state.presence_penalty = valid_decode[0]->presence_penalty;
            state.repeat_last_n = valid_decode[0]->repeat_last_n;
            state.dry_multiplier = valid_decode[0]->dry_multiplier;
            state.dry_base = valid_decode[0]->dry_base;
            state.dry_allowed_length = valid_decode[0]->dry_allowed_length;
            state.dry_penalty_last_n = valid_decode[0]->dry_penalty_last_n;
            if (valid_decode[0]->dry_multiplier > 0.0f &&
                !valid_decode[0]->output_tokens.empty())
                state.host_penalty_tokens = valid_decode[0]->output_tokens.data();
            state.mirostat = valid_decode[0]->mirostat;
            state.mirostat_tau = valid_decode[0]->mirostat_tau;
            state.mirostat_eta = valid_decode[0]->mirostat_eta;
            state.mirostat_mu = valid_decode[0]->mirostat_mu;

            // Upload penalty token history for decode (single-sequence only)
            {
                auto* req0 = valid_decode[0].get();
                bool needs_penalties = (req0->repetition_penalty != 1.0f ||
                                        req0->frequency_penalty != 0.0f ||
                                        req0->presence_penalty != 0.0f);
                if (needs_penalties && !req0->output_tokens.empty() &&
                    gpu_batch.n_sequences == 1) {
                    size_t n = req0->output_tokens.size();
                    if (n > d_penalty_tokens_capacity_) {
                        if (d_penalty_tokens_) cudaFree(d_penalty_tokens_);
                        d_penalty_tokens_capacity_ = std::max(n, (size_t)256);
                        if (cudaMalloc(&d_penalty_tokens_, d_penalty_tokens_capacity_ * sizeof(int32_t)) != cudaSuccess) {
                            IMP_LOG_ERROR("cudaMalloc failed for penalty tokens (%zu)", d_penalty_tokens_capacity_);
                            d_penalty_tokens_ = nullptr;
                            d_penalty_tokens_capacity_ = 0;
                        }
                    }
                    if (d_penalty_tokens_) {
                        cudaMemcpyAsync(d_penalty_tokens_, req0->output_tokens.data(),
                                        n * sizeof(int32_t), cudaMemcpyHostToDevice, dec_stream);
                        state.penalty_tokens = d_penalty_tokens_;
                        state.n_penalty_tokens = static_cast<int>(n);
                    }
                }
            }

            // SSM state for hybrid models (decode uses first sequence's slot)
            if (ssm_state_) {
                state.ssm_state = ssm_state_.get();
                state.ssm_seq_id = valid_decode[0]->id % ssm_state_->max_sequences();
            }

            // Check if any request needs logprobs or json/schema mode
            bool needs_logprobs = false;
            bool needs_json_mode = false;
            bool needs_schema_mode = false;
            for (const auto& r : valid_decode) {
                if (r->logprobs) needs_logprobs = true;
                if (r->json_mode && r->json_schema.empty()) needs_json_mode = true;
                if (!r->json_schema.empty()) needs_schema_mode = true;
            }

            // Lazily initialize JSON constrainer on first json_mode request
            if (needs_json_mode && !json_constrainer_) {
                json_constrainer_ = std::make_unique<JsonConstrainer>();
                Tokenizer* jtok = model_->tokenizer();
                if (!json_constrainer_->init(*jtok)) {
                    IMP_LOG_ERROR("Failed to initialize JSON constrainer");
                    json_constrainer_.reset();
                    needs_json_mode = false;
                }
            }

            // Schema constrainer: reuse existing (state persists across decode steps)
            if (needs_schema_mode && valid_decode.size() == 1 &&
                !valid_decode[0]->json_schema.empty()) {
                if (schema_constrainer_ && schema_constrainer_->is_initialized()) {
                    state.schema_constrainer = schema_constrainer_.get();
                }
            }

            // Set JSON constrainer on InferenceState (single-sequence only)
            if (needs_json_mode && json_constrainer_ &&
                valid_decode.size() == 1 && valid_decode[0]->json_mode) {
                state.json_constrainer = json_constrainer_.get();
            }

            // Per-request sampling: each request uses its own sampling params.
            // The InferenceState 'state' carries valid_decode[0]'s params for the
            // greedy_single check; the actual sampling overrides per request below.
            auto sample_per_request = [&](const Tensor& logits) -> std::vector<int32_t> {
                int n = static_cast<int>(valid_decode.size());
                std::vector<int32_t> result(n);
                for (int i = 0; i < n; i++) {
                    auto& req = valid_decode[i];
                    InferenceState per_state = state;
                    per_state.temperature = req->temperature;
                    per_state.top_p = req->top_p;
                    per_state.top_k = req->top_k;
                    per_state.min_p = req->min_p;
                    per_state.typical_p = req->typical_p;
                    per_state.seed = req->seed;
                    per_state.repetition_penalty = req->repetition_penalty;
                    per_state.frequency_penalty = req->frequency_penalty;
                    per_state.presence_penalty = req->presence_penalty;
                    per_state.repeat_last_n = req->repeat_last_n;
                    per_state.dry_multiplier = req->dry_multiplier;
                    per_state.dry_base = req->dry_base;
                    per_state.dry_allowed_length = req->dry_allowed_length;
                    per_state.dry_penalty_last_n = req->dry_penalty_last_n;
                    if (req->dry_multiplier > 0.0f && !req->output_tokens.empty())
                        per_state.host_penalty_tokens = req->output_tokens.data();
                    per_state.mirostat = req->mirostat;
                    per_state.mirostat_tau = req->mirostat_tau;
                    per_state.mirostat_eta = req->mirostat_eta;
                    per_state.mirostat_mu = req->mirostat_mu;
                    per_state.n_sequences = 1;
                    Tensor seq_logits = logits.slice(i, i + 1);
                    auto t = executor_->sample_from_logits(seq_logits, per_state, dec_stream);
                    result[i] = t[0];
                    if (per_state.mirostat == 2)
                        req->mirostat_mu = per_state.mirostat_mu;
                }
                return result;
            };

            // 3e. Execute batched forward pass (with CUDA Graph when enabled)
            std::vector<int32_t> tokens;
            Tensor decode_logits_out;  // needed when logprobs are requested

            static const bool profiling = (std::getenv("IMP_PROFILE") != nullptr);
            if (config_.use_cuda_graphs && !profiling &&
                gpu_batch.n_sequences > 0 &&
                decode_batch_pool_.is_allocated()) {
                // Invalidate graph when batch config changes
                if (gpu_batch.n_sequences != last_decode_batch_size_ ||
                    gpu_batch.max_blocks_per_seq != last_decode_max_blocks_) {
                    decode_graph_runner_.invalidate();
                    last_decode_batch_size_ = gpu_batch.n_sequences;
                    last_decode_max_blocks_ = gpu_batch.max_blocks_per_seq;
                }

                // Check if we can include greedy sampling in the graph.
                // This captures argmax + D2H memcpy inside the graph, eliminating
                // separate kernel launch + sync overhead per step.
                // Disable when logprobs, json_mode, or penalties are active
                // (penalties modify logits each step with changing token history).
                bool has_penalties = (state.penalty_tokens != nullptr &&
                                     state.n_penalty_tokens > 0 &&
                                     (state.repetition_penalty != 1.0f ||
                                      state.frequency_penalty != 0.0f ||
                                      state.presence_penalty != 0.0f));
                bool greedy_single = (state.temperature <= 0.0f || state.top_k == 1) &&
                                     gpu_batch.n_sequences == 1 &&
                                     h_sample_pinned_ != nullptr &&
                                     executor_->d_sample_result() != nullptr &&
                                     !needs_logprobs && !needs_json_mode &&
                                     !needs_schema_mode && !has_penalties;

                // Invalidate graph if sampling mode changed
                if (greedy_single != graph_includes_sampling_) {
                    decode_graph_runner_.invalidate();
                    graph_includes_sampling_ = greedy_single;
                }

                if (greedy_single) {
                    // Graph captures: forward_logits + argmax + D2H copy
                    decode_graph_runner_.set_decode_fn(
                        [this, &state](cudaStream_t s) {
                            Tensor logits_out;
                            executor_->forward_logits(state, logits_out, s);
                            if (logits_out.data == nullptr)
                                logits_out = executor_->get_logits_view(1);
                            int64_t vshape[1] = {logits_out.shape[logits_out.ndim - 1]};
                            Tensor flat = logits_out.slice(0, 1).reshape(1, vshape);
                            sample_greedy_device(flat, executor_->d_sample_result(),
                                                  h_sample_pinned_, s);
                        });
                    decode_graph_runner_.execute(dec_stream);
                    cudaEventRecord(decode_done_, dec_stream);
                    cudaEventSynchronize(decode_done_);
                    tokens = {*h_sample_pinned_};
                } else {
                    // Forward-only graph + sample outside
                    Tensor logits_out;
                    decode_graph_runner_.set_decode_fn(
                        [this, &state, &logits_out](cudaStream_t s) {
                            executor_->forward_logits(state, logits_out, s);
                        });
                    decode_graph_runner_.execute(dec_stream);

                    if (logits_out.data == nullptr) {
                        logits_out = executor_->get_logits_view(gpu_batch.n_sequences);
                    }
                    tokens = sample_per_request(logits_out);
                    if (needs_logprobs) decode_logits_out = logits_out;
                }
            } else {
                executor_->forward_logits(state, decode_logits_out, dec_stream);
                tokens = sample_per_request(decode_logits_out);
            }

            // Only free if not using pool (pool memory is reused)
            if (!decode_batch_pool_.is_allocated()) {
                gpu_batch.free();
            }

            Tokenizer* tok = model_->tokenizer();

            // 3f. Extract logprobs (D2H copy + CPU computation) before distributing tokens
            if (needs_logprobs && decode_logits_out.data != nullptr) {
                int vocab_size = static_cast<int>(decode_logits_out.shape[decode_logits_out.ndim - 1]);
                executor_->ensure_logits_pinned(vocab_size);

                for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
                    auto& req = valid_decode[i];
                    if (!req->logprobs) continue;

                    // Get pointer to this sequence's logits row
                    const float* d_logits = static_cast<const float*>(decode_logits_out.data)
                        + static_cast<size_t>(i) * vocab_size;

                    // D2H copy
                    cudaMemcpyAsync(executor_->h_logits_pinned(), d_logits,
                                    vocab_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, dec_stream);
                    cudaStreamSynchronize(dec_stream);

                    // CPU logprobs computation
                    LogprobResult lp_result;
                    compute_logprobs_cpu(executor_->h_logits_pinned(), vocab_size,
                                         tokens[i], req->top_logprobs, &lp_result);

                    // Store in request
                    TokenLogprobInfo info;
                    info.logprob = lp_result.sampled_logprob;
                    info.text = tok->decode_token(tokens[i]);
                    info.top.reserve(lp_result.top.size());
                    for (const auto& [tid, tlp] : lp_result.top) {
                        info.top.push_back({tid, tlp, tok->decode_token(tid)});
                    }
                    req->output_logprobs.push_back(std::move(info));
                }
            }

            // Distribute sampled tokens back to requests
            for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
                auto& req = valid_decode[i];
                int32_t next_token = tokens[i];

                req->output_tokens.push_back(next_token);

                IMP_LOG_DEBUG("Decode step %d (ctx=%d, pos=%d): id=%d [%s]",
                              (int)req->output_tokens.size(), req->context_len(),
                              req->context_len() - 1,
                              next_token, tok->decode_token(next_token).c_str());

                // Check EOS and chat template stop tokens
                bool is_stop = false;
                if (!req->ignore_eos) {
                    is_stop = (next_token == tok->eos_id());
                    for (int32_t stop_id : chat_template_.stop_token_ids()) {
                        if (next_token == stop_id) { is_stop = true; break; }
                    }
                }

                if (is_stop ||
                    static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
                    req->status = RequestStatus::FINISHED;
                    kv_manager_->free_sequence(req->id);
                    // Reset constrainer when request finishes
                    if (schema_constrainer_) {
                        schema_constrainer_->reset();
                    } else if (req->json_mode && json_constrainer_) {
                        json_constrainer_->reset();
                    }
                }

                // Update constrainer FSM with the sampled token
                if (schema_constrainer_) {
                    schema_constrainer_->update(next_token);
                } else if (req->json_mode && json_constrainer_) {
                    json_constrainer_->update(next_token);
                }

                kv_manager_->touch(req->id);
            }

            // After first decode with graph ready, launch the async
            // conditional graph loop for all remaining tokens. This runs the
            // entire decode on GPU autonomously — subsequent step() calls just
            // poll from the ring buffer at full GPU speed.
            if (decode_graph_runner_.is_ready() && valid_decode.size() == 1 &&
                !offload_mgr_ && !ssm_state_ && !config_.enable_speculative &&
                config_.use_cuda_graphs && !async_graph_runner_.is_setup() &&
                !needs_logprobs && !needs_json_mode && !needs_schema_mode) {
                auto& dreq = valid_decode[0];
                bool dreq_has_penalties = (dreq->repetition_penalty != 1.0f ||
                                           dreq->frequency_penalty != 0.0f ||
                                           dreq->presence_penalty != 0.0f);
                if (dreq->status == RequestStatus::DECODING &&
                    !dreq->output_tokens.empty() && !dreq->ignore_eos &&
                    !dreq_has_penalties) {
                    int32_t last_token = dreq->output_tokens.back();
                    try_launch_async_graph_loop(dreq, last_token, dec_stream);
                }
            }
        }
    }
decode_done:

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
}

std::string Engine::generate(const std::string& prompt, int max_tokens,
                              float temperature, float top_p,
                              int top_k, int seed,
                              bool apply_chat_template,
                              float min_p,
                              float repetition_penalty,
                              float frequency_penalty,
                              float presence_penalty) {
    Tokenizer* tok = model_->tokenizer();
    if (!tok) {
        return "";
    }

    std::vector<int32_t> tokens;

    if (apply_chat_template && !chat_template_.is_raw()) {
        // Apply detected chat template (with image tokens if vision active)
        std::vector<ChatMessage> messages = {{"user", prompt}};
        if (has_vision_input_ && vision_encoder_) {
            tokens = chat_template_.apply_with_image(*tok, messages,
                                                      vision_model_->config.num_image_tokens);
        } else {
            tokens = chat_template_.apply(*tok, messages);
        }
        IMP_LOG_INFO("Applied %s chat template (%zu tokens%s)",
                     chat_template_family_name(chat_template_.family()),
                     tokens.size(),
                     has_vision_input_ ? ", with image" : "");
    } else {
        // Raw encoding
        tokens = tok->encode(prompt);
        if (tok->add_bos() && (tokens.empty() || tokens[0] != tok->bos_id())) {
            tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
        }
    }

    IMP_LOG_INFO("Encoded %zu tokens", tokens.size());
    // Debug: dump token IDs and their text for verification
    {
        std::string dump;
        for (size_t i = 0; i < tokens.size() && i < 64; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%d", tokens[i]);
            dump += buf;
            if (i + 1 < tokens.size()) dump += ", ";
        }
        if (tokens.size() > 64) dump += "...";
        IMP_LOG_INFO("Token IDs: [%s]", dump.c_str());
    }

    auto req = std::make_shared<Request>();
    req->id = next_request_id_++;
    req->input_tokens = std::move(tokens);
    req->max_tokens = max_tokens;
    req->temperature = temperature;
    req->top_p = top_p;
    req->top_k = top_k;
    req->seed = seed;
    req->min_p = min_p;
    req->repetition_penalty = repetition_penalty;
    req->frequency_penalty = frequency_penalty;
    req->presence_penalty = presence_penalty;
    req->status = RequestStatus::PENDING;

    scheduler_->add_request(req);

    // ---- Step 1: Prefill (always via step()) ----
    while (req->status == RequestStatus::PENDING ||
           req->status == RequestStatus::PREFILLING) {
        bool has_work = step();
        if (!has_work) break;
    }

    // ---- Step 2: Decode — try conditional graph loop, fall back to step() ----
    bool req_has_penalties = (req->repetition_penalty != 1.0f ||
                              req->frequency_penalty != 0.0f ||
                              req->presence_penalty != 0.0f);
    if (req->status == RequestStatus::DECODING && !req->output_tokens.empty() &&
        config_.use_cuda_graphs && !offload_mgr_ && !ssm_state_ &&
        !config_.enable_speculative && !req->ignore_eos && !req_has_penalties) {
        int32_t first_token = req->output_tokens.back();
        Tokenizer* gtok = model_->tokenizer();
        auto graph_tokens = try_graph_loop_decode(req, first_token, decode_stream());
        if (!graph_tokens.empty()) {
            // Check if the graph completed naturally (last token is EOS/stop)
            int32_t last = graph_tokens.back();
            bool hit_stop = (gtok && last == gtok->eos_id());
            if (!hit_stop) {
                for (int32_t stop_id : chat_template_.stop_token_ids()) {
                    if (last == stop_id) { hit_stop = true; break; }
                }
            }
            if (hit_stop) graph_tokens.pop_back();  // strip stop token

            for (int32_t t : graph_tokens) {
                req->output_tokens.push_back(t);
            }

            bool done = hit_stop ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (done) {
                req->status = RequestStatus::FINISHED;
                kv_manager_->free_sequence(req->id);
            }
            // else: graph was capped, fall through to step() loop
        }
        // If graph_tokens is empty, also fall through to step() loop
    }

    // ---- Step 3: Fallback — per-step decode ----
    while (req->status != RequestStatus::FINISHED &&
           req->status != RequestStatus::CANCELLED) {
        bool has_work = step();
        if (!has_work && req->status != RequestStatus::FINISHED &&
            req->status != RequestStatus::CANCELLED) {
            break;
        }
    }

    if (req->output_tokens.empty()) {
        return "";
    }

    // Clear vision state after generation completes
    has_vision_input_ = false;

    std::string result = tok->decode(req->output_tokens);
    return result;
}

std::vector<int32_t> Engine::try_graph_loop_decode(
        std::shared_ptr<Request> req, int32_t first_token, cudaStream_t stream) {
    // Only attempt for single-sequence decode
    Tokenizer* tok = model_->tokenizer();
    if (!tok) return {};

    int remaining = req->max_tokens - static_cast<int>(req->output_tokens.size());
    if (remaining <= 0) return {};

    // Skip conditional graph for large models — the WHILE body has ~600 nodes
    // (48 layers × ~12 kernels) and per-iteration scheduling overhead dominates.
    // Fall back to per-step CudaGraphRunner which captures a flat graph.
    constexpr int kMaxLayersForConditionalGraph = 40;
    if (model_->config().n_layers > kMaxLayersForConditionalGraph) {
        IMP_LOG_INFO("ConditionalGraph: skipping — %d layers exceeds threshold (%d), "
                     "using per-step graph instead",
                     model_->config().n_layers, kMaxLayersForConditionalGraph);
        return {};
    }

    // Skip conditional graph when VRAM is constrained — graph instantiation
    // needs driver-internal memory for the execution plan. With 0 MiB free,
    // the driver uses slow allocation paths that hurt every replay iteration.
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        constexpr size_t kMinVramForGraphMiB = 256;
        if (free_mem < kMinVramForGraphMiB * 1024ULL * 1024) {
            IMP_LOG_INFO("ConditionalGraph: skipping — only %zu MiB VRAM free (need %zu), "
                         "using per-step graph instead",
                         free_mem / (1024 * 1024), kMinVramForGraphMiB);
            return {};
        }
    }

    int ctx_len = req->context_len();
    int position = ctx_len - 1;
    IMP_LOG_DEBUG("try_graph_loop_decode: first_token=%d ctx_len=%d position=%d remaining=%d",
                  first_token, ctx_len, position, remaining);

    // Pre-allocate KV blocks for the full generation — allocate as many as
    // possible and cap max_steps if the cache is too small for the full run.
    // NOTE: no LRU eviction here — evict_lru() could evict the *current*
    // sequence in single-sequence chat mode, destroying its block table.
    {
        int final_ctx = ctx_len + remaining;
        int blocks_needed = (final_ctx + kKVBlockSize - 1) / kKVBlockSize;
        int blocks_have = static_cast<int>(kv_manager_->block_table(req->id).size());

        for (int b = blocks_have; b < blocks_needed; b++) {
            int new_block = kv_manager_->append_block(req->id);
            if (new_block < 0) break;  // pool exhausted, use what we have
        }

        int blocks_got = static_cast<int>(kv_manager_->block_table(req->id).size());
        int max_ctx = blocks_got * kKVBlockSize;
        int capped = max_ctx - ctx_len;
        if (capped <= 0) {
            IMP_LOG_WARN("ConditionalGraph: no KV capacity for new tokens, "
                         "falling back to step() loop");
            return {};
        }
        if (capped < remaining) {
            IMP_LOG_INFO("ConditionalGraph: capping to %d tokens (KV capacity)", capped);
            remaining = capped;
        }
    }

    // Upload full block table
    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    int* d_block_tables = nullptr;
    {
        cudaError_t err = cudaMallocAsync(&d_block_tables, max_blocks_per_seq * sizeof(int), stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalGraph: cudaMallocAsync failed: %s", cudaGetErrorString(err));
            return {};
        }
    }
    cudaMemcpyAsync(d_block_tables, full_bt.data(),
                     max_blocks_per_seq * sizeof(int),
                     cudaMemcpyHostToDevice, stream);

    // Resize workspace for decode (1 token)
    executor_->resize_workspace(1, stream);

    // Build the state template — all pointers are device memory
    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_block_tables;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Configure the conditional runner
    CudaGraphConditionalRunner::Config gcfg;
    gcfg.max_steps = remaining;
    gcfg.initial_context_len = ctx_len;
    gcfg.initial_position = position;
    gcfg.eos_id = tok->eos_id();
    gcfg.stop_ids = chat_template_.stop_token_ids();
    gcfg.temperature = req->temperature;
    gcfg.top_p = req->top_p;
    gcfg.top_k = req->top_k;
    gcfg.seed = req->seed;

    CudaGraphConditionalRunner runner;
    if (!runner.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        cudaFreeAsync(d_block_tables, stream);
        return {};  // setup failed, fall back
    }

    // Launch the graph
    if (!runner.launch(stream)) {
        cudaFreeAsync(d_block_tables, stream);
        return {};
    }

    // Wait for completion and get tokens
    auto tokens = runner.wait_and_get_tokens(stream);

    cudaFreeAsync(d_block_tables, stream);

    // NOTE: EOS/stop tokens are NOT stripped here — the caller checks
    // the last token to distinguish "generation done" from "graph capped".

    IMP_LOG_INFO("ConditionalGraph: generated %zu tokens in graph loop",
                 tokens.size());

    runner.cleanup();
    return tokens;
}

bool Engine::try_launch_async_graph_loop(std::shared_ptr<Request> req,
                                          int32_t first_token, cudaStream_t stream) {
    Tokenizer* tok = model_->tokenizer();
    if (!tok) return false;

    int remaining = req->max_tokens - static_cast<int>(req->output_tokens.size());
    if (remaining <= 0) return false;

    // Skip conditional graph for large models (same rationale as try_graph_loop_decode)
    constexpr int kMaxLayersForConditionalGraph = 40;
    if (model_->config().n_layers > kMaxLayersForConditionalGraph) {
        IMP_LOG_DEBUG("AsyncGraphLoop: skipping — %d layers exceeds threshold (%d)",
                      model_->config().n_layers, kMaxLayersForConditionalGraph);
        return false;
    }

    // Skip when VRAM is constrained
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        constexpr size_t kMinVramForGraphMiB = 256;
        if (free_mem < kMinVramForGraphMiB * 1024ULL * 1024) {
            IMP_LOG_DEBUG("AsyncGraphLoop: skipping — only %zu MiB VRAM free (need %zu)",
                          free_mem / (1024 * 1024), kMinVramForGraphMiB);
            return false;
        }
    }

    int ctx_len = req->context_len();
    int position = ctx_len - 1;
    IMP_LOG_DEBUG("try_launch_async: first_token=%d ctx_len=%d position=%d remaining=%d",
                  first_token, ctx_len, position, remaining);

    // Pre-allocate KV blocks for the full generation — allocate as many as
    // possible and cap max_steps if the cache is too small for the full run.
    // NOTE: no LRU eviction here — evict_lru() could evict the *current*
    // sequence in single-sequence chat mode, destroying its block table.
    {
        int final_ctx = ctx_len + remaining;
        int blocks_needed = (final_ctx + kKVBlockSize - 1) / kKVBlockSize;
        int blocks_have = static_cast<int>(kv_manager_->block_table(req->id).size());

        for (int b = blocks_have; b < blocks_needed; b++) {
            int new_block = kv_manager_->append_block(req->id);
            if (new_block < 0) break;  // pool exhausted, use what we have
        }

        int blocks_got = static_cast<int>(kv_manager_->block_table(req->id).size());
        int max_ctx = blocks_got * kKVBlockSize;
        int capped = max_ctx - ctx_len;
        if (capped <= 0) {
            IMP_LOG_WARN("AsyncGraphLoop: no KV capacity for new tokens");
            return false;
        }
        if (capped < remaining) {
            IMP_LOG_INFO("AsyncGraphLoop: capping to %d tokens (KV capacity)", capped);
            remaining = capped;
        }
    }

    // Upload full block table
    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    int* d_bt = nullptr;
    {
        cudaError_t err = cudaMalloc(&d_bt, max_blocks_per_seq * sizeof(int));
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("AsyncGraphLoop: cudaMalloc failed: %s", cudaGetErrorString(err));
            return false;
        }
    }
    cudaMemcpyAsync(d_bt, full_bt.data(),
                     max_blocks_per_seq * sizeof(int),
                     cudaMemcpyHostToDevice, stream);

    // Resize workspace for decode (1 token)
    executor_->resize_workspace(1, stream);

    // Build state template
    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_bt;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Configure
    CudaGraphConditionalRunner::Config gcfg;
    gcfg.max_steps = remaining;
    gcfg.initial_context_len = ctx_len;
    gcfg.initial_position = position;
    gcfg.eos_id = tok->eos_id();
    gcfg.stop_ids = chat_template_.stop_token_ids();
    gcfg.temperature = req->temperature;
    gcfg.top_p = req->top_p;
    gcfg.top_k = req->top_k;
    gcfg.seed = req->seed;

    if (!async_graph_runner_.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        cudaFree(d_bt);
        IMP_LOG_WARN("AsyncGraphLoop: setup failed, falling back to per-step decode");
        return false;
    }

    if (!async_graph_runner_.launch(stream)) {
        async_graph_runner_.cleanup();
        cudaFree(d_bt);
        IMP_LOG_WARN("AsyncGraphLoop: launch failed, falling back to per-step decode");
        return false;
    }

    // Store state for step() polling
    async_graph_req_ = req;
    async_d_block_tables_ = d_bt;
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;

    IMP_LOG_INFO("AsyncGraphLoop: launched for %d remaining tokens", remaining);
    return true;
}

void Engine::add_request(std::shared_ptr<Request> req) {
    if (scheduler_) {
        req->id = next_request_id_++;
        scheduler_->add_request(std::move(req));
    }
}

} // namespace imp
