// Engine init phase: paged KV cache allocation.
// Decides block geometry (block_size=16), allocates blocks per KV dtype
// (FP16/FP8/INT8/INT4/NVFP4/MXFP4), wires up KVCacheManager. Also
// initialises the BitDecoding residual FP16 cache (opt-in), SSM/GDN
// state, pre-dequant weight caches, decode batch pool, prefill metadata
// pool + pinned staging, and reports VRAM usage.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. Method remains Engine:: with declaration in engine.h.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "memory/kv_cache.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace imp {

bool Engine::init_kv_cache() {
    const auto& mcfg = model_->config();
    int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);

    // Build KV layer mapping for hybrid models
    int n_attn_layers = 0;
    std::vector<int> kv_layer_map(mcfg.n_layers, -1);
    for (int i = 0; i < mcfg.n_layers; i++) {
        if (model_->layer(i).wq.data != nullptr && model_->layer(i).gdn_gate.data == nullptr)
            kv_layer_map[i] = n_attn_layers++;
    }
    if (n_attn_layers == 0) {
        n_attn_layers = mcfg.n_layers;
        for (int i = 0; i < mcfg.n_layers; i++)
            kv_layer_map[i] = i;
    }
    int n_kv_layers = n_attn_layers;
    IMP_LOG_INFO("KV cache layers: %d attention out of %d total", n_kv_layers, mcfg.n_layers);

    // Auto-select block size
    if (config_.kv_block_size <= 0) {
        config_.kv_block_size = (mcfg.n_kv_heads <= 4 && mcfg.n_kv_heads > 0) ? 32 : kKVBlockSize;
        IMP_LOG_INFO("KV block size: auto → %d (n_kv_heads=%d)", config_.kv_block_size, mcfg.n_kv_heads);
    }
    const int kv_bs = config_.kv_block_size;
    int blocks_per_seq = (config_.max_seq_len + kv_bs - 1) / kv_bs;

    // VRAM budget
    auto vram_budget =
        MemoryManager::compute_budget(*model_, config_, n_kv_layers, head_dim, effective_free_vram());
    int max_blocks = config_.kv_cache_max_blocks > 0 ? config_.kv_cache_max_blocks
                                                     : vram_budget.kv_max_blocks;

    {
        QType kv_dtype = config_.kv_cache_dtype;
        size_t block_bytes = static_cast<size_t>(kv_bs) * mcfg.n_kv_heads * head_dim * dtype_size(kv_dtype);
        size_t total_kv = static_cast<size_t>(n_kv_layers) * max_blocks * 2 * block_bytes;
        IMP_LOG_INFO(
            "KV cache: %d blocks (%.0f tokens), %.2f MiB, dtype=%s "
            "(layers=%d/%d, kv_heads=%d, head_dim=%d, block_size=%d)",
            max_blocks, static_cast<double>(max_blocks) * kv_bs,
            static_cast<double>(total_kv) / (1024.0 * 1024.0), dtype_name(kv_dtype), n_kv_layers,
            mcfg.n_layers, mcfg.n_kv_heads, head_dim, kv_bs);
    }

    // Per-layer KV shape path (Gemma 4 dual attention geometry): build per-layer
    // nkv/hd arrays restricted to attention layers (hybrid models may have non-attn layers).
    std::unique_ptr<KVCache> kv_cache;
    if (!mcfg.head_dim_per_layer.empty() && config_.kv_cache_dtype != QType::INT8 &&
        config_.kv_cache_dtype != QType::INT4) {
        std::vector<int> per_layer_nkv(n_kv_layers, 0);
        std::vector<int> per_layer_hd(n_kv_layers, 0);
        for (int l = 0, k = 0; l < mcfg.n_layers && k < n_kv_layers; l++) {
            // Only attention layers get KV cache entries
            int attn_nkv = (l < (int)mcfg.n_kv_heads_per_layer.size()) ? mcfg.n_kv_heads_per_layer[l]
                                                                       : mcfg.n_kv_heads;
            if (attn_nkv <= 0)
                continue;  // non-attention layer (SSM/GDN)
            per_layer_nkv[k] = attn_nkv;
            per_layer_hd[k] = (l < (int)mcfg.head_dim_per_layer.size() && mcfg.head_dim_per_layer[l] > 0)
                                  ? mcfg.head_dim_per_layer[l]
                                  : head_dim;
            k++;
        }
        kv_cache = std::make_unique<KVCache>(n_kv_layers, per_layer_nkv, per_layer_hd, config_.kv_cache_dtype,
                                             max_blocks, kv_bs, &memory_manager_.vram_allocator());
    } else {
        kv_cache = std::make_unique<KVCache>(n_kv_layers, mcfg.n_kv_heads, head_dim, config_.kv_cache_dtype,
                                             max_blocks, kv_bs, &memory_manager_.vram_allocator());
    }
    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));

    // BitDecoding Phase 3: residual FP16 cache (opt-in).
    //
    // Ring state (write_idx / fill_count per slot) lives in device memory
    // (kv_manager_->d_residual_widx_ptr / d_residual_fc_ptr). Updated by a
    // tiny advance_residual_state_kernel at the end of forward_logits; the
    // residual write/read kernels read the state at execution time. This
    // makes the whole path graph-capture-safe — graphs stay enabled.
    {
        const auto& rcfg = runtime_config_;
        int residual_n = rcfg.kv_cache.bitdecoding_residual_tokens;
        if (residual_n > 0 && config_.kv_cache_dtype == QType::NVFP4) {
            int max_seqs = config_.max_batch_size > 0 ? config_.max_batch_size : 1;
            if (kv_manager_->enable_residual_buffer(max_seqs, residual_n, &memory_manager_.vram_allocator())) {
                // Persistent batch→slot lookup buffer (graph-safe). [max_batch_size] ints.
                size_t slot_bytes = static_cast<size_t>(max_seqs) * sizeof(int);
                cudaMalloc(&d_kv_slot_buf_, slot_bytes);
                std::vector<int> init_slots(max_seqs, -1);
                cudaMemcpy(d_kv_slot_buf_, init_slots.data(), slot_bytes, cudaMemcpyHostToDevice);
                d_kv_slot_last_uploaded_.assign(max_seqs, -1);
            }
        } else if (residual_n > 0) {
            IMP_LOG_INFO("kv_cache.bitdecoding_residual_tokens=%d ignored (only active with kv_cache_dtype=NVFP4)",
                         residual_n);
        }
    }

    if (config_.use_prefix_caching) {
        if (mcfg.ssm_inner_size > 0) {
            IMP_LOG_WARN(
                "Prefix caching disabled for recurrent model — "
                "SSM/GDN state requires full sequential prefill");
        } else {
            kv_manager_->set_prefix_caching_enabled(true);
            // cache_control/cache_prompt pin budget: percent of the pool,
            // floor of 1 block when enabled at all.
            int pin_pct = std::min(std::max(config_.prefix_pin_budget_pct, 0), 100);
            int pin_budget =
                pin_pct > 0 ? std::max(1, kv_manager_->kv_cache()->total_blocks() * pin_pct / 100) : 0;
            kv_manager_->set_pin_budget_blocks(pin_budget);
            IMP_LOG_INFO("Prefix caching enabled (pin budget %d blocks)", pin_budget);
            if (!config_.prefix_cache_path.empty()) {
                int restored = kv_manager_->load_prefix_cache(config_.prefix_cache_path, stream_);
                if (restored > 0)
                    IMP_LOG_INFO("Restored %d prefix cache blocks from %s", restored,
                                 config_.prefix_cache_path.c_str());
            }
        }
    }

    executor_->set_kv_layer_map(std::move(kv_layer_map));

    if (offload_mgr_)
        executor_->set_offload_manager(offload_mgr_.get());
    scheduler_->set_kv_manager(kv_manager_.get());

    // SSM state
    if (mcfg.ssm_inner_size > 0) {
        int n_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).ssm_in.data != nullptr)
                n_ssm++;
        if (n_ssm > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int hd = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            ssm_state_ = std::make_unique<SSMState>();
            if (!ssm_state_->init(n_ssm, config_.max_batch_size, conv_ch, mcfg.ssm_conv_kernel, n_heads, hd,
                                  mcfg.ssm_state_size, config_.ssm_state_dtype, &memory_manager_.vram_allocator())) {
                IMP_LOG_WARN("Failed to init SSM state, continuing without it");
                ssm_state_.reset();
            }
        }
    }

    // GDN detection
    {
        if (model_->profile().is_gdn) {
            if (config_.use_cuda_graphs) {
                IMP_LOG_INFO("GDN model: CUDA graphs enabled (recurrent state in-place)");
            } else {
                IMP_LOG_INFO(
                    "GDN model: CUDA graphs disabled (disabled earlier by caller or expert offload)");
            }
            // GDN recurrent state accumulates small precision errors per token.
            // FP8 E4M3 (3-bit mantissa) amplifies these through the delta rule
            // scan, causing degenerate output after ~50 special tokens in
            // multi-turn chat.  Force FP16 weights for GDN prefill.
            if (config_.use_fp8_prefill) {
                if (config_.dual_path_quant) {
                    IMP_LOG_WARN(
                        "GDN + dual-path: attention weights forced to FP16 (not FP8) — "
                        "recurrent state needs FP16 precision. FFN weights still use NVFP4.");
                } else {
                    IMP_LOG_INFO("GDN model: disabling FP8 prefill (recurrent state needs FP16 precision)");
                }
                config_.use_fp8_prefill = 0;
                executor_->disable_fp8_prefill();
            }
        }
    }

    // (Gemma 4 FP8 prefill disabled earlier, before executor init)

    // Detect pure Mamba2 SSM layers (layers with ssm_in but without gdn_gate).
    // GDN-only models (Qwen3.5) are graph-compatible; pure SSM (Nemotron-H) is not yet.
    {
        has_pure_ssm_layers_ = model_->profile().has_pure_ssm;
        if (has_pure_ssm_layers_ && config_.use_cuda_graphs) {
            config_.use_cuda_graphs = false;
            IMP_LOG_INFO("Mamba2 SSM layers detected: disabling CUDA graphs "
                         "(recurrent state not yet graph-safe)");
        }
    }

    // Dequant weights → FP16/FP8/NVFP4 caches
    executor_->pre_dequant_weights(stream_, vram_budget);
    dequant_done_ = true;
    cudaStreamSynchronize(stream_);

    // Pre-allocate the gemm_nvfp4 fallback dequant workspace. Sized from
    // wcache_.nvfp4 which is populated by pre_dequant_weights above, so this
    // call must come AFTER. Lets the M>1 fallback path (used by future
    // multi-token verify / spec-decode) run inside CUDA stream capture
    // without crashing on cudaMalloc.
    (void)executor_->allocate_nvfp4_dequant_workspace();
    // Build the CUTLASS NVFP4 LM head for batched-decode tensor-core GEMM (no-op
    // unless serving with max_batch>1 and the LM head is NVFP4). Must come AFTER
    // pre_dequant_weights so the NVFP4 decode cache exists.
    executor_->build_lm_head_cutlass_(stream_);
    if (config_.use_fp8_prefill)
        IMP_LOG_INFO("Weight cache: FP8 E4M3 (2x prefill throughput on sm_120)");

    // Pre-allocate decode batch pool + penalty buffer
    decode_batch_pool_.allocate(config_.max_batch_size, blocks_per_seq, &memory_manager_.vram_allocator());
    {
        d_penalty_tokens_capacity_ = static_cast<size_t>(config_.max_seq_len);
        d_penalty_tokens_ = static_cast<int32_t*>(
            memory_manager_.vram_allocator().allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
        if (!d_penalty_tokens_) {
            IMP_LOG_WARN("Failed to pre-allocate penalty token buffer");
            d_penalty_tokens_capacity_ = 0;
        }
    }

    // Pre-allocate prefill metadata pool (avoids per-request cudaMallocAsync)
    {
        size_t tok_bytes = config_.max_seq_len * sizeof(int32_t);
        size_t pos_bytes = config_.max_seq_len * sizeof(int);
        // A single request's block_table can grow to the entire KV cache
        // pool (max_blocks), not just max_seq_len/block_size. Size from
        // max_blocks so the H2D copy at the prefill metadata upload site
        // doesn't overflow on long-cumulative-KV requests.
        size_t bt_bytes = static_cast<size_t>(max_blocks) * sizeof(int);
        size_t cl_bytes = sizeof(int);
        prefill_pool_size_ = tok_bytes + pos_bytes + bt_bytes + cl_bytes;
        prefill_pool_ = memory_manager_.vram_allocator().allocate(prefill_pool_size_, "prefill_pool");
        if (prefill_pool_) {
            auto* base = static_cast<char*>(prefill_pool_);
            d_pf_token_ids_ = reinterpret_cast<int32_t*>(base);
            d_pf_positions_ = reinterpret_cast<int*>(base + tok_bytes);
            d_pf_block_tables_ = reinterpret_cast<int*>(base + tok_bytes + pos_bytes);
            d_pf_context_lens_ = reinterpret_cast<int*>(base + tok_bytes + pos_bytes + bt_bytes);
        } else {
            IMP_LOG_WARN("Failed to pre-allocate prefill pool, will use per-request malloc");
        }

        // Pinned host staging buffers for prefill
        if (cudaHostAlloc(&h_pf_positions_, config_.max_seq_len * sizeof(int), cudaHostAllocDefault) !=
            cudaSuccess)
            h_pf_positions_ = nullptr;
        if (cudaHostAlloc(&h_pf_token_ids_, config_.max_seq_len * sizeof(int32_t), cudaHostAllocDefault) !=
            cudaSuccess)
            h_pf_token_ids_ = nullptr;
        if (cudaEventCreateWithFlags(&pf_staging_evt_, cudaEventDisableTiming) != cudaSuccess)
            pf_staging_evt_ = nullptr;
    }

    // Report memory
    {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            IMP_LOG_INFO("GPU memory: %.0f MiB used / %.0f MiB total (%.0f MiB free)",
                         (total_mem - free_mem) / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0),
                         free_mem / (1024.0 * 1024.0));
        memory_manager_.vram_allocator().report();
    }

    return true;
}


}  // namespace imp
