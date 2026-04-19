#include "runtime/engine.h"
#include "runtime/vram_budget.h"
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
#include "core/logging.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <functional>
#include <vector>

namespace imp {

// =====================================================================
// File-local helpers (pure refactoring — no behavior changes)
// =====================================================================
namespace {

// Free prefill metadata buffers when not using the pre-allocated pool.
void free_prefill_buffers(int32_t* d_token_ids, int* d_positions,
                           int* d_block_tables, int* d_context_lens,
                           cudaStream_t stream) {
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, stream));
}

// Compute a deterministic-but-varying seed for each decode step.
// Mixes the request seed (or a hash of the request ID + clock) with
// the current output token count so each step gets a unique RNG draw.
int compute_step_seed(const Request& req) {
    int base_seed = req.seed >= 0 ? req.seed
        : static_cast<int>(std::hash<int>{}(req.id) ^
            std::chrono::steady_clock::now().time_since_epoch().count());
    int step = static_cast<int>(req.output_tokens.size());
    return base_seed + step;
}

// Build a TokenLogprobInfo from raw logits on the host.
TokenLogprobInfo build_logprob_info(const float* h_logits, int vocab_size,
                                            int32_t sampled_token, int top_logprobs,
                                            Tokenizer* tok) {
    LogprobResult lp_result;
    compute_logprobs_cpu(h_logits, vocab_size, sampled_token, top_logprobs, &lp_result);

    TokenLogprobInfo info;
    info.logprob = lp_result.sampled_logprob;
    info.text = tok->decode_token(sampled_token);
    info.top.reserve(lp_result.top.size());
    for (const auto& [tid, tlp] : lp_result.top) {
        info.top.push_back({tid, tlp, tok->decode_token(tid)});
    }
    return info;
}

// Ensure workspace 0 is active (used before prefill and after decode).
void ensure_prefill_workspace(GraphExecutor* executor) {
    if (executor->has_decode_workspace() && executor->active_workspace() != 0) {
        executor->use_workspace(0);
    }
}

} // anonymous namespace

Engine::~Engine() {
    // Save prefix cache to disk before shutdown
    if (kv_manager_ && !config_.prefix_cache_path.empty() &&
        kv_manager_->prefix_caching_enabled()) {
        kv_manager_->save_prefix_cache(config_.prefix_cache_path, stream_);
    }

    gemm_cleanup();
    gemm_grouped_cleanup();
    sampling_cleanup();
    if (async_graph_runner_.is_setup()) {
        async_graph_runner_.cleanup();
    }
    if (async_d_block_tables_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
        async_d_block_tables_ = nullptr;
    }
    if (async_d_banned_tokens_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
        async_d_banned_tokens_ = nullptr;
    }
    if (d_penalty_tokens_) {
        vram_alloc_.free(d_penalty_tokens_);
        d_penalty_tokens_ = nullptr;
    }
    if (h_sample_pinned_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_sample_pinned_));
        h_sample_pinned_ = nullptr;
    }
    if (prefill_pool_) {
        vram_alloc_.free(prefill_pool_);
        prefill_pool_ = nullptr;
    }
    if (h_pf_positions_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_pf_positions_));
        h_pf_positions_ = nullptr;
    }
    if (h_pf_token_ids_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_pf_token_ids_));
        h_pf_token_ids_ = nullptr;
    }
    // stream_, prefill_done_, decode_done_ cleaned up by CudaStream/CudaEvent RAII
    // vision_ cleaned up by VisionPipeline RAII
}

// =====================================================================
// Helper methods
// =====================================================================

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

void Engine::reset_batch_pool_cache() {
    decode_batch_pool_.reset_upload_cache();
}

void Engine::invalidate_graphs() {
    // Preserve decode_graph_pool_ across context resets — the decode step
    // topology (forward_logits) doesn't change between requests. Inputs
    // (token IDs, positions, block tables) are uploaded fresh each step via
    // the batch pool. Per-entry invalidation already handles max_blocks_per_seq
    // changes in step_decode_forward(). Re-capturing on every benchmark rep
    // adds ~100ms overhead per reset.
    //
    // The conditional graph runner MUST be invalidated: it captures the full
    // decode loop including token feedback, stop conditions, and request-specific
    // KV block pointers.
    if (async_graph_runner_.is_setup()) {
        async_graph_runner_.cleanup();
    }
    async_graph_req_ = nullptr;
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;
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

bool Engine::is_stop_token(int32_t token) const {
    Tokenizer* tok = model_->tokenizer();
    if (tok && tok->is_eos(token)) return true;
    for (int32_t stop_id : chat_template_.stop_token_ids()) {
        if (token == stop_id) return true;
    }
    // Banned tokens (e.g. <pad>) should also trigger stop — they indicate
    // the model has degenerated and continuing would produce garbage.
    for (int32_t bid : banned_token_ids_) {
        if (token == bid) return true;
    }
    return false;
}

void Engine::track_think_state(Request& req, int32_t token) const {
    if (token == think_start_id_) req.in_think_block = true;
    else if (token == think_end_id_) req.in_think_block = false;
}

bool Engine::should_stop(Request& req, int32_t token) const {
    if (req.ignore_eos) return false;
    // Inside <think>...</think>: suppress stop tokens so reasoning can complete.
    // The model may generate <|im_end|> during reasoning as part of its internal
    // monologue — stopping here produces empty content (llama.cpp ignores this).
    if (req.in_think_block) return false;
    return is_stop_token(token);
}

void Engine::fill_sampling_params(const Request& req, InferenceState& state) const {
    state.temperature = req.temperature;
    state.top_p = req.top_p;
    state.top_k = req.top_k;
    state.seed = req.seed;
    state.min_p = req.min_p;
    state.typical_p = req.typical_p;
    state.repetition_penalty = req.repetition_penalty;
    state.frequency_penalty = req.frequency_penalty;
    state.presence_penalty = req.presence_penalty;
    state.repeat_last_n = req.repeat_last_n;
    state.dry_multiplier = req.dry_multiplier;
    state.dry_base = req.dry_base;
    state.dry_allowed_length = req.dry_allowed_length;
    state.dry_penalty_last_n = req.dry_penalty_last_n;
    if (req.dry_multiplier > 0.0f && !req.output_tokens.empty())
        state.host_penalty_tokens = req.output_tokens.data();
    state.mirostat = req.mirostat;
    state.mirostat_tau = req.mirostat_tau;
    state.mirostat_eta = req.mirostat_eta;
    state.mirostat_mu = req.mirostat_mu;

    // Logit bias
    if (!req.logit_bias.empty()) {
        state.logit_bias = req.logit_bias.data();
        state.n_logit_bias = static_cast<int>(req.logit_bias.size());
    }

    // Banned tokens (chat template special tokens that must not be generated)
    if (!banned_token_ids_.empty()) {
        state.banned_tokens = banned_token_ids_.data();
        state.n_banned_tokens = static_cast<int>(banned_token_ids_.size());
    }

    // Think budget: force </think> token via logit manipulation when budget exceeded.
    // Count reasoning tokens (between <think> and </think>) from output history.
    // The model generates </think> itself so it lands in the KV cache correctly.
    // Think budget: force </think> via logit manipulation when budget exceeded.
    // Scan output_tokens directly (no dependency on in_think_block tracking).
    state.force_token = -1;
    if (req.think_budget > 0.0f && think_end_id_ >= 0 && !req.output_tokens.empty()) {
        int think_limit = static_cast<int>(req.max_tokens * req.think_budget);
        int n_reasoning = 0;
        bool currently_thinking = false;
        for (int32_t t : req.output_tokens) {
            if (t == think_start_id_) currently_thinking = true;
            else if (t == think_end_id_) currently_thinking = false;
            else if (currently_thinking) n_reasoning++;
        }
        if (currently_thinking && n_reasoning >= think_limit) {
            state.force_token = think_end_id_;
        }
    }
}

void Engine::upload_penalties(const Request& req, InferenceState& state,
                               cudaStream_t stream) {
    bool needs_penalties = (req.repetition_penalty != 1.0f ||
                            req.frequency_penalty != 0.0f ||
                            req.presence_penalty != 0.0f);
    if (!needs_penalties || req.output_tokens.empty()) return;

    size_t n = req.output_tokens.size();
    if (n > d_penalty_tokens_capacity_) {
        if (d_penalty_tokens_) vram_alloc_.free(d_penalty_tokens_);
        d_penalty_tokens_capacity_ = std::max(n, (size_t)256);
        d_penalty_tokens_ = static_cast<int32_t*>(
            vram_alloc_.allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
        if (!d_penalty_tokens_) {
            IMP_LOG_ERROR("VRAMAllocator failed for penalty tokens (%zu)", d_penalty_tokens_capacity_);
            d_penalty_tokens_capacity_ = 0;
            return;
        }
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_penalty_tokens_, req.output_tokens.data(),
                    n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    state.penalty_tokens = d_penalty_tokens_;
    state.n_penalty_tokens = static_cast<int>(n);
}

void Engine::fill_recurrent_state(const Request& req, InferenceState& state,
                                    bool reset, cudaStream_t stream) {
    if (ssm_state_) {
        state.ssm_state = ssm_state_.get();
        state.ssm_seq_id = req.id % ssm_state_->max_sequences();
        if (reset) ssm_state_->reset_sequence(state.ssm_seq_id, stream);
    }
    if (gdn_state_) {
        state.gdn_state = gdn_state_.get();
        state.gdn_seq_id = req.id % gdn_state_->max_sequences();
        if (reset) gdn_state_->reset_sequence(state.gdn_seq_id, stream);
    }
}

void Engine::finish_request(std::shared_ptr<Request>& req) {
    req->status = RequestStatus::FINISHED;
    if (kv_manager_->prefix_caching_enabled()) {
        kv_manager_->register_block_hashes(req->id, req->input_tokens);
    }
    kv_manager_->free_sequence(req->id);
    constraints_.reset();
}

// =====================================================================
// Vision delegation
// =====================================================================

bool Engine::set_image(const std::string& path) {
    return vision_.set_image(path, stream_);
}

bool Engine::set_image_from_memory(const uint8_t* data, size_t len) {
    return vision_.set_image_from_memory(data, len, stream_);
}

void Engine::clear_image() {
    vision_.clear_image();
}

// =====================================================================
// Initialization — decomposed into sub-phases
// =====================================================================

bool Engine::init(std::shared_ptr<Model> model, const EngineConfig& config) {
    if (!model) return false;

    model_ = std::move(model);
    config_ = config;

    const auto& mcfg = model_->config();

    // --- Auto-detect max_seq_len and max_batch_size if not set ---
    if (const char* env_msl = std::getenv("IMP_MAX_SEQ_LEN")) {
        int v = std::atoi(env_msl);
        if (v > 0) { config_.max_seq_len = v; IMP_LOG_INFO("max_seq_len: env IMP_MAX_SEQ_LEN=%d", v); }
    }
    if (config_.max_seq_len <= 0) {
        int model_ctx = mcfg.max_seq_len;  // from GGUF metadata
        // Cap based on available VRAM: reserve ~30% for KV cache
        size_t free_vram = 0, total_vram = 0;
        cudaMemGetInfo(&free_vram, &total_vram);
        int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
        int kv_bytes_per_token = mcfg.n_kv_heads * head_dim * 2 * mcfg.n_layers * 2;  // K+V, FP16
        int max_by_vram = (kv_bytes_per_token > 0)
            ? static_cast<int>(free_vram * 0.3 / kv_bytes_per_token)
            : 131072;
        config_.max_seq_len = std::min(model_ctx, std::max(max_by_vram, 4096));
        IMP_LOG_INFO("max_seq_len: auto → %d (model=%d, vram_cap=%d)",
                     config_.max_seq_len, model_ctx, max_by_vram);
    }

    if (config_.max_batch_size <= 0) {
        // Estimate model weight size from config to determine batch capacity.
        // Rough heuristic: 2 bytes/param for FP16. d_model * d_model * n_layers * ~12 gives
        // approximate total weight bytes for a dense transformer.
        size_t approx_weight_bytes = static_cast<size_t>(mcfg.d_model) * mcfg.d_model
                                     * mcfg.n_layers * 12;  // ~12 matrices per layer
        if (mcfg.n_experts > 0) {
            // MoE: expert weights dominate
            approx_weight_bytes += static_cast<size_t>(mcfg.n_experts) * mcfg.expert_d_ff
                                   * mcfg.d_model * mcfg.n_layers * 2;
        }
        if (approx_weight_bytes > 20ULL * 1024 * 1024 * 1024)
            config_.max_batch_size = 1;   // >20GB models
        else if (approx_weight_bytes > 10ULL * 1024 * 1024 * 1024)
            config_.max_batch_size = 4;   // 10-20GB
        else if (approx_weight_bytes > 5ULL * 1024 * 1024 * 1024)
            config_.max_batch_size = 8;   // 5-10GB
        else
            config_.max_batch_size = 16;  // <5GB
        IMP_LOG_INFO("max_batch_size: auto → %d (approx_weights=%.1f GB)",
                     config_.max_batch_size,
                     approx_weight_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    // --- Auto-detect SSM state dtype for hybrid models ---
    // Nemotron-H and similar: use FP16 for SSM h_state (~50% VRAM savings)
    if (config_.ssm_state_dtype == DType::FP32 && mcfg.ssm_state_size > 0) {
        config_.ssm_state_dtype = DType::FP16;
        IMP_LOG_INFO("SSM state dtype: auto → FP16 (hybrid SSM model, state_size=%d)",
                     mcfg.ssm_state_size);
    }

    // --- Auto-detect KV cache dtype ---
    // Default to FP8 E4M3 for ~50% KV VRAM savings
    if (config_.kv_cache_dtype == DType::FP16) {
        config_.kv_cache_dtype = DType::FP8_E4M3;
        IMP_LOG_INFO("KV cache dtype: auto → FP8_E4M3");
    }

    // --- Auto-detect FP8 prefill ---
    if (!config_.use_fp8_prefill) {
        config_.use_fp8_prefill = true;
        IMP_LOG_INFO("FP8 prefill: auto → enabled");
    }

    // --- Resolve auto-detection flags ---
    // NVFP4 decode mode
    int n_gdn_auto = 0;
    for (int i = 0; i < mcfg.n_layers; i++)
        if (model_->layer(i).gdn_gate.data != nullptr) n_gdn_auto++;

    if (config_.use_nvfp4_decode < 0) {
        if (n_gdn_auto > 0) {
            // GDN models with large d_model: enable NVFP4 for attention + FFN weights,
            // but SSM/GDN projections (ssm_in/ssm_out) will be excluded in
            // pre_dequant_weights to preserve recurrent state precision.
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO("NVFP4 decode: auto → mode 2 (GDN model, %d recurrent layers — "
                         "ssm_in/ssm_out excluded for precision)", n_gdn_auto);
        } else {
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO("NVFP4 decode: auto → mode 2");
        }
    }

    // FP8 prefill auto-disable for sub-8-bit models
    if (config_.use_fp8_prefill) {
        auto qtype = model_->layer(0).wq_qtype;
        bool sub_8bit = (qtype == GGMLQuantType::Q4_0 || qtype == GGMLQuantType::Q4_K ||
                         qtype == GGMLQuantType::Q5_0 || qtype == GGMLQuantType::Q5_K ||
                         qtype == GGMLQuantType::Q3_K || qtype == GGMLQuantType::Q2_K ||
                         qtype == GGMLQuantType::Q4_1 || qtype == GGMLQuantType::Q5_1);
        if (sub_8bit) {
            config_.use_fp8_prefill = 0;
            IMP_LOG_INFO("FP8 prefill cache: auto-disabled (sub-8-bit weights)");
        }
    }

    // Dual-path quant validation: requires NVFP4 decode + FP8 prefill.
    // If either is missing, auto-enable or warn.
    if (config_.dual_path_quant) {
        if (config_.use_nvfp4_decode <= 0) {
            IMP_LOG_WARN("Dual-path quant requires NVFP4 decode — enabling mode 2 (NVFP4 only)");
            config_.use_nvfp4_decode = 2;
        }
        if (!config_.use_fp8_prefill) {
            IMP_LOG_INFO("Dual-path quant: auto-enabling FP8 prefill for attention weight quality");
            config_.use_fp8_prefill = true;
        }
    }

    // Gemma 4: FP8 prefill, NVFP4 prefill, CUTLASS paths, and CUDA graphs all have
    // incompatibilities with the per-layer head_dim + split MoE tensor layout.
    // Force plain FP16 paths for Gemma 4 until proper kernels are added.
    if (model_->config().arch == ModelArch::GEMMA4) {
        // CUDA graphs: enabled for Gemma-4 decode. The MoE decode fast path is fully
        // device-side (dp4a GEMV, no D2H memcpy), so graph capture works.
        // Only the MoE prefill path uses D2H sync, but prefill is never graph-captured.
        if (config_.use_fp8_prefill) {
            IMP_LOG_INFO("Gemma 4: disabling FP8 prefill (per-layer head_dim not yet supported)");
            config_.use_fp8_prefill = 0;
        }
        if (config_.use_nvfp4_decode) {
            IMP_LOG_INFO("Gemma 4: disabling NVFP4 decode cache (per-layer head_dim not yet supported)");
            config_.use_nvfp4_decode = 0;
        }
        if (config_.dual_path_quant) {
            IMP_LOG_INFO("Gemma 4: disabling dual_path_quant");
            config_.dual_path_quant = false;
        }
        // Force FP16 KV cache (FP8 KV cache calibration reads narrow stride incorrectly)
        if (config_.kv_cache_dtype == DType::FP8_E4M3) {
            IMP_LOG_INFO("Gemma 4: forcing FP16 KV cache (FP8 stride mismatch)");
            config_.kv_cache_dtype = DType::FP16;
        }
        // Gemma 4 output_norm has extreme outliers (max=588). Small numeric jitter
        // from cuBLAS algo autotuning / split-K atomics amplifies into wildly
        // different top-1 picks (coherent " Paris" vs garbage "\n"). Force
        // deterministic GEMM paths so generation is stable run-to-run.
        if (!getenv("IMP_DETERMINISTIC_GEMM")) {
            setenv("IMP_DETERMINISTIC_GEMM", "1", 1);
            IMP_LOG_INFO("Gemma 4: enabling IMP_DETERMINISTIC_GEMM (output_norm outliers amplify algo jitter)");
        }
        if (!getenv("CUBLAS_WORKSPACE_CONFIG")) {
            setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
            IMP_LOG_INFO("Gemma 4: setting CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic grouped GEMM");
        }
        // Gemma 4: CUDA graphs are fully enabled. forward_decode_async() now
        // delegates to forward_logits() (the canonical path), eliminating the
        // earlier divergence that caused EOS-at-step-0. Override with
        // IMP_GEMMA4_NO_GRAPHS=1 for bisecting regressions.
        if (getenv("IMP_GEMMA4_NO_GRAPHS")) {
            IMP_LOG_INFO("Gemma 4: disabling all CUDA graphs (IMP_GEMMA4_NO_GRAPHS=1)");
            config_.use_cuda_graphs = false;
        }
        // Enable MMVQ for all weight GEMMs — quantized matmul matching llama.cpp's
        // accumulation behavior, critical for 128-expert MoE precision.
        if (!getenv("IMP_GEMMA4_FORCE_MMVQ")) {
            setenv("IMP_GEMMA4_FORCE_MMVQ", "1", 0);
            IMP_LOG_INFO("Gemma 4: enabling MMVQ for all weight GEMMs (numerical parity with llama.cpp)");
        }
    }

    // --- Core initialization ---
    // 5% headroom (was 10%) — MoE models (30B Q6_K) need every MiB on 32GB.
    // WSL2/WDDM has ~500 MiB driver overhead, 5% of 32GB = 1.6 GB covers it.
    if (!vram_alloc_.init(0.05f)) {
        IMP_LOG_ERROR("Failed to initialize VRAM allocator");
        return false;
    }
    gemm_init();
    scheduler_ = std::make_unique<Scheduler>(config_.max_batch_size);
    (void)stream_.create(cudaStreamNonBlocking);

    // --- Sub-phases ---
    if (!init_weights()) return false;
    if (!init_kv_cache()) return false;
    if (!init_features()) return false;
    if (getenv("IMP_NO_WARMUP")) {
        IMP_LOG_INFO("Warmup SKIPPED (IMP_NO_WARMUP)");
    } else {
        warmup();
    }

    return true;
}

bool Engine::init_weights() {
    const auto& mcfg = model_->config();

    // Initialize graph executor (Phase 1: compute sizes, no GPU allocation)
    executor_ = std::make_unique<GraphExecutor>();
    executor_->set_vram_allocator(&vram_alloc_);
    {
        int eff_batch = config_.max_batch_size;
        if (config_.enable_self_speculative)
            eff_batch = std::max(eff_batch, config_.self_spec_k + 1);
        if (config_.enable_ngram_spec)
            eff_batch = std::max(eff_batch, config_.ngram_spec_k + 1);
        if (!executor_->init(*model_, config_.compute_dtype, config_.use_pdl,
                             eff_batch, config_.max_seq_len,
                             config_.use_fp8_prefill, config_.use_nvfp4_decode,
                             config_.use_mxfp4_prefill))
            return false;

        if (config_.dual_path_quant) {
            executor_->set_dual_path_quant(true);
            IMP_LOG_INFO("Dual-path quant: attention weights → FP8, FFN weights → NVFP4");
        }
    }

    // Reserve L2 persisting cache for decode GEMV
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t max_persist = prop.persistingL2CacheMaxSize;
        if (max_persist > 0) {
            size_t reserve = max_persist * 3 / 4;
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve);
            IMP_LOG_INFO("L2 persisting cache: reserved %zu MB / %zu MB total",
                         reserve >> 20, max_persist >> 20);
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
            if (model_->layer(i).wq.data != nullptr) n_attn++;
        if (n_attn == 0) n_attn = mcfg.n_layers;
        size_t kv_block_bytes = static_cast<size_t>(est_bs) * mcfg.n_kv_heads * head_dim_est * elem_sz;
        size_t kv_est = static_cast<size_t>(blocks_per_seq * config_.max_batch_size) * 2 * n_attn * kv_block_bytes;
        { size_t total_vram = 0, f = 0; cudaMemGetInfo(&f, &total_vram);
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
                if (model_->layer(i).ssm_in.data != nullptr) n_ssm++;
            expert_reserve += static_cast<size_t>(n_ssm) * config_.max_batch_size *
                (conv_ch * std::max(mcfg.ssm_conv_kernel - 1, 0) * sizeof(float) +
                 n_heads * hd_ssm * mcfg.ssm_state_size * dtype_size(config_.ssm_state_dtype));
        }

        size_t safety = 256ULL * 1024 * 1024;  // base safety
        // Only add safety for features that will actually allocate VRAM.
        // On tight VRAM models (Nemotron-30B), every MiB matters for expert coverage.
        if (config_.enable_speculative) safety += 256ULL * 1024 * 1024;
        expert_reserve += safety;

        IMP_LOG_INFO("Expert upload reserve: %.2f MiB (workspace=%.2f, kv=%.2f, ssm+safety=rest)",
                     expert_reserve / (1024.0 * 1024.0),
                     executor_->workspace_estimate() / (1024.0 * 1024.0),
                     kv_est / (1024.0 * 1024.0));
    }

    // Upload weights
    size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);
    IMP_LOG_INFO("GPU memory before weight upload: %zu MiB free / %zu MiB total",
                 free_before / (1024 * 1024), total_before / (1024 * 1024));

    cudaStream_t upload_stream = nullptr;
    IMP_CUDA_CHECK_LOG(cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking));

    if (!model_->upload_weights_gpu(config_.compute_dtype,
                                     upload_stream ? upload_stream : stream_,
                                     expert_reserve)) {
        IMP_LOG_ERROR("Weight upload failed. Try a smaller quantization.");
        if (upload_stream) IMP_CUDA_CHECK_LOG(cudaStreamDestroy(upload_stream));
        return false;
    }

    if (upload_stream) {
        cudaEvent_t upload_done;
        IMP_CUDA_CHECK_LOG(cudaEventCreate(&upload_done));
        IMP_CUDA_CHECK_LOG(cudaEventRecord(upload_done, upload_stream));
        IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream_, upload_done));
        IMP_CUDA_CHECK_LOG(cudaEventDestroy(upload_done));
        IMP_CUDA_CHECK_LOG(cudaStreamDestroy(upload_stream));
    }

    size_t free_after = 0, total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);
    IMP_LOG_INFO("GPU memory after weight upload: %zu MiB free / %zu MiB total (weights ~%zu MiB)",
                 free_after / (1024 * 1024), total_after / (1024 * 1024),
                 (free_before - free_after) / (1024 * 1024));

    // Check for host-resident expert weights
    if (mcfg.n_experts > 0) {
        for (int i = 0; i < mcfg.n_layers; i++) {
            if (model_->layer(i).expert_up_packed.data && !model_->layer(i).expert_up_packed.on_device) {
                experts_on_host_ = true;
                break;
            }
        }
        if (experts_on_host_ && config_.use_cuda_graphs) {
            IMP_LOG_INFO("Disabling CUDA graphs: expert weights on host");
            config_.use_cuda_graphs = false;
        }
        if (getenv("IMP_NO_CUDA_GRAPH") && config_.use_cuda_graphs) {
            IMP_LOG_INFO("Disabling CUDA graphs: IMP_NO_CUDA_GRAPH set");
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

bool Engine::init_kv_cache() {
    const auto& mcfg = model_->config();
    int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);

    // Build KV layer mapping for hybrid models
    int n_attn_layers = 0;
    std::vector<int> kv_layer_map(mcfg.n_layers, -1);
    for (int i = 0; i < mcfg.n_layers; i++) {
        if (model_->layer(i).wq.data != nullptr &&
            model_->layer(i).gdn_gate.data == nullptr)
            kv_layer_map[i] = n_attn_layers++;
    }
    if (n_attn_layers == 0) {
        n_attn_layers = mcfg.n_layers;
        for (int i = 0; i < mcfg.n_layers; i++) kv_layer_map[i] = i;
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
    auto vram_budget = compute_vram_budget(*model_, config_, n_kv_layers,
                                            head_dim, effective_free_vram());
    int max_blocks = config_.kv_cache_max_blocks > 0
        ? config_.kv_cache_max_blocks : vram_budget.kv_max_blocks;

    {
        DType kv_dtype = config_.kv_cache_dtype;
        size_t block_bytes = static_cast<size_t>(kv_bs) * mcfg.n_kv_heads * head_dim * dtype_size(kv_dtype);
        size_t total_kv = static_cast<size_t>(n_kv_layers) * max_blocks * 2 * block_bytes;
        IMP_LOG_INFO("KV cache: %d blocks (%.0f tokens), %.2f MiB, dtype=%s "
                     "(layers=%d/%d, kv_heads=%d, head_dim=%d, block_size=%d)",
                     max_blocks, static_cast<double>(max_blocks) * kv_bs,
                     static_cast<double>(total_kv) / (1024.0 * 1024.0),
                     dtype_name(kv_dtype),
                     n_kv_layers, mcfg.n_layers, mcfg.n_kv_heads, head_dim, kv_bs);
    }

    // Compute sketch_dim for TurboQuant / TurboQuant Lite (0 for other modes)
    int kv_sketch_dim = 0;
    if (config_.kv_cache_dtype == DType::TURBOQUANT) {
        kv_sketch_dim = head_dim;
    } else if (config_.kv_cache_dtype == DType::TURBOQUANT_LITE) {
        int mult = config_.turboquant_sketch_multiplier;
        if (mult <= 0) mult = 2;
        kv_sketch_dim = head_dim * mult;
    }

    // MXFP4 TurboQuant: FP4 E2M1 + UE8M0 micro-scales (requires head_dim % 32 == 0)
    bool tq_use_mxfp4 = false;
    if (config_.kv_cache_dtype == DType::TURBOQUANT && (head_dim % 32 == 0)) {
        tq_use_mxfp4 = true;
        IMP_LOG_INFO("TurboQuant: using MXFP4 FP4 E2M1 + UE8M0 for K directions");
    }

    // Per-layer KV shape path (Gemma 4 dual attention geometry): build per-layer
    // nkv/hd arrays restricted to attention layers (hybrid models may have non-attn layers).
    std::unique_ptr<KVCache> kv_cache;
    if (!mcfg.head_dim_per_layer.empty() &&
        config_.kv_cache_dtype != DType::TURBOQUANT &&
        config_.kv_cache_dtype != DType::TURBOQUANT_LITE &&
        config_.kv_cache_dtype != DType::INT8 &&
        config_.kv_cache_dtype != DType::INT4) {
        std::vector<int> per_layer_nkv(n_kv_layers, 0);
        std::vector<int> per_layer_hd(n_kv_layers, 0);
        for (int l = 0, k = 0; l < mcfg.n_layers && k < n_kv_layers; l++) {
            // Only attention layers get KV cache entries
            int attn_nkv = (l < (int)mcfg.n_kv_heads_per_layer.size())
                           ? mcfg.n_kv_heads_per_layer[l] : mcfg.n_kv_heads;
            if (attn_nkv <= 0) continue;  // non-attention layer (SSM/GDN)
            per_layer_nkv[k] = attn_nkv;
            per_layer_hd[k] = (l < (int)mcfg.head_dim_per_layer.size() && mcfg.head_dim_per_layer[l] > 0)
                              ? mcfg.head_dim_per_layer[l] : head_dim;
            k++;
        }
        kv_cache = std::make_unique<KVCache>(
            n_kv_layers, per_layer_nkv, per_layer_hd,
            config_.kv_cache_dtype, max_blocks, kv_bs, &vram_alloc_);
    } else {
        kv_cache = std::make_unique<KVCache>(
            n_kv_layers, mcfg.n_kv_heads, head_dim,
            config_.kv_cache_dtype, max_blocks, kv_bs, &vram_alloc_, kv_sketch_dim,
            tq_use_mxfp4);
    }
    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));

    if (config_.use_prefix_caching) {
        if (mcfg.ssm_inner_size > 0) {
            IMP_LOG_WARN("Prefix caching disabled for recurrent model — "
                         "SSM/GDN state requires full sequential prefill");
        } else {
            kv_manager_->set_prefix_caching_enabled(true);
            IMP_LOG_INFO("Prefix caching enabled");
            if (!config_.prefix_cache_path.empty()) {
                int restored = kv_manager_->load_prefix_cache(config_.prefix_cache_path, stream_);
                if (restored > 0)
                    IMP_LOG_INFO("Restored %d prefix cache blocks from %s", restored, config_.prefix_cache_path.c_str());
            }
        }
    }

    executor_->set_kv_layer_map(std::move(kv_layer_map));

    // Initialize QJL projection for TurboQuant / TurboQuant Lite KV cache
    if (config_.kv_cache_dtype == DType::TURBOQUANT
        || config_.kv_cache_dtype == DType::TURBOQUANT_LITE) {
        auto& qjl = executor_->qjl_projection();
        int sketch_dim;
        if (config_.kv_cache_dtype == DType::TURBOQUANT_LITE) {
            int mult = config_.turboquant_sketch_multiplier;
            if (mult <= 0) mult = 2;
            sketch_dim = head_dim * mult;
            IMP_LOG_INFO("TurboQuant Lite: sketch_dim=%d (mult=%d, head_dim=%d)", sketch_dim, mult, head_dim);
        } else {
            sketch_dim = head_dim;  // standard TurboQuant: sketch_dim = head_dim
        }
        if (!qjl_init(qjl, head_dim, sketch_dim, /*seed=*/42, stream_)) {
            IMP_LOG_ERROR("Failed to initialize QJL projection for %s",
                          dtype_name(config_.kv_cache_dtype));
            return false;
        }
    }

    if (offload_mgr_) executor_->set_offload_manager(offload_mgr_.get());
    scheduler_->set_kv_manager(kv_manager_.get());

    // SSM state
    if (mcfg.ssm_inner_size > 0) {
        int n_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).ssm_in.data != nullptr) n_ssm++;
        if (n_ssm > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int hd = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            ssm_state_ = std::make_unique<SSMState>();
            if (!ssm_state_->init(n_ssm, config_.max_batch_size, conv_ch, mcfg.ssm_conv_kernel,
                                   n_heads, hd, mcfg.ssm_state_size, config_.ssm_state_dtype, &vram_alloc_)) {
                IMP_LOG_WARN("Failed to init SSM state, continuing without it");
                ssm_state_.reset();
            }
        }
    }

    // GDN detection
    {
        int n_gdn = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).gdn_gate.data != nullptr) n_gdn++;
        if (n_gdn > 0) {
            IMP_LOG_INFO("GDN model: %d layers, CUDA graphs enabled (recurrent state in-place)", n_gdn);
            // GDN recurrent state accumulates small precision errors per token.
            // FP8 E4M3 (3-bit mantissa) amplifies these through the delta rule
            // scan, causing degenerate output after ~50 special tokens in
            // multi-turn chat.  Force FP16 weights for GDN prefill.
            if (config_.use_fp8_prefill) {
                if (config_.dual_path_quant) {
                    IMP_LOG_WARN("GDN + dual-path: attention weights forced to FP16 (not FP8) — "
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
        int n_pure_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).ssm_in.data != nullptr &&
                model_->layer(i).gdn_gate.data == nullptr)
                n_pure_ssm++;
        has_pure_ssm_layers_ = (n_pure_ssm > 0);
    }

    // Dequant weights → FP16/FP8/NVFP4 caches
    executor_->pre_dequant_weights(stream_, vram_budget);
    dequant_done_ = true;
    cudaStreamSynchronize(stream_);
    if (config_.use_fp8_prefill)
        IMP_LOG_INFO("Weight cache: FP8 E4M3 (2x prefill throughput on sm_120)");

    // Pre-allocate decode batch pool + penalty buffer
    decode_batch_pool_.allocate(config_.max_batch_size, blocks_per_seq, &vram_alloc_);
    {
        d_penalty_tokens_capacity_ = static_cast<size_t>(config_.max_seq_len);
        d_penalty_tokens_ = static_cast<int32_t*>(
            vram_alloc_.allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
        if (!d_penalty_tokens_) {
            IMP_LOG_WARN("Failed to pre-allocate penalty token buffer");
            d_penalty_tokens_capacity_ = 0;
        }
    }

    // Pre-allocate prefill metadata pool (avoids per-request cudaMallocAsync)
    {
        size_t tok_bytes = config_.max_seq_len * sizeof(int32_t);
        size_t pos_bytes = config_.max_seq_len * sizeof(int);
        size_t bt_bytes  = blocks_per_seq * sizeof(int);
        size_t cl_bytes  = sizeof(int);
        prefill_pool_size_ = tok_bytes + pos_bytes + bt_bytes + cl_bytes;
        prefill_pool_ = vram_alloc_.allocate(prefill_pool_size_, "prefill_pool");
        if (prefill_pool_) {
            auto* base = static_cast<char*>(prefill_pool_);
            d_pf_token_ids_   = reinterpret_cast<int32_t*>(base);
            d_pf_positions_   = reinterpret_cast<int*>(base + tok_bytes);
            d_pf_block_tables_ = reinterpret_cast<int*>(base + tok_bytes + pos_bytes);
            d_pf_context_lens_ = reinterpret_cast<int*>(base + tok_bytes + pos_bytes + bt_bytes);
        } else {
            IMP_LOG_WARN("Failed to pre-allocate prefill pool, will use per-request malloc");
        }

        // Pinned host staging buffers for prefill
        if (cudaHostAlloc(&h_pf_positions_, config_.max_seq_len * sizeof(int),
                          cudaHostAllocDefault) != cudaSuccess)
            h_pf_positions_ = nullptr;
        if (cudaHostAlloc(&h_pf_token_ids_, config_.max_seq_len * sizeof(int32_t),
                          cudaHostAllocDefault) != cudaSuccess)
            h_pf_token_ids_ = nullptr;
    }

    // Report memory
    {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            IMP_LOG_INFO("GPU memory: %.0f MiB used / %.0f MiB total (%.0f MiB free)",
                         (total_mem - free_mem) / (1024.0 * 1024.0),
                         total_mem / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
        vram_alloc_.report();
    }

    return true;
}

bool Engine::init_features() {
    const auto& mcfg = model_->config();

    // Green contexts
    if (config_.use_green_contexts) {
        if (!green_ctx_.init(0, config_.green_ctx_prefill_ratio)) {
            IMP_LOG_WARN("Green context init failed — falling back to regular streams");
            // Clear the CUDA error state so it doesn't corrupt subsequent operations.
            // Green context failure on sm_120 consumer GPUs is expected (requires
            // data-center features). Without clearing, the stale error causes
            // cublasLtMatmul to fail with CUBLAS_STATUS_INVALID_VALUE.
            cudaGetLastError();
        }
        if (green_ctx_.is_available() && config_.prefill_chunk_size > 0)
            if (executor_->allocate_decode_workspace(stream_, config_.max_batch_size))
                IMP_LOG_INFO("Concurrent prefill/decode overlap enabled");
    }

    // Speculative decoding variants
    if (config_.enable_speculative) {
        if (!init_speculative()) {
            IMP_LOG_WARN("Speculative decoding init failed, continuing without it");
            config_.enable_speculative = false;
        }
    }
    if (config_.enable_self_speculative) {
        if (config_.use_cuda_graphs) {
            IMP_LOG_INFO("Disabling CUDA graphs: self-speculative decoding active");
            config_.use_cuda_graphs = false;
        }
        self_spec_decoder_ = std::make_unique<SelfSpeculativeDecoder>();
        SelfSpecConfig ssc;
        ssc.spec_k = config_.self_spec_k;
        ssc.exit_layer = config_.self_spec_exit_layer;
        ssc.skip_n = config_.self_spec_skip_n;
        int n_kv = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).wq.data && !model_->layer(i).gdn_gate.data) n_kv++;
        if (n_kv == 0) n_kv = mcfg.n_layers;
        if (!self_spec_decoder_->init(executor_.get(), kv_manager_.get(),
                                       kv_cache_raw_, mcfg.n_layers, ssc)) {
            IMP_LOG_WARN("Self-speculative init failed, continuing without it");
            self_spec_decoder_.reset();
            config_.enable_self_speculative = false;
        }
    }
    if (config_.enable_ngram_spec && !config_.enable_speculative && !config_.enable_self_speculative) {
        if (config_.use_cuda_graphs) {
            IMP_LOG_INFO("Disabling CUDA graphs: n-gram speculative decoding active");
            config_.use_cuda_graphs = false;
        }
        int n_kv = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).wq.data && !model_->layer(i).gdn_gate.data) n_kv++;
        if (n_kv == 0) n_kv = mcfg.n_layers;
        ngram_spec_decoder_ = std::make_unique<NgramSpecDecoder>();
        if (!ngram_spec_decoder_->init(executor_.get(), kv_manager_.get(),
                                        kv_cache_raw_, n_kv, config_.ngram_spec_k, config_.ngram_n))
            ngram_spec_decoder_.reset();
    }

    // Chat template
    if (Tokenizer* tok = model_->tokenizer()) {
        auto family = ChatTemplate::detect_family(tok->chat_template_str());
        if (family == ChatTemplateFamily::RAW) {
            family = ChatTemplate::default_family_for_arch(mcfg.arch);
            if (family != ChatTemplateFamily::RAW)
                IMP_LOG_INFO("No chat template in metadata, using %s default for %s",
                             chat_template_family_name(family), model_arch_name(mcfg.arch));
        }
        if (family != ChatTemplateFamily::RAW)
            chat_template_.init(family, *tok, tok->chat_template_str());
    }

    // Build banned token list: special/control tokens that must never appear
    // in generated output.  If the model emits e.g. <|im_start|> or
    // <|endoftext|> mid-generation it starts a phantom new turn or hallucinates
    // a continuation, causing output degeneration.  llama.cpp blocks all
    // control tokens via llama_token_is_control(); we scan the vocabulary for
    // tokens matching known special-token patterns.
    {
        banned_token_ids_.clear();
        auto add_if_valid = [this](int32_t id) {
            if (id >= 0) banned_token_ids_.push_back(id);
        };

        // Collect IDs that must NOT be banned (stop tokens + EOS + think tokens).
        // The model must be able to generate these for correct operation.
        std::vector<int32_t> keep_ids;
        Tokenizer* tok = model_->tokenizer();
        if (tok) {
            for (int32_t eid : tok->eos_ids()) keep_ids.push_back(eid);
        }
        for (int32_t sid : chat_template_.stop_token_ids())
            keep_ids.push_back(sid);
        // Think tokens must not be banned — think models generate these to
        // enter/exit reasoning mode. Support both Qwen (<think>/<\/think>) and
        // Gemma-4 (<|think|>/<|/think|>) naming conventions.
        if (tok) {
            for (const char* name : {"<think>", "</think>", "<|think|>", "<|/think|>"}) {
                int32_t tid = tok->find_token(name);
                if (tid >= 0) keep_ids.push_back(tid);
            }
        }
        auto is_kept = [&](int32_t id) {
            return std::find(keep_ids.begin(), keep_ids.end(), id) != keep_ids.end();
        };

        // Chat template start-of-turn delimiters (never valid in output)
        if (!is_kept(chat_template_.im_start_id()))
            add_if_valid(chat_template_.im_start_id());
        if (!is_kept(chat_template_.start_header_id()))
            add_if_valid(chat_template_.start_header_id());
        if (!is_kept(chat_template_.end_header_id()))
            add_if_valid(chat_template_.end_header_id());

        // Scan vocab for control tokens, excluding stop/EOS tokens.
        if (tok) {
            int vocab_size = tok->vocab_size();
            if (tok->has_token_types()) {
                // Authoritative: use token_type metadata from GGUF.
                // CONTROL=3 tokens are special tokens that should not appear in output.
                for (int i = 0; i < vocab_size; i++) {
                    if (is_kept(static_cast<int32_t>(i))) continue;
                    if (tok->is_control_token(i)) {
                        add_if_valid(static_cast<int32_t>(i));
                    }
                }
            } else {
                // Fallback: heuristic pattern matching for GGUF files without token_type.
                for (int i = 0; i < vocab_size; i++) {
                    if (is_kept(static_cast<int32_t>(i))) continue;
                    const std::string& t = tok->token_text(i);
                    if (t.size() < 3 || t[0] != '<' || t.back() != '>') continue;
                    if (t.size() >= 4 && t[1] == '|' && t[t.size()-2] == '|') {
                        add_if_valid(static_cast<int32_t>(i));
                        continue;
                    }
                    if (t == "<pad>" || t == "<unk>" || t == "<mask>" ||
                        t == "<unused0>" || t == "<start_of_turn>" ||
                        t == "<end_of_turn>" || t == "<start_of_image>" ||
                        t == "<end_of_image>") {
                        add_if_valid(static_cast<int32_t>(i));
                    }
                }
            }
        }

        // Deduplicate
        std::sort(banned_token_ids_.begin(), banned_token_ids_.end());
        banned_token_ids_.erase(
            std::unique(banned_token_ids_.begin(), banned_token_ids_.end()),
            banned_token_ids_.end());

        if (!banned_token_ids_.empty()) {
            IMP_LOG_INFO("Banned %zu special tokens from generation",
                         banned_token_ids_.size());
            if (tok) {
                std::string bl;
                for (int32_t bid : banned_token_ids_) {
                    bl += std::to_string(bid) + "(" + tok->token_text(bid) + ") ";
                }
                IMP_LOG_INFO("  banned: %s", bl.c_str());
            }
        }
    }

    // Cache think token IDs for stop-suppression during reasoning.
    // Only treat as think model if <think> is a CONTROL token (from GGUF metadata),
    // not a regular text piece. Nemotron has "<think>" at ID 12 as normal text.
    {
        Tokenizer* ptok = model_->tokenizer();
        if (ptok) {
            int32_t ts = ptok->find_token("<think>");
            int32_t te = ptok->find_token("</think>");
            int vocab = ptok->vocab_size();
            bool is_special = (ts >= 0) && (ptok->has_token_types()
                ? ptok->is_control_token(ts)
                : ts > vocab * 99 / 100);
            if (is_special) {
                think_start_id_ = ts;
                think_end_id_ = te;
            }
        }
    }

    // Vision
    if (!config_.mmproj_path.empty()) {
        if (!vision_.init(config_.mmproj_path, mcfg.d_model, model_.get(), vram_alloc_, stream_))
            return false;
    }

    // Pinned sample buffer for CUDA graphs
    if (!h_sample_pinned_) {
        cudaError_t err = cudaHostAlloc(&h_sample_pinned_, sizeof(int32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostAlloc for sample buffer failed: %s", cudaGetErrorString(err));
            if (config_.use_cuda_graphs) config_.use_cuda_graphs = false;
            h_sample_pinned_ = nullptr;
        }
    }
    if (!decode_done_)
        (void)decode_done_.create(cudaEventDisableTiming);

    // Pre-allocate DRY penalty buffers to avoid cudaStreamSynchronize on first
    // use during inference (the lazy-alloc path blocks the decode stream).
    sampling_preallocate_dry(config_.max_seq_len, decode_stream());

    return true;
}

void Engine::warmup() {
    // Skip warmup for MXFP4 models — the warmup forward pass triggers
    // illegal memory access due to kernel paths that bypass the FP16 cache
    // and attempt to use raw MXFP4 data as FP16 weights.
    bool has_mxfp4_weights = false;
    for (int i = 0; i < model_->config().n_layers && !has_mxfp4_weights; i++) {
        if (model_->layer(i).wq_qtype == GGMLQuantType::MXFP4) has_mxfp4_weights = true;
    }
    if (has_mxfp4_weights) {
        IMP_LOG_INFO("Warmup skipped (MXFP4 model)");
        return;
    }

    // Gemma-4 has outlier-heavy output_norm activations that amplify cuBLAS
    // algo jitter — warming up with BOS-filled buffers pins an algo that
    // produces wrong logits under real inputs and drives decode into
    // backtick/markdown degeneration. IMP_NO_WARMUP=1 was the manual
    // mitigation; make it automatic for the arch.
    if (model_->config().arch == ModelArch::GEMMA4) {
        IMP_LOG_INFO("Warmup skipped (Gemma-4 algo-jitter protection)");
        return;
    }

    Tokenizer* tok = model_->tokenizer();
    int32_t warmup_id = tok ? tok->bos_id() : 1;
    if (warmup_id < 0) warmup_id = 1;

    for (int prompt_len : {16, 32}) {
        auto req = std::make_shared<Request>();
        req->id = next_request_id_++;
        req->input_tokens.resize(prompt_len, warmup_id);
        req->max_tokens = 2;
        req->temperature = 0.0f;
        req->ignore_eos = true;
        scheduler_->add_request(req);

        for (int i = 0; i < 8 && req->status != RequestStatus::FINISHED; i++)
            (void)step();

        kv_manager_->free_sequence(req->id);
        reset_ssm_state(req->id);
        while (kv_manager_->evict_cached_block()) {}
        req->status = RequestStatus::CANCELLED;
    }

    for (int i = 0; i < kMaxGraphPoolSize; i++) decode_graph_pool_[i].invalidate();
    decode_batch_pool_.reset_upload_cache();
    if (async_graph_runner_.is_setup()) async_graph_runner_.cleanup();
    if (async_d_block_tables_) { IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_)); async_d_block_tables_ = nullptr; }
    if (async_d_banned_tokens_) { IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_)); async_d_banned_tokens_ = nullptr; }
    async_graph_req_ = nullptr;
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) IMP_LOG_ERROR("warmup CUDA error: %s", cudaGetErrorString(e)); }
    // Clear any stale CUDA errors from warmup (e.g. green context reconfigure
    // failure on consumer GPUs — the error propagates to cuBLAS otherwise).
    cudaGetLastError();
    cudaDeviceSynchronize();  // ensure all weight upload/dequant kernels are done
    IMP_LOG_INFO("Warmup complete");
}

// init_speculative: moved to engine_speculative.cpp
// set_draft_model: moved to engine_speculative.cpp

// =====================================================================
// step() — main inference loop
// =====================================================================

bool Engine::step() {
    // Fast path: async conditional graph loop completed on GPU.
    int async_result = step_async_graph_resume();
    if (async_result == 1) return true;   // still running
    if (async_result == -1) {
        return scheduler_->has_pending() || scheduler_->active_count() > 0;
    }

    // Schedule prefill/decode batches and reconfigure green contexts.
    if (!step_schedule()) return false;

    // Process prefill requests.
    if (!sched_prefill_batch_.empty()) {
        step_prefill(prefill_stream());
        ensure_prefill_workspace(executor_.get());
    }

    // Process decode requests (batched).
    if (!sched_decode_batch_.empty()) {
        step_decode(decode_stream());
        ensure_prefill_workspace(executor_.get());
    }

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
}

// =====================================================================
// step_async_graph_resume — handle async conditional graph loop
// Returns: 0 = no graph active, 1 = still running, -1 = generation done
// =====================================================================

int Engine::step_async_graph_resume() {
    if (async_graph_runner_.is_setup() && async_graph_req_) {
        auto& req = async_graph_req_;

        if (async_pending_tokens_.empty() && async_pending_cursor_ == 0) {
            cudaStream_t dec_stream = decode_stream();
            async_pending_tokens_ = async_graph_runner_.wait_and_get_tokens(dec_stream);
        }

        int32_t token = -1;
        if (async_pending_cursor_ < static_cast<int>(async_pending_tokens_.size())) {
            token = async_pending_tokens_[async_pending_cursor_++];
        }

        bool generation_done = false;
        if (token >= 0) {
            req->output_tokens.push_back(token);
            track_think_state(*req, token);
            bool is_stop = should_stop(*req, token);
            generation_done = is_stop ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (!generation_done) return 1;
        }

        auto saved_req = async_graph_req_;

        async_graph_runner_.cleanup();
        if (async_d_block_tables_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
            async_d_block_tables_ = nullptr;
        }
        if (async_d_banned_tokens_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
            async_d_banned_tokens_ = nullptr;
        }
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;

        if (generation_done) {
            finish_request(saved_req);
            return -1;
        }

        IMP_LOG_DEBUG("AsyncGraphLoop: graph tokens exhausted, continuing with step decode");
    }

    // Clean up stale async graph state
    if (async_graph_req_ && !async_graph_runner_.is_setup()) {
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;
    }

    return 0;
}

// =====================================================================
// step_schedule — call scheduler, reconfigure green contexts
// Returns true if there is work to do.
// =====================================================================

bool Engine::step_schedule() {
    sched_prefill_batch_.clear();
    sched_decode_batch_.clear();
    scheduler_->schedule(sched_prefill_batch_, sched_decode_batch_);

    if (sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
        return false;
    }

    // Dynamic Green Context SM reconfiguration
    if (config_.use_green_contexts && green_ctx_.is_available() &&
        green_ctx_.has_green_contexts()) {
        float target_ratio = config_.green_ctx_prefill_ratio;
        if (sched_prefill_batch_.empty() && !sched_decode_batch_.empty()) {
            target_ratio = 0.0f;
        } else if (!sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
            target_ratio = 1.0f;
        }
        if (std::abs(target_ratio - green_ctx_.prefill_ratio()) > 0.1f) {
            green_ctx_.reconfigure(target_ratio);
        }
    }

    return true;
}

// =====================================================================
// step_prefill — process all prefill requests
// =====================================================================

void Engine::step_prefill(cudaStream_t stream) {
    int effective_chunk = config_.prefill_chunk_size > 0
        ? config_.prefill_chunk_size : executor_->max_tokens();
    if (kv_manager_) {
        int bs = kv_manager_->kv_cache()->block_size();
        if (effective_chunk > bs)
            effective_chunk = (effective_chunk / bs) * bs;
    }

    for (auto& req : sched_prefill_batch_) {
        step_prefill_one(req, effective_chunk, stream);
        kv_manager_->touch(req->id);
    }
}

// =====================================================================
// step_prefill_one — process a single prefill request
// =====================================================================

void Engine::step_prefill_one(std::shared_ptr<Request>& req, int effective_chunk,
                              cudaStream_t pf_stream) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    int total_input = static_cast<int>(req->input_tokens.size());
    int offset = req->prefill_offset;

    // Determine chunk boundaries
    int chunk_len = total_input - offset;
    bool is_last_chunk = true;
    if (chunk_len > effective_chunk) {
        chunk_len = effective_chunk;
        is_last_chunk = false;
    }

    int ctx_len = offset + chunk_len;
    (void)executor_->resize_workspace(chunk_len, pf_stream);

    int num_blocks = (ctx_len + kv_bs - 1) / kv_bs;

    // Allocate KV cache blocks
    int prefix_reused = 0;
    int existing = static_cast<int>(kv_manager_->block_table(req->id).size());

    if (kv_manager_->prefix_caching_enabled() && existing == 0 && offset == 0) {
        int total_blocks_needed = (total_input + kv_bs - 1) / kv_bs;
        prefix_reused = kv_manager_->allocate_blocks_with_prefix(
            req->id, req->input_tokens);
        if (prefix_reused < 0) {
            while (kv_manager_->num_free_blocks() < total_blocks_needed) {
                int evicted = kv_manager_->evict_lru();
                if (evicted < 0) break;
            }
            prefix_reused = kv_manager_->allocate_blocks_with_prefix(
                req->id, req->input_tokens);
            if (prefix_reused < 0) {
                req->status = RequestStatus::CANCELLED;
                return;
            }
        }

        if (prefix_reused > 0) {
            int effective_reused = (prefix_reused > 1) ? prefix_reused - 1 : 0;
            int skip_tokens = effective_reused * kv_bs;
            if (skip_tokens >= total_input) {
                skip_tokens = (total_input / kv_bs) * kv_bs;
                if (skip_tokens >= total_input) {
                    skip_tokens = total_input - 1;
                }
            }
            if (skip_tokens > offset) {
                IMP_LOG_INFO("PrefixCache: seq %d skipping %d/%d prefill tokens (%d blocks reused)",
                             req->id, skip_tokens, total_input, prefix_reused);
                req->cached_tokens = skip_tokens;
                offset = skip_tokens;
                req->prefill_offset = offset;
                chunk_len = total_input - offset;
                is_last_chunk = true;
                if (chunk_len > effective_chunk) {
                    chunk_len = effective_chunk;
                    is_last_chunk = false;
                }
                ctx_len = offset + chunk_len;
                (void)executor_->resize_workspace(chunk_len, pf_stream);
            }
        }
    } else {
        int additional = num_blocks - existing;
        if (additional > 0) {
            if (!kv_manager_->allocate_blocks(req->id, additional)) {
                while (kv_manager_->num_free_blocks() < additional) {
                    int evicted = kv_manager_->evict_lru();
                    if (evicted < 0) break;
                }
                if (!kv_manager_->allocate_blocks(req->id, additional)) {
                    kv_manager_->free_sequence(req->id);
                    req->status = RequestStatus::CANCELLED;
                    return;
                }
            }
        }
    }

    const auto& block_table = kv_manager_->block_table(req->id);

    // Upload prefill metadata to device (pre-allocated pool or fallback malloc)
    int32_t* d_token_ids = nullptr;
    int* d_positions = nullptr;
    int* d_block_tables = nullptr;
    int* d_context_lens = nullptr;
    bool pf_pool_used = false;

    auto check = [&req](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Engine::step prefill %s failed: %s", op, cudaGetErrorString(err));
            req->status = RequestStatus::CANCELLED;
        }
        return err == cudaSuccess;
    };

    if (prefill_pool_ && chunk_len <= config_.max_seq_len) {
        d_token_ids   = d_pf_token_ids_;
        d_positions   = d_pf_positions_;
        d_block_tables = d_pf_block_tables_;
        d_context_lens = d_pf_context_lens_;
        pf_pool_used = true;
    } else {
        if (!check(cudaMallocAsync(&d_token_ids, chunk_len * sizeof(int32_t), pf_stream), "malloc token_ids") ||
            !check(cudaMallocAsync(&d_positions, chunk_len * sizeof(int), pf_stream), "malloc positions") ||
            !check(cudaMallocAsync(&d_block_tables, block_table.size() * sizeof(int), pf_stream), "malloc block_tables") ||
            !check(cudaMallocAsync(&d_context_lens, sizeof(int), pf_stream), "malloc context_lens")) {
            if (d_token_ids) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, pf_stream));
            if (d_positions) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, pf_stream));
            if (d_block_tables) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, pf_stream));
            if (d_context_lens) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, pf_stream));
            kv_manager_->free_sequence(req->id);
            return;
        }
    }

    // Use pinned staging buffers when available (avoids internal pageable->pinned copy)
    if (h_pf_token_ids_ && chunk_len <= config_.max_seq_len) {
        memcpy(h_pf_token_ids_, req->input_tokens.data() + offset,
               chunk_len * sizeof(int32_t));
        check(cudaMemcpyAsync(d_token_ids, h_pf_token_ids_,
                        chunk_len * sizeof(int32_t),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy token_ids");
    } else {
        check(cudaMemcpyAsync(d_token_ids, req->input_tokens.data() + offset,
                        chunk_len * sizeof(int32_t),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy token_ids");
    }

    if (h_pf_positions_ && chunk_len <= config_.max_seq_len) {
        for (int i = 0; i < chunk_len; i++)
            h_pf_positions_[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, h_pf_positions_,
                        chunk_len * sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy positions");
    } else {
        std::vector<int> positions(chunk_len);
        for (int i = 0; i < chunk_len; i++)
            positions[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, positions.data(),
                        chunk_len * sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream), "memcpy positions");
    }

    check(cudaMemcpyAsync(d_block_tables, block_table.data(),
                    block_table.size() * sizeof(int),
                    cudaMemcpyHostToDevice, pf_stream), "memcpy block_tables");
    check(cudaMemcpyAsync(d_context_lens, &ctx_len, sizeof(int),
                    cudaMemcpyHostToDevice, pf_stream), "memcpy context_lens");

    // Build InferenceState
    InferenceState state;
    state.token_ids = d_token_ids;
    state.positions = d_positions;
    state.n_tokens = chunk_len;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_block_tables;
    state.context_lens = d_context_lens;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = 0;
    state.is_prefill = true;
    fill_sampling_params(*req, state);

    // Constraints via ConstraintManager
    constraints_.prepare(req->json_mode, req->json_schema, model_->tokenizer());
    state.json_constrainer = constraints_.json_constrainer();
    state.schema_constrainer = constraints_.schema_constrainer();

    // Penalties
    upload_penalties(*req, state, pf_stream);

    // Recurrent state (SSM/GDN)
    // Reset on the first chunk of a new request so previous-request state
    // doesn't leak in.  Subsequent chunks must NOT reset — the recurrent
    // state built during earlier chunks must carry forward.
    fill_recurrent_state(*req, state, /*reset=*/(offset == 0), pf_stream);

    // Vision embeddings on first chunk
    if (vision_.has_input() && vision_.is_available() && offset == 0) {
        state.vision_embeddings = vision_.embeddings();
        state.vision_token_id = vision_.soft_token_id();
        state.n_vision_tokens = vision_.num_image_tokens();
    }

    if (!is_last_chunk) {
        if (executor_->has_decode_workspace()) {
            executor_->use_workspace(0);
        }
        Tensor logits_out;
        executor_->forward_logits(state, logits_out, pf_stream);

        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions,
                                 d_block_tables, d_context_lens, pf_stream);
        }

        req->prefill_offset = offset + chunk_len;
        IMP_LOG_DEBUG("Chunked prefill: req %d chunk [%d, %d) of %d",
                      req->id, offset, offset + chunk_len, total_input);
    } else {
        // Last chunk: forward + sample
        int32_t next_token;
        bool use_event_sync = (h_sample_pinned_ != nullptr &&
                               executor_->d_sample_result() != nullptr &&
                               (state.temperature <= 0.0f || state.top_k == 1) &&
                               !req->logprobs &&
                               !state.json_constrainer &&
                               !state.schema_constrainer);

        Tensor prefill_logits_out;

        if (use_event_sync) {
            Tensor logits_out;
            executor_->forward_logits(state, logits_out, pf_stream);
            Tensor last_logits = logits_out.slice(0, 1);
            int64_t vocab_shape[1] = {last_logits.shape[1]};
            last_logits = last_logits.reshape(1, vocab_shape);
            sample_greedy_device(last_logits, executor_->d_sample_result(),
                                  h_sample_pinned_, pf_stream);

            if (!prefill_done_) (void)prefill_done_.create();
            cudaEventRecord(prefill_done_, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions,
                                     d_block_tables, d_context_lens, pf_stream);
            }

            cudaEventSynchronize(prefill_done_);
            next_token = *h_sample_pinned_;
        } else if (req->logprobs) {
            executor_->forward_logits(state, prefill_logits_out, pf_stream);
            auto sampled = executor_->sample_from_logits(prefill_logits_out, state, pf_stream);
            next_token = sampled[0];

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions,
                                     d_block_tables, d_context_lens, pf_stream);
            }
        } else {
            next_token = executor_->forward(state, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions,
                                     d_block_tables, d_context_lens, pf_stream);
            }
        }

        if (req->mirostat == 2)
            req->mirostat_mu = state.mirostat_mu;

        // Extract logprobs
        if (req->logprobs && prefill_logits_out.data != nullptr) {
            int vocab_size = static_cast<int>(prefill_logits_out.shape[prefill_logits_out.ndim - 1]);
            executor_->ensure_logits_pinned(vocab_size);

            const float* d_logits = static_cast<const float*>(prefill_logits_out.data);
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(executor_->h_logits_pinned(), d_logits,
                            vocab_size * sizeof(float),
                            cudaMemcpyDeviceToHost, pf_stream));
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(pf_stream));

            req->output_logprobs.push_back(
                build_logprob_info(executor_->h_logits_pinned(), vocab_size,
                                   next_token, req->top_logprobs,
                                   model_->tokenizer()));
        }

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        Tokenizer* tok = model_->tokenizer();
        IMP_LOG_DEBUG("Prefill -> token %d (ctx=%d): id=%d [%s]",
                      (int)req->output_tokens.size(), req->context_len(),
                      next_token, tok->decode_token(next_token).c_str());

        // Update constraint FSM
        constraints_.update(next_token);

        if (should_stop(*req, next_token) ||
            static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        } else {
            req->status = RequestStatus::DECODING;
            if (kv_manager_->prefix_caching_enabled()) {
                kv_manager_->register_block_hashes(req->id, req->input_tokens);
            }
        }
    }
}

// =====================================================================
// step_decode — process all decode requests (batched)
// =====================================================================

void Engine::step_decode(cudaStream_t dec_stream) {
    auto& decode_batch = sched_decode_batch_;
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // SSM/GDN: limit decode batch to 1 sequence
    if ((ssm_state_ || gdn_state_) && decode_batch.size() > 1) {
        decode_batch.resize(1);
    }

    // Allocate new KV blocks where needed
    valid_decode_.clear();
    auto& valid_decode = valid_decode_;

    for (auto& req : decode_batch) {
        int ctx_len = req->context_len();
        int blocks_needed = (ctx_len + kv_bs - 1) / kv_bs;
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
                    kv_manager_->free_sequence(req->id);
                    req->status = RequestStatus::CANCELLED;
                    continue;
                }
            }
        }
        valid_decode.push_back(req);
    }

    if (!valid_decode.empty()) {
        // Speculative decode shortcut (self-spec, n-gram)
        if (try_speculative_decode(valid_decode, dec_stream))
            return;

        step_decode_forward(valid_decode, dec_stream);
    }
}

// =====================================================================
// step_decode_forward — build batch, run forward pass, sample, process
// =====================================================================

void Engine::step_decode_forward(std::vector<std::shared_ptr<Request>>& valid_decode,
                                  cudaStream_t dec_stream) {
    // Switch workspace for decode
    if (executor_->has_decode_workspace() && valid_decode.size() == 1) {
        executor_->use_workspace(1);
    } else {
        if (executor_->active_workspace() == 1) executor_->use_workspace(0);
        (void)executor_->resize_workspace(static_cast<int>(valid_decode.size()), dec_stream);
    }

    // Build batched decode
    decode_builder_.reset();

    int max_ctx = 0;
    for (auto& req : valid_decode) {
        int ctx_len = req->context_len();
        max_ctx = std::max(max_ctx, ctx_len);

        int32_t last_token = req->output_tokens.empty()
            ? req->input_tokens.back()
            : req->output_tokens.back();
        int position = ctx_len - 1;

        const auto& bt = kv_manager_->block_table(req->id);
        decode_builder_.add_decode_sequence(last_token, position,
                                           bt.data(), static_cast<int>(bt.size()),
                                           ctx_len);
    }

    Batch batch = decode_builder_.build();

    // Upload to GPU using pre-allocated pool
    GPUBatch gpu_batch;
    if (decode_batch_pool_.is_allocated()) {
        int pool_max = decode_batch_pool_.max_blocks_per_seq();
        if (batch.max_blocks_per_seq < pool_max) {
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
            batch.block_tables.swap(padded_block_table_);
            batch.max_blocks_per_seq = pool_max;
        }
        gpu_batch = decode_batch_pool_.upload_into_pool(batch, dec_stream);
    } else {
        gpu_batch.upload(batch, dec_stream);
    }

    // Build InferenceState
    InferenceState state;
    state.token_ids = gpu_batch.d_token_ids;
    state.positions = gpu_batch.d_positions;
    state.n_tokens = gpu_batch.total_tokens;
    state.n_sequences = gpu_batch.n_sequences;
    state.max_blocks_per_seq = gpu_batch.max_blocks_per_seq;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = gpu_batch.d_block_tables;
    state.context_lens = gpu_batch.d_context_lens;
    state.max_context_len = max_ctx;
    state.is_prefill = false;
    fill_sampling_params(*valid_decode[0], state);

    // Derive per-step seed: mix request seed with output count so each
    // decode step gets a different random draw.  Without this, seed=-1
    // falls back to a fixed 42 on every step, producing identical RNG
    // values and causing repetition loops on long structured outputs.
    state.seed = compute_step_seed(*valid_decode[0]);

    // Penalties (single-sequence only)
    if (gpu_batch.n_sequences == 1) {
        upload_penalties(*valid_decode[0], state, dec_stream);
    }

    // Recurrent state
    fill_recurrent_state(*valid_decode[0], state, false, dec_stream);

    // Check if any request needs logprobs or constrained mode
    bool needs_logprobs = false;
    bool needs_json_mode = false;
    bool needs_schema_mode = false;
    for (const auto& r : valid_decode) {
        if (r->logprobs) needs_logprobs = true;
        if (r->json_mode && r->json_schema.empty()) needs_json_mode = true;
        if (!r->json_schema.empty()) needs_schema_mode = true;
    }

    // Schema/JSON constraints for decode (reuse state from prefill)
    if (needs_schema_mode && valid_decode.size() == 1 &&
        !valid_decode[0]->json_schema.empty()) {
        if (constraints_.has_schema()) {
            state.schema_constrainer = constraints_.schema_constrainer();
        }
    }
    if (needs_json_mode && valid_decode.size() == 1 && valid_decode[0]->json_mode) {
        // Lazily init if needed (decode might be first step with json_mode)
        if (!constraints_.has_json() && !constraints_.has_schema()) {
            constraints_.prepare(true, "", model_->tokenizer());
        }
        state.json_constrainer = constraints_.json_constrainer();
    }

    // Per-request sampling lambda
    auto sample_per_request = [&](const Tensor& logits) -> std::vector<int32_t> {
        int n = static_cast<int>(valid_decode.size());

        if (n == 1) {
            auto& req = valid_decode[0];
            int32_t tok = executor_->sample_single_from_logits(logits, state, dec_stream);
            if (state.mirostat == 2)
                req->mirostat_mu = state.mirostat_mu;
            return {tok};
        }

        std::vector<int32_t> result(n);
        for (int i = 0; i < n; i++) {
            auto& req = valid_decode[i];
            InferenceState per_state = state;
            fill_sampling_params(*req, per_state);
            // Per-step seed (same fix as single-sequence path)
            per_state.seed = compute_step_seed(*req);
            per_state.penalty_tokens = nullptr;
            per_state.n_penalty_tokens = 0;
            bool req_needs_pen = (req->repetition_penalty != 1.0f ||
                                  req->frequency_penalty != 0.0f ||
                                  req->presence_penalty != 0.0f);
            if (req_needs_pen && !req->output_tokens.empty() && d_penalty_tokens_) {
                size_t rn = req->output_tokens.size();
                if (rn <= d_penalty_tokens_capacity_) {
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_penalty_tokens_, req->output_tokens.data(),
                                    rn * sizeof(int32_t), cudaMemcpyHostToDevice, dec_stream));
                    per_state.penalty_tokens = d_penalty_tokens_;
                    per_state.n_penalty_tokens = static_cast<int>(rn);
                }
            }
            per_state.n_sequences = 1;
            Tensor seq_logits = logits.slice(i, i + 1);
            result[i] = executor_->sample_single_from_logits(seq_logits, per_state, dec_stream);
            if (per_state.mirostat == 2)
                req->mirostat_mu = per_state.mirostat_mu;
        }
        return result;
    };

    // Execute forward pass (piecewise CUDA Graph: forward in graph,
    // sampling always eager — per-batch-size graph pool avoids
    // re-capture when continuous batching changes batch size)
    std::vector<int32_t> tokens;
    Tensor decode_logits_out;

    static const bool profiling = (std::getenv("IMP_PROFILE") != nullptr);
    int graph_idx = gpu_batch.n_sequences - 1;
    if (config_.use_cuda_graphs && !profiling &&
        gpu_batch.n_sequences > 0 &&
        graph_idx < kMaxGraphPoolSize &&
        decode_batch_pool_.is_allocated()) {
        auto& graph_runner = decode_graph_pool_[graph_idx];

        if (gpu_batch.max_blocks_per_seq != last_decode_max_blocks_per_graph_[graph_idx]) {
            graph_runner.invalidate();
            last_decode_max_blocks_per_graph_[graph_idx] = gpu_batch.max_blocks_per_seq;
        }

        // Graph captures ONLY forward_logits — sampling runs eager after
        Tensor logits_out;
        graph_runner.set_decode_fn(
            [this, &state, &logits_out](cudaStream_t s) {
                executor_->forward_logits(state, logits_out, s);
            });
        graph_runner.execute(dec_stream);

        if (logits_out.data == nullptr) {
            logits_out = executor_->get_logits_view(gpu_batch.n_sequences);
        }
        // Eager sampling (handles all modes: greedy, top-k/p, penalties,
        // force_token, constraints, logprobs, mirostat)
        tokens = sample_per_request(logits_out);
        if (needs_logprobs) decode_logits_out = logits_out;
    } else {
        executor_->forward_logits(state, decode_logits_out, dec_stream);
        tokens = sample_per_request(decode_logits_out);
    }

    if (!decode_batch_pool_.is_allocated()) {
        gpu_batch.free();
    }

    // Process outputs: logprobs extraction + token distribution
    step_decode_process_outputs(valid_decode, tokens, decode_logits_out,
                                needs_logprobs, needs_json_mode, needs_schema_mode,
                                dec_stream);
}

// =====================================================================
// step_decode_process_outputs — extract logprobs, distribute tokens,
//                                try async graph loop
// =====================================================================

void Engine::step_decode_process_outputs(
        std::vector<std::shared_ptr<Request>>& valid_decode,
        const std::vector<int32_t>& tokens,
        const Tensor& decode_logits_out,
        bool needs_logprobs, bool needs_json_mode, bool needs_schema_mode,
        cudaStream_t dec_stream) {
    Tokenizer* tok = model_->tokenizer();

    // Extract logprobs
    if (needs_logprobs && decode_logits_out.data != nullptr) {
        int vocab_size = static_cast<int>(decode_logits_out.shape[decode_logits_out.ndim - 1]);
        int n_lp = 0;
        for (const auto& r : valid_decode) if (r->logprobs) n_lp++;
        executor_->ensure_logits_pinned(vocab_size * n_lp);
        float* h_base = executor_->h_logits_pinned();

        int slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            if (!valid_decode[i]->logprobs) continue;
            const float* d_logits = static_cast<const float*>(decode_logits_out.data)
                + static_cast<size_t>(i) * vocab_size;
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_base + static_cast<size_t>(slot) * vocab_size,
                            d_logits, vocab_size * sizeof(float),
                            cudaMemcpyDeviceToHost, dec_stream));
            slot++;
        }
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(dec_stream));

        slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            auto& req = valid_decode[i];
            if (!req->logprobs) continue;

            float* h_logits = h_base + static_cast<size_t>(slot) * vocab_size;
            req->output_logprobs.push_back(
                build_logprob_info(h_logits, vocab_size, tokens[i],
                                   req->top_logprobs, tok));
            slot++;
        }
    }

    // Distribute sampled tokens back to requests
    for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
        auto& req = valid_decode[i];
        int32_t next_token = tokens[i];

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        IMP_LOG_DEBUG("Decode step %d (ctx=%d, pos=%d): id=%d [%s]",
                      (int)req->output_tokens.size(), req->context_len(),
                      req->context_len() - 1,
                      next_token, tok->decode_token(next_token).c_str());

        if (should_stop(*req, next_token) ||
            static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        }

        constraints_.update(next_token);
        kv_manager_->touch(req->id);
    }

    // Try async graph loop after first decode step.
    // Think budget is now handled device-side in post_decode_step_kernel.
    if (decode_graph_pool_[0].is_ready() && valid_decode.size() == 1 &&
        !offload_mgr_ &&
        !config_.enable_speculative &&
        config_.use_cuda_graphs && !async_graph_runner_.is_setup() &&
        !needs_logprobs && !needs_json_mode && !needs_schema_mode) {
        auto& dreq = valid_decode[0];
        // forward_decode_async only implements banned_tokens + rep/freq/presence
        // penalties device-side. Any sampling feature that requires host-side
        // logic (logit_bias, mirostat, typical_p, min_p, DRY) would be silently
        // skipped inside the captured graph — stay on the eager path instead.
        const bool async_compatible =
            dreq->logit_bias.empty() &&
            dreq->mirostat == 0 &&
            dreq->dry_multiplier == 0.0f &&
            dreq->min_p == 0.0f &&
            dreq->typical_p >= 1.0f;
        if (async_compatible &&
            dreq->status == RequestStatus::DECODING &&
            !dreq->output_tokens.empty()) {
            int32_t last_token = dreq->output_tokens.back();
            try_launch_async_graph_loop(dreq, last_token, dec_stream);
        }
    }
}

// =====================================================================
// generate()
// =====================================================================

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
        std::vector<ChatMessage> messages = {{"user", prompt}};
        if (vision_.has_input() && vision_.is_available()) {
            tokens = chat_template_.apply_with_image(*tok, messages,
                                                      vision_.num_image_tokens());
        } else {
            tokens = chat_template_.apply(*tok, messages);
        }
        IMP_LOG_INFO("Applied %s chat template (%zu tokens%s)",
                     chat_template_family_name(chat_template_.family()),
                     tokens.size(),
                     vision_.has_input() ? ", with image" : "");
    } else {
        tokens = tok->encode(prompt);
        if (tok->add_bos() && (tokens.empty() || tokens[0] != tok->bos_id())) {
            tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
        }
    }

    IMP_LOG_INFO("Encoded %zu tokens", tokens.size());
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

    // Prefill
    while (req->status == RequestStatus::PENDING ||
           req->status == RequestStatus::PREFILLING) {
        bool has_work = step();
        if (!has_work) break;
    }

    // Decode — try conditional graph loop, fall back to step()
    // Think budget is now enforced device-side in post_decode_step_kernel.
    // Penalties are applied device-side via apply_penalties_device_count in the graph loop.
    if (req->status == RequestStatus::DECODING && !req->output_tokens.empty() &&
        config_.use_cuda_graphs && !offload_mgr_ &&
        !config_.enable_speculative) {
        int32_t first_token = req->output_tokens.back();
        auto graph_tokens = try_graph_loop_decode(req, first_token, decode_stream());
        if (!graph_tokens.empty()) {
            int32_t last = graph_tokens.back();
            // Track think state through all graph tokens
            for (int32_t t : graph_tokens) track_think_state(*req, t);
            bool hit_stop = should_stop(*req, last);
            if (hit_stop) graph_tokens.pop_back();

            for (int32_t t : graph_tokens) {
                req->output_tokens.push_back(t);
            }

            bool done = hit_stop ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (done) {
                req->status = RequestStatus::FINISHED;
                kv_manager_->free_sequence(req->id);
            }
        }
    }

    // Fallback — per-step decode
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

    vision_.clear_image();

    std::string result = tok->decode(req->output_tokens);
    return result;
}

// try_speculative_decode: moved to engine_speculative.cpp
// prepare_graph_loop: moved to engine_speculative.cpp
// build_graph_config: moved to engine_speculative.cpp
// try_graph_loop_decode: moved to engine_speculative.cpp
// try_launch_async_graph_loop: moved to engine_speculative.cpp

void Engine::add_request(std::shared_ptr<Request> req) {
    if (scheduler_) {
        req->id = next_request_id_++;
        scheduler_->add_request(std::move(req));
    }
}

} // namespace imp
