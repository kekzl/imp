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
    cudaFreeAsync(d_token_ids, stream);
    cudaFreeAsync(d_positions, stream);
    cudaFreeAsync(d_block_tables, stream);
    cudaFreeAsync(d_context_lens, stream);
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
        cudaFree(async_d_block_tables_);
        async_d_block_tables_ = nullptr;
    }
    if (async_d_banned_tokens_) {
        cudaFree(async_d_banned_tokens_);
        async_d_banned_tokens_ = nullptr;
    }
    if (d_penalty_tokens_) {
        vram_alloc_.free(d_penalty_tokens_);
        d_penalty_tokens_ = nullptr;
    }
    if (h_sample_pinned_) {
        cudaFreeHost(h_sample_pinned_);
        h_sample_pinned_ = nullptr;
    }
    if (prefill_pool_) {
        vram_alloc_.free(prefill_pool_);
        prefill_pool_ = nullptr;
    }
    if (h_pf_positions_) {
        cudaFreeHost(h_pf_positions_);
        h_pf_positions_ = nullptr;
    }
    if (h_pf_token_ids_) {
        cudaFreeHost(h_pf_token_ids_);
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
    for (int i = 0; i < kMaxGraphPoolSize; i++)
        decode_graph_pool_[i].invalidate();
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
    cudaMemcpyAsync(d_penalty_tokens_, req.output_tokens.data(),
                    n * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
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

    // --- Resolve auto-detection flags ---
    // NVFP4 decode mode
    int n_gdn_auto = 0;
    for (int i = 0; i < mcfg.n_layers; i++)
        if (model_->layer(i).gdn_gate.data != nullptr) n_gdn_auto++;

    if (config_.use_nvfp4_decode < 0) {
        int sm = get_device_sm_version();
        if (n_gdn_auto > 0) {
            config_.use_nvfp4_decode = 0;
            IMP_LOG_INFO("NVFP4 decode: auto → disabled (GDN model, %d recurrent layers)", n_gdn_auto);
        } else if (mcfg.d_model < 4096 && mcfg.n_experts == 0) {
            config_.use_nvfp4_decode = 0;
            IMP_LOG_INFO("NVFP4 decode: auto → disabled (d_model=%d < 4096)", mcfg.d_model);
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

    // --- Core initialization ---
    // 5% headroom (was 10%) — MoE models (30B Q6_K) need every MiB on 32GB.
    // WSL2/WDDM has ~500 MiB driver overhead, 5% of 32GB = 1.6 GB covers it.
    if (!vram_alloc_.init(0.05f)) {
        IMP_LOG_ERROR("Failed to initialize VRAM allocator");
        return false;
    }
    gemm_init();
    scheduler_ = std::make_unique<Scheduler>(config_.max_batch_size);
    stream_.create(cudaStreamNonBlocking);

    // --- Sub-phases ---
    if (!init_weights()) return false;
    if (!init_kv_cache()) return false;
    if (!init_features()) return false;
    warmup();

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
        { size_t total_vram = 0, f = 0; cudaMemGetInfo(&f, &total_vram); kv_est = std::min(kv_est, total_vram / 5); }
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
    cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking);

    if (!model_->upload_weights_gpu(config_.compute_dtype,
                                     upload_stream ? upload_stream : stream_,
                                     expert_reserve)) {
        IMP_LOG_ERROR("Weight upload failed. Try a smaller quantization.");
        if (upload_stream) cudaStreamDestroy(upload_stream);
        return false;
    }

    if (upload_stream) {
        cudaEvent_t upload_done;
        cudaEventCreate(&upload_done);
        cudaEventRecord(upload_done, upload_stream);
        cudaStreamWaitEvent(stream_, upload_done);
        cudaEventDestroy(upload_done);
        cudaStreamDestroy(upload_stream);
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
    }

    // Phase 2: allocate GPU workspace
    executor_->allocate_workspaces(experts_on_host_);

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

    // Detect sm_120+ for MXFP4 TurboQuant: FP4 E2M1 + UE8M0 micro-scales
    bool tq_use_mxfp4 = false;
    if (config_.kv_cache_dtype == DType::TURBOQUANT) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int sm_ver = prop.major * 10 + prop.minor;
        if (sm_ver >= 120 && (head_dim % 32 == 0)) {
            tq_use_mxfp4 = true;
            IMP_LOG_INFO("TurboQuant: sm_%d detected, using MXFP4 FP4 E2M1 + UE8M0 for K directions", sm_ver);
        }
    }

    auto kv_cache = std::make_unique<KVCache>(
        n_kv_layers, mcfg.n_kv_heads, head_dim,
        config_.kv_cache_dtype, max_blocks, kv_bs, &vram_alloc_, kv_sketch_dim,
        tq_use_mxfp4);
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
                IMP_LOG_INFO("GDN model: disabling FP8 prefill (recurrent state needs FP16 precision)");
                config_.use_fp8_prefill = 0;
                executor_->disable_fp8_prefill();
            }
        }
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
        // Think tokens (<think>/<\/think>) must not be banned — think models
        // generate these to enter/exit reasoning mode. Banning them causes
        // immediate stop on structured output prompts.
        if (tok) {
            int32_t ts = tok->find_token("<think>");
            int32_t te = tok->find_token("</think>");
            if (ts >= 0) keep_ids.push_back(ts);
            if (te >= 0) keep_ids.push_back(te);
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
        decode_done_.create(cudaEventDisableTiming);

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
            step();

        kv_manager_->free_sequence(req->id);
        reset_ssm_state(req->id);
        while (kv_manager_->evict_cached_block()) {}
        req->status = RequestStatus::CANCELLED;
    }

    for (int i = 0; i < kMaxGraphPoolSize; i++) decode_graph_pool_[i].invalidate();
    decode_batch_pool_.reset_upload_cache();
    if (async_graph_runner_.is_setup()) async_graph_runner_.cleanup();
    if (async_d_block_tables_) { cudaFree(async_d_block_tables_); async_d_block_tables_ = nullptr; }
    if (async_d_banned_tokens_) { cudaFree(async_d_banned_tokens_); async_d_banned_tokens_ = nullptr; }
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

// =====================================================================
// Speculative decoding init
// =====================================================================

bool Engine::init_speculative() {
    if (config_.draft_model_path.empty()) {
        IMP_LOG_ERROR("Speculative decoding enabled but no draft model path provided");
        return false;
    }

    auto draft_unique = load_gguf(config_.draft_model_path);
    if (!draft_unique) {
        IMP_LOG_ERROR("Failed to load draft model: %s", config_.draft_model_path.c_str());
        return false;
    }
    draft_model_ = std::move(draft_unique);

    if (!draft_model_->upload_weights_gpu(config_.compute_dtype, stream_)) {
        IMP_LOG_ERROR("Failed to upload draft model weights");
        return false;
    }

    const auto& dcfg = draft_model_->config();
    int hd = dcfg.head_dim > 0 ? dcfg.head_dim : (dcfg.d_model / dcfg.n_heads);
    int draft_max_blocks = std::max(kv_cache_raw_->total_blocks() / 4, 64);
    auto draft_kv = std::make_unique<KVCache>(
        dcfg.n_layers, dcfg.n_kv_heads, hd,
        config_.compute_dtype, draft_max_blocks);
    draft_kv_manager_ = std::make_unique<KVCacheManager>(std::move(draft_kv));

    auto draft_exec = std::make_unique<GraphExecutor>();
    if (!draft_exec->init(*draft_model_, config_.compute_dtype, config_.use_pdl)) {
        IMP_LOG_ERROR("Failed to init draft executor");
        return false;
    }

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

// =====================================================================
// step() — main inference loop
// =====================================================================

bool Engine::step() {
    // Ensure all async weight dequant / upload operations are complete.
    // Without this, MXFP4 CPU-side dequant H2D copies may race with
    // prefill memcpy on the same device memory (first call only).
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // ====================================================================
    // Fast path: async conditional graph loop completed on GPU.
    // ====================================================================
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
            if (!generation_done) return true;
        }

        auto saved_req = async_graph_req_;

        async_graph_runner_.cleanup();
        if (async_d_block_tables_) {
            cudaFree(async_d_block_tables_);
            async_d_block_tables_ = nullptr;
        }
        if (async_d_banned_tokens_) {
            cudaFree(async_d_banned_tokens_);
            async_d_banned_tokens_ = nullptr;
        }
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;

        if (generation_done) {
            finish_request(saved_req);
            return scheduler_->has_pending() || scheduler_->active_count() > 0;
        }

        IMP_LOG_DEBUG("AsyncGraphLoop: graph tokens exhausted, continuing with step decode");
    }

    // Clean up stale async graph state
    if (async_graph_req_ && !async_graph_runner_.is_setup()) {
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;
    }

    // 1. Call scheduler
    sched_prefill_batch_.clear();
    sched_decode_batch_.clear();
    scheduler_->schedule(sched_prefill_batch_, sched_decode_batch_);
    auto& prefill_batch = sched_prefill_batch_;
    auto& decode_batch = sched_decode_batch_;

    if (prefill_batch.empty() && decode_batch.empty()) {
        return false;
    }

    // Dynamic Green Context SM reconfiguration
    if (config_.use_green_contexts && green_ctx_.is_available() &&
        green_ctx_.has_green_contexts()) {
        float target_ratio = config_.green_ctx_prefill_ratio;
        if (prefill_batch.empty() && !decode_batch.empty()) {
            target_ratio = 0.0f;
        } else if (!prefill_batch.empty() && decode_batch.empty()) {
            target_ratio = 1.0f;
        }
        if (std::abs(target_ratio - green_ctx_.prefill_ratio()) > 0.1f) {
            green_ctx_.reconfigure(target_ratio);
        }
    }

    // ====================================================================
    // 2. Process prefill requests
    // ====================================================================
    cudaStream_t pf_stream = prefill_stream();

    for (auto& req : prefill_batch) {
        int total_input = static_cast<int>(req->input_tokens.size());
        int offset = req->prefill_offset;

        // Determine chunk boundaries
        int chunk_len = total_input - offset;
        bool is_last_chunk = true;
        int effective_chunk = config_.prefill_chunk_size > 0
            ? config_.prefill_chunk_size : executor_->max_tokens();
        if (kv_manager_) {
            int bs = kv_manager_->kv_cache()->block_size();
            if (effective_chunk > bs)
                effective_chunk = (effective_chunk / bs) * bs;
        }
        if (chunk_len > effective_chunk) {
            chunk_len = effective_chunk;
            is_last_chunk = false;
        }

        int ctx_len = offset + chunk_len;
        executor_->resize_workspace(chunk_len, pf_stream);

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
                    continue;
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
                    executor_->resize_workspace(chunk_len, pf_stream);
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
                        continue;
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
                if (d_token_ids) cudaFreeAsync(d_token_ids, pf_stream);
                if (d_positions) cudaFreeAsync(d_positions, pf_stream);
                if (d_block_tables) cudaFreeAsync(d_block_tables, pf_stream);
                if (d_context_lens) cudaFreeAsync(d_context_lens, pf_stream);
                kv_manager_->free_sequence(req->id);
                continue;
            }
        }

        // Use pinned staging buffers when available (avoids internal pageable→pinned copy)
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

                if (!prefill_done_) prefill_done_.create();
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
                cudaMemcpyAsync(executor_->h_logits_pinned(), d_logits,
                                vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, pf_stream);
                cudaStreamSynchronize(pf_stream);

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

        kv_manager_->touch(req->id);
    }

    // ====================================================================
    // 2b. Restore workspace
    // ====================================================================
    ensure_prefill_workspace(executor_.get());

    // ====================================================================
    // 3. Process decode requests (BATCHED)
    // ====================================================================
    if (!decode_batch.empty()) {
        cudaStream_t dec_stream = decode_stream();

        // SSM/GDN: limit decode batch to 1 sequence
        if ((ssm_state_ || gdn_state_) && decode_batch.size() > 1) {
            decode_batch.resize(1);
        }

        // 3a. Pre-process: allocate new blocks where needed
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
                goto decode_done;

            // Switch workspace for decode
            if (executor_->has_decode_workspace() && valid_decode.size() == 1) {
                executor_->use_workspace(1);
            } else {
                if (executor_->active_workspace() == 1) executor_->use_workspace(0);
                executor_->resize_workspace(static_cast<int>(valid_decode.size()), dec_stream);
            }

            // 3b. Build batched decode
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

            // 3c. Upload to GPU using pre-allocated pool
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

            // 3d. Build InferenceState
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

            // Per-request sampling
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
                            cudaMemcpyAsync(d_penalty_tokens_, req->output_tokens.data(),
                                            rn * sizeof(int32_t), cudaMemcpyHostToDevice, dec_stream);
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

            // 3e. Execute forward pass (with CUDA Graph when enabled)
            std::vector<int32_t> tokens;
            Tensor decode_logits_out;

            // 3e. Execute forward pass (piecewise CUDA Graph: forward in graph,
            //     sampling always eager — per-batch-size graph pool avoids
            //     re-capture when continuous batching changes batch size)
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

            Tokenizer* tok = model_->tokenizer();

            // 3f. Extract logprobs
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
                    cudaMemcpyAsync(h_base + static_cast<size_t>(slot) * vocab_size,
                                    d_logits, vocab_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, dec_stream);
                    slot++;
                }
                cudaStreamSynchronize(dec_stream);

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

    // Restore prefill workspace
    ensure_prefill_workspace(executor_.get());

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
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
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
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
    bool req_has_penalties = (req->repetition_penalty != 1.0f ||
                              req->frequency_penalty != 0.0f ||
                              req->presence_penalty != 0.0f);
    // Think budget is now enforced device-side in post_decode_step_kernel.
    if (req->status == RequestStatus::DECODING && !req->output_tokens.empty() &&
        config_.use_cuda_graphs && !offload_mgr_ && !ssm_state_ && !gdn_state_ &&
        !config_.enable_speculative && !req->ignore_eos && !req_has_penalties) {
        int32_t first_token = req->output_tokens.back();
        Tokenizer* gtok = model_->tokenizer();
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

// =====================================================================
// Speculative decode shortcut
// =====================================================================

bool Engine::try_speculative_decode(
        std::vector<std::shared_ptr<Request>>& valid_decode, cudaStream_t stream) {
    if (valid_decode.size() != 1) return false;
    auto& req = valid_decode[0];

    auto accept_tokens = [&](const std::vector<int32_t>& tokens) {
        for (int32_t t : tokens) {
            req->output_tokens.push_back(t);
            track_think_state(*req, t);
            if (should_stop(*req, t) ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
                finish_request(req);
                break;
            }
        }
        kv_manager_->touch(req->id);
    };

    // Self-speculative
    if (self_spec_decoder_ && config_.enable_self_speculative) {
        int32_t last_token = req->output_tokens.empty()
            ? req->input_tokens.back() : req->output_tokens.back();
        auto tokens = self_spec_decoder_->step(
            last_token, req->context_len() - 1, req->id,
            req->temperature, req->top_p, req->top_k, req->seed, stream);
        accept_tokens(tokens);
        return true;
    }

    // N-gram speculative
    if (ngram_spec_decoder_ &&
        static_cast<int>(req->output_tokens.size()) >= config_.ngram_n) {
        int32_t last_token = req->output_tokens.back();
        auto sr = ngram_spec_decoder_->step(
            req, last_token, req->context_len() - 1, req->id, stream);
        if (!sr.tokens.empty()) {
            accept_tokens(sr.tokens);
            IMP_LOG_DEBUG("N-gram spec: drafted=%d accepted=%d (rate=%.0f%%)",
                          sr.n_drafted, sr.n_accepted,
                          ngram_spec_decoder_->acceptance_rate() * 100.0f);
            return true;
        }
    }

    return false;
}

// =====================================================================
// CUDA Graph decode helpers
// =====================================================================

int Engine::prepare_graph_loop(std::shared_ptr<Request>& req) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    int remaining = req->max_tokens - static_cast<int>(req->output_tokens.size());
    if (remaining <= 0) return 0;

    constexpr int kMaxLayersForConditionalGraph = 128;
    if (model_->config().n_layers > kMaxLayersForConditionalGraph) return 0;

    { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
      if (f < 256ULL * 1024 * 1024) return 0; }

    // Pre-allocate KV blocks
    int ctx_len = req->context_len();
    int final_ctx = ctx_len + remaining;
    int blocks_needed = (final_ctx + kv_bs - 1) / kv_bs;
    int blocks_have = static_cast<int>(kv_manager_->block_table(req->id).size());

    for (int b = blocks_have; b < blocks_needed; b++) {
        if (kv_manager_->append_block(req->id) < 0) break;
    }

    int blocks_got = static_cast<int>(kv_manager_->block_table(req->id).size());
    int capped = blocks_got * kv_bs - ctx_len;
    if (capped <= 0) return 0;
    return std::min(capped, remaining);
}

CudaGraphConditionalRunner::Config Engine::build_graph_config(
        const Request& req, int remaining) const {
    Tokenizer* tok = model_->tokenizer();
    CudaGraphConditionalRunner::Config gcfg;
    gcfg.max_steps = remaining;
    gcfg.initial_context_len = req.context_len();
    gcfg.initial_position = req.context_len() - 1;
    gcfg.eos_id = tok ? tok->eos_id() : -1;
    gcfg.stop_ids = chat_template_.stop_token_ids();
    // Banned tokens (e.g. <pad>, <unk>) should also stop the graph loop.
    // The ban_logits_kernel sets them to -1e30 before sampling, but if
    // they still leak through (FP32 precision edge cases with 262K vocab),
    // the stop check catches them as a safety net.
    for (int32_t bid : banned_token_ids_) {
        gcfg.stop_ids.push_back(bid);
    }
    gcfg.temperature = req.temperature;
    gcfg.top_p = req.top_p;
    gcfg.top_k = req.top_k;
    gcfg.seed = req.seed;
    // Think budget: device-side enforcement in post_decode_step_kernel
    if (req.think_budget > 0.0f && think_end_id_ >= 0) {
        gcfg.think_budget_limit = static_cast<int>(req.max_tokens * req.think_budget);
        gcfg.think_start_id = think_start_id_;
        gcfg.think_end_id = think_end_id_;
        gcfg.initial_in_think = req.in_think_block;
    }
    return gcfg;
}

std::vector<int32_t> Engine::try_graph_loop_decode(
        std::shared_ptr<Request> req, int32_t first_token, cudaStream_t stream) {
    int remaining = prepare_graph_loop(req);
    if (remaining <= 0) return {};

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    int* d_block_tables = nullptr;
    if (cudaMallocAsync(&d_block_tables, max_blocks_per_seq * sizeof(int), stream) != cudaSuccess)
        return {};
    cudaMemcpyAsync(d_block_tables, full_bt.data(),
                     max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice, stream);

    executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_block_tables;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Upload banned tokens for graph-captured logit masking
    int32_t* d_banned = nullptr;
    if (!banned_token_ids_.empty()) {
        if (cudaMallocAsync(&d_banned, banned_token_ids_.size() * sizeof(int32_t), stream) == cudaSuccess) {
            cudaMemcpyAsync(d_banned, banned_token_ids_.data(),
                            banned_token_ids_.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            state_template.d_banned_tokens = d_banned;
            state_template.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
        }
    }

    auto gcfg = build_graph_config(*req, remaining);

    CudaGraphConditionalRunner runner;
    if (!runner.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (d_banned) cudaFreeAsync(d_banned, stream);
        cudaFreeAsync(d_block_tables, stream);
        return {};
    }
    if (!runner.launch(stream)) {
        if (d_banned) cudaFreeAsync(d_banned, stream);
        cudaFreeAsync(d_block_tables, stream);
        return {};
    }

    auto tokens = runner.wait_and_get_tokens(stream);
    if (d_banned) cudaFreeAsync(d_banned, stream);
    cudaFreeAsync(d_block_tables, stream);
    IMP_LOG_INFO("ConditionalGraph: generated %zu tokens in graph loop", tokens.size());
    runner.cleanup();
    return tokens;
}

bool Engine::try_launch_async_graph_loop(std::shared_ptr<Request> req,
                                          int32_t first_token, cudaStream_t stream) {
    int remaining = prepare_graph_loop(req);
    if (remaining <= 0) return false;

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    int* d_bt = nullptr;
    if (cudaMalloc(&d_bt, max_blocks_per_seq * sizeof(int)) != cudaSuccess)
        return false;
    cudaMemcpyAsync(d_bt, full_bt.data(),
                     max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice, stream);

    executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_bt;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Upload banned tokens to device for graph-captured logit masking
    int32_t* d_banned = nullptr;
    if (!banned_token_ids_.empty()) {
        if (cudaMalloc(&d_banned, banned_token_ids_.size() * sizeof(int32_t)) == cudaSuccess) {
            cudaMemcpyAsync(d_banned, banned_token_ids_.data(),
                            banned_token_ids_.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            state_template.d_banned_tokens = d_banned;
            state_template.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
        }
    }

    auto gcfg = build_graph_config(*req, remaining);

    if (!async_graph_runner_.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (d_banned) cudaFree(d_banned);
        cudaFree(d_bt);
        return false;
    }
    if (!async_graph_runner_.launch(stream)) {
        async_graph_runner_.cleanup();
        if (d_banned) cudaFree(d_banned);
        cudaFree(d_bt);
        return false;
    }

    async_graph_req_ = req;
    async_d_block_tables_ = d_bt;
    async_d_banned_tokens_ = d_banned;
    IMP_LOG_DEBUG("AsyncGraphLoop: launched with %d banned tokens",
                  state_template.n_d_banned_tokens);
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
