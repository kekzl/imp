#include "runtime/engine.h"
#include "runtime/speculative.h"
#include "runtime/batch.h"
#include "memory/kv_cache.h"
#include "model/gguf_loader.h"
#include "model/chat_template.h"
#include "compute/gemm.h"
#include "core/logging.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace imp {

Engine::~Engine() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

cudaStream_t Engine::prefill_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available())
           ? green_ctx_.prefill_stream() : stream_;
}

cudaStream_t Engine::decode_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available())
           ? green_ctx_.decode_stream() : stream_;
}

bool Engine::init(std::shared_ptr<Model> model, const EngineConfig& config) {
    if (!model) {
        return false;
    }

    model_ = std::move(model);
    config_ = config;

    const auto& mcfg = model_->config();

    // --- Pre-allocate cuBLAS/cuBLASLt workspace while GPU memory is plentiful ---
    gemm_init();

    // --- Initialize scheduler ---
    scheduler_ = std::make_unique<Scheduler>(config_.max_batch_size);

    // --- Initialize graph executor (allocates activation workspace) ---
    executor_ = std::make_unique<GraphExecutor>();
    if (!executor_->init(*model_, config_.compute_dtype, config_.use_pdl,
                         config_.max_batch_size, config_.max_seq_len)) {
        return false;
    }

    // --- Create CUDA stream ---
    cudaError_t stream_err = cudaStreamCreate(&stream_);
    if (stream_err != cudaSuccess) {
        stream_ = nullptr;
    }

    // --- Upload model weights to GPU ---
    {
        size_t free_before = 0, total_before = 0;
        cudaMemGetInfo(&free_before, &total_before);
        IMP_LOG_INFO("GPU memory before weight upload: %zu MiB free / %zu MiB total",
                     free_before / (1024 * 1024), total_before / (1024 * 1024));

        if (!model_->upload_weights_gpu(config_.compute_dtype, stream_)) {
            IMP_LOG_ERROR("Weight upload failed. Model may be too large for GPU. "
                          "Try a smaller quantization (e.g. Q4_K_M instead of Q6_K).");
            return false;
        }

        size_t free_after = 0, total_after = 0;
        cudaMemGetInfo(&free_after, &total_after);
        IMP_LOG_INFO("GPU memory after weight upload: %zu MiB free / %zu MiB total "
                     "(weights used ~%zu MiB)",
                     free_after / (1024 * 1024), total_after / (1024 * 1024),
                     (free_before - free_after) / (1024 * 1024));
    }

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
        size_t elem_size = dtype_size(config_.compute_dtype);
        size_t single_block_bytes = static_cast<size_t>(kKVBlockSize) *
                                    mcfg.n_kv_heads * head_dim * elem_size;
        size_t per_block_total = single_block_bytes * 2 * n_kv_layers;

        // Target: 2x headroom for batching / prefix caching
        int target_blocks = needed_blocks * 2;

        size_t free_mem = 0, total_mem = 0;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err != cudaSuccess || per_block_total == 0) {
            max_blocks = needed_blocks;
        } else {
            // Don't exceed 80% of remaining free VRAM
            size_t kv_budget = static_cast<size_t>(free_mem * 0.8);
            int budget_blocks = static_cast<int>(kv_budget / per_block_total);
            max_blocks = std::min(target_blocks, budget_blocks);
        }

        max_blocks = std::max(max_blocks, needed_blocks);  // at least what's needed
        max_blocks = std::max(max_blocks, 16);
    }

    {
        size_t elem_size = dtype_size(config_.compute_dtype);
        size_t block_bytes = static_cast<size_t>(kKVBlockSize) *
                             mcfg.n_kv_heads * head_dim * elem_size;
        size_t total_kv = static_cast<size_t>(n_kv_layers) * max_blocks * 2 * block_bytes;
        IMP_LOG_INFO("KV cache: %d blocks (%.0f tokens), %.2f MiB "
                     "(layers=%d/%d, kv_heads=%d, head_dim=%d, block_size=%d)",
                     max_blocks,
                     static_cast<double>(max_blocks) * kKVBlockSize,
                     static_cast<double>(total_kv) / (1024.0 * 1024.0),
                     n_kv_layers, mcfg.n_layers, mcfg.n_kv_heads, head_dim, kKVBlockSize);
    }

    auto kv_cache = std::make_unique<KVCache>(
        n_kv_layers, mcfg.n_kv_heads, head_dim,
        config_.compute_dtype, max_blocks);

    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));

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
        if (!green_ctx_.init(0, 0.8f)) {
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

bool Engine::step() {
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
        int ctx_len = req->context_len();
        int n_tokens = static_cast<int>(req->input_tokens.size());

        // Resize workspace for prefill token count (may be large)
        executor_->resize_workspace(n_tokens, pf_stream);

        int num_blocks = (ctx_len + kKVBlockSize - 1) / kKVBlockSize;

        // Allocate KV cache blocks
        if (!kv_manager_->allocate_blocks(req->id, num_blocks)) {
            while (kv_manager_->num_free_blocks() < num_blocks) {
                int evicted = kv_manager_->evict_lru();
                if (evicted < 0) break;
            }
            if (!kv_manager_->allocate_blocks(req->id, num_blocks)) {
                req->status = RequestStatus::CANCELLED;
                continue;
            }
        }

        const auto& block_table = kv_manager_->block_table(req->id);

        std::vector<int> positions(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            positions[i] = i;
        }

        // Upload to device
        int32_t* d_token_ids = nullptr;
        int* d_positions = nullptr;
        int* d_block_tables = nullptr;
        int* d_context_lens = nullptr;

        cudaMalloc(&d_token_ids, n_tokens * sizeof(int32_t));
        cudaMalloc(&d_positions, n_tokens * sizeof(int));
        cudaMalloc(&d_block_tables, block_table.size() * sizeof(int));
        cudaMalloc(&d_context_lens, sizeof(int));

        cudaMemcpyAsync(d_token_ids, req->input_tokens.data(),
                        n_tokens * sizeof(int32_t),
                        cudaMemcpyHostToDevice, pf_stream);
        cudaMemcpyAsync(d_positions, positions.data(),
                        n_tokens * sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream);
        cudaMemcpyAsync(d_block_tables, block_table.data(),
                        block_table.size() * sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream);
        cudaMemcpyAsync(d_context_lens, &ctx_len, sizeof(int),
                        cudaMemcpyHostToDevice, pf_stream);

        // Build InferenceState (single-sequence prefill)
        InferenceState state;
        state.token_ids = d_token_ids;
        state.positions = d_positions;
        state.n_tokens = n_tokens;
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

        // SSM state for hybrid models
        if (ssm_state_) {
            state.ssm_state = ssm_state_.get();
            // Use request ID mod max_sequences as SSM sequence slot
            state.ssm_seq_id = req->id % ssm_state_->max_sequences();
            ssm_state_->reset_sequence(state.ssm_seq_id, pf_stream);
        }

        int32_t next_token = executor_->forward(state, pf_stream);

        cudaStreamSynchronize(pf_stream);

        cudaFree(d_token_ids);
        cudaFree(d_positions);
        cudaFree(d_block_tables);
        cudaFree(d_context_lens);

        req->output_tokens.push_back(next_token);

        Tokenizer* tok = model_->tokenizer();
        IMP_LOG_INFO("Prefill -> token %d (ctx=%d): id=%d [%s]",
                     (int)req->output_tokens.size(), req->context_len(),
                     next_token, tok->decode_token(next_token).c_str());

        // Check EOS and chat template stop tokens
        bool is_stop = (next_token == tok->eos_id());
        for (int32_t stop_id : chat_template_.stop_token_ids()) {
            if (next_token == stop_id) { is_stop = true; break; }
        }

        if (is_stop ||
            static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            req->status = RequestStatus::FINISHED;
            kv_manager_->free_sequence(req->id);
        } else {
            req->status = RequestStatus::DECODING;
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

            // SSM state for hybrid models (decode uses first sequence's slot)
            if (ssm_state_) {
                state.ssm_state = ssm_state_.get();
                state.ssm_seq_id = valid_decode[0]->id % ssm_state_->max_sequences();
            }

            // 3e. Execute batched forward pass (with CUDA Graph when enabled)
            std::vector<int32_t> tokens;

            if (config_.use_cuda_graphs && gpu_batch.n_sequences > 0 &&
                decode_batch_pool_.is_allocated()) {
                // Invalidate graph when batch config changes
                if (gpu_batch.n_sequences != last_decode_batch_size_ ||
                    gpu_batch.max_blocks_per_seq != last_decode_max_blocks_) {
                    decode_graph_runner_.invalidate();
                    last_decode_batch_size_ = gpu_batch.n_sequences;
                    last_decode_max_blocks_ = gpu_batch.max_blocks_per_seq;
                }

                // Set the compute-only forward pass as the graph body
                Tensor logits_out;
                decode_graph_runner_.set_decode_fn(
                    [this, &state, &logits_out](cudaStream_t s) {
                        executor_->forward_logits(state, logits_out, s);
                    });
                decode_graph_runner_.execute(dec_stream);

                // Sample outside the graph (sampling has host sync)
                tokens = executor_->forward_batch(state, dec_stream);
            } else {
                tokens = executor_->forward_batch(state, dec_stream);
            }

            cudaStreamSynchronize(dec_stream);

            // Only free if not using pool (pool memory is reused)
            if (!decode_batch_pool_.is_allocated()) {
                gpu_batch.free();
            }

            // 3f. Distribute sampled tokens back to requests
            Tokenizer* tok = model_->tokenizer();
            for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
                auto& req = valid_decode[i];
                int32_t next_token = tokens[i];

                req->output_tokens.push_back(next_token);

                IMP_LOG_INFO("Decode step %d (ctx=%d, pos=%d): id=%d [%s]",
                             (int)req->output_tokens.size(), req->context_len(),
                             req->context_len() - 1,
                             next_token, tok->decode_token(next_token).c_str());

                // Check EOS and chat template stop tokens
                bool is_stop = (next_token == tok->eos_id());
                for (int32_t stop_id : chat_template_.stop_token_ids()) {
                    if (next_token == stop_id) { is_stop = true; break; }
                }

                if (is_stop ||
                    static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
                    req->status = RequestStatus::FINISHED;
                    kv_manager_->free_sequence(req->id);
                }

                kv_manager_->touch(req->id);
            }
        }
    }

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
}

std::string Engine::generate(const std::string& prompt, int max_tokens,
                              float temperature, float top_p,
                              int top_k, int seed,
                              bool apply_chat_template) {
    Tokenizer* tok = model_->tokenizer();
    if (!tok) {
        return "";
    }

    std::vector<int32_t> tokens;

    if (apply_chat_template && !chat_template_.is_raw()) {
        // Apply detected chat template
        std::vector<ChatMessage> messages = {{"user", prompt}};
        tokens = chat_template_.apply(*tok, messages);
        IMP_LOG_INFO("Applied %s chat template (%zu tokens)",
                     chat_template_family_name(chat_template_.family()),
                     tokens.size());
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
    req->status = RequestStatus::PENDING;

    scheduler_->add_request(req);

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

    std::string result = tok->decode(req->output_tokens);
    return result;
}

void Engine::add_request(std::shared_ptr<Request> req) {
    if (scheduler_) {
        req->id = next_request_id_++;
        scheduler_->add_request(std::move(req));
    }
}

} // namespace imp
