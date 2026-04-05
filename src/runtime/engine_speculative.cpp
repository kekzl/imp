#include "runtime/engine.h"
#include "runtime/speculative.h"
#include "runtime/self_speculative.h"
#include "runtime/cuda_graph.h"
#include "memory/kv_cache.h"
#include "model/gguf_loader.h"
#include "model/chat_template.h"
#include "compute/sampling.h"
#include "core/logging.h"

#include <cstring>
#include <algorithm>
#include <vector>

namespace imp {

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
    gcfg.ignore_eos = req.ignore_eos;
    // Penalty parameters for device-side application inside the graph loop
    gcfg.repetition_penalty = req.repetition_penalty;
    gcfg.frequency_penalty = req.frequency_penalty;
    gcfg.presence_penalty = req.presence_penalty;
    gcfg.repeat_last_n = req.repeat_last_n;
    // Seed penalty history from existing output tokens
    if (req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
        req.presence_penalty != 0.0f) {
        if (!req.output_tokens.empty()) {
            gcfg.penalty_history = req.output_tokens;
        }
    }
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
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_block_tables, full_bt.data(),
                     max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice, stream));

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
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_banned, banned_token_ids_.data(),
                            banned_token_ids_.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream));
            state_template.d_banned_tokens = d_banned;
            state_template.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
        }
    }

    auto gcfg = build_graph_config(*req, remaining);

    CudaGraphConditionalRunner runner;
    if (!runner.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (d_banned) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
        return {};
    }
    if (!runner.launch(stream)) {
        if (d_banned) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
        return {};
    }

    auto tokens = runner.wait_and_get_tokens(stream);
    if (d_banned) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
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
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_bt, full_bt.data(),
                     max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice, stream));

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
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_banned, banned_token_ids_.data(),
                            banned_token_ids_.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream));
            state_template.d_banned_tokens = d_banned;
            state_template.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
        }
    }

    auto gcfg = build_graph_config(*req, remaining);

    if (!async_graph_runner_.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (d_banned) IMP_CUDA_CHECK_LOG(cudaFree(d_banned));
        IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
        return false;
    }
    if (!async_graph_runner_.launch(stream)) {
        async_graph_runner_.cleanup();
        if (d_banned) IMP_CUDA_CHECK_LOG(cudaFree(d_banned));
        IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
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

} // namespace imp
