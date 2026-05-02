#include "runtime/engine.h"
#include "runtime/cuda_graph.h"
#include "memory/kv_cache.h"
#include "model/chat_template.h"
#include "compute/sampling.h"
#include "core/logging.h"

#include <cstring>
#include <algorithm>
#include <vector>

namespace imp {

// =====================================================================
// CUDA Graph decode helpers
// (Filename is historical; the speculative-decode variants that originally
// lived here were removed when proven broken/unused. The async-graph and
// conditional-graph helpers below are the production decode path.)
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

    (void)executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_block_tables;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Recurrent state for SSM/GDN layers — pointers are constant for
    // single-sequence decode, so they're safe to bake into the graph.
    fill_recurrent_state(*req, state_template, /*reset=*/false, stream);

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

    (void)executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_bt;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Recurrent state for SSM/GDN layers — pointers are constant for
    // single-sequence decode, so they're safe to bake into the graph.
    fill_recurrent_state(*req, state_template, /*reset=*/false, stream);

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
